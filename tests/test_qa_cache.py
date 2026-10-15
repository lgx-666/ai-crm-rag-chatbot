# tests/test_qa_cache.py —— 纯本地单测：假 Redis 替身 + monkeypatch，不打网络、不调 LLM
import json
import sys
import qa_cache as qc
from config import PROMPT_TEMPLATES

class FakeRedis:
    """最小替身：只实现 qa_cache 用到的 get/setex/dbsize，用来断言 key/TTL/payload。"""

    def __init__(self, error=None):
        self.store = {}
        self.ttl = {}
        self.error = error

    def _boom(self):
        if self.error is not None:
            raise self.error

    def get(self, key):
        self._boom()
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self._boom()
        self.store[key] = value
        self.ttl[key] = ttl
        return True

    def dbsize(self):
        self._boom()
        return len(self.store)


def _use_fake(monkeypatch, fake=None):
    """开缓存 + 换成替身。重置 _resolved/_client 是为了绕过 _redis 的懒连接单例，
    否则前一个用例建的连接会漏到下一个用例。"""
    fake = fake if fake is not None else FakeRedis()
    monkeypatch.setattr(qc, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(qc, "_resolved", False)
    monkeypatch.setattr(qc, "_client", None)
    monkeypatch.setattr(qc, "_redis", lambda: fake)
    return fake



# ---------- 归一化与 key ----------
def test_normalize_等价打字差异():
    assert qc.normalize("  RAG  是什么？ ") == qc.normalize("RAG是什么")
    assert qc.normalize("RAG是什么") == qc.normalize("RAG是什么!!")


def test_normalize_全半角与大小写():
    assert qc.normalize("ＪＡＶＡ八股") == qc.normalize("java八股")


def test_key_同义问题同key_不同问题异key():
    assert qc._key("RAG 是什么？") == qc._key("rag是什么")
    assert qc._key("rag是什么") != qc._key("rag怎么调优")
    assert qc._key("rag是什么").startswith(qc.NAMESPACE)


# ---------- 开关：没配 REDIS_URL 时不能碰网络 ----------
def test_未配置REDIS_URL时不建连接(monkeypatch):
    """门禁在 _redis() 内部第 48 行，所以要测的是真 _redis 本身，不能拿桩替换它
    （替换掉就等于把要验证的东西绕过去了）。
    证法：塞一个假 redis 模块进 sys.modules，谁真去 from_url 就当场炸。
    get/put/size 会调 _redis() 是正确行为，_redis() 在建连之前就 return None。"""
    monkeypatch.setattr(qc, "REDIS_URL", "")
    monkeypatch.setattr(qc, "_resolved", False)
    monkeypatch.setattr(qc, "_client", None)

    calls = []

    class BannedRedis:
        @staticmethod
        def from_url(*args, **kwargs):
            calls.append(1)
            raise AssertionError("关缓存时不该建 Redis 连接")

    monkeypatch.setitem(sys.modules, "redis", BannedRedis)
    assert qc.enabled() is False
    assert qc.get("rag是什么") is None
    assert qc.put("rag是什么", "答案", 1) is False
    assert qc.size() == 0
    assert calls == []


def test_redis连不上时降级成没缓存而不是报错(monkeypatch):
    """覆盖 _redis() 的 except 分支（第 56-57 行）：URL 配了但 Redis 没起，
    业务侧只该看到"缓存没命中"，不该看到异常，也不该每次都重试建连拖慢响应。"""
    monkeypatch.setattr(qc, "REDIS_URL", "redis://127.0.0.1:6399/0")
    monkeypatch.setattr(qc, "_resolved", False)
    monkeypatch.setattr(qc, "_client", None)

    class UnreachableRedis:
        @staticmethod
        def from_url(*args, **kwargs):
            raise ConnectionError("Connection refused")

    monkeypatch.setitem(sys.modules, "redis", UnreachableRedis)
    assert qc._redis() is None
    assert qc._redis() is None          # 第二次走单例缓存，不再重建连
    assert qc.get("rag是什么") is None
    assert qc.put("rag是什么", "答案", 1) is False



# ---------- 命中路径 ----------
def test_put然后get命中且TTL与intent正确(monkeypatch):
    fake = _use_fake(monkeypatch)
    assert qc.put("RAG 是什么？", "RAG 是检索增强生成。", 1) is True

    hit = qc.get("rag是什么")          # 打字差异要能命中同一条
    assert hit is not None
    assert hit["answer"] == "RAG 是检索增强生成。"
    assert hit["intent"] == 1
    assert list(fake.ttl.values()) == [qc.TTL_SECONDS]
    assert json.loads(list(fake.store.values())[0])["query"] == "RAG 是什么？"


def test_写入门禁_非意图1不落缓存(monkeypatch):
    fake = _use_fake(monkeypatch)
    for intent in (0, 2, 3, 4):
        assert qc.put("你好", "你好呀", intent) is False
    assert fake.store == {}


def test_空答案不写(monkeypatch):
    fake = _use_fake(monkeypatch)
    assert qc.put("rag是什么", "   ", 1) is False
    assert qc.put("rag是什么", None, 1) is False
    assert fake.store == {}


# ---------- 降级：脏数据与 Redis 故障都不许抛 ----------
def test_缓存内容损坏时按未命中处理(monkeypatch):
    fake = _use_fake(monkeypatch)
    fake.store[qc._key("rag是什么")] = "{不是合法json"
    assert qc.get("rag是什么") is None


def test_意图被篡改时不采信(monkeypatch):
    fake = _use_fake(monkeypatch)
    fake.store[qc._key("rag是什么")] = json.dumps({"answer": "答案", "intent": 3})
    assert qc.get("rag是什么") is None


def test_redis抛异常时get与put都不带崩调用方(monkeypatch):
    _use_fake(monkeypatch, FakeRedis(error=ConnectionError("redis down")))
    assert qc.get("rag是什么") is None
    assert qc.put("rag是什么", "答案", 1) is False
    assert qc.size() == 0



# ---------- 画像签名（scope）：防跨用户串内容 ----------
def test_同问题不同画像不共享缓存(monkeypatch):
    fake = _use_fake(monkeypatch)
    assert qc.put("rag是什么", "答案A", 1, scope="aaa") is True
    assert qc.get("rag是什么", scope="bbb") is None      # 不许把 A 画像生成的答案推给 B
    hit = qc.get("rag是什么", scope="aaa")
    assert hit is not None and hit["answer"] == "答案A"


def test_兜底道歉文案不进缓存(monkeypatch):
    """_stream_llm 抛异常时道歉是当成答案 yield 的（response_generator.py:204），
    它在 state["answer"] 里和正常答案长得一样，必须在写端按内容拦掉。"""
    fake = _use_fake(monkeypatch)
    apology = PROMPT_TEMPLATES["fallback"]
    assert qc.put("rag是什么", apology, 1) is False
    assert qc.put("rag是什么", "前半段是正常答案。" + apology, 1) is False
    assert fake.store == {}


def test_clear_返回条数并让缓存失效(monkeypatch):
    class FakeWithFlush(FakeRedis):
        def __init__(self):
            super().__init__()
            self.flushed = 0

        def flushdb(self):
            self.flushed += 1
            n = len(self.store)
            self.store.clear()
            return n

    fake = FakeWithFlush()
    _use_fake(monkeypatch, fake)
    qc.put("rag是什么", "答案", 1)
    assert qc.size() == 1
    assert qc.clear() == 1
    assert fake.flushed == 1
    assert qc.get("rag是什么") is None    