# tests/test_graph_cache.py —— 图级接线验证：打桩意图分类与 LLM 生成，零 token、不碰 DB/网络
import json
import pytest

import graph_flow as gf
import qa_cache as qc


class DictRedis:
    """内存版 Redis，只实现 qa_cache 用到的三个方法。"""

    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def dbsize(self):
        return len(self.store)


@pytest.fixture
def stub(monkeypatch):
    """把 graph_flow 里"会花钱/会落盘"的协作者全部换成假货。
    注意：chat_graph 在 import 时就把节点函数编进图了，所以只能打桩节点*内部调用*的名字
    （detect_intent_code / stream_response_from_vectorstore / 那几个 db 函数），
    打桩 gf.rag_node 本身是无效的——它已经不在调用路径上了。"""
    box = {"redis": DictRedis(), "classify": 0, "rag": 0,
           "history": {}, "written": []}

    monkeypatch.setattr(gf, "fetch_chat_history",
                        lambda phone, limit=None: list(box["history"].get(phone, [])))
    monkeypatch.setattr(gf, "count_chat_history",
                        lambda phone: len(box["history"].get(phone, [])))

    def fake_update(phone, role, content):
        box["written"].append((phone, role, content))
        box["history"].setdefault(phone, []).append({"role": role, "content": content})

    monkeypatch.setattr(gf, "update_chat_history", fake_update)
    monkeypatch.setattr(gf, "update_and_get_summary", lambda phone: "")
    # 尾号 2 的用户填过姓名 => 画像不同 => 缓存不该共享
    monkeypatch.setattr(gf, "get_user_info_from_db", lambda phone: {
        "name": "张三" if phone.endswith("2") else "",
        "age": None, "occupation": "", "interest": ""})

    def fake_classify(query):
        box["classify"] += 1
        return 1

    def fake_stream(query, phone_number, history=None, summary=""):
        box["rag"] += 1
        yield from ["RAG", " 是", "检索增强生成。"]

    monkeypatch.setattr(gf, "detect_intent_code", fake_classify)
    monkeypatch.setattr(gf, "stream_response_from_vectorstore", fake_stream)
    monkeypatch.setattr(gf, "trace_meta", lambda *a, **kw: None)

    monkeypatch.setattr(qc, "REDIS_URL", "redis://fake:6379/0")
    monkeypatch.setattr(qc, "_resolved", False)
    monkeypatch.setattr(qc, "_client", None)
    monkeypatch.setattr(qc, "_redis", lambda: box["redis"])
    return box


def _run(query, phone="13800000001"):
    """消费整条 custom 流，返回前端实际拼到的文本。"""
    return "".join(chunk.get("token", "") for chunk in
                   gf.chat_graph.stream({"phone_number": phone, "query": query},
                                        stream_mode="custom"))


ANSWER = "RAG 是检索增强生成。"


def test_首轮未命中_走完整链路并写入缓存(stub):
    assert _run("RAG 是什么？") == ANSWER
    assert (stub["classify"], stub["rag"]) == (1, 1)
    assert stub["redis"].dbsize() == 1
    # key 必须含画像签名：A 四项全空，签名由 profile_scope 自己算，不写死常量
    assert qc._key("RAG 是什么？", gf.profile_scope("13800000001")) in stub["redis"].store
    assert json.loads(list(stub["redis"].store.values())[0])["intent"] == 1


def test_同问题再问一次_命中且两次LLM调用全省掉(stub):
    _run("RAG 是什么？")
    stub["classify"] = stub["rag"] = 0

    # 打字差异（大小写/空格/问号）也要命中，这正是 normalize 的职责
    assert _run("  rag是什么 ") == ANSWER
    assert (stub["classify"], stub["rag"]) == (0, 0)
    assert stub["redis"].dbsize() == 1              # 命中不重复写


def test_命中时历史里存的和流出去的一字不差(stub):
    _run("RAG 是什么？")
    stub["written"].clear()                         # 只看命中这一轮写了什么
    assert _run("RAG 是什么？") == ANSWER
    assert stub["written"] == [
        ("13800000001", "user", "RAG 是什么？"),
        ("13800000001", "assistant", ANSWER),
    ]


def test_非首轮的答案不写缓存(stub):
    _run("RAG 是什么？")                            # 首轮，写 1 条
    assert _run("JVM 的栈和堆有什么区别？") == ANSWER   # 已有历史 => 不写
    assert stub["rag"] == 2
    assert stub["redis"].dbsize() == 1


def test_画像不同的用户不共享缓存(stub):
    _run("RAG 是什么？", phone="13800000001")       # A：资料全空
    stub["classify"] = stub["rag"] = 0
    assert _run("RAG 是什么？", phone="13800000002") == ANSWER   # B：填过姓名
    assert (stub["classify"], stub["rag"]) == (1, 1)
    assert len(stub["redis"].store) == 2            # 各存一条，互不覆盖


def test_没配Redis时行为与改前逐字等价(stub, monkeypatch):
    """宿主和 CI 走的正是这条路：REDIS_URL 空 => cache_lookup 恒未命中。
    连续两次同样的问题都要真生成，一次缓存都不能有。"""
    monkeypatch.setattr(qc, "_redis", lambda: None)
    for _ in range(2):
        assert _run("RAG 是什么？") == ANSWER
    assert (stub["classify"], stub["rag"]) == (2, 2)
    assert stub["redis"].store == {}