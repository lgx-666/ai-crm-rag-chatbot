# qa_cache.py —— 问答缓存：把"问过的问题"直接映射到"已经生成好的答案"。
#
# 为什么读缓存要放在 classify 之前：意图分类本身要调一次 LLM，命中就同时省掉
# 意图分类 + 答案生成两次调用，是唯一既降延迟又降账单的位置。
# 安全性依据：detect_intent_code(query) 只吃 query、不看历史，是 query 的纯函数；
# 又只缓存首轮，所以"同一问题 => 同一意图 => 同一答案"成立，读端不必先分类。
#
# 开关与降级（照 observability.py 的写法）：REDIS_URL 为空时全部方法退化成 no-op；
# Redis 任何异常一律吞掉当成"没命中"。缓存不能把一次本来能成功的回答变成失败。
# 缓存边界（两条硬约束）：
#   1) 答案文本受"用户画像"影响（response_generator.py:89-93、102），所以画像签名必须
#      进 key（scope）；不带 scope 的全局缓存会把 A 的带画像答案推给 B。
#   2) LLM 调用失败时 _stream_llm 会把兜底道歉当成答案 yield 出来
#      （response_generator.py:201-204），这种"负面结果"一律不写缓存。
import hashlib
import json
import time
import unicodedata

from config import REDIS_URL, PROMPT_TEMPLATES

# v1 写进前缀：归一化规则或 payload 结构以后改了，换前缀就能让旧缓存整体失效
NAMESPACE = "aicrm:qa:v1:"
TTL_SECONDS = 3600
CACHEABLE_INTENTS = {1}      # 写入门禁：只缓存意图1（知识库问答）

_client = None
_resolved = False


def normalize(query):
    """把同一个问题的各种打字差异归一到同一个串，作为 key 的输入。
    顺序：NFKC（全角→半角）→ casefold → 去掉所有空白 → 去掉首尾标点。
    去空白是因为中文问题里 "RAG 是什么" 和 "RAG是什么" 必须命中同一条；
    代价是只差空格的两个英文问题会撞在一起，可以接受。"""
    s = unicodedata.normalize("NFKC", query or "").casefold()
    s = "".join(s.split())
    return s.strip("？！。，、；：,.;:!?~～\"'“”‘’()（）[]【】<>《》")


def _key(query, scope=""):
    """定长、无特殊字符，且原始问题不出现在 key 里（问题可能带手机号）。
    \x00 做分隔符：避免"问题尾部字符 + 签名首字符"拼出同一个串的歧义。"""
    raw = normalize(query) + "\x00" + (scope or "")
    return NAMESPACE + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]



def _redis():
    """懒连接单例。不在 import 时建连：那样 Redis 一缺席，app.py 直接起不来，
    CI 也得被迫装一个 Redis。0.5 秒超时是刻意的——读缓存不该拖慢首 token。"""
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True
    if not REDIS_URL:
        return None
    try:
        import redis
        client = redis.from_url(REDIS_URL, decode_responses=True,
                                socket_connect_timeout=0.5, socket_timeout=0.5)
        client.ping()
        _client = client
    except Exception:
        _client = None
    return _client


def enabled():
    return bool(REDIS_URL)


def get(query, scope=""):
    """命中返回 payload dict（含 answer/intent/created_at），未命中或降级返回 None。"""
    client = _redis()
    if client is None:
        return None
    try:
        raw = client.get(_key(query, scope))
        if not raw:
            return None
        payload = json.loads(raw)
    except Exception:
        return None
    if payload.get("intent") not in CACHEABLE_INTENTS or not payload.get("answer"):
        return None
    return payload


def put(query, answer, intent, scope=""):
    """写缓存。四道门禁：Redis 在、意图可缓存、答案非空、答案不含兜底文案。"""
    client = _redis()
    if client is None or intent not in CACHEABLE_INTENTS:
        return False
    text = (answer or "").strip()
    if not text:
        return False
    # 负面结果进缓存 = 把偶发故障固化成一小时；比空答案更危险，因为它看起来像缓存生效
    if PROMPT_TEMPLATES["fallback"] in text:
        return False
    try:
        return bool(client.setex(_key(query, scope), TTL_SECONDS, json.dumps(
            {"answer": text, "intent": intent, "query": query,
            "created_at": time.time()}, ensure_ascii=False)))
    except Exception:
        return False

def size():
    """给验收脚本看的条目数；未开启/异常都算 0。"""
    client = _redis()
    if client is None:
        return 0
    try:
        return client.dbsize()
    except Exception:
        return 0
    



def clear():
    """清空本服务用的 db0（compose 里 REDIS_URL 就是 /0，这个库由问答缓存独占）。
    只在验收脚本和"知识库灌完要让旧答案失效"时用，业务路径不调它。"""
    client = _redis()
    if client is None:
        return 0
    try:
        return int(client.flushdb())
    except Exception:
        return 0