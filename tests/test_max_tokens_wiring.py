# tests/test_max_tokens_wiring.py
# 0 token：_stream_llm 与 bocha_search 都换成桩，不出网、不打模型、不碰 sqlite。
# 为什么值得钉：#38 之后 finish_reason/truncated 已经能筛了，但"哪一档接到哪个调用点"
# 一直没人守——把 response_generator.py:134 的 CHAT_MAX_TOKENS 误写成 RAG_MAX_TOKENS，
# 全量测试仍然全绿。第 3 条钉住"改模块级变量会真的传下去"，因为付费跑的阳性对照
# 正是靠这个手段制造必然截断；若它是 import 时绑死的，那个对照就是空的。
import pytest

import response_generator as rg


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_stream_llm(messages, temperature, max_tokens):
        calls.append((temperature, max_tokens))
        yield ""

    monkeypatch.setattr(rg, "_stream_llm", fake_stream_llm)
    monkeypatch.setattr(rg, "get_user_info_from_db", lambda p: None)
    # response_generator.py:142 在调 bocha_search 之前先判 BOCHA_API_KEY，CI 上没有 .env
    # ⇒ 只打桩 bocha_search 会短路成"未配置"、captured 为空（run #27 实测：本地绿、CI 红）。
    monkeypatch.setattr(rg, "BOCHA_API_KEY", "test-key-not-real")
    monkeypatch.setattr(rg, "bocha_search", lambda q, count=5: {
        "code": 200,
        "data": {"webPages": {"value": [
            {"name": "t", "summary": "s", "url": "u", "siteName": "n"}]}}})
    return calls


def test_闲聊档接到CHAT上限(captured):
    list(rg.stream_other_intents("q", "eval_user", history=[], summary=""))

    assert captured == [(0.7, rg.CHAT_MAX_TOKENS)]


def test_联网档接到WEB上限(captured):
    list(rg.stream_web_search("q", "eval_user", history=[], summary=""))

    assert captured == [(0.5, rg.WEB_MAX_TOKENS)]


def test_改模块级CHAT上限会真的传下去(captured, monkeypatch):
    monkeypatch.setattr(rg, "CHAT_MAX_TOKENS", 5)

    list(rg.stream_other_intents("q", "eval_user", history=[], summary=""))

    assert captured == [(0.7, 5)]

# ---------------------------------------------------------- #59 追加：意图档/摘要档接线
# 0 token：client.chat.completions.create 与 generation 全是桩，不出网、不连 Langfuse。
# 为什么值得钉：#38 之后 RAG/闲聊/联网三档有 finish_reason/truncated 可筛，但意图档和摘要档
# 的上限既写死在调用点、又各在"记进 trace 的参数"里重复写了一遍（双写）——两处不同步时，
# Langfuse 显示的上限跟真正生效的上限就不是同一个数。这两条用例把"两处同步"钉成不变量。
# 一个真实的坑：intent.py 的 `from observability import generation` 写在函数体里，打桩必须
# 打在 observability 上；response_generator.py 是模块级 import，必须打在 rg 上。打错目标，
# 一个静默用真函数（ENABLED=False 时空操作，测不出），另一个 AttributeError。
import types as _types

import intent as it_mod
import observability


class _GenSink:
    """替掉 observability.generation：parameters 与 end(metadata) 都留下来。"""

    def __init__(self, box):
        self.box = box

    def end(self, output, usage=None, error=None, metadata=None):
        self.box["output"] = output
        self.box["metadata"] = metadata or {}


def _stub_gen(monkeypatch, box, target):
    def generation(name, model, messages, parameters=None):
        box["name"] = name
        box["parameters"] = dict(parameters or {})
        return _GenSink(box)

    monkeypatch.setattr(target, "generation", generation)


def _stub_openai(monkeypatch, mod, box, content="1", finish="stop"):
    """假端点：实参记进 box["call"]，finish_reason 摆在 choice 上（真端点就是这么给的）。"""
    def create(**kw):
        box["call"] = kw
        return _types.SimpleNamespace(
            choices=[_types.SimpleNamespace(
                message=_types.SimpleNamespace(content=content),
                finish_reason=finish)],
            usage=None)

    monkeypatch.setattr(mod.client, "chat", _types.SimpleNamespace(
        completions=_types.SimpleNamespace(create=create)))


def _stub_summary(monkeypatch, box, finish="stop"):
    """把 update_and_get_summary 推到"真的调一次 LLM"那条分支所需的状态全配齐。"""
    saved = []
    _stub_gen(monkeypatch, box, rg)
    _stub_openai(monkeypatch, rg, box, content="新摘要", finish=finish)
    monkeypatch.setattr(rg, "get_conversation_summary", lambda p: ("旧摘要", 5))
    monkeypatch.setattr(rg, "count_chat_history", lambda p: 21)      # > SUMMARY_THRESHOLD
    monkeypatch.setattr(rg, "fetch_history_to_summarize",
                        lambda p, upto, w: ([{"role": "user", "content": "x"}], upto + 1))
    monkeypatch.setattr(rg, "save_conversation_summary",
                        lambda p, s, u: saved.append((p, s, u)))
    return saved


def test_意图档接到INTENT上限且改常量两处同步(monkeypatch):
    box = {}
    _stub_gen(monkeypatch, box, observability)
    _stub_openai(monkeypatch, it_mod, box)

    assert it_mod.detect_intent_code("你好") == 1
    assert box["call"]["max_tokens"] == it_mod.INTENT_MAX_TOKENS
    assert box["parameters"]["max_tokens"] == it_mod.INTENT_MAX_TOKENS

    # 抬一次：实参与记录必须一起跟着走。只补其中一处的那半边红在这里。
    monkeypatch.setattr(it_mod, "INTENT_MAX_TOKENS", 7)
    assert it_mod.detect_intent_code("你好") == 1
    assert box["call"]["max_tokens"] == 7
    assert box["parameters"]["max_tokens"] == 7


def test_意图档被上限切掉时记为truncated(monkeypatch):
    box = {}
    _stub_gen(monkeypatch, box, observability)
    _stub_openai(monkeypatch, it_mod, box, content="1", finish="length")

    it_mod.detect_intent_code("你好")

    assert box["metadata"]["finish_reason"] == "length"
    assert box["metadata"]["truncated"] is True


def test_摘要档接到SUMMARY上限且改常量两处同步(monkeypatch):
    box = {}
    saved = _stub_summary(monkeypatch, box)

    assert rg.update_and_get_summary("eval_user") == "新摘要"
    assert box["call"]["max_tokens"] == rg.SUMMARY_MAX_TOKENS
    assert box["parameters"]["max_tokens"] == rg.SUMMARY_MAX_TOKENS
    assert saved == [("eval_user", "新摘要", 6)]

    monkeypatch.setattr(rg, "SUMMARY_MAX_TOKENS", 9)
    assert rg.update_and_get_summary("eval_user") == "新摘要"
    assert box["call"]["max_tokens"] == 9
    assert box["parameters"]["max_tokens"] == 9


def test_摘要被上限切掉时记为truncated(monkeypatch):
    box = {}
    saved = _stub_summary(monkeypatch, box, finish="length")

    rg.update_and_get_summary("eval_user")

    assert box["metadata"]["finish_reason"] == "length"
    assert box["metadata"]["truncated"] is True
    assert saved, "截断的摘要照样落库（这次只加观测，不改这个行为）"