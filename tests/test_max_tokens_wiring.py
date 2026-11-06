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