# tests/test_truncation_signal.py
# 0 token：既不出网也不连 Langfuse（generation 被换成记录器）。
# 为什么值得单独测：max_tokens 截断过去是隐形的——finish_reason 全项目没人读，
# Langfuse 里只能靠"答案刚好停在 500 token"反推。这里钉住"截断会被记成 truncated=True"。
from types import SimpleNamespace

import response_generator


class _Recorder:
    """替掉 observability.generation，只把 end() 收到的 metadata 记下来。"""

    def __init__(self, sink):
        self._sink = sink

    def end(self, output, usage=None, error=None, metadata=None):
        self._sink.append(metadata or {})


def _run(monkeypatch, events):
    """跑一次 _stream_llm，返回 (拼出的正文, gen.end 收到的 metadata)。"""
    sink = []
    monkeypatch.setattr(response_generator, "generation",
                        lambda *a, **kw: _Recorder(sink))
    monkeypatch.setattr(response_generator.client, "chat", SimpleNamespace(
        completions=SimpleNamespace(create=lambda **kw: iter(events))))
    text = "".join(response_generator._stream_llm(
        [{"role": "user", "content": "hi"}], 0.7, 500))
    return text, sink[0]


def _events(finish_reason):
    return [
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="你好"), finish_reason=None)]),
        SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content=""), finish_reason=finish_reason)]),
        SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=10, completion_tokens=2, total_tokens=12)),
    ]


def test_被上限截断时记为truncated(monkeypatch):
    text, meta = _run(monkeypatch, _events("length"))

    assert text == "你好"
    assert meta["finish_reason"] == "length"
    assert meta["truncated"] is True


def test_正常收尾不算截断(monkeypatch):
    text, meta = _run(monkeypatch, _events("stop"))

    assert text == "你好"
    assert meta["finish_reason"] == "stop"
    assert meta["truncated"] is False


def test_事件没有finish_reason字段也不炸(monkeypatch):
    """护栏：test_thinking_toggle.py 的 _FakeStream 就是这种形状。"""
    events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="你"))]),
        SimpleNamespace(choices=[], usage=SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2)),
    ]

    text, meta = _run(monkeypatch, events)

    assert text == "你"
    assert meta["finish_reason"] is None
    assert meta["truncated"] is False