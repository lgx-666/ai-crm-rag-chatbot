# tests/test_thinking_toggle.py
# 0 token：只替掉 intent.client.chat.completions.create 这一层出网调用，
# 意图编号的解析仍走真实的 parse_intent。
# 为什么值得单独测：百炼兼容端点只认 extra_body={"enable_thinking": False}，
# 顶层同名字段实测直接 TypeError（_lab_thinking_probe.py 三变体对照）。
# 传错位置不会报错、只会静默多想 —— 线上表现为"答案变慢变贵但没人知道为什么"。
from types import SimpleNamespace

import config
import intent


def _stub(monkeypatch, flag):
    """换成假 client，记下真实传进去的关键字参数，返回 [已收到的 kwargs]。"""
    captured = []

    def fake_create(**kwargs):
        captured.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=154, completion_tokens=1, total_tokens=155)
        message = SimpleNamespace(content="1")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    monkeypatch.setattr(intent, "QWEN_DISABLE_THINKING", flag)
    monkeypatch.setattr(intent.client, "chat",
                        SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    return captured


def test_开关打开时思考模式参数以extra_body传出(monkeypatch):
    captured = _stub(monkeypatch, True)

    assert intent.detect_intent_code("什么是RAG") == 1

    assert len(captured) == 1, "意图分类应该只发一次请求"
    assert captured[0]["extra_body"] == {"enable_thinking": False}
    # 回归护栏：关思考不该动语义参数
    assert captured[0]["temperature"] == 0
    assert captured[0]["max_tokens"] == 10
    assert captured[0]["model"] == config.MODEL_NAME


def test_开关关闭时完全不传extra_body(monkeypatch):
    captured = _stub(monkeypatch, False)

    assert intent.detect_intent_code("什么是RAG") == 1
    # 默认必须"不改变现有行为"：没开开关时，请求体和改动前逐字一致
    assert "extra_body" not in captured[0]