# tests/test_observability_mask.py —— mask 的纯函数层：不连 Langfuse。
# _mask_node/_mask_pii 故意定义在 if ENABLED: 之外，CI 里 ENABLED=False 也能直接 import。
from observability import _mask_node, _mask_pii, trace_user_id

FAKE = "13900000000"


def test_state_key_becomes_stable_hash():
    out = _mask_node({"phone_number": FAKE, "query": "hi"})
    assert out["phone_number"] == trace_user_id(FAKE)
    assert out["query"] == "hi"
    assert FAKE not in str(out)


def test_phone_inside_text_and_nested_containers_is_masked():
    out = _mask_node({"answer": f"联系 {FAKE} 谢谢", "ctx": [{"t": FAKE}], "n": ("a", FAKE)})
    assert FAKE not in str(out)
    assert "<masked>" in out["answer"]


def test_longer_digit_runs_are_not_eaten():
    s = "编号 13900000000123 和 013900000000"
    assert _mask_node({"x": s})["x"] == s


def test_other_types_pass_through():
    assert _mask_node(None) is None
    assert _mask_node(42) == 42
    assert _mask_pii({"phone_number": FAKE})["phone_number"] == trace_user_id(FAKE)