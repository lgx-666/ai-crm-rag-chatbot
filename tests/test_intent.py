# tests/test_intent.py
from intent import parse_intent

def test_parse_clean_digit():
    assert parse_intent("3") == 3

def test_parse_with_whitespace():
    """LLM 常带换行/空格，strip 后要能正常解析"""
    assert parse_intent("  2\n") == 2

def test_parse_digit_in_sentence():
    """正则抓第一个数字序列"""
    assert parse_intent("意图是 1 号") == 1

def test_parse_out_of_range_falls_back():
    """超出 0-4 范围 → 回退 0（比如 LLM 抽风返回 9）"""
    assert parse_intent("9") == 0

def test_parse_no_digit_falls_back():
    """没有数字 → 0"""
    assert parse_intent("我不知道") == 0

def test_parse_empty_and_none():
    """空串/None 不能崩，回退 0"""
    assert parse_intent("") == 0
    assert parse_intent(None) == 0

def test_parse_first_number_only():
    """有多个数字时只取第一个（re.search 语义）"""
    assert parse_intent("1和2") == 1