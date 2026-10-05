# tests/test_password.py
from db.sqlite_db import verify_password, _HASH_PREFIXES
from werkzeug.security import generate_password_hash

# verify_password(stored, plain) 是纯函数：输入两个字符串，输出 bool，不碰 DB/网络。
# 纯函数最好测——不需要 fixture、不需要 mock，直接断言。

def test_hash_correct_password():
    """新数据（哈希）+ 正确明文 → True"""
    stored = generate_password_hash("mypassword123")
    assert verify_password(stored, "mypassword123") is True

def test_hash_wrong_password():
    """新数据（哈希）+ 错误明文 → False（哈希校验的核心安全属性）"""
    stored = generate_password_hash("mypassword123")
    assert verify_password(stored, "wrongpassword") is False

def test_legacy_plaintext_match():
    """历史明文账号 + 明文匹配 → True（兼容老数据，走的是 stored == plain 分支）"""
    assert verify_password("oldplain123", "oldplain123") is True

def test_legacy_plaintext_mismatch():
    """历史明文账号 + 不匹配 → False"""
    assert verify_password("oldplain123", "different") is False

def test_empty_stored_returns_false():
    """stored 为空/None 时不能崩，且应判 False（防 None.startswith 报错）"""
    assert verify_password("", "anything") is False
    assert verify_password(None, "anything") is False

def test_hash_actually_uses_prefix():
    """确认 generate_password_hash 产出的前缀确实在 _HASH_PREFIXES 里，
    否则 verify_password 会把哈希误判成明文，直接字符串比对 → 永远 False。
    这条测的是'分支走对了'，防止将来 werkzeug 升级换了前缀导致静默失效。"""
    stored = generate_password_hash("x")
    assert stored.startswith(_HASH_PREFIXES)