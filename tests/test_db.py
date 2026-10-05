# tests/test_db.py
import db.sqlite_db as db

# 每个用例参数里的 temp_db 就是 conftest.py 那个 fixture：
# 它已经把 DB 指向临时库并建好表，用例之间互不干扰。

# ---------- 用户 / 密码 ----------
def test_create_user_stores_hash_not_plaintext(temp_db):
    """注册后库里必须是哈希，绝不能是明文（阶段1 安全加固的回归保护）"""
    db.create_user("13800000000", "secret123")
    user = db.get_user_by_phone("13800000000")
    assert user is not None
    assert user["password"] != "secret123"                       # 不是明文
    assert user["password"].startswith(("pbkdf2:", "scrypt:"))   # 是哈希
    assert db.verify_password(user["password"], "secret123")     # 但能校验通过

def test_get_user_by_phone_missing_returns_none(temp_db):
    """查不存在的用户返回 None，不崩"""
    assert db.get_user_by_phone("no_such_user") is None

# ---------- 对话历史 ----------
def test_chat_history_order_preserved(temp_db):
    """取回顺序应与存入顺序一致（fetch 内部 DESC 取再 reversed）"""
    db.update_chat_history("139", "user", "第一句")
    db.update_chat_history("139", "assistant", "第二句")
    db.update_chat_history("139", "user", "第三句")
    history = db.fetch_chat_history("139")
    assert [h["content"] for h in history] == ["第一句", "第二句", "第三句"]
    assert [h["role"] for h in history] == ["user", "assistant", "user"]

def test_chat_history_limit_returns_most_recent(temp_db):
    """limit 返回'最近 N 条'，且仍保持时间正序"""
    for i in range(5):
        db.update_chat_history("139", "user", f"msg{i}")
    history = db.fetch_chat_history("139", limit=2)
    assert [h["content"] for h in history] == ["msg3", "msg4"]

def test_chat_history_isolated_by_phone(temp_db):
    """安全属性：用户 A 绝不能看到用户 B 的对话（越权隔离的回归保护）"""
    db.update_chat_history("111", "user", "A的秘密")
    db.update_chat_history("222", "user", "B的秘密")
    a_history = db.fetch_chat_history("111")
    assert len(a_history) == 1
    assert a_history[0]["content"] == "A的秘密"

def test_count_chat_history(temp_db):
    db.update_chat_history("139", "user", "x")
    db.update_chat_history("139", "assistant", "y")
    assert db.count_chat_history("139") == 2
    assert db.count_chat_history("other") == 0

# ---------- 滚动摘要 ----------
def test_summary_default_when_empty(temp_db):
    """无摘要记录时返回 ("", 0)，不是 None 也不崩"""
    assert db.get_conversation_summary("139") == ("", 0)

def test_summary_upsert(temp_db):
    """save 两次应更新（upsert），不是插两行"""
    db.save_conversation_summary("139", "第一版摘要", 5)
    db.save_conversation_summary("139", "第二版摘要", 10)
    summary, upto = db.get_conversation_summary("139")
    assert summary == "第二版摘要"
    assert upto == 10

# ---------- 摘要窗口游标逻辑（最有价值的一组）----------
def test_summarize_below_window_does_nothing(temp_db):
    """新增消息数 <= keep_recent：不总结，返回 ([], 原游标)"""
    for i in range(3):
        db.update_chat_history("139", "user", f"m{i}")
    msgs, upto = db.fetch_history_to_summarize("139", summarized_upto=0, keep_recent=5)
    assert msgs == []
    assert upto == 0

def test_summarize_excludes_recent_window(temp_db):
    """超窗口时：尾部 keep_recent 条保留，前面的折叠，游标推进到最后一条被总结的 id"""
    for i in range(6):
        db.update_chat_history("139", "user", f"m{i}")   # id 1..6
    msgs, upto = db.fetch_history_to_summarize("139", summarized_upto=0, keep_recent=2)
    assert [m["content"] for m in msgs] == ["m0", "m1", "m2", "m3"]  # 前4条
    assert upto == 4                                                  # 游标=被总结的最后id

def test_summarize_respects_cursor_incremental(temp_db):
    """增量：只考虑游标之后的新消息，已总结的不重复"""
    for i in range(6):
        db.update_chat_history("139", "user", f"m{i}")   # id 1..6
    # 已总结到 id=4，之后只剩 id 5,6 共2条 <= keep_recent=3 → 不再总结
    msgs, upto = db.fetch_history_to_summarize("139", summarized_upto=4, keep_recent=3)
    assert msgs == []
    assert upto == 4

# ---------- 懒迁移（阶段1 安全核心，必须覆盖）----------
def test_upgrade_password_hash_migrates_plaintext(temp_db):
    """老账号明文密码 → upgrade 后变哈希，且新哈希仍能校验通过"""
    conn = db.get_db_connection()          # 读的是被 monkeypatch 的临时库
    conn.execute("INSERT INTO user_info (phone_number, password) VALUES (?, ?)",
                ("old_user", "plain123"))
    conn.commit(); conn.close()
    assert db.get_user_by_phone("old_user")["password"] == "plain123"   # 初始明文

    db.upgrade_password_hash("old_user", "plain123")                    # 执行懒迁移
    stored = db.get_user_by_phone("old_user")["password"]
    assert stored != "plain123"
    assert stored.startswith(("pbkdf2:", "scrypt:"))
    assert db.verify_password(stored, "plain123")

# ---------- get_user_info_from_db ----------
def test_get_user_info_missing_returns_none(temp_db):
    assert db.get_user_info_from_db("no_such") is None

# ---------- upsert_user_info（有 insert/update 分支）----------
def test_upsert_insert_new(temp_db):
    """用户不存在 + 有字段 → INSERT，返回 True"""
    assert db.upsert_user_info("139", name="张三", age=25) is True
    info = db.get_user_info_from_db("139")
    assert info["name"] == "张三" and info["age"] == 25

def test_upsert_update_existing_keeps_other_fields(temp_db):
    """用户已存在 → 只改提供的字段，其余保持原值"""
    db.upsert_user_info("139", name="张三", age=25, occupation="学生")
    db.upsert_user_info("139", age=26)               # 只改 age
    info = db.get_user_info_from_db("139")
    assert info["age"] == 26
    assert info["name"] == "张三"                     # 没提供的不动
    assert info["occupation"] == "学生"

def test_upsert_all_none_new_returns_false(temp_db):
    """用户不存在 + 全 None → 无可插，返回 False"""
    assert db.upsert_user_info("139") is False

# ---------- update_user_profile ----------
def test_update_user_profile_partial(temp_db):
    db.upsert_user_info("139", name="张三", age=25, occupation="学生")
    db.update_user_profile("139", occupation="工程师")   # 只改职业
    info = db.get_user_info_from_db("139")
    assert info["occupation"] == "工程师"
    assert info["name"] == "张三" and info["age"] == 25  # 其余保持

def test_upsert_and_profile_all_fields(temp_db):
    """一次传满 name/age/occupation/interest，覆盖所有字段拼接分支"""
    db.upsert_user_info("139", name="张三", age=25, occupation="学生", interest="篮球")
    db.update_user_profile("139", name="李四", age=30, occupation="工程师", interest="足球")
    info = db.get_user_info_from_db("139")
    assert info == {"name":"李四","age":30,"occupation":"工程师","interest":"足球"}