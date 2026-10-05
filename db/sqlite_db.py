# db/sqlite_db.py
import sqlite3
from typing import List, Dict, Optional
from config import SQLITE_DB_PATH
from utils.logging_config import setup_logging
from werkzeug.security import generate_password_hash, check_password_hash

logger = setup_logging()

# Python 操作 SQLite 数据库的标准两步法。
def init_sqlite_db():
    #创建数据库连接
    conn = sqlite3.connect(SQLITE_DB_PATH)
    # 从连接创建游标对象。
    cursor = conn.cursor()
    #  使用游标执行 SQL 语句。
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # cursor.execute('''
    #     CREATE TABLE IF NOT EXISTS user_info (
    #         phone_number TEXT PRIMARY KEY,
    #         name TEXT,
    #         age INTEGER,
    #         occupation TEXT,
    #         interest TEXT,
    #         last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP
    #     )
    # ''')
    # 创建用户信息表，增加 password 字段
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_info (
            phone_number TEXT PRIMARY KEY,
            password TEXT,                -- 新增密码字段
            name TEXT,
            age INTEGER,
            occupation TEXT,
            interest TEXT,
            last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # 对话摘要表：保存每个用户"窗口之外"旧对话的滚动摘要
    # summarized_upto 记录已折叠进摘要的最后一条 chat_history.id，用于增量压缩
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversation_summary (
            phone_number TEXT PRIMARY KEY,
            summary TEXT,
            summarized_upto INTEGER DEFAULT 0,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("SQLite 数据库初始化完成（路径：%s）", SQLITE_DB_PATH)

def get_db_connection():
    return sqlite3.connect(SQLITE_DB_PATH)

def fetch_chat_history(phone_number: str, limit: int = 100) -> List[Dict[str, str]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE phone_number = ? ORDER BY id DESC LIMIT ?",
        (phone_number, limit)
    )
    rows = cursor.fetchall()
    conn.close()
    history = [{"role": row[0], "content": row[1]} for row in reversed(rows)]
    return history

def update_chat_history(phone_number: str, role: str, content: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (phone_number, role, content) VALUES (?, ?, ?)",
        (phone_number, role, content)
    )
    conn.commit()
    conn.close()
    logger.info("聊天记录已保存: %s - %s", phone_number, role)

def get_user_info_from_db(phone_number: str) -> Optional[Dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, occupation, interest FROM user_info WHERE phone_number = ?", (phone_number,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {"name": row[0], "age": row[1], "occupation": row[2], "interest": row[3]}
    return None

def upsert_user_info(phone_number: str, name: str = None, age: int = None,
                    occupation: str = None, interest: str = None) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT phone_number FROM user_info WHERE phone_number = ?", (phone_number,))
    exists = cursor.fetchone()
    try:
        if exists:
            updates = []
            values = []
            if name is not None:
                updates.append("name = ?"); values.append(name)
            if age is not None:
                updates.append("age = ?"); values.append(age)
            if occupation is not None:
                updates.append("occupation = ?"); values.append(occupation)
            if interest is not None:
                updates.append("interest = ?"); values.append(interest)
            if updates:
                values.append(phone_number)
                sql = f"UPDATE user_info SET {', '.join(updates)}, last_interaction = CURRENT_TIMESTAMP WHERE phone_number = ?"
                cursor.execute(sql, values)
        else:
            if name is None and age is None and occupation is None and interest is None:
                return False
            fields = ["phone_number"]
            placeholders = ["?"]
            values = [phone_number]
            if name is not None:
                fields.append("name"); placeholders.append("?"); values.append(name)
            if age is not None:
                fields.append("age"); placeholders.append("?"); values.append(age)
            if occupation is not None:
                fields.append("occupation"); placeholders.append("?"); values.append(occupation)
            if interest is not None:
                fields.append("interest"); placeholders.append("?"); values.append(interest)
            sql = f"INSERT INTO user_info ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(sql, values)
        conn.commit()
        return True
    except Exception as e:
        logger.error("更新用户信息失败: %s", e)
        return False
    finally:
        conn.close()

def get_user_by_phone(phone_number: str):
    """根据手机号获取用户信息（含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT phone_number, password, name, age, occupation, interest FROM user_info WHERE phone_number = ?", (phone_number,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return {
            "phone_number": row[0],
            "password": row[1],
            "name": row[2],
            "age": row[3],
            "occupation": row[4],
            "interest": row[5]
        }
    return None


def create_user(phone_number: str, password: str, name: str = None, age: int = None, occupation: str = None, interest: str = None):
    """创建新用户（注册）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO user_info (phone_number, password, name, age, occupation, interest) VALUES (?, ?, ?, ?, ?, ?)",
        (phone_number, generate_password_hash(password), name, age, occupation, interest)  # ← 存哈希不存明文
    )
    conn.commit()
    conn.close()

_HASH_PREFIXES = ("pbkdf2:", "scrypt:")   # werkzeug 哈希的固定前缀，用来区分"新哈希"和"历史明文"

def verify_password(stored: str, plain: str) -> bool:
    """校验密码：新数据是哈希用 check_password_hash；历史明文直接比对（兼容老账号）。"""
    if stored and stored.startswith(_HASH_PREFIXES):
        return check_password_hash(stored, plain)
    return stored == plain

def upgrade_password_hash(phone_number: str, plain: str):
    """懒迁移：把老账号的明文密码就地升级成哈希。"""
    conn = get_db_connection()
    conn.execute("UPDATE user_info SET password = ? WHERE phone_number = ?",
                (generate_password_hash(plain), phone_number))
    conn.commit()
    conn.close()

def update_user_profile(phone_number: str, name: str = None, age: int = None, occupation: str = None, interest: str = None):
    """更新用户个人信息（不包含密码）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    updates = []
    values = []
    if name is not None:
        updates.append("name = ?"); values.append(name)
    if age is not None:
        updates.append("age = ?"); values.append(age)
    if occupation is not None:
        updates.append("occupation = ?"); values.append(occupation)
    if interest is not None:
        updates.append("interest = ?"); values.append(interest)
    if updates:
        values.append(phone_number)
        sql = f"UPDATE user_info SET {', '.join(updates)}, last_interaction = CURRENT_TIMESTAMP WHERE phone_number = ?"
        cursor.execute(sql, values)
        conn.commit()
    conn.close()


def count_chat_history(phone_number: str) -> int:
    """统计某用户历史总条数，用于判断是否触发摘要压缩"""
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chat_history WHERE phone_number = ?", (phone_number,))
    n = cursor.fetchone()[0]
    conn.close()
    return n

def get_conversation_summary(phone_number: str):
    """返回 (摘要文本, 已总结到的消息id)。没有记录时返回 ("", 0)"""
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("SELECT summary, summarized_upto FROM conversation_summary WHERE phone_number = ?",
                (phone_number,))
    row = cursor.fetchone(); conn.close()
    if row:
        return (row[0] or "", row[1] or 0)
    return ("", 0)

def save_conversation_summary(phone_number: str, summary: str, summarized_upto: int):
    """写入/更新滚动摘要，同时推进游标 summarized_upto"""
    conn = get_db_connection(); cursor = conn.cursor()
    # ON CONFLICT：主键已存在就更新，实现 upsert（有则改、无则插）
    cursor.execute('''
        INSERT INTO conversation_summary (phone_number, summary, summarized_upto, updated_at)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(phone_number) DO UPDATE SET
            summary = excluded.summary,
            summarized_upto = excluded.summarized_upto,
            updated_at = CURRENT_TIMESTAMP
    ''', (phone_number, summary, summarized_upto))
    conn.commit(); conn.close()

def fetch_history_to_summarize(phone_number: str, summarized_upto: int, keep_recent: int):
    """
    取出"需要被折叠进摘要"的旧消息。
    逻辑：拿到 id > summarized_upto 的所有新消息（即上次总结之后新增的），
        但最近 keep_recent 条要留在窗口里逐字保留、不总结，
        所以把这部分尾部排除，剩下的才交给 LLM 压缩。
    返回 (待总结消息列表, 新游标id)；若不够总结则返回 ([], 原游标)。
    """
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute(
        "SELECT id, role, content FROM chat_history WHERE phone_number = ? AND id > ? ORDER BY id ASC",
        (phone_number, summarized_upto)
    )
    rows = cursor.fetchall(); conn.close()
    if len(rows) <= keep_recent:      # 新增的还没超过窗口，全都留在窗口里，无需总结
        return [], summarized_upto
    to_summarize = rows[:-keep_recent] if keep_recent > 0 else rows  # 去掉尾部窗口部分
    new_upto = to_summarize[-1][0]    # 新游标 = 本批最后一条的 id
    msgs = [{"id": r[0], "role": r[1], "content": r[2]} for r in to_summarize]
    return msgs, new_upto