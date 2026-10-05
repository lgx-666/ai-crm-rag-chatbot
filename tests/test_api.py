# tests/test_api.py
import json
import datetime
import jwt
import db.sqlite_db as db
from config import JWT_SECRET

# ---------- /login 注册与登录 ----------
def test_login_register_new_user(client):
    """新手机号 → 注册成功，is_new=True，返回 token，且响应不泄露密码"""
    r = client.post("/login", json={"phone_number": "13800000000", "password": "pw123"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["is_new"] is True
    assert data["token"]                       # 返回了 JWT
    assert "password" not in data["user"]      # 安全回归：safe_user 必须剔除密码/哈希

def test_login_existing_user_correct_password(client):
    """已注册用户 + 正确密码 → is_new=False"""
    client.post("/login", json={"phone_number": "13800000000", "password": "pw123"})   # 先注册
    r = client.post("/login", json={"phone_number": "13800000000", "password": "pw123"})  # 再登录
    assert r.status_code == 200
    assert r.get_json()["is_new"] is False

def test_login_wrong_password(client):
    client.post("/login", json={"phone_number": "13800000000", "password": "pw123"})
    r = client.post("/login", json={"phone_number": "13800000000", "password": "wrong"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "密码错误"

def test_login_missing_fields(client):
    """缺 password → 400，在查库之前就拦截"""
    r = client.post("/login", json={"phone_number": "138"})
    assert r.status_code == 400
    assert r.get_json()["error"] == "缺少手机号或密码"

# ---------- /chat 鉴权闸门（不碰 LLM）----------
def test_chat_requires_auth(client):
    """无 token → 401 未登录，且不进入业务逻辑"""
    r = client.post("/chat", json={"query": "hi"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "未登录"

def test_chat_rejects_invalid_token(client):
    r = client.post("/chat", json={"query": "hi"},
                    headers={"Authorization": "Bearer garbage.token.here"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "无效凭证"

def test_chat_rejects_expired_token(client):
    """手工构造已过期 token，命中 ExpiredSignatureError 分支"""
    expired = jwt.encode(
        {"phone": "139", "exp": datetime.datetime.utcnow() - datetime.timedelta(hours=1)},
        JWT_SECRET, algorithm="HS256")
    r = client.post("/chat", json={"query": "hi"},
                    headers={"Authorization": f"Bearer {expired}"})
    assert r.status_code == 401
    assert r.get_json()["error"] == "登录已过期"

def test_chat_missing_query_400(client):
    """合法 token 但缺 query → 400（在调用 graph 之前拦截，无需 mock）"""
    token = client.post("/login", json={"phone_number": "139", "password": "pw"}).get_json()["token"]
    r = client.post("/chat", json={}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400
    assert r.get_json()["error"] == "缺少 query 字段"

# ---------- /chat SSE 管道（用假 graph 替换真 LLM）----------
def test_chat_streams_tokens_via_sse(client, monkeypatch):
    """核心：验证 SSE 逐 token 外发 + [DONE] 收尾，但不真调 LLM。
    用 FakeGraph 顶替 app.chat_graph——它模拟节点里 writer 外发的 custom 数据。"""
    token = client.post("/login", json={"phone_number": "139", "password": "pw"}).get_json()["token"]

    class FakeGraph:
        def stream(self, inputs, stream_mode=None):
            for tok in ["你", "好", "呀"]:
                yield {"token": tok}           # 和真 graph 的 custom 输出结构一致

    # patch 使用处 app.chat_graph：chat_api 的 generate() 在调用时才查这个模块全局
    monkeypatch.setattr("app.chat_graph", FakeGraph())

    r = client.post("/chat", json={"query": "hi"},
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.content_type.startswith("text/event-stream")

    # 解析 SSE：逐行还原 token，拼回完整答案
    body = r.get_data(as_text=True)
    tokens = []
    for line in body.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            tokens.append(json.loads(line[len("data: "):]).get("content", ""))
    assert "".join(tokens) == "你好呀"          # 逐 token 累积成完整文本
    assert "data: [DONE]" in body               # 流正常收尾

# ---------- 越权隔离（API 层安全回归）----------
def test_chat_history_isolated_by_token(client):
    """/chat_history 只认 token 里的 phone：A 的 token 拿不到 B 的历史"""
    token_a = client.post("/login", json={"phone_number": "111", "password": "pw"}).get_json()["token"]
    db.update_chat_history("111", "user", "A的私密对话")
    db.update_chat_history("222", "user", "B的私密对话")   # 另一个用户的数据

    r = client.post("/chat_history", headers={"Authorization": f"Bearer {token_a}"})
    contents = [h["content"] for h in r.get_json()["history"]]
    assert "A的私密对话" in contents
    assert "B的私密对话" not in contents        # 关键：越权拿不到