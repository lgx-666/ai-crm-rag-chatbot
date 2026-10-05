# auth.py（新增）
import jwt, datetime
from functools import wraps
from flask import request, jsonify
from config import JWT_SECRET   # 放 .env，别硬编码

def create_token(phone_number: str) -> str:
    payload = {
        "phone": phone_number,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=2),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not token:
            return jsonify({"error": "未登录"}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "登录已过期"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "无效凭证"}), 401
        request.phone_number = payload["phone"]   # ← 可信身份挂到 request 上
        return f(*args, **kwargs)
    return wrapper