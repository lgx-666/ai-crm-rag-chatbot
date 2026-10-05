# tests/conftest.py
import pytest
import db.sqlite_db as db

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """把 DB 指向一个全新的临时文件并建好表结构。
    每个请求它的用例都拿到一个干净、隔离、用完即弃的数据库。"""
    test_db = tmp_path / "test_crm.db"
    # 关键：patch 使用处 db.sqlite_db.SQLITE_DB_PATH（不是 config 里的）
    monkeypatch.setattr(db, "SQLITE_DB_PATH", str(test_db))
    db.init_sqlite_db()          # 在临时库里建表
    return test_db
    # 不需要 yield 后清理——tmp_path 由 pytest 自动删

@pytest.fixture
def client(temp_db):
    """Flask 测试客户端。依赖 temp_db → 所有请求打的都是临时库。
    import app 会触发 chromadb/langchain/langgraph 重导入（首次慢几秒），
    但不会加载嵌入模型、不会连真实 crm.db（init_data 只在 __main__ 跑）。"""
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c