# config.py
import os
import secrets
from dotenv import load_dotenv  # 若未安装 python-dotenv，可删除此行

# 加载 .env 文件（如果存在）
load_dotenv()

# 数据存储根目录
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# OpenAI 配置（从环境变量读取）
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")

# 博查搜索API配置
BOCHA_API_KEY = os.getenv("BOCHA_API_KEY", "")

# Langfuse 可观测性：三项齐全才开启埋点。CI 上没有 .env，这里全是空字符串，
# observability.ENABLED 就是 False，所有埋点退化成空操作。
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")

JWT_SECRET = os.getenv("JWT_SECRET")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    print("[警告] 未设置 JWT_SECRET，本次启动使用随机密钥："
        "服务重启后所有 token 失效，且多 worker 部署会导致会话不互通。")

# 向量数据库路径
# 向量库持久化目录：chroma_db_v2 由修复后的切块算法重建，旧的碎片库已清理。
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db_v2")
COLLECTION_MARKET = "market_knowledge"
COLLECTION_COURSE = "java_knowledge"

# SQLite 数据库路径----->根据目录创建数据库文件
SQLITE_DB_PATH = os.path.join(DATA_DIR, "crm.db")

# 聊天记录配置
MEMORY_WINDOW = 10          # 注入最近多少条消息（10 条 ≈ 5 轮对话）
MEMORY_MAX_CHARS = 6000     # 历史总字符预算，超出则从最旧开始丢弃（token 兜底）
SUMMARY_THRESHOLD = 20      # 历史总条数超过它，才启动"滚动摘要"压缩更早的对话


# Prompt 模板
# config.py
PROMPT_TEMPLATES = {
    # ...
    "greeting": "您好！我是 AI 知识助手，可以为您解答大模型原理和 Java 后端开发的相关问题，也能帮您查询/修改个人信息。请问有什么可以帮您？",
    "fallback": "抱歉，我现在无法处理您的请求，请稍后再试。",
    "intent_prompt": (
        "你是一个意图识别助手。请根据用户输入，判断其意图编号：\n"
        "0: 常规回答（如闲聊、问候、无关问题等，不需要特别知识）\n"
        "1: RAG知识库检索（如大模型原理、Java八股文等专业知识点）\n"
        "2: 网络检索插件（需要实时信息，如新闻、天气、最新资讯等）\n"
        "3: 数据库信息查询（查询用户自己的个人信息、系统统计数据等）\n"
        "4: 操作个人信息（如修改姓名、年龄、职业、兴趣等）\n"   # 新增
        "用户输入：{query}\n"
        "只输出数字编号（0/1/2/3/4）。"
    )
}