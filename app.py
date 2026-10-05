# app.py
import json
import os
import uuid
from flask import send_from_directory, Flask, request, jsonify, Response, stream_with_context
from werkzeug.utils import secure_filename

from db.sqlite_db import (init_sqlite_db,get_user_by_phone,create_user,
        update_user_profile,fetch_chat_history,verify_password,upgrade_password_hash
)
from db.vector_db import (init_vectorstores, add_documents_to_collection, get_market_collection, 
                        get_course_collection
)
from graph_flow import chat_graph
from utils.logging_config import setup_logging
from file_loader import load_and_chunk_file

from auth import login_required, create_token   # 补 create_token

# LangChain 相关导入
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from config import SQLITE_DB_PATH, OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME,COLLECTION_MARKET, COLLECTION_COURSE

logger = setup_logging()
app = Flask(__name__)

# ---------- 页面路由 ----------
@app.route('/')
def login_page():
    """登录页面"""
    return send_from_directory('static/HTML', 'login.html')

@app.route('/chat_page')
def chat_page():
    """聊天页面（登录后跳转）"""
    return send_from_directory('static/HTML', 'chat.html')

@app.route('/profile')
def profile_page():
    """个人信息页面"""
    return send_from_directory('static/HTML', 'profile.html')


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if not data or 'phone_number' not in data or 'password' not in data:
        return jsonify({"error": "缺少手机号或密码"}), 400

    phone = data['phone_number']
    password = data['password']

    user = get_user_by_phone(phone)
    if user:
        # 用 verify_password 兼容"哈希/历史明文"两种存储
        if verify_password(user['password'], password):
            # 若库里还是历史明文，趁登录成功升级为哈希（懒迁移）
            if not user['password'].startswith(("pbkdf2:", "scrypt:")):
                upgrade_password_hash(phone, password)
            token = create_token(phone)
            safe_user = {k: v for k, v in user.items() if k != 'password'}
            return jsonify({"status":"success","user":safe_user,"token":token,"is_new":False}), 200
        else:
            return jsonify({"error": "密码错误"}), 401
    else:
        create_user(phone, password)
        new_user = get_user_by_phone(phone)
        token = create_token(phone)
        safe_user = {k: v for k, v in new_user.items() if k != 'password'}
        return jsonify({"status":"success","user":safe_user,"token":token,"is_new":True}), 200

# ---------- 初始化数据 ----------
def init_data():
    """
    初始化数据库和向量库，并在首次启动时填充示例知识。
    现在项目领域为：市场知识（大模型） 和 课程知识（Java八股文）。
    """
    init_sqlite_db()
    init_vectorstores()

    market_coll = get_market_collection()
    if market_coll.count() == 0:
        sample_market = [
            "Transformer 模型的核心是自注意力机制（Self-Attention），它允许模型在处理序列时关注所有位置的信息，解决了 RNN 的长期依赖问题。",
            "大型语言模型（LLM）的预训练通常采用无监督学习，通过海量文本预测下一个词（语言建模）。微调（Fine-tuning）则使用特定任务数据调整模型参数。",
            "检索增强生成（RAG）结合了信息检索和生成模型，先检索外部知识库，再将检索结果作为上下文输入 LLM 生成回答，有效减少幻觉。"
        ]
        add_documents_to_collection(
            COLLECTION_MARKET,
            sample_market,
            metadatas=[{"source": "sample_init"}] * len(sample_market),
            ids=[f"market_{i}" for i in range(len(sample_market))]
        )
        logger.info("大模型知识集合已填充示例数据（3条）")

    course_coll = get_course_collection()
    if course_coll.count() == 0:
        sample_course = [
            "JVM 内存结构包括：程序计数器、虚拟机栈、本地方法栈、堆、方法区。堆是线程共享的，存储对象实例；栈是线程私有的，存储局部变量表。",
            "Java 并发编程中，synchronized 关键字用于实现互斥锁，ReentrantLock 提供了更灵活的锁机制（如公平锁、可中断锁）。",
            "Spring 框架的核心是 IoC（控制反转）和 AOP（面向切面编程）。IoC 容器管理对象的生命周期和依赖关系，AOP 用于横切关注点的模块化（如日志、事务）。"
        ]
        add_documents_to_collection(
            COLLECTION_COURSE,
            sample_course,
            metadatas=[{"source": "sample_init"}] * len(sample_course),
            ids=[f"course_{i}" for i in range(len(sample_course))]
        )
        logger.info("Java 八股文知识集合已填充示例数据（3条）")

# ---------- 聊天接口（流式） ----------
@app.route('/chat', methods=['POST'])
@login_required
def chat_api():
    phone_number = request.phone_number
    data = request.get_json() or {}
    query = data.get('query')
    if not query:
        return jsonify({"error": "缺少 query 字段"}), 400

    def generate():
        try:
            # stream_mode="custom"：只收节点里 writer(...) 外发的自定义数据（即逐 token）
            for chunk in chat_graph.stream({"phone_number": phone_number, "query": query},
                                        stream_mode="custom"):
                tok = chunk.get("token", "")
                if tok:
                    yield f"data: {json.dumps({'content': tok}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("流式输出失败: %s", e)
            yield f"data: {json.dumps({'error': '服务异常'}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type='text/event-stream',                  # ← SSE 的 MIME 类型
        headers={
            'Cache-Control': 'no-cache',                   # 禁止缓存流
            'X-Accel-Buffering': 'no',                     # 关键：阻止 Nginx 等反代缓冲，否则会被攒够才发
            'Connection': 'keep-alive',
        }
    )


# ---------- 向量库更新接口（JSON方式） ----------
@app.route('/update_vectorstore', methods=['POST'])
@login_required
def update_vectorstore():
    data = request.get_json() or {}
    collection = data.get('collection')
    documents = data.get('documents')
    if not collection or not documents:
        return jsonify({"error": "缺少 collection 或 documents 字段"}), 400

    metadatas = data.get('metadatas')
    ids = data.get('ids')

    # 修改为（使用常量）
    if collection not in (COLLECTION_MARKET, COLLECTION_COURSE):
        return jsonify({"error": f"collection 必须是 '{COLLECTION_MARKET}' 或 '{COLLECTION_COURSE}'"}), 400
    if not isinstance(documents, list) or len(documents) == 0:
        return jsonify({"error": "documents 必须是非空列表"}), 400

    try:
        add_documents_to_collection(collection, documents, metadatas, ids)
        return jsonify({"status": "success", "added": len(documents)}), 200
    except Exception as e:
        logger.error("向量数据库更新失败: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------- SQL Agent 接口 ----------
@app.route('/sql_query', methods=['POST'])
@login_required
def sql_agent_query():
    data = request.get_json() or {}
    question = data.get('question')
    if not question:
        return jsonify({"error": "缺少 question 字段"}), 400

    try:
        db = SQLDatabase.from_uri(f"sqlite:///{SQLITE_DB_PATH}")
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0
        )
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        agent = create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=True,
            handle_parsing_errors=True
        )
        result = agent.invoke({"input": question})
        answer = result.get('output', '无结果')
        return jsonify({"question": question, "answer": answer}), 200
    except Exception as e:
        logger.error("SQL Agent 执行失败: %s", e)
        return jsonify({"error": str(e)}), 500

# ---------- 文件上传导入知识库接口 ----------
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.md', '.html', '.htm', '.txt'}
def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route('/upload_knowledge', methods=['POST'])
@login_required
def upload_knowledge():
    collection = request.form.get('collection')
    if collection not in (COLLECTION_MARKET, COLLECTION_COURSE):
        return jsonify({"error": f"collection 必须为 '{COLLECTION_MARKET}' 或 '{COLLECTION_COURSE}'"}), 400

    if 'files' not in request.files:
        return jsonify({"error": "缺少 files 字段"}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "未选择任何文件"}), 400

    all_chunks = []
    metadata_list = []
    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"error": f"不支持的文件类型: {file.filename}"}), 400

        filename = secure_filename(file.filename)
        # 使用系统临时目录（Windows下可用 tempfile 模块或固定路径，这里简单处理）
        temp_dir = os.environ.get('TEMP', '/tmp') if os.name != 'nt' else os.environ.get('TEMP', 'C:\\Temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        try:
            chunks = load_and_chunk_file(temp_path, chunk_size=500, overlap=50)
            for chunk in chunks:
                all_chunks.append(chunk)
                metadata_list.append({"source": filename, "collection": collection})
        except Exception as e:
            os.remove(temp_path)
            return jsonify({"error": f"解析文件 {filename} 失败: {str(e)}"}), 500
        os.remove(temp_path)

    if not all_chunks:
        return jsonify({"error": "未提取到任何文本内容"}), 400

    try:
        ids = [str(uuid.uuid4()) for _ in all_chunks]
        add_documents_to_collection(collection, all_chunks, metadatas=metadata_list, ids=ids)
        return jsonify({"status": "success", "total_chunks": len(all_chunks)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    phone = request.phone_number
    if not phone:
        return jsonify({"error": "缺少手机号"}), 400

    # 可选的字段
    data = request.get_json() or {}
    name = data.get('name')
    age = data.get('age')
    occupation = data.get('occupation')
    interest = data.get('interest')

    # 检查用户是否存在（可选）
    user = get_user_by_phone(phone)
    if not user:
        return jsonify({"error": "用户不存在"}), 404

    update_user_profile(phone, name, age, occupation, interest)
    # 返回更新后的信息
    updated_user = get_user_by_phone(phone)
    return jsonify({"status": "success", "user": updated_user}), 200

#---------------获取历史记录的接口--------------
@app.route('/chat_history', methods=['POST'])
@login_required
def get_chat_history():
    phone = request.phone_number      # ← 从 token 来，不从 body 来
    history = fetch_chat_history(phone, limit=1000)
    return jsonify({"history": history}), 200

# ---------- 启动 ----------
if __name__ == '__main__':
    init_data()
    # 如果设置了 FLASK_DEBUG=0 或 DEBUGPY_RUNNING 环境变量，则关闭 Flask 的调试模式
    use_debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    # 也可通过判断是否在调试模式下来决定
    app.run(host='0.0.0.0', port=5000, debug=use_debug, use_reloader=False)