# response_generator.py

import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from openai import OpenAI
from config import (
    OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, BOCHA_API_KEY, 
    SQLITE_DB_PATH,
    MEMORY_WINDOW, MEMORY_MAX_CHARS, SUMMARY_THRESHOLD      # ← 记忆编排用得到，摘 COLLECTION 时别把它们一起带走
)
from db.vector_db import query_collection
from kb_router import retrieval_plan
from db.sqlite_db import (                             
    get_user_info_from_db,
    count_chat_history, get_conversation_summary,
    save_conversation_summary, fetch_history_to_summarize
)
from utils.logging_config import setup_logging
from search_utils import bocha_search

logger = setup_logging()
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def build_messages(system_prompt: str, history: list, query: str,
                summary: str = "", max_chars: int = MEMORY_MAX_CHARS) -> list:
    """
    组装最终发给 LLM 的 messages：
    [系统提示] + [更早对话摘要] + [最近窗口历史] + [当前问题]
    history: [{"role","content"}, ...]，时间正序，且不含当前 query。
    """
    msgs = [{"role": "system", "content": system_prompt}]
    if summary:   # 有滚动摘要就作为一条 system 消息带上，补足窗口之外的长期记忆
        msgs.append({"role": "system", "content": "以下是你与用户更早对话的摘要，供参考：\n" + summary})

    # 字符预算裁剪：从最新往回累加，一旦超过 max_chars 就停止，保证最近的对话一定保留
    trimmed, total = [], 0
    for m in reversed(history or []):
        c = m.get("content", "")
        if total + len(c) > max_chars:
            break
        trimmed.append({"role": m["role"], "content": c})
        total += len(c)
    msgs.extend(reversed(trimmed))       # 恢复成时间正序

    msgs.append({"role": "user", "content": query})   # 当前问题永远放最后
    return msgs


# RAG核心函数，负责从向量检索知识并调用LLM生产答案
def stream_response_from_vectorstore(query, phone_number, history=None, summary=""):
    """意图1：知识库问答（流式）。先按主题路由选出该查的集合，只检索该域知识块
    拼进 system_prompt，再逐 token 产出回答。
    """
    # 1. 主题路由 + 按路由结果逐集合检索。
    #    改前这里是"无条件查 market top-2 + java top-2"，router_lab.py 实测：
    #    40 个上下文槽里 20 个是跨域无关内容（50%），且 market 永远拼在前面，
    #    所以 Java 问题即使检到正文也只能落在 [3][4]。
    #    现在不该参与的集合整个不查：跨域槽 50% → 0%，目标文档回到 [1]。
    pairs = []
    for coll_name, k in retrieval_plan(query):
        try:
            docs, metas = query_collection(coll_name, query, n_results=k)
            pairs.extend(zip(docs, metas))
        except Exception as e:
            # 单个集合检索失败只少一份参考材料，不该让整个回答挂掉
            logger.error("知识库检索失败(%s): %s", coll_name, e)

    # 2. 按文档文本去重（相同内容只留一份，并保留它的来源）
    seen, unique_pairs = set(), []
    for doc, meta in pairs:
        if doc not in seen:
            seen.add(doc)
            unique_pairs.append((doc, meta))

    # 3. 给每条知识编号并附上来源，模型就能在答案里用 [编号] 指代它
    lines = []
    for i, (doc, meta) in enumerate(unique_pairs, start=1):
        src = (meta or {}).get("source", "未知来源")
        lines.append(f"[{i}] {doc}（来源：{src}）")
    context = "\n".join(lines) if lines else "暂无相关知识。"
    
    # 4. 获取用户信息（用于个性化）
    user_info = get_user_info_from_db(phone_number)
    if user_info:
        user_info_str = f"姓名: {user_info['name']}, 年龄: {user_info['age']}, 职业: {user_info['occupation']}, 兴趣: {user_info['interest']}"
    else:
        user_info_str = "新用户，暂无记录。"
    
    # 5. 构建系统提示词：
    #   “根据以下知识库内容” → 要求模型以知识库为基础。
    #   “尽量引用” → 允许模型在知识库基础上做总结和扩展（而非严格复制）。
    #   模型会优先使用知识库事实，同时利用自身语言能力进行润色、结构化和补充，这通常能提升回答的可读性和完整性。
    system_prompt = (
    "你是一个专业的技术知识助手。根据以下知识库内容和用户信息，回答用户的问题。\n"
    "知识库内容：\n" + context + "\n\n"
    "用户信息：\n" + user_info_str + "\n"
    "请以清晰、准确、结构化的方式回答，尽量引用知识库中的内容。\n"
    "引用知识库某条内容时，请在该句末尾用方括号标注其编号，如 ……[1]；\n"
    "只允许使用上面知识库中出现过的编号，不要编造。"
    )

    yield from _stream_llm(build_messages(system_prompt, history, query, summary), 0.7, 500)


def stream_other_intents(query, phone_number, history=None, summary=""):
    """意图0：闲聊/兜底（流式）。不查知识库，只带上用户信息做个性化寒暄。"""
    # 1. 获取用户信息（同样用于个性化）
    user_info = get_user_info_from_db(phone_number)
    if user_info:
        user_info_str = f"姓名: {user_info['name']}, 年龄: {user_info['age']}, 职业: {user_info['occupation']}, 兴趣: {user_info['interest']}"
    else:
        user_info_str = "新用户"
    
    # 2. 构建提示词（无知识库）
    system_prompt = f"你是一个友好的技术助手。用户信息：{user_info_str}。请恰当回复用户。"

    yield from _stream_llm(build_messages(system_prompt, history, query, summary), 0.7, 300)


# 网络检索
def stream_web_search(query, phone_number, history=None, summary=""):
    """意图2：联网搜索（流式）。调用博查 API 拿实时结果，拼成上下文交给 LLM 总结。
    注意：本函数是生成器，所有"要给用户看的输出"都必须用 yield，
    用 return <值> 不会把值发出去（只会抛 StopIteration），会导致空回复。"""
    if not BOCHA_API_KEY:
        yield "网络搜索功能未配置，请联系管理员。"; return    # ← 原 return 改成 yield+return
    result = bocha_search(query, count=5)
    if "error" in result:
        yield f"搜索失败: {result['error']}"; return
    
    # 检查返回码
    if result.get('code') != 200:
        yield f"搜索服务返回错误: {result.get('msg', '未知错误')}"; return
    
    # 提取搜索结果
    data = result.get('data', {})
    web_pages = data.get('webPages', {}).get('value', [])
    
    if not web_pages:
        yield "未找到与您问题相关的信息。"; return
    
    # 构建搜索结果上下文
    context = "以下是网络搜索到的相关信息：\n\n"
    for idx, page in enumerate(web_pages, start=1):
        title = page.get('name', '无标题')
        snippet = page.get('summary', page.get('snippet', '无摘要'))
        url = page.get('url', '')
        site = page.get('siteName', '未知来源')
        
        context += f"【{idx}】{title}\n"
        context += f"来源: {site}\n"
        context += f"摘要: {snippet}\n"
        if url:
            context += f"链接: {url}\n"
        context += "\n"
    
    # 调用大模型生成最终回答
    system_prompt = (
        "你是一个专业的技术知识助手。请根据以下网络搜索结果，回答用户的问题。\n"
        "要求：\n"
        "1. 综合多个搜索结果，给出全面准确的回答\n"
        "2. 如果搜索结果有矛盾，请说明不同来源的观点\n"
        "3. 如果搜索结果不足以回答问题，请如实告知用户\n"
        "4. 引用搜索结果时，请标注来源\n\n"
        f"{context}"
    )

    yield from _stream_llm(build_messages(system_prompt, history, query, summary), 0.5, 800)

def _stream_llm(messages, temperature, max_tokens):
    """统一的流式 LLM 调用：把 OpenAI stream 逐 token 产出。
    三个 stream_* 共用它，避免各写一遍 for-event-yield 循环。
    （它虽定义在调用者之后，但没关系：模块加载完才会真正调用，那时它早已存在。）"""
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            temperature=temperature, max_tokens=max_tokens, stream=True)
        for event in stream:
            # 兼容端点的收尾 chunk 可能 choices 为空（只带 usage/结束标记），
            # 直接取 [0] 会 IndexError，必须先跳过空 choices
            if not event.choices:
                continue
            delta = event.choices[0].delta.content or ""
            if delta:
                yield delta
    except Exception as e:
        logger.error("流式生成失败: %s", e)
        yield "抱歉，我现在无法处理您的请求，请稍后再试。"

# 数据库检索
def handle_db_query(query, phone_number):
    """意图3：让 SQL Agent 回答"我的信息"类问题。
    安全关键：绝不让 Agent 直接连主库，否则它能查到所有用户的数据（数据越权）。
    做法是临时建一个"只装了当前用户一行、且不含 password 列"的内存库喂给 Agent，
    从物理上把它的可见范围锁死在这一个用户身上。"""
    # sqlite:// 是纯内存库；StaticPool 保证多次连接复用同一个内存实例，
    # 否则每次 connect 都会得到一个全新的空库，前面建的表就丢了。
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                        poolclass=StaticPool)          # 关键：共享同一内存库
    with engine.begin() as conn:
        # 只建 5 个字段，故意不建 password 列 —— 即便 Agent 想查密码也无从查起
        conn.execute(text("""CREATE TABLE my_info(
            phone_number TEXT, name TEXT, age INTEGER,
            occupation TEXT, interest TEXT)"""))          # 无 password 列
    # 从真实主库里，只捞出"当前登录用户"这一行、且只捞非敏感列
    src = sqlite3.connect(SQLITE_DB_PATH)
    rows = src.execute(
        "SELECT phone_number,name,age,occupation,interest "
        "FROM user_info WHERE phone_number=?", (phone_number,)).fetchall()
    src.close()
    # 把这一行灌进内存库，Agent 后续能看到的数据就只有它
    with engine.begin() as conn:
        conn.executemany(
            text("INSERT INTO my_info VALUES (:p,:n,:a,:o,:i)"),
            [{"p":r[0],"n":r[1],"a":r[2],"o":r[3],"i":r[4]} for r in rows])

    db = SQLDatabase(engine)                              # ✅ 传 engine
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL, temperature=0)
    # SQL Agent：让 LLM 自己把自然语言翻译成 SQL 在这个隔离库上执行
    agent = create_sql_agent(llm=llm, toolkit=SQLDatabaseToolkit(db=db, llm=llm),
                            verbose=True, handle_parsing_errors=True)
    return agent.invoke({"input": query}).get('output', '无结果')


def update_and_get_summary(phone_number: str) -> str:
    """
    返回当前用户的滚动摘要；若历史够长且有新消息未总结，则调用 LLM 增量压缩后再返回。
    """
    summary, upto = get_conversation_summary(phone_number)
    total = count_chat_history(phone_number)
    if total <= SUMMARY_THRESHOLD:        # 对话还短，窗口装得下，不浪费一次 LLM 调用
        return summary

    msgs, new_upto = fetch_history_to_summarize(phone_number, upto, MEMORY_WINDOW)
    if not msgs:                          # 没有窗口之外的新消息，无需总结
        return summary

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    prompt = (
        "你是对话摘要助手。请把【已有摘要】和【新增对话】合并，压缩成一段简洁中文摘要，"
        "保留关键事实（用户身份/需求、讨论过的技术点、结论）。不要遗漏重要信息，不要编造。\n\n"
        f"【已有摘要】\n{summary or '（无）'}\n\n【新增对话】\n{convo}\n\n只输出更新后的摘要："
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=500
        )
        new_summary = resp.choices[0].message.content.strip()
        save_conversation_summary(phone_number, new_summary, new_upto)   # 存库并推进游标
        return new_summary
    except Exception as e:
        logger.error("历史摘要压缩失败: %s", e)
        return summary        # 压缩失败不影响主流程，退回旧摘要