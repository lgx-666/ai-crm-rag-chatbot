# graph_flow.py —— 用 LangGraph 状态图编排整个对话流程（替代 main_flow 的 if/else）

import hashlib
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer          # 节点内逐 token 外发的写入口

from db.sqlite_db import (fetch_chat_history, update_chat_history, count_chat_history,
                        get_user_info_from_db)
from config import PROMPT_TEMPLATES, MEMORY_WINDOW
from intent import detect_intent_code
from response_generator import (
    handle_db_query, update_and_get_summary,
    stream_response_from_vectorstore, stream_web_search, stream_other_intents,
)
from function_calling_handler import handle_function_calling
from observability import trace_meta
import qa_cache
from utils.logging_config import setup_logging

logger = setup_logging()


# ---------- 图的状态：在节点之间流转的共享数据 ----------
class ChatState(TypedDict, total=False):
    phone_number: str
    query: str
    history: List[Dict[str, str]]   # 最近窗口历史
    summary: str                    # 窗口之外的滚动摘要
    is_new_user: bool
    prefix: str                     # 新用户欢迎语（要最先流出去）
    intent: int
    answer: str                     # handler 产出的完整答案
    scope: str                    # 用户画像签名，进缓存 key，防跨用户串答案
    cache_answer: str             # 非空即代表缓存命中


def _stream_out(gen) -> str:
    """消费一个 token 生成器：边收边用 writer 外发（实现 SSE 逐字流式），并累积完整文本返回。"""
    writer = get_stream_writer()
    parts = []
    for tok in gen:
        parts.append(tok)
        writer({"token": tok})       # 每个 token 推给 stream_mode="custom" 的消费端
    return "".join(parts)


def profile_scope(phone: str) -> str:
    """把"会改变答案文本的个性化输入"压成短签名，混进缓存 key。
    依据：stream_response_from_vectorstore 把姓名/年龄/职业/兴趣拼进了 system_prompt
    （response_generator.py:89-93、102）。不带签名的全局缓存会把 A 的带画像答案推给 B。
    四项全空时签名恒定 => 没填资料的访客之间仍能跨用户命中（演示与评估的常态）。
    手机号绝不进签名：那等于零跨用户命中，还把 PII 写进 key。"""
    info = get_user_info_from_db(phone) or {}
    parts = [str(info.get(k) or "") for k in ("name", "age", "occupation", "interest")]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:12]


# ---------- 节点 ----------
def prepare(state: ChatState) -> dict:
    """取历史/判新用户/取摘要/存 user 消息。

    改前这里对新用户无条件先 writer 一句欢迎语，但此刻意图还没分类，
    于是"新用户的第一个知识问答"前面也挂了一句『……请问有什么可以帮您？』，
    读起来像两句问候叠在一起。现在只标记 is_new_user，
    发不发交给闲聊分支决定：问的是实质问题就直接给答案，不塞寒暄。
    """
    phone, query = state["phone_number"], state["query"]
    trace_meta("chat-request", phone) 
    history = fetch_chat_history(phone, limit=MEMORY_WINDOW)
    # 必须在 update_chat_history 之前数：存完这条 user 消息就不再是 0 了
    is_new_user = (count_chat_history(phone) == 0)
    summary = update_and_get_summary(phone)
    update_chat_history(phone, "user", query)
    return {"history": history, "summary": summary,
        "is_new_user": is_new_user, "prefix": "", "scope": profile_scope(phone)}


def classify(state: ChatState) -> dict:
    return {"intent": detect_intent_code(state["query"])}


def cache_lookup(state: ChatState) -> dict:
    """读缓存，位置在 classify 之前：命中就同时省掉意图分类 + 答案生成两次 LLM 调用。
    读端不设"仅首轮"限制——能命中的条目必然由首轮写入（见 rag_node 的写端门禁），
    它本来就与对话上下文无关，所以老用户重复提问也该秒回。"""
    hit = qa_cache.get(state["query"], state.get("scope", ""))
    if hit:
        logger.info("[问答缓存] 命中")
        return {"cache_answer": hit["answer"]}
    logger.info("[问答缓存] 未命中")
    return {"cache_answer": ""}


def route_cache(state: ChatState) -> str:
    return "cache_hit" if state.get("cache_answer") else "classify"


def cache_hit_node(state: ChatState) -> dict:
    """命中也必须走 _stream_out，不能另开一条 return 路径：
    不变式是"流出去的字 == 存进历史的字"，绕开它 finalize 就拿不到答案，
    历史里会凭空少一大段，下一轮的记忆是残的。"""
    return {"answer": _stream_out(iter([state["cache_answer"]]))}


def rag_node(state: ChatState) -> dict:      # 意图1 知识库
    gen = stream_response_from_vectorstore(state["query"], state["phone_number"],
                                        state["history"], state["summary"])
    return {"answer": _stream_out(gen)}

def web_node(state: ChatState) -> dict:      # 意图2 联网
    gen = stream_web_search(state["query"], state["phone_number"],
                            state["history"], state["summary"])
    return {"answer": _stream_out(gen)}

def db_node(state: ChatState) -> dict:       # 意图3 查库（非流式，包成单元素生成器）
    return {"answer": _stream_out(iter([handle_db_query(state["query"], state["phone_number"])]))}

def fc_node(state: ChatState) -> dict:       # 意图4 改个人信息（非流式）
    return {"answer": _stream_out(iter([handle_function_calling(
        state["query"], state["phone_number"], state["history"], state["summary"])]))}

def chat_node(state: ChatState) -> dict:     # 意图0 闲聊
    """只有寒暄类首问才发欢迎语。"""
    prefix = ""
    if state.get("is_new_user"):
        prefix = PROMPT_TEMPLATES["greeting"] + "\n\n"
        get_stream_writer()({"token": prefix})   # 欢迎语必须排在答案之前
    gen = stream_other_intents(state["query"], state["phone_number"],
                            state["history"], state["summary"])
    return {"answer": _stream_out(gen), "prefix": prefix}


def finalize(state: ChatState) -> dict:
    """流结束后把 欢迎语+答案 存进历史，然后才写缓存。

    顺序是刻意的：缓存写在 rag_node 里会早历史一个节点，客户端恰好在这两步
    之间断开（Ctrl+C、关窗口、断网）就留下"缓存有、历史缺"——缓存内容本身没错，
    但提问者自己的下一轮记忆里挂着一个没有回答的问题。挪到历史之后，最坏退化成
    "历史全、缓存无"，代价只是下次多付一次生成。

    写入门禁三条：意图1（只缓存知识库问答）；首轮（history 为空 => summary 也空，
    答案 = query + 画像 + 知识库 的确定性函数，不带对话痕迹，才允许跨用户共享）；
    命中路径不写（它没走 classify，intent 缺席，None != 1 天然跳过，
    所以热条目不会被反复续期、绕过 TTL）。"""
    update_chat_history(state["phone_number"], "assistant",
                        state.get("prefix", "") + state.get("answer", ""))
    if state.get("intent") == 1 and not state.get("history"):
        qa_cache.put(state["query"], state.get("answer", ""),
                    state["intent"], state.get("scope", ""))
    return {}


def route_intent(state: ChatState) -> str:
    """条件边：按 intent 选下一个节点。"""
    return {1: "rag", 2: "web", 3: "db", 4: "fc"}.get(state.get("intent"), "chat")


# ---------- 建图 ----------
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("prepare", prepare)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("cache_hit", cache_hit_node)
    g.add_node("classify", classify)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("db", db_node)
    g.add_node("fc", fc_node)
    g.add_node("chat", chat_node)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "cache_lookup")
    g.add_conditional_edges("cache_lookup", route_cache, ["cache_hit", "classify"])
    g.add_conditional_edges("classify", route_intent, ["rag", "web", "db", "fc", "chat"])
    for n in ("rag", "web", "db", "fc", "chat", "cache_hit"):
        g.add_edge(n, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


chat_graph = build_graph()   # 模块级编译一次，app.py 直接复用