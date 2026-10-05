# graph_flow.py —— 用 LangGraph 状态图编排整个对话流程（替代 main_flow 的 if/else）


from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer          # 节点内逐 token 外发的写入口

from db.sqlite_db import fetch_chat_history, update_chat_history, count_chat_history
from config import PROMPT_TEMPLATES, MEMORY_WINDOW
from intent import detect_intent_code
from response_generator import (
    handle_db_query, update_and_get_summary,
    stream_response_from_vectorstore, stream_web_search, stream_other_intents,
)
from function_calling_handler import handle_function_calling
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


def _stream_out(gen) -> str:
    """消费一个 token 生成器：边收边用 writer 外发（实现 SSE 逐字流式），并累积完整文本返回。"""
    writer = get_stream_writer()
    parts = []
    for tok in gen:
        parts.append(tok)
        writer({"token": tok})       # 每个 token 推给 stream_mode="custom" 的消费端
    return "".join(parts)


# ---------- 节点 ----------
def prepare(state: ChatState) -> dict:
    """取历史/判新用户/取摘要/存 user 消息。

    改前这里对新用户无条件先 writer 一句欢迎语，但此刻意图还没分类，
    于是"新用户的第一个知识问答"前面也挂了一句『……请问有什么可以帮您？』，
    读起来像两句问候叠在一起。现在只标记 is_new_user，
    发不发交给闲聊分支决定：问的是实质问题就直接给答案，不塞寒暄。
    """
    phone, query = state["phone_number"], state["query"]
    history = fetch_chat_history(phone, limit=MEMORY_WINDOW)
    # 必须在 update_chat_history 之前数：存完这条 user 消息就不再是 0 了
    is_new_user = (count_chat_history(phone) == 0)
    summary = update_and_get_summary(phone)
    update_chat_history(phone, "user", query)
    return {"history": history, "summary": summary,
            "is_new_user": is_new_user, "prefix": ""}


def classify(state: ChatState) -> dict:
    return {"intent": detect_intent_code(state["query"])}


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
    """流结束后把 欢迎语+答案 一起存进历史。"""
    update_chat_history(state["phone_number"], "assistant",
                        state.get("prefix", "") + state.get("answer", ""))
    return {}


def route_intent(state: ChatState) -> str:
    """条件边：按 intent 选下一个节点。"""
    return {1: "rag", 2: "web", 3: "db", 4: "fc"}.get(state.get("intent"), "chat")


# ---------- 建图 ----------
def build_graph():
    g = StateGraph(ChatState)
    g.add_node("prepare", prepare)
    g.add_node("classify", classify)
    g.add_node("rag", rag_node)
    g.add_node("web", web_node)
    g.add_node("db", db_node)
    g.add_node("fc", fc_node)
    g.add_node("chat", chat_node)
    g.add_node("finalize", finalize)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "classify")
    g.add_conditional_edges("classify", route_intent, ["rag", "web", "db", "fc", "chat"])
    for n in ("rag", "web", "db", "fc", "chat"):
        g.add_edge(n, "finalize")
    g.add_edge("finalize", END)
    return g.compile()


chat_graph = build_graph()   # 模块级编译一次，app.py 直接复用