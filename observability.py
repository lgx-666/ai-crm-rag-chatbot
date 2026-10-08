# observability.py —— Langfuse 埋点的唯一入口。
# ENABLED 为 False 时下面每个函数都返回空操作对象，pytest / GitHub CI 不会因为
# 没有 Langfuse 服务而失败，连一行多余日志都不会打印。
import hashlib
from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

ENABLED = bool(LANGFUSE_HOST and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if ENABLED:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler


def trace_user_id(phone_number):
    # 原始手机号不进 trace：同一号码恒定映射到同一个短 id，能按人聚合会话，又不落 PII
    return hashlib.sha256(phone_number.encode("utf-8")).hexdigest()[:12]


def callback_handler():
    """一次 HTTP 请求 new 一个，挂到 graph.stream 的 config 上 => 一次请求一条 trace，
    每个节点自动变成一个 span。没开启时返回 None，调用方据此不传 config。"""
    return CallbackHandler() if ENABLED else None


def trace_meta(name, phone_number):
    """给当前 trace 打名字/用户/会话。必须在节点函数内部调用（要 LangGraph 的运行上下文）。"""
    if not ENABLED:
        return
    try:
        uid = trace_user_id(phone_number)
        Langfuse().update_current_trace(name=name, user_id=uid, session_id=uid)
    except Exception:
        pass          # 埋点自己挂了，绝不能带崩业务


def usage_from(u):
    """openai 的 usage 对象 -> Langfuse 的 usage_details。
    端点没回 usage 就返回 None：不拿字符数估一个假数字填进报表。"""
    if not u:
        return None
    return {"input": u.prompt_tokens or 0,
            "output": u.completion_tokens or 0,
            "total": u.total_tokens or 0}


class _Generation:
    """一次 LLM 调用的记录器。只写 messages / response / token 数，
    绝不写请求头——Authorization 里就是密钥，写进去等于把 #30 重演一遍。"""

    def __init__(self, obs):
        self._obs, self._done = obs, False

    def end(self, output, usage=None, error=None):
        if self._obs is None or self._done:
            return
        self._done = True
        try:
            self._obs.update(output=output)
            if error is not None:
                self._obs.update(level="ERROR", status_message=str(error)[:300])
            if usage:
                self._obs.update(usage_details=usage)
            self._obs.end()
        except Exception:
            pass


def generation(name, model, messages, parameters=None):
    """在节点内部调用会自动挂到该节点的 span 下（嵌套关系已实测）。"""
    if not ENABLED:
        return _Generation(None)
    try:
        obs = Langfuse().start_observation(name=name, as_type="generation", model=model,
                                        input=messages, model_parameters=parameters or {})
        return _Generation(obs)
    except Exception:
        return _Generation(None)