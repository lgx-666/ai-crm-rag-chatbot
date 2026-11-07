# observability.py —— Langfuse 埋点的唯一入口。
# ENABLED 为 False 时下面每个函数都返回空操作对象，pytest / GitHub CI 不会因为
# 没有 Langfuse 服务而失败，连一行多余日志都不会打印。
import hashlib
import re
from contextlib import contextmanager, ExitStack
from config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY

ENABLED = bool(LANGFUSE_HOST and LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)

if ENABLED:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler


def trace_user_id(phone_number):
    # 原始手机号不进 trace：同一号码恒定映射到同一个短 id，能按人聚合会话，又不落 PII
    return hashlib.sha256(phone_number.encode("utf-8")).hexdigest()[:12]

# 手机号有两类落点：作为 state 的键（app.py:132 把 {"phone_number": 明文} 当图的输入），
# 以及被拼进正文（日志串、用户在提问里自己打的号码）。前者按键名换，后者用正则兜。
# 负向环视是防止误伤长数字串里的一段（技术文档里的 11 位常量）。
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def _mask_node(data):
    if isinstance(data, dict):
        return {k: (trace_user_id(v) if k == "phone_number" and isinstance(v, str)
                    else _mask_node(v))
                for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_mask_node(x) for x in data]
    if isinstance(data, str):
        return _PHONE_RE.sub("<masked>", data)
    return data


def _mask_pii(data):
    """SDK 以 mask(data=...) 关键字调用 ⇒ 形参名必须叫 data。
    只作用于上报给 Langfuse 的那份数据，发给用户的 SSE 正文不经这里。
    出错就整体打码：宁可丢一条 trace 的正文，也不能把明文送出去。"""
    try:
        return _mask_node(data)
    except Exception:
        return "<masked>"


if ENABLED:
    # mask 只在【第一次】初始化时生效：resource_manager.py:101-102 对同一 public_key
    # 直接返回已注册实例并忽略新参数，client.py:289 的 self._mask 又取自那个共享实例
    # ⇒ 这里先建带 mask 的，之后 trace_meta/generation/span 里的无参 Langfuse() 会继承；
    # 反过来后补则永久失效。CallbackHandler 没有 mask 参数，靠 get_client() 继承（get_client.py:52）。
    Langfuse(mask=_mask_pii)

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

    def end(self, output, usage=None, error=None, metadata=None):
        if self._obs is None or self._done:
            return
        self._done = True
        try:
            self._obs.update(output=output)
            if metadata:
                try:
                    self._obs.update(metadata=metadata)   # 新参数名单独 try：不支持就只丢元数据
                except Exception:
                    pass
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
    

class _Span(_Generation):
    """检索这类"不花 token 但花时间"的步骤：与 _Generation 同一套收尾，只是没有 usage。"""

    def end(self, output=None, usage=None, error=None, metadata=None):
        super().end(output, usage=usage, error=error, metadata=metadata)


def span(name, input=None):
    """挂到当前节点的 span 下（与 generation() 同一条已实测的嵌套路径，只换 as_type）。
    没开启埋点时返回空操作对象，调用方不用写 if。"""
    if not ENABLED:
        return _Span(None)
    try:
        return _Span(Langfuse().start_observation(name=name, as_type="span", input=input))
    except Exception:
        return _Span(None) 
    

@contextmanager
def eval_span(name, input=None):
    """给离线脚本用的父 span：yield 出 trace_id，没开埋点时 yield ""。
    为什么要它：generation() 在没有上下文时自建根 trace，调用方拿不到那个 id
    ⇒ CSV 里没有 trace_id 列，分数只能靠内容指纹反查。显式开一条父 span，
    里面的 generation 会自动挂进来（_lab_traceid_53.py 实测嵌套成立）。
    用 ExitStack 而不是把 with 整个塞进 try：主体抛异常时不能吞掉它，
    否则 @contextmanager 会 yield 两次并报一个看不出所以然的 RuntimeError。"""
    if not ENABLED:
        yield ""
        return
    stack = ExitStack()
    tid = ""
    try:
        span = stack.enter_context(Langfuse().start_as_current_span(name=name, input=input))
        tid = span.trace_id or ""
    except Exception:
        tid = ""
    try:
        yield tid
    finally:
        stack.close()       # 关掉 span；主体若抛异常，异常在 close 之后继续往外传   