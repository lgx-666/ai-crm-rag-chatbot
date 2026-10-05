# function_calling_handler.py
import logging
from functools import partial
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from db.sqlite_db import get_user_info_from_db, upsert_user_info
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, MEMORY_WINDOW

logger = logging.getLogger(__name__)

def handle_function_calling(query: str, phone_number: str,
                            history: list = None, summary: str = "") -> str:
    """
    使用原生 Function Calling 处理用户指令（意图4）
    """
    # 定义原始工具函数（绑定手机号）
    def _get_user_info() -> str:
        info = get_user_info_from_db(phone_number)
        if info:
            return f"姓名: {info['name']}, 年龄: {info['age']}, 职业: {info['occupation']}, 兴趣: {info['interest']}"
        return f"未找到手机号 {phone_number} 的用户信息"

    def _update_user_info(name: str = None, age: int = None, 
                        occupation: str = None, interest: str = None) -> str:
        success = upsert_user_info(phone_number, name, age, occupation, interest)
        if success:
            return f"用户 {phone_number} 信息更新成功"
        return f"用户 {phone_number} 信息更新失败（可能未提供任何字段）"

    # 包装为 LangChain 工具
    tools = [
        StructuredTool.from_function(
            func=_get_user_info,
            name="get_user_info",
            description="查询当前登录用户的个人信息，不需要任何参数。"
        ),
        StructuredTool.from_function(
            func=_update_user_info,
            name="update_user_info",
            description="更新当前登录用户的个人信息。参数：name(姓名), age(年龄), occupation(职业), interest(兴趣)，至少提供一个字段。"
        )
    ]

    # 初始化 LLM 并绑定工具
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0
    )
    llm_with_tools = llm.bind_tools(tools)

    # 构建消息：系统说明 + 摘要 + 最近历史 + 当前指令
    system_prompt = "你是个人信息管理助手，可调用工具查询或修改当前用户的姓名/年龄/职业/兴趣。"
    messages = [{"role": "system", "content": system_prompt}]
    if summary:
        messages.append({"role": "system", "content": "更早对话摘要：" + summary})
    for m in (history or [])[-MEMORY_WINDOW:]:        # 只取最近窗口，防止干扰工具判断
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": query})

    # 第一次调用：模型决定是否调用工具
    response = llm_with_tools.invoke(messages)

    # 如果模型决定调用工具
    if response.tool_calls:
        messages.append(response)  # 添加模型的响应到历史
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "get_user_info":
                result = _get_user_info()
            elif tool_name == "update_user_info":
                result = _update_user_info(**tool_args)
            else:
                result = f"未知工具: {tool_name}"

            # 将工具执行结果添加为 tool 角色消息
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call["id"]
            })

        # 第二次调用：将工具结果返回给模型，生成最终回答
        final_response = llm.invoke(messages)
        return final_response.content
    else:
        # 未调用工具，直接返回
        return response.content
    


