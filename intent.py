# intent.py
import re
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, PROMPT_TEMPLATES
from utils.logging_config import setup_logging

logger = setup_logging()
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# intent.py 新增（放在 detect_intent_code 之前）
def parse_intent(raw: str) -> int:
    """从 LLM 原始输出里解析意图编号：抓第一个数字，校验在 0-4 内，否则回退 0。
    这是纯函数——不联网、确定性输出，因此可被单元测试直接覆盖。"""
    if not raw:
        return 0
    match = re.search(r'\d+', raw.strip())
    if match:
        code = int(match.group())
        if code in (0, 1, 2, 3, 4):
            return code
    return 0

def detect_intent_code(query: str) -> int:
    from observability import generation, usage_from    # 意图调用是裸 openai client，不在 LangChain 体系内，要手工记
    prompt = PROMPT_TEMPLATES["intent_prompt"].format(query=query)
    msgs = [{"role": "user", "content": prompt}]
    gen = generation("intent-classify", MODEL_NAME, msgs,
                    {"temperature": 0, "max_tokens": 10})
    try:
        # 调用AI模型 API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            # 构造对话消息列表
            messages=msgs,
            # 控制生成文本的随机性。设为 0 表示确定性输出，即每次调用相同输入会得到最可能的结果（贪心解码）。
            temperature=0,
            # 限制模型输出最多 10 个 token（约 7-10 个字符）。
            max_tokens=10
        )
        result = response.choices[0].message.content
        gen.end(result, usage=usage_from(getattr(response, "usage", None)))
        return parse_intent(result)      # ← 解析交给纯函数
    except Exception as e:
        logger.error("意图识别失败: %s", e)
        gen.end(None, error=e)
        return 0

