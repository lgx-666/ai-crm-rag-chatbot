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
    prompt = PROMPT_TEMPLATES["intent_prompt"].format(query=query)
    try:
        # 调用AI模型 API
        response = client.chat.completions.create(
            model=MODEL_NAME,
            # 构造对话消息列表
            messages=[{"role": "user", "content": prompt}],
            # 控制生成文本的随机性。设为 0 表示确定性输出，即每次调用相同输入会得到最可能的结果（贪心解码）。
            temperature=0,
            # 限制模型输出最多 10 个 token（约 7-10 个字符）。
            max_tokens=10
        )
        # 解析模型输出
        # response.choices[0].message.content：从 API 响应中提取模型生成的文本内容（即意图编号的字符串，可能包含多余空格或换行）。
        result = response.choices[0].message.content
        return parse_intent(result)      # ← 解析交给纯函数
    except Exception as e:
        logger.error("意图识别失败: %s", e)
        return 0