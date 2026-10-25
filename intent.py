# intent.py
import re
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, PROMPT_TEMPLATES, QWEN_DISABLE_THINKING
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
    # 实测 2026-10-25：百炼兼容端点只认 extra_body，顶层 enable_thinking 直接 TypeError；
    # 生效后 completion 86→44、可见正文反而 61→73 字（_lab_thinking_probe.py 三变体对照）。
    # 这里关它零风险：思考产出的 reasoning_content 本来就被丢掉，而 baseline 实测烧掉 105 token。
    kwargs = {}
    if QWEN_DISABLE_THINKING:
        kwargs["extra_body"] = {"enable_thinking": False}
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=msgs,
            temperature=0,
            max_tokens=10,
            **kwargs
        )
        result = response.choices[0].message.content
        gen.end(result, usage=usage_from(getattr(response, "usage", None)))
        return parse_intent(result)      # ← 解析交给纯函数
    except Exception as e:
        logger.error("意图识别失败: %s", e)
        gen.end(None, error=e)
        return 0

