# kb_router.py
"""主题路由：在向量检索之前，用零成本的关键词打分决定"该查哪个向量集合"。

背景（router_lab.py 实测）：改前 response_generator 对每个 query 都无条件查
market top-2 + java top-2，40 个上下文槽里有 20 个是跨域无关内容（50%），
而且 market 的两条永远被拼在前面，导致 Java 问题即使检到正文也只能排 [3]。
本模块把"不该参与的集合"整个去掉：实测跨域槽 50% → 0%。

设计约束：
1. 纯函数、不联网、不加载模型 —— 判定耗时≈0、不花钱、结果可复现，可直接被 pytest 覆盖；
2. 误判方向必须偏"多查"而不是"少查"：独判只省一次廉价检索，
    而判错会让该查的集合一个都没查、上下文直接为空，所以只放过"顺带提及"。
"""

from config import COLLECTION_MARKET, COLLECTION_COURSE

# 只收"出现即强烈指向该域"的词，宁窄勿宽。
# 已知偏差：按子串计数，所以 "rag" 会同时命中 "rag" 和 "rag-fusion"，
# 让分数整体偏高一点点。这里要的是两侧谁更有证据，不是精确词频，偏差无害。
LLM_KEYWORDS = [
    "大模型", "小模型", "模型", "训练", "推理", "微调", "预训练", "量化", "显存",
    "参数", "梯度", "注意力", "自回归", "嵌入", "向量", "语义", "召回", "重排",
    "分块", "chunk", "提示", "prompt", "幻觉", "评测", "评估",
    "rag", "rag-fusion", "rrf", "bm25", "embedding", "llm", "gpt", "bert",
    "transformer", "token", "lora", "peft", "qlora", "adalora", "zero",
    "sft", "rlhf", "langchain", "langgraph", "agent", "智能体",
    "多轮对话", "长期记忆", "流式", "sse", "意图", "强化学习", "自然语言",
]
JAVA_KEYWORDS = [
    "java", "jvm", "jdk", "字节码", "类加载", "面向对象", "抽象类", "接口",
    "多态", "重载", "重写", "final", "static", "volatile", "transient",
    "synchronized", "锁", "线程", "并发", "协程", "线程池", "死锁",
    "垃圾回收", "gc", "内存泄漏", "引用队列", "可达性",
    "hashmap", "concurrenthashmap", "arraylist", "linkedlist", "集合框架",
    "单例", "工厂模式", "代理模式", "设计模式", "spring", "springboot",
    "mybatis", "ioc", "aop", "bean", "事务", "索引", "mysql", "sql",
]

TOP_K_SINGLE = 4   # 独判：一个域独占全部槽位，取 4 条
TOP_K_SHARED = 2   # 双查：两域各取 2 条，总数仍是 4，不把 prompt 撑大
LOSER_DUAL_THRESHOLD = 2   # 落选侧命中 ≥ 此数 → 视为两侧都有实质证据


def score_domains(query: str):
    """返回 (llm得分, java得分)：各自关键词在 query 里的出现次数之和。"""
    q = (query or "").lower()
    return (sum(q.count(k) for k in LLM_KEYWORDS),
            sum(q.count(k) for k in JAVA_KEYWORDS))


def route_collections(query: str):
    """返回本次该查哪些集合。三条规则按顺序判：

    1. 同分（含 0/0 的"完全没关键词"）→ 两个都查。不知道派给谁就别派。
    2. 落选侧命中 ≥ 2 → 真跨域问题（如"用 LangChain 做向量检索时怎么加锁"），两个都查。
    3. 其余情况（落选侧 0 或 1 次，属顺带提及）→ 只查得分更高的那个域。
    """
    llm, java = score_domains(query)
    if llm == java or min(llm, java) >= LOSER_DUAL_THRESHOLD:
        return [COLLECTION_MARKET, COLLECTION_COURSE]
    return [COLLECTION_MARKET if llm > java else COLLECTION_COURSE]


def retrieval_plan(query: str):
    """给调用方的唯一入口：返回 [(集合名, 该取几条), ...]。

    生产链路（response_generator）和离线评估（eval_rag.retrieve_contexts）
    必须都用这个函数生成检索计划 —— 否则评估测的就不是线上那条链路了。
    """
    cols = route_collections(query)
    k = TOP_K_SHARED if len(cols) > 1 else TOP_K_SINGLE
    return [(c, k) for c in cols]


if __name__ == '__main__':
    # 自检：不碰向量库、不加载模型，只验判定逻辑
    for q in ["LoRA 的原理和主要特点是什么？", "HashMap 的底层实现原理是什么？",
            "用 LangChain 做向量检索时怎么加锁防止并发问题？",
            "RAG 系统里向量库的接口怎么设计？", "扫码加老师领取更多资料"]:
        llm, java = score_domains(q)
        names = {"market_knowledge": "LLM", "java_knowledge": "JAVA"}
        plan = ", ".join(f"{names.get(c, c)}×{k}" for c, k in retrieval_plan(q))
        print(f"  {llm}/{java}  ->  {plan:<12} {q}")