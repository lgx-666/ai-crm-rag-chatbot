# eval_rag.py — RAGAS 离线评估脚本（开发工具，不进生产镜像）
# 复用项目真实检索(query_collection)与生成(stream_response_from_vectorstore)，
# 用百炼 qwen 当裁判 LLM、本地 bge 当 embedding，给 RAG 质量打可量化分数。
#
# 四个指标各自回答一个问题：
#   Faithfulness          生成的答案是否只基于检索到的上下文（查"幻觉"）
#   AnswerRelevancy       答案是否切题（靠反向生成问题再算向量相似度）
#   ContextPrecision      检索回来的上下文是否相关、是否排得靠前（需要 reference）
#   ContextRecall         回答 reference 所需的信息是否都被检索到（需要 reference）
# 后两个是"检索质量"指标，与生成本身无关；它们必须有 ground truth，
# reference 已按 knowledge/llm 下对应 PDF 原文逐条核对，不能拿本项目的回答当答案。

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME, EMBEDDING_MODEL
from db.vector_db import init_vectorstores, query_collection
from kb_router import retrieval_plan
from response_generator import stream_response_from_vectorstore

# 前两个指标在 ragas 0.2.x 里的类名；若你装的版本叫别的名字，
# 用 `python -c "import ragas.metrics as m; print([x for x in dir(m) if 'Context' in x])"` 查，
# 然后把下面元组里的名字换成实际存在的。
_CTX_PRECISION_NAMES = ("LLMContextPrecisionWithReference", "ContextPrecision")
_CTX_RECALL_NAMES = ("LLMContextRecall", "ContextRecall")

import ragas.metrics as _ragas_metrics

def _pick_metric_name(candidates):
    """从候选类名里挑出当前 ragas 版本真的有的那个，都没有就报错提示"""
    for name in candidates:
        if hasattr(_ragas_metrics, name):
            return getattr(_ragas_metrics, name)
    raise ImportError(
        f"当前 ragas 版本没有 {candidates} 中的任何一个指标类，"
        f"请用 dir(ragas.metrics) 查实际类名后改 _CTX_*_NAMES"
    )

# 评估问题集：(问题, reference 标准答案) 元组列表。
# reference 的事实点均来自 knowledge/llm 下的对应 PDF，已逐条核对原文：
#   Q1 -> 24-RAG 优化策略 RAG-Fusion 篇
#   Q2 -> 52-图解分布式训练（八）ZeRO 学习
#   Q3 -> 29-LoRA 系列篇（+ 26-PEFT 面的 QLoRA/AdaLoRA 小节）
#   Q4 -> 12-多轮对话中让AI保持长期记忆的8种优化方式篇
#   Q5 -> 18-RAG 版面分析——文本分块面
#   Q6 -> 26-大模型（LLMs）参数高效微调(PEFT) 面
QUESTIONS = [
    ("RAG-Fusion 是怎么做的？RRF 起什么作用？",
    "RAG-Fusion 用来解决单一查询的局限：用户不擅长向搜索系统表达意图，线性地把查询映射到答案会漏掉顶部结果之外的信息。"
    "它的做法是先利用提示工程让大模型基于原始查询生成多个查询，这些查询不是随机变化，而是提供原始问题的不同视角，"
    "每个查询各自做向量检索得到一个排序列表；再用 RRF（逆向排名融合 / Reciprocal Rank Fusion）"
    "把这些搜索结果列表的排名结合成单一统一排名："
    "函数接收一个字典，键是查询、值是按该查询相关性排名的文档 ID 列表，"
    "RRF 基于每个文档在不同列表中的排名为它计算一个新分数，再按这些融合分数降序排序，得到最终重排列表。"
    "RRF 起的作用正是这个融合环节——它不依赖各检索系统分配的绝对分数（不同查询结果的分数尺度和分布不可比），"
    "只依赖相对排名，因此最适合把多个结果列表合并，让在多个列表中都靠前的文档出现在最终列表顶部。"
    "生成输出时会把重排后的文档和所有查询一起放进 LLM 提示，并在提示工程中更重视原始查询以保留用户意图。"
    "代价是输出可能过于冗长，且多查询与多文档集会挤压上下文窗口。"),

    ("ZeRO 的三个阶段分别优化了什么？",
    "ZeRO 是一种显存优化的数据并行方案，动机是 DataParallel 需要每张卡都存一整个模型，显存成为制约模型规模的主要因素"
    "（例：GPT-2 1.5B 参数用 fp16 只需 3GB，但模型状态实际耗费 24GB）。"
    "核心思想是去除数据并行中的冗余参数，使每张卡只存储一部分模型状态，从而减少显存占用。"
    "每张卡显存分为两类：模型状态（参数、梯度、优化器状态，其中优化器状态占 75%，是首要优化对象）和剩余状态（激活值、临时缓冲区、显存碎片）。"
    "对模型状态的优化方法是分片：每张卡只存 1/N 的模型状态量，系统内只维护一份模型状态。"
    "三个优化阶段分别对应模型状态三要素的分片："
    "ZeRO-1 优化器状态分区（Pos），内存减少 4 倍，通信量与数据并行相同；"
    "ZeRO-2 在此基础上添加梯度分区（Pos+g），内存减少 8 倍，通信量仍与数据并行相同；"
    "ZeRO-3 添加参数分区（Pos+g+p），内存减少与数据并行度 Nd 成线性关系（如 64 卡减少到 1/64），GPU 通信量略有增加约 50%。"),

    ("LoRA 的原理和主要特点是什么？",
    "LoRA 通过低秩分解来模拟参数的改变量，从而以极小的参数量实现大模型的间接训练。"
    "实现思想是冻结一个预训练模型的矩阵参数，选择用 A 和 B 两个矩阵替代，在下游任务微调时只更新 A 和 B；"
    "A 是降维矩阵、B 是升维矩阵（先降维再升维），训练时原模型固定。"
    "初始化方式：A 用高斯分布初始化，B 初始化为全 0，保证训练开始时旁路是 0 矩阵、维持网络原有输出；"
    "若 A、B 全 0 初始化会像深度网络全 0 初始化一样容易梯度消失，全高斯初始化则一开始就引入过大偏移 ΔW 和噪声、难以收敛。"
    "特点：推理时可将 BA 加到原参数 W 上，不引入额外推理延迟；可插拔式切换任务（当前任务 W0+B1A1，减掉 LoRA 换成 B2A2）；"
    "一个中心模型服务多个下游任务、节省参数存储量；与其它参数高效微调方法正交、可有效组合。"
    "Rank 选取：作者对比 1-64，Rank 在 4-8 之间最好，再高没有提升，指令微调因指令分布广需测 8 以上；"
    "alpha 是缩放参数、本质与 learning rate 相同，可简化为 alpha=rank 只调 lr。"
    "作用于哪些矩阵：把可微调参数全放进 attention 的某一个矩阵效果不好，平均分配到 Wq 和 Wk 效果最好，秩仅取 4 就能在 ΔW 中获得足够信息。"
    "QLoRA 是先用高精度技术把预训练模型量化为 4bit，再添加一小组可学习的低秩适配器权重、通过量化权重的反向传播梯度微调，显著降低显存要求但训练速度慢于 LoRA；"
    "AdaLoRA 根据重要性评分动态分配参数预算，给关键增量矩阵高秩、给不重要的降秩以防过拟合省预算。"
    "缺点：参与训练的参数量只有百万到千万级，效果比全量微调差，且不节省训练时间。"),

    ("多轮对话中让大模型保持长期记忆有哪些方式？",
    "在大模型 Agent 中，长期记忆的状态维护是关键组件之一，LangChain 提供了 8 种记忆维护方式，按场景选用："
    "1) ConversationBufferMemory 获取全量历史对话，适合客服这类需要记住整段对话（如先问账单再问网络）的场景；"
    "2) ConversationBufferWindowMemory 用滑动窗口只保留最近 k 轮（如 k=1 只留最后一次互动），适合商品咨询这类需要聚焦最新问题的场景；"
    "3) ConversationEntityMemory 提取历史对话中的实体及其属性，适合法律咨询记住案件名、条款、当事人；"
    "4) ConversationKGMemory 用知识图谱保存历史对话中的实体及其联系，适合医疗咨询把症状与病史关联；"
    "5) ConversationSummaryMemory 对历史对话做阶段性总结摘要，适合教育辅导累积答疑要点；"
    "6) ConversationSummaryBufferMemory 兼顾最近几次交互的详细信息与更早历史的摘要，适合长期技术故障排查；"
    "7) ConversationTokenBufferMemory 按 token 数回溯最近和最关键的对话信息，避免记忆过多造成信息混淆，适合金融咨询；"
    "8) VectorStoreRetrieverMemory 把对话存入向量库、按当前问题检索最相关片段，即使这些信息在历史中不是最新的也能召回，适合新闻事件问答。"),

    ("RAG 中的文本分块有哪些策略？",
    "为什么需要分块：一次性提取整篇长文档的嵌入会有信息丢失风险（捕捉了整体上下文却忽略特定主题的重要信息），且分块大小是关键限制因素（如 GPT-4 有 32K 窗口限制），因此处理长文档要分块而非整篇处理。"
    "常见策略：1) 一般按限制长度切分，用 chunk_size 步进切片，缺点是长句子容易被从中间切开；"
    "2) 正则拆分，用中文标点（句号、问号、感叹号、分号）和换行作为句子结束标志切句，实现简单，多数情况下足够；"
    "3) Spacy Text Splitter，借助 NLP 库的 doc.sents 分句，能在分割的同时保留上下文信息；"
    "4) langchain CharacterTextSplitter，主要参数 chunk_size、chunk_overlap、separator、strip_whitespace；"
    "5) langchain 递归字符切分 RecursiveCharacterTextSplitter，不需手动设分隔符，默认按 \\n\\n（段落）、\\n（换行）、空格、字符逐级尝试，块太大就换下一个分隔符；"
    "6) HTMLHeaderTextSplitter，结构感知，在 HTML 元素级别拆分并为每块附加对应标题元数据，只提取 headers_to_split_on 指定的标题；"
    "7) MarkdownHeaderTextSplitter，按 Markdown 语法规则（标题、代码块、图片、列表）分块；"
    "8) PythonCodeTextSplitter 与 9) LatexTextSplitter 属于代码/专用格式拆分，按类、函数、章节、小节等逻辑单元建块，其中代码分块的 chunk_overlap 要设为 0，因为任何重叠的代码都可能完全改变其原有含义。"
    "分块后需提取嵌入并存入向量数据库（文中示例用无需配置、开源、无服务器、数据落盘的 LanceDB）。"),

    ("什么是 PEFT（参数高效微调）？",
    "PEFT（参数高效微调）旨在通过最小化微调参数的数量和计算复杂度来提高预训练模型在新任务上的性能，缓解大型预训练模型的训练成本，使计算资源受限时也能借助预训练知识快速适应新任务，实现高效迁移学习。"
    "动机：面对下游任务时全参微调（fine-tune，全部参数权重参与更新，效果好）过于低效，而固定部分层只微调接近任务的几层参数又难以达到好效果。"
    "优点：在提高模型效果的同时大大缩短训练时间和计算成本；并能缓解全量微调带来的灾难性遗忘问题。"
    "方法分类：1) 增加额外参数，如 Prefix Tuning、Prompt Tuning、Adapter Tuning 及其变体；"
    "2) 选取一部分参数更新，如 BitFit；"
    "3) 引入重参数化，如 LoRA、AdaLoRA、QLoRA；"
    "4) 混合高效微调，如 MAM Adapter、UniPELT。"
    "与全量微调的区别：LoRA 这类低秩方法本来就只能改变风格、难以对模型产生决定性改变，全量微调才可以改变知识；实测上 FT 效果稍好于 LoRA，而 AdaLoRA 效果稍好于 FT。"
    "选型：P-Tuning v2、LoRA 综合评估不错，显存有限可考虑 QLoRA，简单任务可用 P-Tuning、Prompt Tuning。"
    "现存问题：相比全参数微调，大部分高效微调技术推理速度会变慢、模型精度会变差；且因参数计算口径不一致、缺乏对模型大小的考虑、缺少统一测量基准与评价标准、代码可读性差，不同方法之间难以直接比较。"),
]


def retrieve_contexts(query):
    """复用项目真实检索：先按主题路由拿检索计划，再按文本去重。

    这里必须和 response_generator.stream_response_from_vectorstore 完全同构 ——
    评估工具的意义是"量线上那条链路"，如果它自己还硬编码双库各 top-2，
    分数测的就是已经不存在的旧管线，结论无效。
    """
    docs = []
    for coll_name, k in retrieval_plan(query):
        part, _ = query_collection(coll_name, query, n_results=k)
        docs.extend(part)
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def build_dataset():
    """对每个问题跑一遍真实 RAG，收集 question/contexts/answer/reference 四要素"""
    init_vectorstores()
    samples = []
    for q, ref in QUESTIONS:                    # ← 现在是 (问题, 标准答案) 元组，要解包
        contexts = retrieve_contexts(q)
        # 复用真实生成函数（流式），拼成完整答案。
        # phone 用占位符：get_user_info 查不到 → 走"新用户"分支，不影响知识问答。
        answer = "".join(stream_response_from_vectorstore(q, phone_number="eval_user"))
        print(f"[{q}] 命中知识 {len(contexts)} 条，答案 {len(answer)} 字")
        samples.append(SingleTurnSample(
            user_input=q,
            retrieved_contexts=contexts,
            response=answer,
            reference=ref,                      # ← 新增：ground truth，两个检索指标要用
        ))
    return EvaluationDataset(samples=samples)


def main():
    dataset = build_dataset()

    # 裁判 LLM = 百炼 qwen（OpenAI 兼容端点），temperature=0 保证打分可复现
    llm = LangchainLLMWrapper(ChatOpenAI(
        model=MODEL_NAME, api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL, temperature=0))
    # embedding = 你已下载的本地 bge，免费、离线，不额外花 qwen 的钱
    embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))

    # Faithfulness 只需裁判 LLM；AnswerRelevancy 还要 embedding（靠向量算切题度）
    faithfulness = Faithfulness()
    faithfulness.llm = llm
    relevancy = AnswerRelevancy()
    relevancy.llm = llm
    relevancy.embeddings = embeddings

    # 两个"检索质量"指标：只看召回的上下文好不好，与生成本身无关。
    # 都依赖 reference，只喂 llm 即可，不需要 embeddings。
    ctx_precision = _pick_metric_name(_CTX_PRECISION_NAMES)()
    ctx_precision.llm = llm
    ctx_recall = _pick_metric_name(_CTX_RECALL_NAMES)()
    ctx_recall.llm = llm

    result = evaluate(
        dataset,
        metrics=[faithfulness, relevancy, ctx_precision, ctx_recall],
        run_config=RunConfig(max_workers=1, timeout=600),
    )

    print("\n===== RAGAS 总分（0~1，越高越好）=====")
    print(result)

    df = result.to_pandas()
    # 带时间戳落盘，不覆盖历史结果：改前基线是唯一的对照物，一旦被覆盖就无法复现（项目没有 git 兜底）。
    out_path = f"data/ragas_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("\n逐条明细已写入 " + out_path)
    print(df.to_string())


if __name__ == "__main__":
    main()