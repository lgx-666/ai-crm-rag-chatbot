# tests/test_kb_router.py
"""主题路由的纯函数层测试。

为什么这批用例值得写：kb_router 不联网、不加载模型、不碰数据库，是确定性的，
所以能把"路由规则"本身钉死成回归测试——以后任何人改关键词表或改阈值，
只要行为一变这里立刻红，而不是等到线上出现"答案引用了 Java 文档"才发现。

断言的取舍：
- 主断言落在【路由结果】上（查哪些集合），这是对外行为，稳定且有业务含义；
- 只对两个"已知性质"断言具体分数（子串重复计分、大小写归一），
    因为分数绝对值本身没有业务含义（它是子串计数的噪声累加，不是相关性），
    给普通 query 写死分数只会让改词表时到处误报。
- 所有期望值都由 `python kb_router.py` 同款逻辑实测得出，不是估的。
"""

from kb_router import (
    score_domains, route_collections, retrieval_plan,
    COLLECTION_MARKET, COLLECTION_COURSE,
    TOP_K_SINGLE, TOP_K_SHARED,
)

BOTH = [COLLECTION_MARKET, COLLECTION_COURSE]


# ---------- 1. 单域问题：独判，且把 4 个槽位全给这一域 ----------
def test_pure_llm_query_routes_to_market_only():
    assert route_collections("LoRA 的原理和主要特点是什么？") == [COLLECTION_MARKET]

def test_pure_java_query_routes_to_course_only():
    assert route_collections("HashMap 的底层实现原理是什么？") == [COLLECTION_COURSE]

def test_single_domain_plan_gets_full_budget():
    """独判时一个域独占全部槽位：取 TOP_K_SINGLE 条，不再固定 2 条"""
    assert retrieval_plan("什么是 PEFT（参数高效微调）？") == [(COLLECTION_MARKET, TOP_K_SINGLE)]


# ---------- 2. "顺带提及"应当被忽略，不能因此退化成双查 ----------
# "接口"在 Java 表里，但这题的语义主体是大模型；落选侧只有 1 次命中 → 独判。
def test_incidental_other_domain_hit_does_not_force_both():
    q = "RAG 系统里向量库的接口怎么设计？"
    assert score_domains(q) == (2, 1)
    assert route_collections(q) == [COLLECTION_MARKET]

def test_incidental_llm_hit_still_routes_to_java():
    """对称情况：大模型侧只有 1 次命中时，同样独判 Java"""
    assert route_collections("向量 线程 锁") == [COLLECTION_COURSE]


# ---------- 3. 真跨域问题：两侧都有实质证据，必须双查 ----------
def test_real_cross_domain_query_queries_both():
    q = "用 LangChain 做向量检索时怎么加锁防止并发问题？"
    assert score_domains(q) == (2, 3)
    assert route_collections(q) == BOTH

def test_loser_with_two_hits_is_the_boundary_for_both():
    """阈值边界：落选侧恰好 2 次命中（≥LOSER_DUAL_THRESHOLD）→ 双查。
    和上一条 test_incidental（落选侧 1 次 → 独判）成对，把阈值两侧都钉住。"""
    assert route_collections("大模型微调 线程 锁") == BOTH


# ---------- 4. 兜底路径：任何"判不出来"的情况都必须回到双查 ----------
def test_zero_hit_query_falls_back_to_both():
    """没有任何领域关键词（省略句、寒暄、公司名等），不能出现"一个库都不查"""
    assert route_collections("这个怎么弄？") == BOTH

def test_equal_scores_fall_back_to_both():
    """同分（含跨域对比题）→ 双查"""
    assert score_domains("LoRA 和 HashMap 的区别") == (1, 1)
    assert route_collections("LoRA 和 HashMap 的区别") == BOTH

def test_shared_plan_splits_budget():
    assert retrieval_plan("这个怎么弄？") == [
        (COLLECTION_MARKET, TOP_K_SHARED), (COLLECTION_COURSE, TOP_K_SHARED)]


# ---------- 5. 脏输入不能崩（这是路由唯一的失败模式：上下文直接为空） ----------
def test_empty_and_none_query_still_return_a_plan():
    for bad in ("", None):
        cols = route_collections(bad)
        assert cols == BOTH
        assert sum(k for _, k in retrieval_plan(bad)) == TOP_K_SHARED * 2


# ---------- 6. 两个已知性质：故意用断言写下来，防止被"顺手优化"掉 ----------
def test_scoring_is_case_insensitive_for_ascii():
    """实现靠 query.lower() 比，所以关键词表里 ASCII 词必须写小写；
    写成 "RAG" 会永远命中不了——这条测试就是这个坑的哨兵。"""
    assert score_domains("LoRA") == score_domains("LORA") == (1, 0)

def test_substring_keywords_double_count():
    """"大模型" 会同时命中 "大模型" 和 "模型" 两个表项 → 计 2 分。
    这是有意的已知偏差（判定只看两侧谁更有证据），不是 bug；
    但如果哪天有人改成分词计数，这条会红，提醒他确认影响面。"""
    assert score_domains("大模型")[0] == 2


# ---------- 7. 结构性不变式：不管怎么改，这几条必须始终成立 ----------
def test_plan_invariants_over_a_corpus_of_queries():
    """总槽位恒为 4（不让 prompt 无限撑大）；集合名合法、不重复、条数为正。"""
    queries = [
        "LoRA 的原理和主要特点是什么？", "HashMap 的底层实现原理是什么？",
        "用 LangChain 做向量检索时怎么加锁防止并发问题？",
        "这个怎么弄？", "大模型微调 线程 锁", "", None,
    ]
    for q in queries:
        plan = retrieval_plan(q)
        assert 1 <= len(plan) <= 2
        names = [c for c, _ in plan]
        assert set(names) <= {COLLECTION_MARKET, COLLECTION_COURSE}
        assert len(set(names)) == len(names)          # 同一集合不得出现两次
        assert all(k > 0 for _, k in plan)
        assert sum(k for _, k in plan) == TOP_K_SINGLE  # 4 = 单域4 / 双域2+2