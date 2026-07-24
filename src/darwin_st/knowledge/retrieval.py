"""跨域类比检索 (Cross-Domain Analogical Retrieval) —— P3 知识系统的检索核心。

实现 docs/TIER2_RESEARCH_FINDINGS.md §3,§7 的检索逻辑:
  瓶颈 → 分解成前提 → MAC(语义召回) + FAC(前提图重排) → 互补集选择(子模覆盖)
  → 跨域拉回【互补机制集】(非单个最相似), 让 LLM 融合。

两阶段 MAC/FAC (结构映射理论):
  - MAC: 语义向量召回 (embedder + GraphStore.vector_search)。
    关键: 只在 abstract_function+preconditions 上算相似(embedding_text), 绝不含表面/域。
  - FAC: 按【共享前提数】沿前提图重排(结构对齐), 非余弦 —— 这才是跨域类比的判据。

互补集选择 (Q3 升级): 不返回单个最相似, 而是贪心子模最大覆盖, 让选出的机制集
  覆盖瓶颈分解出的【多个前提】(IA-Select/Lin-Bilmes 思路, (1-1/e) 保证)。

跨域 (origin_domain <> target) 是可选硬过滤: 强制返回他域机制以驱动跨域涌现。

对外主接口 = find_cross_domain_analogy (GraphStore + Embedder 注入, 后端可插拔:
内存/Neo4j 存储 × Hash/SentenceTransformer 嵌入)。旧版 token 重叠 demo
(cross_domain_analogy) 已删除 —— 全仓库无引用, 且相似度判据与正式版不一致。
"""

from __future__ import annotations

from dataclasses import dataclass

from darwin_st.knowledge.ontology import Mechanism, PRECONDITION_VOCAB

__all__ = ["RetrievalResult", "decompose_to_preconditions", "find_cross_domain_analogy"]


@dataclass
class RetrievalResult:
    """一次检索的结果: 互补机制集 + 解释。"""

    mechanisms: list[Mechanism]
    covered_preconditions: set[str]
    target_preconditions: list[str]
    target_domain: str | None
    rationale: list[str]   # 每个机制为何入选 (覆盖了哪个前提)


# ---------------------------------------------------------------------------
# 瓶颈 → 前提分解 (demo 用关键词规则; 真实版用 LLM Self-Ask)
# ---------------------------------------------------------------------------

# 瓶颈描述里的线索词 → 标准前提 (PRECONDITION_VOCAB)
_BOTTLENECK_HINTS = {
    "long_range_dependency": ["长程", "长时", "long", "远距离", "长期依赖", "长距离"],
    "label_scarcity": ["标签", "无监督", "自监督", "label", "少样本", "稀缺"],
    "redundant_structure": ["冗余", "redundan", "低秩", "相关性", "可压缩"],
    "non_euclidean_topology": ["图", "拓扑", "graph", "邻接", "非欧"],
    "spatial_smoothness": ["平滑", "空间相关", "邻近", "smooth"],
    "temporal_periodicity": ["周期", "periodic", "日周期", "规律"],
    "multi_scale_structure": ["多尺度", "scale", "尺度", "分辨率"],
    "distribution_shift": ["漂移", "shift", "非平稳", "分布变化"],
    "node_indistinguishability": ["不可区分", "节点身份", "indistinguish", "同质"],
    "high_dimensionality": ["高维", "维度", "dimension"],
    "sequential_order": ["序列", "时序", "顺序", "sequen"],
    "sparse_interaction": ["稀疏", "sparse"],
    "heterogeneity": ["异质", "heterogen", "不同节点", "差异"],
    "noise_corruption": ["噪声", "noise", "污染", "扰动"],
}


def decompose_to_preconditions(bottleneck: str) -> list[str]:
    """把瓶颈描述分解成标准前提集 (demo 用关键词规则)。

    真实版: LLM Self-Ask/Least-to-Most 分解, 归一到 PRECONDITION_VOCAB。
    """
    low = bottleneck.lower()
    hit = []
    for precond, cues in _BOTTLENECK_HINTS.items():
        if any(c.lower() in low for c in cues):
            hit.append(precond)
    return hit


# ---------------------------------------------------------------------------
# 正式版: GraphStore + embedding 支撑的跨域检索 (MAC + FAC + 互补集)
# ---------------------------------------------------------------------------


def find_cross_domain_analogy(
    bottleneck: str,
    store,                        # GraphStore (InMemory / Neo4j)
    embedder,                     # Embedder (Hash / SentenceTransformer)
    target_domain: str | None = "ST",
    cross_domain_only: bool = True,
    max_results: int = 4,
    mac_top_k: int = 8,
    override_preconditions: list[str] | None = None,
) -> RetrievalResult:
    """正式跨域类比检索 (P3 对外主接口)。

    MAC: 把瓶颈嵌入 → 向量搜索召回 mac_top_k 候选 (语义模糊容错)。
    FAC: 把瓶颈分解成前提 → 沿前提图拉"共享前提"的机制 (跨域结构桥)。
    合并 MAC∪FAC 候选 → 贪心子模最大覆盖目标前提 → 互补机制集。

    相似度走真实 embedding + 图遍历召回, 语义模糊匹配更准, 且 FAC 保证"前提共享"的
    机制即使语义不相似也能被拉到。

    override_preconditions: Tier-2 LLM 诊断直出的受控前提词 (∈ PRECONDITION_VOCAB)。给定时
    **跳过** decompose_to_preconditions 的脆弱关键词匹配, 直接用 LLM 前提词做 FAC 图召回
    (MAC 语义召回仍走 bottleneck 字符串, 双路保留)。非法词在调用方 (diagnosis) 已过滤。
    """
    if override_preconditions:
        target_preconds = [p for p in override_preconditions if p in PRECONDITION_VOCAB]
        if not target_preconds:                       # 全非法 → 退回关键词分解
            target_preconds = decompose_to_preconditions(bottleneck)
    else:
        target_preconds = decompose_to_preconditions(bottleneck)
    excl = target_domain if (cross_domain_only and target_domain) else None

    # MAC: 语义向量召回
    q_emb = embedder.embed([bottleneck])[0]
    mac_hits = store.vector_search(q_emb, top_k=mac_top_k, exclude_domain=excl)
    sim_by_name = {m.name: s for m, s in mac_hits}
    candidates: dict[str, Mechanism] = {m.name: m for m, _ in mac_hits}

    # FAC: 沿前提图拉共享前提的机制 (即使语义不相似也召回 —— 结构桥)
    for p in target_preconds:
        for m in store.mechanisms_sharing_precondition(p, exclude_domain=excl):
            candidates.setdefault(m.name, m)
            sim_by_name.setdefault(m.name, 0.0)

    # 贪心子模最大覆盖: 选覆盖最多【未覆盖】目标前提的机制, 平局看相似度
    selected: list[Mechanism] = []
    rationale: list[str] = []
    covered: set[str] = set()
    pool = list(candidates.values())

    while pool and len(selected) < max_results:
        def gain(m: Mechanism):
            new_cover = (set(m.preconditions) & set(target_preconds)) - covered
            return (len(new_cover), sim_by_name.get(m.name, 0.0))

        pool.sort(key=gain, reverse=True)
        best = pool[0]
        new_cover = (set(best.preconditions) & set(target_preconds)) - covered
        sim = sim_by_name.get(best.name, 0.0)

        if not new_cover and sim < 0.05 and selected:
            break

        selected.append(best)
        covered |= (set(best.preconditions) & set(target_preconds))
        if new_cover:
            rationale.append(f"{best.name} (来自 {best.origin_domain}): 覆盖前提 {sorted(new_cover)}, 语义 {sim:.2f}")
        else:
            rationale.append(f"{best.name} (来自 {best.origin_domain}): 语义相关 {sim:.2f} (补充)")
        pool.pop(0)

        if target_preconds and covered >= set(target_preconds):
            break

    return RetrievalResult(
        mechanisms=selected,
        covered_preconditions=covered,
        target_preconditions=target_preconds,
        target_domain=target_domain,
        rationale=rationale,
    )
