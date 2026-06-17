"""经验蒸馏与检索 (Experience Distillation & Retrieval) —— ExpeL/Reflexion 式记忆服务。

把原始试验记录加工成 Agent 决策时可直接消费的「经验上下文」, 是连接
记忆库 (store.py) 与演化决策 (Tier 2 Agent) 的桥。

本模块**确定性、无 LLM 调用、无外部依赖**——真正的自然语言语义蒸馏由 Agent 在
Tier 2 完成 (它读这里产出的结构化聚合再做创造性归纳)。这里只负责:
  - build_experience_context: 汇总当前最优/SOTA差距/失败模式/相似成功经验 → 给 Agent 的上下文
  - rank_by_relevance:       recency × importance × similarity 三因子排序 (Generative Agents 式)
  - summarize_graveyard:     复发失败模式聚合 (哪些死因最常见 → 重点规避)
  - propose_insight_candidates: 基于历史的候选洞察 (规则式, 供 Agent 复核后入库)

参考: Reflexion(arXiv:2303.11366) / ExpeL(arXiv:2308.10144) /
      Generative Agents(arXiv:2304.03442, recency×importance×relevance)
详见 docs/BLUEPRINT.md §3。
"""

from __future__ import annotations

import math
from collections import Counter

from darwin_st.memory.store import MemoryStore, _descriptor_distance

__all__ = [
    "build_experience_context",
    "rank_by_relevance",
    "summarize_graveyard",
    "propose_insight_candidates",
]


def summarize_graveyard(store: MemoryStore, dataset: str | None = None) -> dict[str, int]:
    """聚合失败死因分布 {fail_reason: 次数}, 降序。供 Agent 识别复发陷阱重点规避。"""
    clauses, params = ["status IN ('CRASH','DISCARD')"], []
    if dataset is not None:
        clauses.append("dataset=?")
        params.append(dataset)
    where = "WHERE " + " AND ".join(clauses)
    rows = store.conn.execute(
        f"SELECT fail_reason FROM experiments {where}", params
    ).fetchall()
    counter: Counter[str] = Counter()
    for r in rows:
        reason = r["fail_reason"] or "unspecified"
        counter[reason] += 1
    return dict(counter.most_common())


def rank_by_relevance(
    store: MemoryStore,
    behavior_descriptor: dict,
    dataset: str | None = None,
    k: int = 5,
    now_id: int | None = None,
) -> list[dict]:
    """按 recency × importance × similarity 三因子对 KEEP 经验排序, 取前 k。

    - similarity: 行为描述子距离的反函数 (越近越相关)
    - importance: 该试验相对其数据集 baseline 的改进幅度 (越大越重要)
    - recency:    id 越新越近 (id 单调递增, 充当时间序; now_id 为当前参照点)

    三者乘积为综合得分, 降序返回。这是 Generative Agents 的记忆检索范式。
    """
    clauses, params = ["status='KEEP'", "val_mae IS NOT NULL"], []
    if dataset is not None:
        clauses.append("dataset=?")
        params.append(dataset)
    where = "WHERE " + " AND ".join(clauses)
    rows = [dict_row for dict_row in (_as_dict(store, r) for r in
            store.conn.execute(f"SELECT * FROM experiments {where}", params).fetchall())]
    if not rows:
        return []

    max_id = now_id if now_id is not None else max(r["id"] for r in rows)
    # baseline 取该批次最差 KEEP 值作为改进幅度的参照尺度
    worst_mae = max(r["val_mae"] for r in rows)
    best_mae = min(r["val_mae"] for r in rows)
    mae_span = max(worst_mae - best_mae, 1e-6)

    scored = []
    for r in rows:
        sim = 1.0 / (1.0 + _descriptor_distance(behavior_descriptor, r.get("behavior_descriptor", {})))
        importance = (worst_mae - r["val_mae"]) / mae_span  # 0..1, 越优越重要
        importance = max(importance, 0.05)                  # 给个下限, 别把老经验压成 0
        # recency: 指数衰减, 以 id 距离当尺度
        age = max(max_id - r["id"], 0)
        recency = math.exp(-age / max(max_id, 1.0))
        score = sim * importance * recency
        scored.append((score, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for _, r in scored[:k]]


def build_experience_context(
    store: MemoryStore,
    dataset: str,
    behavior_descriptor: dict | None = None,
    sota_mae: float | None = None,
    k_neighbors: int = 3,
) -> dict:
    """汇总一个 Agent 决策即用的经验上下文包 (供 Tier 2 注入提示)。

    返回结构化字典, 含: 当前最优、与 SOTA 差距、结局统计、失败模式 Top、
    相似成功经验。Agent 据此做有依据的变异 (而非凭空)。
    """
    best = store.best_so_far(dataset)
    best_mae = best["val_mae"] if best else None

    ctx: dict = {
        "dataset": dataset,
        "best_so_far": best,
        "best_mae": best_mae,
        "stats": store.get_stats(dataset),
        "graveyard_patterns": summarize_graveyard(store, dataset),
        "insights": store.get_insights(dataset),
    }

    # 与 SOTA 的差距 (终极目标判定)
    if sota_mae is not None and best_mae is not None:
        gap = best_mae - sota_mae
        ctx["sota_mae"] = sota_mae
        ctx["sota_gap"] = gap
        ctx["beats_sota"] = gap < 0

    # 相似成功经验
    if behavior_descriptor is not None:
        ctx["similar_successes"] = rank_by_relevance(
            store, behavior_descriptor, dataset=dataset, k=k_neighbors
        )

    return ctx


def propose_insight_candidates(store: MemoryStore, dataset: str | None = None) -> list[str]:
    """基于历史规则式生成候选洞察文本 (供 Agent 复核后再决定是否 add_insight 入库)。

    纯启发式、确定性。不臆造数值, 只陈述可从统计直接读出的模式。
    """
    candidates: list[str] = []
    grave = summarize_graveyard(store, dataset)
    if grave:
        top_reason, cnt = next(iter(grave.items()))
        if cnt >= 2:
            candidates.append(
                f"失败模式『{top_reason}』已复发 {cnt} 次, 后续变异应优先规避其触发条件。"
            )

    stats = store.get_stats(dataset)
    decided = stats["KEEP"] + stats["DISCARD"]
    if decided >= 5:
        keep_rate = stats["KEEP"] / decided
        if keep_rate < 0.2:
            candidates.append(
                f"保留率偏低 ({keep_rate:.0%}), 变异步子可能过大; 考虑更小步长的精细微调。"
            )
        elif keep_rate > 0.6:
            candidates.append(
                f"保留率较高 ({keep_rate:.0%}), 当前搜索方向富矿; 可在此邻域加密探索。"
            )
    return candidates


def _as_dict(store: MemoryStore, row) -> dict:
    """复用 store 的 Row→dict 解析 (含 JSON 字段)。"""
    from darwin_st.memory.store import _row_to_dict

    return _row_to_dict(row)
