"""P3-b 机制库质检 (QC) 确定性内核 —— 无 LLM/网络, 可单测。

全量建图产出 ~624 张自动卡, 颗粒度偏细 (44% variant, 光 attention 95 张)。本模块
为"质检报告生成器"提供确定性骨架: 把卡按中心词聚类 (让 LLM 一次看全同概念卡才能
判同义) + 纯规则廉价标记 (空前提/短功能/无公式) + 解析 LLM 裁决 + 渲染 Markdown 报告。

设计立场 (用户决策): **保留细粒度**。variant 多是"基元+论文创新"(gravity_informed_
attention / neural_ode_attention), 是组合创新的素材, 默认 KEEP。质检只标出真问题:
纯同义改名 (可合并)、整模型没拆开 (可拆)、空泛无信息 (可删)。报告是**建议性、只读**,
不改库 —— apply 是用户看完报告后的后续步骤。

复用 extraction 的确定性件 (中心词归一 / 截断容错 JSON 解析 / 覆盖统计), 不重造。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass

from darwin_st.knowledge.extraction import (
    _NOISE_TOKENS,
    _extract_json,
    canonical_name,
    coverage_report,
)
from darwin_st.knowledge.ontology import Mechanism

__all__ = [
    "concept_head",
    "cluster_by_concept",
    "rule_based_flags",
    "parse_verdicts",
    "render_report",
    "mechanisms_from_cards",
    "CRITIC_QC_SYSTEM",
    "VERDICTS",
]

# 合法裁决集合 (LLM 输出归一到此)
VERDICTS = {"keep", "merge", "split", "drop"}

# 规则标记阈值
_SHORT_AF_CHARS = 40  # abstract_function 短于此 → 标记 (信息不足, 跨域检索弱)


# ---------------------------------------------------------------------------
# 1. 中心词聚类 (把同概念卡聚到一起, 供 LLM 整组判同义)
# ---------------------------------------------------------------------------


def concept_head(name: str) -> str:
    """取机制名的"中心词" = 剥噪声修饰后的**末位实义 token**。

    复合机制名的中心词在末位 (英文偏正结构): dynamic_graph_attention→attention,
    graph_diffusion_convolution→convolution, masked_autoencoder_module→autoencoder
    (module 是噪声尾词, 剥掉)。这样所有 *_attention 聚到一组, LLM 一次看全 95 张
    才能判断哪些是同义改名、哪些是带创新的真变体。

    全是噪声词或空名 → 回退到 canonical 全名 (自成一组, 不误并)。
    """
    toks = [t for t in canonical_name(name).split("_") if t and t not in _NOISE_TOKENS]
    if not toks:
        return canonical_name(name) or "unnamed"
    return toks[-1]


def cluster_by_concept(mechs: list[Mechanism]) -> dict[str, list[Mechanism]]:
    """按中心词把卡聚类。返回 {concept_head: [按名排序的卡]}。

    单卡组 (中心词独一份) 也保留 —— 它们多半本就正交, LLM 可快速判 keep。
    """
    groups: dict[str, list[Mechanism]] = defaultdict(list)
    for m in mechs:
        groups[concept_head(m.name)].append(m)
    # 组内按名排序, 组按"成员多→少"排 (重复重灾区优先)
    return {
        head: sorted(members, key=lambda m: m.name)
        for head, members in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    }


# ---------------------------------------------------------------------------
# 2. 规则标记 (廉价、确定性, 与 LLM 互补的独立视角)
# ---------------------------------------------------------------------------


def rule_based_flags(mechs: list[Mechanism]) -> dict[str, list[str]]:
    """纯规则的廉价标记。返回 {机制名: [标记...]}, 只含被标记的卡。

    标记项 (都是"客观可数"的缺陷, 不做语义判断):
      - empty_preconditions: 无前提 → 无法挂到类比钩子上, 跨域检索检不到 (死卡)
      - short_abstract_function: 功能描述过短 → 信息不足, 向量检索弱
      - no_math_structure: 无核心数学 → 可复现性缺失 (合成时无从下手)
    """
    flags: dict[str, list[str]] = {}
    for m in mechs:
        fs: list[str] = []
        if not m.preconditions:
            fs.append("empty_preconditions")
        if len((m.abstract_function or "").strip()) < _SHORT_AF_CHARS:
            fs.append("short_abstract_function")
        if not (m.math_structure or "").strip():
            fs.append("no_math_structure")
        if fs:
            flags[m.name] = fs
    return flags


# ---------------------------------------------------------------------------
# 3. LLM 质检 prompt + 裁决解析
# ---------------------------------------------------------------------------


CRITIC_QC_SYSTEM = """你是跨域机制知识库的【质检员】。下面给你**同一中心词**下的一组机制卡 \
(它们名字里有共同的核心词, 如都含 attention/convolution)。库的价值 = 互斥正交 + 覆盖全面。

【重要立场: 默认保留 (keep)】本库刻意保留细粒度 —— 一个"标准基元 + 某篇论文的实质创新" \
(如 gravity_informed_attention 用引力先验、neural_ode_attention 用连续时间) 是有价值的 \
组合创新素材, **应 keep**。只在下面三种情况给非 keep:

1. merge: 两张卡其实是**同一基元的纯同义改名**, 没有实质机制差异 (如 self_attention 与 \
   scaled_dot_product_attention 指同一件事)。给出 merge_into = 应保留的那张卡的 name \
   (选更通用/更学界通名的)。
2. split: 这张卡其实是**一个完整模型/多基元堆叠**, 没拆成正交基元 (如把整个 Informer \
   抽成一张卡)。reason 里说该拆成哪些基元。
3. drop: 这张卡**空泛无信息 / 不是可移植基元** (如 "feature_extraction" 这种放之四海皆准 \
   的空壳, 或纯应用描述)。

对**每一张**给我裁决。只输出 JSON:
{"verdicts": [{"name": "<卡名>", "verdict": "keep|merge|split|drop", \
"merge_into": "<仅 merge 时给目标卡名, 否则空串>", "reason": "<一句话英文理由>"}, ...]}"""


def parse_verdicts(resp: str) -> list[dict]:
    """解析 LLM 裁决 JSON (复用 _extract_json 的截断容错)。

    返回归一后的裁决列表: [{name, verdict, merge_into, reason}]。
    非法/缺字段的条目丢弃; verdict 不在 VERDICTS 的归到 keep (保守, 不误删)。
    """
    data = _extract_json(resp)
    if data is None:
        return []
    if isinstance(data, dict):
        # "verdicts"/"cards" 是正常键; "mechanisms" 是 _extract_json 截断抢救时的固定包装键
        rows = data.get("verdicts") or data.get("cards") or data.get("mechanisms") or []
    elif isinstance(data, list):
        rows = data
    else:
        return []
    out: list[dict] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = canonical_name(str(r.get("name", "")))
        if name == "unnamed":
            continue
        verdict = str(r.get("verdict", "keep")).strip().lower()
        if verdict not in VERDICTS:
            verdict = "keep"  # 未知裁决保守当 keep, 绝不误删
        out.append({
            "name": name,
            "verdict": verdict,
            "merge_into": canonical_name(str(r.get("merge_into", ""))) if r.get("merge_into") else "",
            "reason": str(r.get("reason", "")).strip()[:300],
        })
    return out


# ---------------------------------------------------------------------------
# 4. 报告渲染
# ---------------------------------------------------------------------------


def render_report(
    mechs: list[Mechanism],
    clusters: dict[str, list[Mechanism]],
    verdicts: list[dict],
    rule_flags: dict[str, list[str]],
) -> str:
    """渲染人类可读 Markdown 质检报告。纯字符串拼接, 无副作用。"""
    cov = coverage_report(mechs)
    by_verdict: dict[str, list[dict]] = defaultdict(list)
    for v in verdicts:
        by_verdict[v["verdict"]].append(v)

    multi = {h: g for h, g in clusters.items() if len(g) > 1}
    L: list[str] = []
    L.append("# 机制库质检报告")
    L.append("")
    L.append(f"- 总卡数: **{cov.n_total}** (concept {cov.n_concept} / variant {cov.n_variant})")
    L.append(f"- 中心词聚类: **{len(clusters)}** 组, 其中 **{len(multi)}** 组含多张卡")
    L.append(f"- 域分布: {cov.by_domain}")
    L.append(f"- 未用前提 (词表死角): {cov.unused_preconditions or '无'}")
    if verdicts:
        counts = {k: len(by_verdict.get(k, [])) for k in ("keep", "merge", "split", "drop")}
        L.append(f"- LLM 裁决: keep {counts['keep']} / merge {counts['merge']} / "
                 f"split {counts['split']} / drop {counts['drop']}")
    L.append("")

    # --- 非 keep 裁决 (用户最该看的, 置顶) ---
    for verdict, title in (("merge", "建议合并 (纯同义改名)"),
                           ("split", "建议拆分 (整模型未拆成基元)"),
                           ("drop", "建议删除 (空泛/非可移植基元)")):
        rows = by_verdict.get(verdict, [])
        L.append(f"## {title} — {len(rows)} 张")
        if not rows:
            L.append("(无)")
        for r in rows:
            tgt = f" → `{r['merge_into']}`" if r["merge_into"] else ""
            L.append(f"- `{r['name']}`{tgt}: {r['reason']}")
        L.append("")

    # --- 规则标记 (我的独立视角, 与 LLM 互补) ---
    L.append(f"## 规则标记 (确定性, 与 LLM 互补) — {len(rule_flags)} 张")
    if not rule_flags:
        L.append("(无)")
    for name in sorted(rule_flags):
        L.append(f"- `{name}`: {', '.join(rule_flags[name])}")
    L.append("")

    # --- 多卡聚类一览 (供用户核对 LLM 判断) ---
    L.append(f"## 多卡聚类一览 ({len(multi)} 组)")
    for head, group in multi.items():
        L.append(f"### `{head}` — {len(group)} 张")
        for m in group:
            L.append(f"- `{m.name}` [{m.origin_domain}/{m.abstraction_level}]: "
                     f"{(m.abstract_function or '')[:80]}")
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# 5. 从 mechanism_cards.json 重建 Mechanism (QC 只需核心字段)
# ---------------------------------------------------------------------------


def mechanisms_from_cards(cards: list[dict]) -> list[Mechanism]:
    """把 mechanism_cards.json 的 dict 列表重建为 Mechanism 对象。

    QC 只用 name/abstract_function/preconditions/math_structure/domain/abstraction_level,
    故跳过 evidence/related 等嵌套字段 (仿 graph_store._row_to_mech)。
    """
    out: list[Mechanism] = []
    for c in cards:
        if not isinstance(c, dict) or not c.get("name"):
            continue
        out.append(Mechanism(
            name=str(c["name"]),
            abstract_function=str(c.get("abstract_function", "")),
            preconditions=list(c.get("preconditions", []) or []),
            causal_behavior=str(c.get("causal_behavior", "")),
            function_tags=list(c.get("function_tags", []) or []),
            math_structure=str(c.get("math_structure", "")),
            consequences=str(c.get("consequences", "")),
            origin_domain=str(c.get("origin_domain", "ST")),
            abstraction_level=str(c.get("abstraction_level", "concept")),
            provenance=str(c.get("provenance", "")),
        ))
    return out
