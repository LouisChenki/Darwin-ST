"""机制卡抽取的确定性后处理 (Deterministic Post-Processing for Mechanism Extraction)。

P3-b 建图 pipeline 的"纯函数核心": 不触网、不读外部语料, 只对【已抽取的草稿机制卡】
做确定性变换。把这些逻辑从 scripts/build_kg.py 抽到这里, 是为了:
  1. 可单测 (tests/test_extraction.py) —— 去重/归一/schema 校验是建库质量的命脉。
  2. 迭代时只改 prompt/LLM 逻辑, 后处理规则稳定可回归。

四类纯函数:
  - parse_mechanism_cards: 把 LLM 返回的 JSON 草稿解析成 Mechanism (容错 + schema 对齐)。
  - normalize_preconditions: 把自由前提词归一到 PRECONDITION_VOCAB (互斥性的关键)。
  - dedup_mechanisms: 按规范名 + 抽象功能近似做语义去重 (正交性的关键)。
  - coverage_report: 按 origin_domain / abstraction_level 统计覆盖 (全面性自评)。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass

from darwin_st.knowledge.ontology import (
    DOMAINS,
    FUNCTION_VOCAB,
    PRECONDITION_VOCAB,
    Evidence,
    Mechanism,
)

__all__ = [
    "PRECONDITION_ALIASES",
    "DOMAIN_ALIASES",
    "canonical_name",
    "normalize_preconditions",
    "normalize_domain",
    "normalize_function_tags",
    "parse_mechanism_cards",
    "dedup_mechanisms",
    "coverage_report",
    "CoverageReport",
]

# ---------------------------------------------------------------------------
# 受控词表别名映射 (LLM 自由产出 → 受控 VOCAB)
# 迭代时在此扩充; 克制——新增一个 alias 前先确认它不是该登记进 VOCAB 的新前提。
# ---------------------------------------------------------------------------

# 把 LLM 可能吐出的同义/近义前提词, 归一到 PRECONDITION_VOCAB 的规范键。
PRECONDITION_ALIASES: dict[str, str] = {
    # redundant_structure
    "redundancy": "redundant_structure",
    "low_rank": "redundant_structure",
    "low_rank_structure": "redundant_structure",
    "spatial_redundancy": "redundant_structure",
    "temporal_redundancy": "redundant_structure",
    "compressible": "redundant_structure",
    # label_scarcity
    "unlabeled": "label_scarcity",
    "unlabeled_data": "label_scarcity",
    "few_labels": "label_scarcity",
    "limited_labels": "label_scarcity",
    "self_supervision_available": "label_scarcity",
    "data_scarcity": "label_scarcity",
    # long_range_dependency
    "long_range": "long_range_dependency",
    "long_term_dependency": "long_range_dependency",
    "long_horizon": "long_range_dependency",
    "global_dependency": "long_range_dependency",
    "long_sequence": "long_range_dependency",
    # non_euclidean_topology
    "graph_structure": "non_euclidean_topology",
    "graph_structured": "non_euclidean_topology",
    "relational_structure": "non_euclidean_topology",
    "irregular_topology": "non_euclidean_topology",
    "non_grid": "non_euclidean_topology",
    # spatial_smoothness
    "smoothness": "spatial_smoothness",
    "local_correlation": "spatial_smoothness",
    "spatial_correlation": "spatial_smoothness",
    "neighborhood_similarity": "spatial_smoothness",
    # temporal_periodicity
    "periodicity": "temporal_periodicity",
    "seasonality": "temporal_periodicity",
    "cyclic_pattern": "temporal_periodicity",
    "daily_weekly_pattern": "temporal_periodicity",
    # multi_scale_structure
    "multi_scale": "multi_scale_structure",
    "multiscale": "multi_scale_structure",
    "hierarchical_structure": "multi_scale_structure",
    "multi_resolution": "multi_scale_structure",
    "scale_variation": "multi_scale_structure",
    # distribution_shift
    "non_stationarity": "distribution_shift",
    "non_stationary": "distribution_shift",
    "domain_shift": "distribution_shift",
    "covariate_shift": "distribution_shift",
    "concept_drift": "distribution_shift",
    # node_indistinguishability
    "permutation_invariance": "node_indistinguishability",
    "anonymous_nodes": "node_indistinguishability",
    "indistinguishable_entities": "node_indistinguishability",
    "identical_nodes": "node_indistinguishability",
    # high_dimensionality
    "high_dimensional": "high_dimensionality",
    "many_variables": "high_dimensionality",
    "many_channels": "high_dimensionality",
    "wide_features": "high_dimensionality",
    # sequential_order
    "ordered_sequence": "sequential_order",
    "temporal_order": "sequential_order",
    "causal_order": "sequential_order",
    "autoregressive": "sequential_order",
    # sparse_interaction
    "sparsity": "sparse_interaction",
    "sparse_dependency": "sparse_interaction",
    "few_relevant": "sparse_interaction",
    "local_interaction": "sparse_interaction",
    # heterogeneity
    "heterogeneous": "heterogeneity",
    "multi_modal": "heterogeneity",
    "multimodal": "heterogeneity",
    "diverse_patterns": "heterogeneity",
    "node_heterogeneity": "heterogeneity",
    # noise_corruption
    "noisy": "noise_corruption",
    "noise": "noise_corruption",
    "missing_values": "noise_corruption",
    "corruption": "noise_corruption",
    "outliers": "noise_corruption",
}

# origin_domain 归一 (metadata domain / LLM 自由域名 → DOMAINS 规范键)
DOMAIN_ALIASES: dict[str, str] = {
    "spatio-temporal": "ST",
    "spatiotemporal": "ST",
    "spatio_temporal": "ST",
    "st": "ST",
    "traffic": "ST",
    "cv": "CV",
    "vision": "CV",
    "computer_vision": "CV",
    "image": "CV",
    "nlp": "NLP",
    "language": "NLP",
    "text": "NLP",
    "graph": "GraphLearning",
    "graphlearning": "GraphLearning",
    "graph_learning": "GraphLearning",
    "gnn": "GraphLearning",
    "ssm": "SSM",
    "sequence": "SSM",
    "sequence_modeling": "SSM",
    "selfsupervised": "SelfSupervised",
    "self_supervised": "SelfSupervised",
    "self-supervised": "SelfSupervised",
    "ssl": "SelfSupervised",
    "timeseries": "TimeSeries",
    "time_series": "TimeSeries",
    "time-series": "TimeSeries",
    "forecasting": "TimeSeries",
    "optimization": "Optimization",
    "training": "Optimization",
    "optim": "Optimization",
    "generative": "Generative",
    "generation": "Generative",
    "diffusion": "Generative",
    "control": "Control",
    "dynamical_system": "Control",
    "dynamics": "Control",
}


# ---------------------------------------------------------------------------
# 名称规范化 + 去重
# ---------------------------------------------------------------------------

_NAME_SUB = re.compile(r"[^a-z0-9]+")
# 去重时忽略的"修饰前缀/后缀"——让 "spatial_attention"/"temporal_attention"/"attention_module"
# 不被无脑合并, 但 "self-attention"/"self_attention_mechanism" 能合并。
_NOISE_TOKENS = {
    "mechanism", "module", "block", "layer", "network", "net", "model",
    "based", "method", "approach", "technique", "operator", "the", "a",
    "learning",
}


def canonical_name(name: str) -> str:
    """把机制名归一为 snake_case 规范键 (小写, 非字母数字→下划线, 去首尾下划线)。"""
    s = _NAME_SUB.sub("_", name.strip().lower()).strip("_")
    return s or "unnamed"


def _name_key(name: str) -> str:
    """去重用的"语义骨架": canonical 后去噪声修饰词 + 词干化 + 排序 + 去分隔符。

    目标: "masked_autoencoding" / "masked_autoencoder_module" / "masked_auto_encoder"
    命中同一 key (autoencoder 不论怎么切分/派生都等价), 但保留 "spatial"/"temporal"/
    "causal" 等有区分度的限定词 → spatial_attention 与 temporal_attention 仍不同。

    去分隔符是关键: 让 "auto_encod" 与 "autoencod" 折叠成同一骨架。
    """
    toks = [t for t in canonical_name(name).split("_") if t and t not in _NOISE_TOKENS]
    stemmed = []
    for t in toks:
        for suf in ("ation", "ing", "er", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
        stemmed.append(t)
    # 排序后直接拼接 (无分隔符): 折叠 auto_encod≡autoencod, 同时保留全部区分性 token
    return "".join(sorted(stemmed))


def normalize_preconditions(raw: list[str]) -> list[str]:
    """把自由前提词列表归一到 PRECONDITION_VOCAB。

    规则: 先 canonical_name, 命中 VOCAB 直接留; 否则查 alias 表; 都不中则丢弃
    (宁缺毋滥——保持前提词表互斥)。返回去重且保序的列表。
    """
    out: list[str] = []
    seen: set[str] = set()
    for p in raw or []:
        key = canonical_name(p)
        if key in PRECONDITION_VOCAB:
            canon = key
        elif key in PRECONDITION_ALIASES:
            canon = PRECONDITION_ALIASES[key]
        else:
            continue  # 未知前提丢弃, 不污染词表
        if canon not in seen:
            seen.add(canon)
            out.append(canon)
    return out


def normalize_domain(raw: str) -> str:
    """把来源域归一到 DOMAINS。未知→默认 'ST' (本域)。"""
    if not raw:
        return "ST"
    if raw in DOMAINS:
        return raw
    key = canonical_name(raw)
    if key in DOMAIN_ALIASES:
        return DOMAIN_ALIASES[key]
    # canonical 后可能等于某 DOMAINS 成员的小写
    for d in DOMAINS:
        if key == d.lower():
            return d
    return "ST"


def normalize_function_tags(raw: list[str]) -> list[str]:
    """function_tags 归一到 FUNCTION_VOCAB; 未知丢弃。"""
    out: list[str] = []
    seen: set[str] = set()
    for f in raw or []:
        key = canonical_name(f)
        if key in FUNCTION_VOCAB and key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---------------------------------------------------------------------------
# LLM JSON 草稿 → Mechanism
# ---------------------------------------------------------------------------


def _coerce_evidence(raw) -> list[Evidence]:
    out: list[Evidence] = []
    if not isinstance(raw, list):
        return out
    for e in raw:
        if not isinstance(e, dict):
            continue
        out.append(
            Evidence(
                dataset=str(e.get("dataset", ""))[:80],
                metric=str(e.get("metric", ""))[:40],
                delta=str(e.get("delta", ""))[:120],
                grade=str(e.get("grade", "moderate")) or "moderate",
                source=str(e.get("source", ""))[:120],
            )
        )
    return out


def parse_mechanism_cards(raw_json: str, provenance: str = "") -> list[Mechanism]:
    """把 LLM 返回的 JSON 文本解析为 Mechanism 列表 (容错 + 归一, 不 validate)。

    LLM 约定返回 {"mechanisms": [ {...}, ... ]} 或裸 [ {...} ]。本函数:
      - 抽出第一段 JSON (容忍 markdown ```json 围栏 / 前后噪声)
      - 字段缺失给安全默认; preconditions/domain/function_tags 归一
      - 丢弃明显非机制的空卡 (无 name 或无 abstract_function)
    不抛异常 (除非 JSON 完全无法解析→返回 [])。validate 留给 build_kg / graph_store。
    """
    data = _extract_json(raw_json)
    if data is None:
        return []
    if isinstance(data, dict):
        cards = data.get("mechanisms") or data.get("cards") or []
    elif isinstance(data, list):
        cards = data
    else:
        return []

    out: list[Mechanism] = []
    for c in cards:
        if not isinstance(c, dict):
            continue
        name = canonical_name(str(c.get("name", "")))
        af = str(c.get("abstract_function", "")).strip()
        if name == "unnamed" or not af:
            continue
        abstraction = str(c.get("abstraction_level", "concept")).strip().lower()
        if abstraction not in ("concept", "variant"):
            abstraction = "concept"
        m = Mechanism(
            name=name,
            abstract_function=af[:500],
            preconditions=normalize_preconditions(_as_list(c.get("preconditions"))),
            causal_behavior=str(c.get("causal_behavior", "")).strip()[:600],
            function_tags=normalize_function_tags(_as_list(c.get("function_tags"))),
            math_structure=str(c.get("math_structure", "")).strip()[:600],
            consequences=str(c.get("consequences", "")).strip()[:400],
            origin_domain=normalize_domain(str(c.get("origin_domain", ""))),
            abstraction_level=abstraction,
            evidence=_coerce_evidence(c.get("evidence")),
            anti_patterns=[str(a)[:200] for a in _as_list(c.get("anti_patterns"))][:4],
            provenance=str(c.get("provenance", "") or provenance)[:200],
            inferred=True,
        )
        out.append(m)
    return out


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return [x for x in v if x is not None]
    return [v]


def _extract_json(text: str):
    """从可能含 markdown 围栏/噪声的文本里抽出第一段合法 JSON。"""
    if not text:
        return None
    # 去 ```json ... ``` 围栏
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    candidate = fence.group(1) if fence else text
    # 直接尝试
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # 退而求其次: 抓第一个平衡的 {...} 或 [...]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except Exception:
                        break
    return None


# ---------------------------------------------------------------------------
# 去重 (正交性的命脉)
# ---------------------------------------------------------------------------


def dedup_mechanisms(
    mechs: list[Mechanism], seed_names: set[str] | None = None
) -> tuple[list[Mechanism], int]:
    """按"语义骨架名" + (origin_domain, 前提集合) 合并近重复机制卡。

    合并策略 (确定性, 与输入顺序无关地稳定):
      - key = _name_key(name)。同 key 视为同一机制。
      - 同 key 内, 选"信息量最大"的代表 (math_structure+causal_behavior+evidence 更全),
        其余卡的 evidence / anti_patterns / preconditions 合并进代表 (扩充而不重复)。
      - 若 seed_names 给定: 草稿卡若与某 seed 同 key, 视为已被种子覆盖 → 丢弃 (种子是标杆,
        不被自动抽取的卡覆盖)。

    返回 (去重后列表, 被去掉/合并的卡数)。
    """
    seed_keys = {_name_key(n) for n in (seed_names or set())}
    groups: dict[str, list[Mechanism]] = {}
    order: list[str] = []
    for m in mechs:
        k = _name_key(m.name)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(m)

    kept: list[Mechanism] = []
    removed = 0
    for k in order:
        group = groups[k]
        if k in seed_keys:
            removed += len(group)  # 全被种子覆盖
            continue
        rep = _pick_representative(group)
        for other in group:
            if other is rep:
                continue
            _merge_into(rep, other)
            removed += 1
        kept.append(rep)
    return kept, removed


def _info_score(m: Mechanism) -> int:
    return (
        len(m.math_structure)
        + len(m.causal_behavior)
        + len(m.consequences)
        + 50 * len(m.evidence)
        + 10 * len(m.preconditions)
        + 5 * len(m.anti_patterns)
    )


def _pick_representative(group: list[Mechanism]) -> Mechanism:
    # 信息量最大者; 并列时取名字字典序最小 (确定性)
    return max(group, key=lambda m: (_info_score(m), -_str_ord(m.name)))


def _str_ord(s: str) -> int:
    # 字典序的稳定数值代理 (前 8 字符)
    v = 0
    for ch in s[:8].ljust(8):
        v = v * 256 + ord(ch)
    return v


def _merge_into(rep: Mechanism, other: Mechanism) -> None:
    for p in other.preconditions:
        if p not in rep.preconditions:
            rep.preconditions.append(p)
    for f in other.function_tags:
        if f not in rep.function_tags:
            rep.function_tags.append(f)
    have = {(e.dataset, e.metric, e.source) for e in rep.evidence}
    for e in other.evidence:
        if (e.dataset, e.metric, e.source) not in have:
            rep.evidence.append(e)
            have.add((e.dataset, e.metric, e.source))
    for a in other.anti_patterns:
        if a not in rep.anti_patterns:
            rep.anti_patterns.append(a)
    if not rep.math_structure and other.math_structure:
        rep.math_structure = other.math_structure


# ---------------------------------------------------------------------------
# 覆盖统计 (全面性自评)
# ---------------------------------------------------------------------------


@dataclass
class CoverageReport:
    n_total: int
    by_domain: dict[str, int]
    by_abstraction: dict[str, int]
    by_precondition: dict[str, int]
    unused_preconditions: list[str]
    n_concept: int
    n_variant: int

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "by_domain": self.by_domain,
            "by_abstraction": self.by_abstraction,
            "by_precondition": self.by_precondition,
            "unused_preconditions": self.unused_preconditions,
        }


def coverage_report(mechs: list[Mechanism]) -> CoverageReport:
    """按域 / 抽象层 / 前提统计覆盖 (全面性 + 词表使用率)。"""
    by_domain = Counter(m.origin_domain for m in mechs)
    by_abs = Counter(m.abstraction_level for m in mechs)
    by_pre: Counter = Counter()
    for m in mechs:
        for p in m.preconditions:
            by_pre[p] += 1
    unused = sorted(PRECONDITION_VOCAB - set(by_pre))
    return CoverageReport(
        n_total=len(mechs),
        by_domain=dict(sorted(by_domain.items())),
        by_abstraction=dict(sorted(by_abs.items())),
        by_precondition=dict(sorted(by_pre.items(), key=lambda kv: -kv[1])),
        unused_preconditions=unused,
        n_concept=by_abs.get("concept", 0),
        n_variant=by_abs.get("variant", 0),
    )
