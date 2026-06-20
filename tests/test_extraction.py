"""extraction.py 纯函数的回归测试 (P3-b 建图后处理的命脉)。

只测确定性后处理 (不触网/不读语料/不调 LLM):
  - canonical_name / _name_key 名称归一
  - normalize_preconditions: 别名→VOCAB, 未知丢弃, 去重保序
  - normalize_domain / normalize_function_tags
  - parse_mechanism_cards: 容错解析 LLM JSON (markdown 围栏/裸数组/字段缺失)
  - dedup_mechanisms: 近重复合并 + 种子覆盖丢弃 + evidence 合并
  - coverage_report: 域/层/前提统计
"""

from __future__ import annotations

from darwin_st.knowledge.extraction import (
    canonical_name,
    coverage_report,
    dedup_mechanisms,
    normalize_domain,
    normalize_function_tags,
    normalize_preconditions,
    parse_mechanism_cards,
)
from darwin_st.knowledge.ontology import (
    PRECONDITION_VOCAB,
    Evidence,
    Mechanism,
)


# ---------------------------------------------------------------------------
# 名称归一
# ---------------------------------------------------------------------------


def test_canonical_name_snake_case():
    assert canonical_name("Masked Auto-Encoding") == "masked_auto_encoding"
    assert canonical_name("  Self-Attention!! ") == "self_attention"
    assert canonical_name("") == "unnamed"


# ---------------------------------------------------------------------------
# 前提归一 (互斥性命脉)
# ---------------------------------------------------------------------------


def test_normalize_preconditions_alias_to_vocab():
    out = normalize_preconditions(["redundancy", "unlabeled_data", "non_stationarity"])
    assert out == ["redundant_structure", "label_scarcity", "distribution_shift"]
    for p in out:
        assert p in PRECONDITION_VOCAB


def test_normalize_preconditions_passthrough_canonical():
    out = normalize_preconditions(["long_range_dependency", "spatial_smoothness"])
    assert out == ["long_range_dependency", "spatial_smoothness"]


def test_normalize_preconditions_drops_unknown():
    out = normalize_preconditions(["totally_made_up_precond", "redundancy"])
    assert out == ["redundant_structure"]  # 未知丢弃, 别名保留


def test_normalize_preconditions_dedup_preserves_order():
    out = normalize_preconditions(["long_range", "long_range_dependency", "redundancy"])
    assert out == ["long_range_dependency", "redundant_structure"]


def test_normalize_domain_and_function_tags():
    assert normalize_domain("spatio-temporal") == "ST"
    assert normalize_domain("vision") == "CV"
    assert normalize_domain("graph") == "GraphLearning"
    assert normalize_domain("nonsense") == "ST"  # 未知→默认本域
    assert normalize_function_tags(["temporal_modeling", "bogus_tag"]) == ["temporal_modeling"]


# ---------------------------------------------------------------------------
# LLM JSON 解析容错
# ---------------------------------------------------------------------------


def test_parse_cards_with_markdown_fence():
    raw = """好的, 这是结果:
```json
{"mechanisms": [
  {"name": "Patching", "abstract_function": "切块降序列长度",
   "preconditions": ["long_range", "redundancy"], "causal_behavior": "分块→token",
   "origin_domain": "nlp", "abstraction_level": "concept"}
]}
```
以上。"""
    cards = parse_mechanism_cards(raw)
    assert len(cards) == 1
    c = cards[0]
    assert c.name == "patching"
    assert c.origin_domain == "NLP"
    assert c.preconditions == ["long_range_dependency", "redundant_structure"]
    c.validate()  # 归一后应可通过 schema 校验


def test_parse_cards_bare_array():
    raw = '[{"name":"x","abstract_function":"y","preconditions":[],"causal_behavior":"z"}]'
    cards = parse_mechanism_cards(raw)
    assert len(cards) == 1 and cards[0].name == "x"


def test_parse_cards_skips_empty_and_garbage():
    raw = '{"mechanisms": [{"name":"", "abstract_function":"a"}, {"name":"ok","abstract_function":""}]}'
    assert parse_mechanism_cards(raw) == []
    assert parse_mechanism_cards("not json at all") == []


def test_parse_cards_invalid_abstraction_defaults_concept():
    raw = '{"mechanisms":[{"name":"m","abstract_function":"f","abstraction_level":"weird"}]}'
    cards = parse_mechanism_cards(raw)
    assert cards[0].abstraction_level == "concept"


# ---------------------------------------------------------------------------
# 去重 (正交性命脉)
# ---------------------------------------------------------------------------


def _mk(name, dom="CV", af="fn", pre=None, ev=None, math=""):
    return Mechanism(name=name, abstract_function=af, preconditions=pre or [],
                     causal_behavior="cb", origin_domain=dom, math_structure=math,
                     evidence=ev or [])


def test_dedup_merges_name_variants():
    ms = [_mk("masked_autoencoding"), _mk("Masked Autoencoder Module"),
          _mk("masked_auto_encoder")]
    kept, removed = dedup_mechanisms(ms)
    assert len(kept) == 1
    assert removed == 2


def test_dedup_keeps_distinct_qualified_names():
    # spatial vs temporal attention 是不同基元, 不应被合并
    ms = [_mk("spatial_attention"), _mk("temporal_attention")]
    kept, _ = dedup_mechanisms(ms)
    assert len(kept) == 2


def test_dedup_picks_richest_representative_and_merges_evidence():
    poor = _mk("state_space_model", af="state space")
    rich = _mk("State-Space-Model", af="state space", math="h_t=Ah+Bx",
               ev=[Evidence("LRA", "Acc", "+5", "high", "S4")])
    kept, removed = dedup_mechanisms([poor, rich])
    assert len(kept) == 1 and removed == 1
    assert kept[0].math_structure == "h_t=Ah+Bx"  # 选了信息多的代表
    assert len(kept[0].evidence) == 1


def test_dedup_drops_seed_covered():
    ms = [_mk("self_attention"), _mk("new_unique_mech")]
    kept, removed = dedup_mechanisms(ms, seed_names={"self_attention"})
    names = {m.name for m in kept}
    assert "new_unique_mech" in names
    assert "self_attention" not in names
    assert removed == 1


def test_dedup_deterministic_regardless_of_order():
    a = _mk("state_space_model", math="big_math_structure_here")
    b = _mk("State Space Model")
    k1, _ = dedup_mechanisms([a, b])
    k2, _ = dedup_mechanisms([b, a])
    assert k1[0].name == k2[0].name == "state_space_model"


# ---------------------------------------------------------------------------
# 覆盖统计 (全面性自评)
# ---------------------------------------------------------------------------


def test_coverage_report_counts_domains_and_unused_preconditions():
    ms = [_mk("a", dom="CV", pre=["redundant_structure"]),
          _mk("b", dom="NLP", pre=["long_range_dependency"]),
          _mk("c", dom="ST", pre=["redundant_structure"])]
    rep = coverage_report(ms)
    assert rep.n_total == 3
    assert rep.by_domain == {"CV": 1, "NLP": 1, "ST": 1}
    assert rep.by_precondition["redundant_structure"] == 2
    # 未用前提应包含没出现过的词
    assert "spatial_smoothness" in rep.unused_preconditions
    assert "redundant_structure" not in rep.unused_preconditions
