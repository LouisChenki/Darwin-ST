"""qc.py 确定性内核的回归测试 (无 LLM/网络)。

锁住质检骨架的契约:
  - concept_head: 取末位实义 token, 剥噪声尾词
  - cluster_by_concept: 同中心词聚到一组, 组按大小排
  - rule_based_flags: 空前提/短功能/无公式
  - parse_verdicts: 正常 JSON + 截断容错 + 未知裁决保守归 keep
  - render_report: 各裁决分区 + 规则标记 + 聚类一览
  - mechanisms_from_cards: dict → Mechanism 重建
"""

from __future__ import annotations

from darwin_st.knowledge.ontology import Mechanism
from darwin_st.knowledge.qc import (
    cluster_by_concept,
    concept_head,
    mechanisms_from_cards,
    parse_verdicts,
    render_report,
    rule_based_flags,
)


def _m(name, af="一个足够长的抽象功能描述文本用于稳妥通过短文本规则检查这一项目从而不会被误标记成内容过短",
       pre=("long_range_dependency",), math="y=Wx", domain="ST", level="concept"):
    # pre 用元组默认值: 显式传 pre=[] 时不会被 `or` 替换 (空列表是 falsy)
    return Mechanism(
        name=name, abstract_function=af, preconditions=list(pre),
        causal_behavior="c", math_structure=math, origin_domain=domain,
        abstraction_level=level,
    )


# ---------------------------------------------------------------------------
# 中心词
# ---------------------------------------------------------------------------


def test_concept_head_takes_last_meaningful_token():
    assert concept_head("dynamic_graph_attention") == "attention"
    assert concept_head("graph_diffusion_convolution") == "convolution"
    assert concept_head("self_attention") == "attention"


def test_concept_head_strips_noise_suffix():
    # module/model/block/network/learning 是噪声尾词, 应剥掉取前一个实义词
    assert concept_head("masked_autoencoder_module") == "autoencoder"
    assert concept_head("graph_convolution_network") == "convolution"
    assert concept_head("contrastive_learning") == "contrastive"


def test_concept_head_all_noise_falls_back():
    # 全噪声词 → 回退 canonical 全名 (自成一组, 不误并)
    assert concept_head("module_block") == "module_block"
    assert concept_head("") == "unnamed"


# ---------------------------------------------------------------------------
# 聚类
# ---------------------------------------------------------------------------


def test_cluster_groups_same_head():
    mechs = [_m("self_attention"), _m("graph_attention"), _m("temporal_attention"),
             _m("dilated_convolution")]
    clusters = cluster_by_concept(mechs)
    assert len(clusters["attention"]) == 3
    assert len(clusters["convolution"]) == 1


def test_cluster_sorted_by_size_desc():
    mechs = [_m("a_convolution"), _m("self_attention"), _m("graph_attention")]
    heads = list(cluster_by_concept(mechs).keys())
    assert heads[0] == "attention"  # 2 张的组排前


def test_cluster_members_sorted_by_name():
    mechs = [_m("zzz_attention"), _m("aaa_attention")]
    g = cluster_by_concept(mechs)["attention"]
    assert [m.name for m in g] == ["aaa_attention", "zzz_attention"]


# ---------------------------------------------------------------------------
# 规则标记
# ---------------------------------------------------------------------------


def test_rule_flags_empty_preconditions():
    flags = rule_based_flags([_m("op_a", pre=[])])
    assert "empty_preconditions" in flags["op_a"]


def test_rule_flags_short_abstract_function():
    flags = rule_based_flags([_m("op_b", af="太短")])
    assert "short_abstract_function" in flags["op_b"]


def test_rule_flags_no_math():
    flags = rule_based_flags([_m("op_c", math="")])
    assert "no_math_structure" in flags["op_c"]


def test_rule_flags_clean_card_not_flagged():
    # 字段齐全的卡不应被标记
    assert rule_based_flags([_m("good_op")]) == {}


# ---------------------------------------------------------------------------
# 裁决解析
# ---------------------------------------------------------------------------


def test_parse_verdicts_normal():
    resp = ('{"verdicts": [{"name": "self_attention", "verdict": "keep", "reason": "r"},'
            '{"name": "scaled_dot_product_attention", "verdict": "merge",'
            ' "merge_into": "self_attention", "reason": "same"}]}')
    vs = parse_verdicts(resp)
    assert len(vs) == 2
    assert vs[1]["verdict"] == "merge"
    assert vs[1]["merge_into"] == "self_attention"


def test_parse_verdicts_unknown_verdict_is_keep():
    # 未知裁决保守归 keep, 绝不误删
    resp = '{"verdicts": [{"name": "x", "verdict": "nuke", "reason": "r"}]}'
    assert parse_verdicts(resp)[0]["verdict"] == "keep"


def test_parse_verdicts_salvages_truncated():
    # 被 max_tokens 截断 → 救回截断前完整裁决 (复用 _extract_json 截断容错)
    truncated = ('{"verdicts": [\n'
                 '{"name": "a", "verdict": "keep", "reason": "ok"},\n'
                 '{"name": "b", "verdict": "drop", "reason": "empty shell"},\n'
                 '{"name": "c", "verdict": "mer')  # 截断
    vs = parse_verdicts(truncated)
    assert [v["name"] for v in vs] == ["a", "b"]


def test_parse_verdicts_garbage():
    assert parse_verdicts("not json") == []


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def test_render_report_sections_present():
    mechs = [_m("self_attention"), _m("graph_attention", pre=[])]
    clusters = cluster_by_concept(mechs)
    verdicts = [{"name": "graph_attention", "verdict": "merge",
                 "merge_into": "self_attention", "reason": "dup"}]
    flags = rule_based_flags(mechs)
    md = render_report(mechs, clusters, verdicts, flags)
    assert "# 机制库质检报告" in md
    assert "建议合并" in md
    assert "graph_attention" in md
    assert "empty_preconditions" in md  # 规则标记进了报告
    assert "`attention`" in md  # 聚类一览


def test_render_report_empty_verdicts_no_crash():
    mechs = [_m("solo_op")]
    md = render_report(mechs, cluster_by_concept(mechs), [], {})
    assert "(无)" in md


# ---------------------------------------------------------------------------
# 卡重建
# ---------------------------------------------------------------------------


def test_mechanisms_from_cards():
    cards = [{"name": "op_a", "abstract_function": "f", "preconditions": ["heterogeneity"],
              "origin_domain": "CV", "abstraction_level": "variant"},
             {"name": "", "abstract_function": "skip me"},  # 无名跳过
             {"not_a": "dict"}]
    mechs = mechanisms_from_cards(cards)
    assert len(mechs) == 1
    assert mechs[0].name == "op_a"
    assert mechs[0].origin_domain == "CV"
