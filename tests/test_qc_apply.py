"""qc_apply.py 确定性内核的回归测试 (无 IO)。

锁住改库逻辑的契约, 重点是**种子保护铁律**:
  - plan_actions: 种子源 merge 被 skip、drop/split 入 drops、tgt 不存在 skip、空前提入 drops
  - apply_plan: merge 后源消失且目标继承证据/前提、drop 后消失、种子内容不被改、计数正确
"""

from __future__ import annotations

from darwin_st.knowledge.qc_apply import apply_plan, plan_actions


def _card(name, pre=None, ev=None, math="", af="f", domain="ST", level="concept"):
    return {"name": name, "abstract_function": af, "preconditions": pre or [],
            "function_tags": [], "anti_patterns": [], "evidence": ev or [],
            "math_structure": math, "origin_domain": domain, "abstraction_level": level}


PROTECTED = {"self_attention", "identity_embedding"}


# ---------------------------------------------------------------------------
# plan_actions
# ---------------------------------------------------------------------------


def test_plan_merge_normal():
    cards = [_card("luong_attention"), _card("softmax_attention")]
    verdicts = [{"name": "luong_attention", "verdict": "merge", "merge_into": "softmax_attention"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert plan.merges == [("luong_attention", "softmax_attention")]
    assert plan.drops == []


def test_plan_seed_source_merge_skipped():
    # 种子卡作为 merge 源 → 跳过 (铁律)
    cards = [_card("self_attention"), _card("softmax_attention")]
    verdicts = [{"name": "self_attention", "verdict": "merge", "merge_into": "softmax_attention"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert plan.merges == []
    assert any("受保护" in r for _, r in plan.skipped)


def test_plan_seed_as_merge_target_allowed():
    # 种子卡作为 merge 目标 → 允许 (种子继承别人, 自身保留)
    cards = [_card("auto_dilated_conv"), _card("dilated_causal_convolution")]
    protected = {"dilated_causal_convolution"}
    verdicts = [{"name": "auto_dilated_conv", "verdict": "merge",
                 "merge_into": "dilated_causal_convolution"}]
    plan = plan_actions(cards, verdicts, protected)
    assert plan.merges == [("auto_dilated_conv", "dilated_causal_convolution")]


def test_plan_drop_and_split_become_drops():
    cards = [_card("data_augmentation"), _card("masked_contrastive_learning")]
    verdicts = [{"name": "data_augmentation", "verdict": "drop"},
                {"name": "masked_contrastive_learning", "verdict": "split"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert set(plan.drops) == {"data_augmentation", "masked_contrastive_learning"}


def test_plan_seed_drop_skipped():
    cards = [_card("self_attention")]
    verdicts = [{"name": "self_attention", "verdict": "drop"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert plan.drops == []
    assert any("受保护" in r for _, r in plan.skipped)


def test_plan_merge_target_missing_skipped():
    cards = [_card("op_a")]
    verdicts = [{"name": "op_a", "verdict": "merge", "merge_into": "ghost"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert plan.merges == []
    assert any("不存在" in r for _, r in plan.skipped)


def test_plan_name_not_in_library_skipped():
    cards = [_card("op_a")]
    verdicts = [{"name": "vanished", "verdict": "drop"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    assert plan.drops == []
    assert any("已不在库" in r for _, r in plan.skipped)


def test_plan_empty_precondition_drops():
    cards = [_card("dead_card", pre=[]), _card("good", pre=["heterogeneity"])]
    plan = plan_actions(cards, [], PROTECTED, empty_precondition_drops=["dead_card"])
    assert plan.drops == ["dead_card"]


def test_plan_empty_precondition_seed_protected():
    cards = [_card("identity_embedding", pre=[])]
    plan = plan_actions(cards, [], PROTECTED, empty_precondition_drops=["identity_embedding"])
    assert plan.drops == []


# ---------------------------------------------------------------------------
# apply_plan
# ---------------------------------------------------------------------------


def test_apply_merge_removes_source_target_inherits():
    cards = [
        _card("luong_attention", pre=["long_range_dependency"], math="QK^T",
              ev=[{"dataset": "D", "metric": "MAE", "source": "S"}]),
        _card("softmax_attention", pre=["sequential_order"]),
    ]
    verdicts = [{"name": "luong_attention", "verdict": "merge", "merge_into": "softmax_attention"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    new_cards, stats = apply_plan(cards, plan)
    names = {c["name"] for c in new_cards}
    assert "luong_attention" not in names  # 源删
    assert "softmax_attention" in names
    tgt = next(c for c in new_cards if c["name"] == "softmax_attention")
    # 目标继承了源的前提 + 证据 + math (目标原本无 math)
    assert "long_range_dependency" in tgt["preconditions"]
    assert "sequential_order" in tgt["preconditions"]
    assert len(tgt["evidence"]) == 1
    assert tgt["math_structure"] == "QK^T"
    assert stats["merged"] == 1 and stats["after"] == 1


def test_apply_merge_does_not_overwrite_target_identity():
    # 目标的 name/abstract_function/domain 不被源覆盖
    cards = [_card("src_op", af="源功能", domain="CV"),
             _card("tgt_op", af="目标功能", domain="ST")]
    verdicts = [{"name": "src_op", "verdict": "merge", "merge_into": "tgt_op"}]
    plan = plan_actions(cards, verdicts, PROTECTED)
    new_cards, _ = apply_plan(cards, plan)
    tgt = next(c for c in new_cards if c["name"] == "tgt_op")
    assert tgt["abstract_function"] == "目标功能"
    assert tgt["origin_domain"] == "ST"


def test_apply_drop_removes_card():
    cards = [_card("data_augmentation"), _card("keepme")]
    plan = plan_actions(cards, [{"name": "data_augmentation", "verdict": "drop"}], PROTECTED)
    new_cards, stats = apply_plan(cards, plan)
    assert {c["name"] for c in new_cards} == {"keepme"}
    assert stats["dropped"] == 1


def test_apply_does_not_mutate_input():
    cards = [_card("a"), _card("b")]
    plan = plan_actions(cards, [{"name": "a", "verdict": "drop"}], PROTECTED)
    apply_plan(cards, plan)
    assert len(cards) == 2  # 原列表不变


def test_apply_empty_plan_noop():
    cards = [_card("a")]
    new_cards, stats = apply_plan(cards, plan_actions(cards, [], PROTECTED))
    assert len(new_cards) == 1 and stats["after"] == 1
