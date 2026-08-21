"""v2 精炼熔断器 (refine_breaker) 测试。

契约锚点 (方案 P0-3 / F3):
  - 多候选按 action 聚合: 任一 child 达标 = 一次成功; 一败一 pending 不得开闸;
    全基建中断 = neutral (解 pending 不罚族); 终态无 MAE 不得永久 pending;
  - 观察时刻: breaker 在第 3 个失败被观察到时打开, cooldown 只计此后终态动作;
  - 父代相对操作性改善: child < parent_at_proposal × (1 - min_rel_delta) (提出时刻值);
  - pending 族禁发; 半开须等回填; failures=0 全策略关闭 = 旧行为。
"""

from __future__ import annotations

import pytest

from darwin_st.creation.action_ledger import ActionLedger
from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator
from darwin_st.creation.creation_loop import CreationConfig, CreationLoop
from darwin_st.creation.refine_breaker import (
    family_selectable,
    refine_action_result,
    replay_refine_breakers,
)
from darwin_st.creation.registry import OperatorRegistry
from darwin_st.search import operators as ops_mod


@pytest.fixture(autouse=True)
def _clean_spatial_ops():
    """每个测试后清理注入的 synth 算子, 不污染全局 SPATIAL_OPS。"""
    before = set(ops_mod.SPATIAL_OPS)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before:
            ops_mod.SPATIAL_OPS.pop(k, None)

_CODE = "import torch\nclass Foo(torch.nn.Module):\n    pass\n"


def _mkop(name, real_mae=None):
    plan = FusionPlan(operator_name=name, rationale="r", shared_structure="s",
                      composition="additive_residual", source_mechanisms=["m1"],
                      expected_effect="e")
    return SynthesizedOperator(name=name, code=_CODE.replace("Foo", name), plan=plan,
                               needs_adj=False, validated=True, real_mae=real_mae)


def _refine_action(led, seq, family, parent_mae, cand=1, status="succeeded"):
    aid = led.start(seq, "exp/t", requested_action_type="operator_refine",
                    action_type="operator_refine", decision_reason="refine_first",
                    target_refine_family=family, parent_operator=f"{family}_v1",
                    parent_mae_at_proposal=parent_mae)
    led.finish(aid, seq, status, candidate_count=cand)
    return aid


def _non_aux_action(led, seq):
    aid = led.start(seq, "exp/t", requested_action_type="operator_create",
                    action_type="operator_create", decision_reason="create_fallback")
    led.finish(aid, seq, "failed", candidate_count=0)
    return aid


def _outcome(led, aid, seq, mae=None, kind="resolved", observed_at=None):
    led.outcome(aid, seq, "synth_x", mae, kind,
                resolved_after_action_seq=observed_at if observed_at is not None else seq)


# ---------------------------------------------------------------------------
# refine_action_result: 多候选按动作聚合
# ---------------------------------------------------------------------------

def _act(seq, cand, status="succeeded", family="famA", parent_mae=18.5):
    led_path = ":memory:"
    led = ActionLedger.__new__(ActionLedger)      # 不落盘, 只要 Tier2Action
    from darwin_st.creation.action_ledger import Tier2Action
    return Tier2Action(action_id=f"a{seq}", action_seq=seq, action_type="operator_refine",
                       target_refine_family=family, parent_mae_at_proposal=parent_mae,
                       terminal_status=status, candidate_count=cand)


def test_aggregate_two_fail_one_success_is_success():
    """一次动作 3 个 child: 两败一胜 = 一次成功 (不得算 2 次连败)。"""
    act = _act(0, cand=3)
    outcomes = [{"outcome_kind": "resolved", "seed_val_mae": 18.9, "resolved_after_action_seq": 0},
                {"outcome_kind": "resolved", "seed_val_mae": 18.7, "resolved_after_action_seq": 0},
                {"outcome_kind": "resolved", "seed_val_mae": 18.4, "resolved_after_action_seq": 0}]
    res, _ = refine_action_result(act, outcomes, 0.001)
    assert res == "success"                          # 18.4 < 18.5×0.999


def test_aggregate_pending_blocks():
    """一个 child 已失败、另一个仍 pending → 动作 pending, 不得开闸。"""
    act = _act(0, cand=2)
    outcomes = [{"outcome_kind": "resolved", "seed_val_mae": 19.0, "resolved_after_action_seq": 0}]
    res, obs = refine_action_result(act, outcomes, 0.001)
    assert res == "pending" and obs is None


def test_aggregate_all_infra_neutral():
    """全部基础设施中断 → neutral (解 pending, 不增连败也不重置)。"""
    act = _act(0, cand=2)
    outcomes = [{"outcome_kind": "neutral", "seed_val_mae": None, "resolved_after_action_seq": 0},
                {"outcome_kind": "neutral", "seed_val_mae": None, "resolved_after_action_seq": 0}]
    res, _ = refine_action_result(act, outcomes, 0.001)
    assert res == "neutral"


def test_terminal_without_mae_is_operational_failure_not_pending():
    """终态无有限 MAE (NaN/模型错误) → operational failure, 不得永久 pending。"""
    act = _act(0, cand=1)
    outcomes = [{"outcome_kind": "operational_failure", "seed_val_mae": None,
                 "resolved_after_action_seq": 0}]
    res, obs = refine_action_result(act, outcomes, 0.001)
    assert res == "failure" and obs == 0


def test_synthesis_stage_failure_counts_once():
    """合成/验证阶段失败 (无 seed): 可归因失败, 观察时刻=动作自身序号。"""
    act = _act(0, cand=0, status="failed")
    res, obs = refine_action_result(act, [], 0.001)
    assert res == "failure" and obs == 0


def test_relative_threshold_uses_proposal_time_parent():
    """父代相对操作性改善: child 18.49 vs parent 18.50 (0.054% < 0.1%) → 不算改善。"""
    act = _act(0, cand=1, parent_mae=18.5)
    outcomes = [{"outcome_kind": "resolved", "seed_val_mae": 18.49, "resolved_after_action_seq": 0}]
    res, _ = refine_action_result(act, outcomes, 0.001)
    assert res == "failure"                          # 18.49 > 18.5×0.999=18.4815
    outcomes[0]["seed_val_mae"] = 18.47
    res, _ = refine_action_result(act, outcomes, 0.001)
    assert res == "success"


# ---------------------------------------------------------------------------
# replay_refine_breakers: 状态机 + 观察时刻
# ---------------------------------------------------------------------------

def test_three_failures_open_then_half_open_after_cooldown(tmp_path):
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    # 族 A 连续 3 次失败 (每次观察时刻即动作序号)
    for i in range(3):
        aid = _refine_action(led, i, "famA", parent_mae=18.5)
        _outcome(led, aid, i, mae=18.9, observed_at=i)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "cooldown"
    assert states["famA"]["opened_at_seq"] == 2
    assert not family_selectable(states, "famA")
    # 冷却只计 opened 之后完成的终态动作: 2 个后 → 半开
    _non_aux_action(led, 3)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "cooldown"     # 才 1 个, 不够
    _non_aux_action(led, 4)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "half_open"
    assert family_selectable(states, "famA")


def test_probe_success_closes_probe_failure_recools(tmp_path):
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    for i in range(3):
        aid = _refine_action(led, i, "famA", parent_mae=18.5)
        _outcome(led, aid, i, mae=18.9, observed_at=i)
    _non_aux_action(led, 3)
    _non_aux_action(led, 4)                          # 冷却满足 → 半开
    aid = _refine_action(led, 5, "famA", parent_mae=18.5)
    _outcome(led, aid, 5, mae=18.4, observed_at=5)   # 探针改善
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "closed"
    # 再来 3 连败 → 又开; 探针失败 → 立即重冷却 (不等 3 次)
    for i in range(6, 9):
        aid = _refine_action(led, i, "famA", parent_mae=18.5)
        _outcome(led, aid, i, mae=18.9, observed_at=i)
    _non_aux_action(led, 9)
    _non_aux_action(led, 10)
    aid = _refine_action(led, 11, "famA", parent_mae=18.5)   # 探针
    _outcome(led, aid, 11, mae=18.9, observed_at=11)         # 探针再败
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "cooldown"
    assert states["famA"]["opened_at_seq"] == 11


def test_pending_family_not_selectable_and_neutral_unblocks(tmp_path):
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    for i in range(2):                               # 2 连败 (未达阈值)
        aid = _refine_action(led, i, "famA", parent_mae=18.5)
        _outcome(led, aid, i, mae=18.9, observed_at=i)
    aid = _refine_action(led, 2, "famA", parent_mae=18.5)    # 第 3 次: 无 outcome = pending
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["has_pending"]
    assert not family_selectable(states, "famA")             # pending 禁发
    _outcome(led, aid, 2, kind="neutral", observed_at=3)     # 基建中断回填
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "closed"               # neutral 解 pending 不罚族
    assert family_selectable(states, "famA")
    assert states["famA"]["consecutive_failures"] == 2       # 连败保留 (不增不重置)


def test_out_of_order_backfill_opens_at_observation(tmp_path):
    """乱序回填: action 1 的失败在 action 4 完成后才被观察到 → 开闸点在观察时刻。"""
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    aid0 = _refine_action(led, 0, "famA", parent_mae=18.5)
    _outcome(led, aid0, 0, mae=18.9, observed_at=0)
    aid1 = _refine_action(led, 1, "famA", parent_mae=18.5)
    _outcome(led, aid1, 1, mae=18.9, observed_at=1)
    aid2 = _refine_action(led, 2, "famA", parent_mae=18.5)   # 第 3 次精炼
    _non_aux_action(led, 3)                                  # 后续动作先完成
    _non_aux_action(led, 4)
    # 第 3 次失败结果此时才被观察到 (观察时已到 action 4)
    _outcome(led, aid2, 2, mae=18.9, observed_at=4)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["opened_at_seq"] == 4              # 开闸在观察时刻, 非动作序号 2
    assert states["famA"]["state"] == "cooldown"             # 3/4 在观察前完成, 不计入冷却
    _non_aux_action(led, 5)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "cooldown"             # 只有 action 5 计入
    _non_aux_action(led, 6)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "half_open"


def test_success_resets_streak(tmp_path):
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    for i, mae in enumerate([18.9, 18.9, 18.4, 18.9]):
        aid = _refine_action(led, i, "famA", parent_mae=18.5)
        _outcome(led, aid, i, mae=mae, observed_at=i)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["famA"]["state"] == "closed"
    assert states["famA"]["consecutive_failures"] == 1       # 成功后被重置, 只算最近一次


# ---------------------------------------------------------------------------
# CreationLoop._pick_refine_candidate 集成: 熔断过滤 + 最久未精炼优先 + 关闭=旧行为
# ---------------------------------------------------------------------------


def _loop_with_families(tmp_path, breaker=3, cooldown=2):
    reg = OperatorRegistry()
    reg.register(_mkop("FamA", real_mae=18.5))
    reg.register(_mkop("FamB", real_mae=18.6))
    reg.register(_mkop("FamC", real_mae=18.7))
    ledger = ActionLedger(str(tmp_path / "l.jsonl"))
    loop = CreationLoop(None, None, None, reg,
                        config=CreationConfig(refine_breaker_failures=breaker,
                                              refine_breaker_cooldown=cooldown),
                        action_ledger=ledger)
    return loop, ledger


def test_pick_excludes_cooled_family_and_rotates(tmp_path):
    loop, ledger = _loop_with_families(tmp_path)
    # FamA 3 连败 → 冷却
    for i in range(3):
        aid = _refine_action(ledger, i, "synth_FamA", parent_mae=18.5)
        _outcome(ledger, aid, i, mae=18.9, observed_at=i)
    pick = loop._pick_refine_candidate(18.5)
    assert pick[0] == "synth_FamB"                           # FamA 被熔断排除
    # FamB 被精炼过一次后, 最久未精炼优先 → 轮到 FamC
    aid = _refine_action(ledger, 3, "synth_FamB", parent_mae=18.6)
    _outcome(ledger, aid, 3, mae=18.55, observed_at=3)       # 改善 (不关熔断事)
    pick = loop._pick_refine_candidate(18.5)
    assert pick[0] == "synth_FamC"


def test_breaker_disabled_keeps_greedy_behavior(tmp_path):
    """failures=0: 熔断/pending/轮换全关, 旧贪心 (族 best 最小者) —— 即垄断旧行为。"""
    loop, ledger = _loop_with_families(tmp_path, breaker=0)
    for i in range(5):                                       # FamA 5 连败也照选 (旧行为)
        aid = _refine_action(ledger, i, "synth_FamA", parent_mae=18.5)
        _outcome(ledger, aid, i, mae=18.9, observed_at=i)
    pick = loop._pick_refine_candidate(18.5)
    assert pick[0] == "synth_FamA"


def test_all_families_cooled_returns_none(tmp_path):
    loop, ledger = _loop_with_families(tmp_path, cooldown=100)  # 长冷却: 后续动作不解锁
    seq = 0
    for fam in ["synth_FamA", "synth_FamB", "synth_FamC"]:
        for _ in range(3):
            aid = _refine_action(ledger, seq, fam, parent_mae=18.5)
            _outcome(ledger, aid, seq, mae=18.9, observed_at=seq)
            seq += 1
    assert loop._pick_refine_candidate(18.5) is None         # 全灭 → 回落从零创造


def test_llm_exception_stage_is_neutral_not_family_fault(tmp_path):
    """LLM 超时/服务异常 (exception:* 或 llm 阶段) 无 seed → neutral, 不罚族。"""
    from darwin_st.creation.action_ledger import Tier2Action
    act = Tier2Action(action_id="a0", action_seq=0, action_type="operator_refine",
                      target_refine_family="famA", parent_mae_at_proposal=18.5,
                      terminal_status="failed", failure_stage="exception:TimeoutError",
                      candidate_count=0)
    res, _ = refine_action_result(act, [], 0.001)
    assert res == "neutral"
    act.failure_stage = "validate"
    res, obs = refine_action_result(act, [], 0.001)
    assert res == "failure" and obs == 0                      # 验证门败: 可归因


def test_outcome_kind_enum_classification():
    """_outcome_kind 枚举: 候选自身 OOM 归候选 (operational), 中断归基建 (neutral)。"""
    from darwin_st.optim.orchestrator import _outcome_kind
    assert _outcome_kind("CRASH", float("nan"), "cuda oom: out of memory") \
        == "operational_failure"                            # 候选 OOM 是架构信号
    assert _outcome_kind("CRASH", float("nan"), "KeyboardInterrupt 中断") == "neutral"
    assert _outcome_kind("KEEP", 18.5, None) == "resolved"
    assert _outcome_kind("CRASH", float("nan"), "all_hpo_trials_failed") \
        == "operational_failure"


def test_real_chain_refiner_llm_timeout_is_neutral(tmp_path):
    """真实链路: OperatorRefiner 的 llm.chat 超时 → last_error 带 LLM_INFRA 标记
    → _failure_stage_of=llm → terminal → refine_action_result=neutral (不误罚家族)。"""
    from darwin_st.creation import CreationConfig, CreationLoop, MockLLM, OperatorSynthesizer
    from darwin_st.creation.synthesizer import SynthesisConfig
    from darwin_st.optim.orchestrator import Orchestrator

    reg = OperatorRegistry()
    reg.register(_mkop("FamA", real_mae=18.5))
    llm = MockLLM(lambda msgs: (_ for _ in ()).throw(TimeoutError("API 超时")))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    loop = CreationLoop(None, None, synth, reg, config=CreationConfig())
    outcome = loop.maybe_refine(round_idx=0, current_best_mae=18.5)
    assert outcome is not None and not outcome.success
    assert "LLM_INFRA: TimeoutError" in outcome.reason          # 结构化标记穿透包装文本
    assert Orchestrator._failure_stage_of(outcome) == "llm"

    # 走完账本: terminal(failed, llm) → 聚合 neutral
    from darwin_st.creation.action_ledger import ActionLedger
    led = ActionLedger(str(tmp_path / "l.jsonl"))
    aid = led.start(0, "exp/t", requested_action_type="operator_refine",
                    action_type="operator_refine", decision_reason="refine_first",
                    target_refine_family="synth_FamA", parent_operator="synth_FamA",
                    parent_mae_at_proposal=18.5)
    led.finish(aid, 0, "failed", failure_stage="llm", candidate_count=0)
    states = replay_refine_breakers(led, failures=3, cooldown=2)
    assert states["synth_FamA"]["consecutive_failures"] == 0   # 不罚族
    assert states["synth_FamA"]["state"] == "closed"


def test_outcome_kind_broken_process_pool_is_neutral():
    """scheduler 的 BrokenProcessPool (worker 池故障) → neutral; 候选 shape/NaN → operational。"""
    from darwin_st.optim.orchestrator import _outcome_kind
    assert _outcome_kind("CRASH", float("nan"),
                         "BrokenProcessPool: A process in the pool was terminated abruptly") \
        == "neutral"
    assert _outcome_kind("CRASH", float("nan"), "shape mismatch: expected 307x12") \
        == "operational_failure"
    assert _outcome_kind("CRASH", float("nan"), "loss is nan") == "operational_failure"
