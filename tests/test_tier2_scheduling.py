"""v2 Tier-2 三段调度 (orchestrator._try_creation) 测试。

契约锚点:
  - 决策顺序: aux_due → refine-first → create; 一次触发 = 一个动作 = 账本 started+terminal;
  - aux preflight 无卡才回落 (aux_unavailable_fallback, 配额不重置);
  - aux 一旦启动, 成败均收尾, 不追加 refine/create;
  - create 自然路由进 aux → terminal final_action_type=aux_create (配额重置);
  - AUX_FORCE_EVERY=0 (默认) → 不强制, 行为同 v1; ledger=None → 不写账本。
"""

from __future__ import annotations

import pytest

from darwin_st.creation.action_ledger import ActionLedger
from darwin_st.creation.creation_loop import CreationOutcome
from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
from darwin_st.search.genotype import Genotype, STBlock


class _FakeLoop:
    """mock CreationLoop: 记录调用, 按脚本返回 outcome。"""

    def __init__(self, tmp_path, aux_cards=True, refine_candidate=True):
        self.action_ledger = ActionLedger(str(tmp_path / "tier2_actions.jsonl"))
        self._aux_cards = aux_cards
        self._refine_candidate = refine_candidate
        self.calls: list[str] = []
        self._last_used_insight_ids: list = []
        self.create_action_type = "operator_create"     # create 自然路由 aux 时改 "aux_create"
        self.outcome_maker = self._default_outcome

    def _default_outcome(self, action_type):
        g = Genotype(blocks=[STBlock("gcn", "tcn")])
        return CreationOutcome(True, operator_names=["synth_x"], seed_genotypes=[g],
                               n_hypotheses=1, n_success=1, action_type=action_type)

    # CreationLoop 接口面
    def aux_cards_available(self):
        return self._aux_cards

    def pick_aux_mechanisms(self, bottleneck="", max_pick=2):
        from types import SimpleNamespace
        return [SimpleNamespace(name="masked_autoencoding", abstract_function="",
                                preconditions=[])] if self._aux_cards else []

    def preselect_refine_candidate(self, _mae):
        if not self._refine_candidate:
            return None
        return ("synth_Foo", {"reg_name": "synth_Foo_v1", "real_mae": 18.5, "code": "x"},
                [{"reg_name": "synth_Foo_v1"}])

    def maybe_create_aux_direct(self, *_a, **kw):
        self.calls.append("aux")
        self.aux_mechanisms_received = kw.get("mechanisms")
        return self.outcome_maker("aux_create")

    def maybe_refine(self, *_a, **_kw):
        self.calls.append("refine")
        return self.outcome_maker("operator_refine")

    def maybe_create(self, *_a, **_kw):
        self.calls.append("create")
        return self.outcome_maker(self.create_action_type)

    def set_action_ctx(self, *_a):
        pass


def _orch(tmp_path, loop, aux_force_every=0):
    cfg = OrchestratorConfig(dataset="PeMS04", population_size=4, tournament_size=2,
                             max_rounds=1, target_mae=0.0, run_tag="exp/test",
                             aux_force_every=aux_force_every)
    orch = Orchestrator(cfg, lambda *a, **k: None, devices=1)
    orch.creation_loop = loop
    return orch


def _fill_non_aux(loop, n):
    """账本里预置 n 个 non-aux 终态动作。"""
    for i in range(n):
        aid = loop.action_ledger.start(i, "exp/test", requested_action_type="operator_refine",
                                       action_type="operator_refine",
                                       decision_reason="refine_first")
        loop.action_ledger.finish(aid, i, "failed")


# ---------------------------------------------------------------------------

def test_forced_aux_after_quota(tmp_path):
    """5 个 non-aux 终态后 → 强制 aux; 账本能追到 forced_aux_quota。"""
    loop = _FakeLoop(tmp_path)
    _fill_non_aux(loop, 5)
    orch = _orch(tmp_path, loop, aux_force_every=5)
    assert orch._try_creation() is True
    assert loop.calls == ["aux"]
    act = loop.action_ledger.replay()[-1]
    assert act.action_type == "aux_create" and act.decision_reason == "forced_aux_quota"
    assert act.terminal_status == "succeeded"
    assert act.selected_mechanism_ids == ["masked_autoencoding"]   # 真实选卡进 started (轮转闭环)
    assert [m.name for m in loop.aux_mechanisms_received] == ["masked_autoencoding"]
    assert loop.action_ledger.non_aux_since_last_aux() == 0     # 配额重置


def test_aux_failure_does_not_fall_back(tmp_path):
    """aux 启动后失败 → 本次触发结束, 不追加 refine/create。"""
    loop = _FakeLoop(tmp_path)
    loop.outcome_maker = lambda t: CreationOutcome(False, reason="未过防泄漏验证门",
                                                   action_type=t)
    _fill_non_aux(loop, 5)
    orch = _orch(tmp_path, loop, aux_force_every=5)
    assert orch._try_creation() is False
    assert loop.calls == ["aux"]                                # 没有 refine/create
    act = loop.action_ledger.replay()[-1]
    assert act.terminal_status == "failed" and act.failure_stage == "validate"
    assert loop.action_ledger.non_aux_since_last_aux() == 0     # 失败 aux 也重置


def test_aux_unavailable_falls_back_no_reset(tmp_path):
    """aux 配额到期但无卡 → 回落 refine, requested=aux_create, 配额不重置。"""
    loop = _FakeLoop(tmp_path, aux_cards=False)
    _fill_non_aux(loop, 5)
    orch = _orch(tmp_path, loop, aux_force_every=5)
    assert orch._try_creation() is True
    assert loop.calls == ["refine"]
    act = loop.action_ledger.replay()[-1]
    assert act.requested_action_type == "aux_create"
    assert act.action_type == "operator_refine"
    assert act.decision_reason == "aux_unavailable_fallback"
    assert loop.action_ledger.non_aux_since_last_aux() == 6     # 计 non-aux, 不重置


def test_no_quota_defaults_to_refine_first(tmp_path):
    """AUX_FORCE_EVERY=0: 永不强制 aux; 有候选族走 refine, 无候选回落 create。"""
    loop = _FakeLoop(tmp_path)
    orch = _orch(tmp_path, loop, aux_force_every=0)
    assert orch._try_creation() is True and loop.calls == ["refine"]

    loop2 = _FakeLoop(tmp_path / "b", refine_candidate=False)
    orch2 = _orch(tmp_path / "b", loop2, aux_force_every=0)
    assert orch2._try_creation() is True and loop2.calls == ["create"]
    act = loop2.action_ledger.replay()[-1]
    assert act.decision_reason == "create_fallback"


def test_natural_aux_records_final_type(tmp_path):
    """create 自然路由进 aux → terminal 记 final_action_type=aux_create + natural_aux, 配额重置。"""
    loop = _FakeLoop(tmp_path, refine_candidate=False)
    loop.create_action_type = "aux_create"
    _fill_non_aux(loop, 3)
    orch = _orch(tmp_path, loop, aux_force_every=5)
    assert orch._try_creation() is True and loop.calls == ["create"]
    act = loop.action_ledger.replay()[-1]
    assert act.action_type == "operator_create"                 # started 意图
    assert act.effective_action_type == "aux_create"            # terminal 修正
    assert act.effective_decision_reason == "natural_aux"
    assert loop.action_ledger.non_aux_since_last_aux() == 0


def test_action_ctx_cleared_and_one_action_per_trigger(tmp_path):
    """一次触发只产生一个动作 (started+terminal 两行)。"""
    loop = _FakeLoop(tmp_path)
    orch = _orch(tmp_path, loop)
    orch._try_creation()
    n_lines = len(open(loop.action_ledger.path).readlines())
    assert n_lines == 2
    assert loop.action_ledger.replay()[0].used_insight_ids == []


def test_exception_marks_failed_and_continues(tmp_path):
    """动作内异常 → terminal failed (exception:<type>) + creation_failed history, 不炸进化。"""
    loop = _FakeLoop(tmp_path)

    def boom(*_a, **_kw):
        raise RuntimeError("LLM 超时")
    loop.maybe_refine = boom
    orch = _orch(tmp_path, loop)
    assert orch._try_creation() is False
    act = loop.action_ledger.replay()[-1]
    assert act.terminal_status == "failed" and act.failure_stage == "exception:RuntimeError"
    assert any(h.get("event") == "creation_failed" for h in orch.state.history)


def test_no_ledger_keeps_v1_behavior(tmp_path):
    """ledger=None (自定义 loop 无 action_ledger): 不写账本, refine-first 旧语义。"""
    loop = _FakeLoop(tmp_path)
    loop.action_ledger = None
    orch = _orch(tmp_path, loop, aux_force_every=5)             # 配额无账本 → 不强制
    assert orch._try_creation() is True and loop.calls == ["refine"]


def test_used_insight_ids_cleared_between_actions(tmp_path):
    """create 用的 insight 不得串记到下一个 refine 动作 (动作级 collector 初始化)。"""
    loop = _FakeLoop(tmp_path)
    loop._last_used_insight_ids = [7, 9]                      # 上一动作遗留
    orch = _orch(tmp_path, loop)
    # 假 loop 模拟真实行为: set_action_ctx 时清空 (真实 CreationLoop 的实现)
    real_set = loop.set_action_ctx

    def _set_and_clear(*a):
        loop._last_used_insight_ids = []
        real_set(*a)
    loop.set_action_ctx = _set_and_clear
    orch._try_creation()                                      # refine 动作 (不注入 insight)
    act = loop.action_ledger.replay()[-1]
    assert act.action_type == "operator_refine"
    assert act.used_insight_ids == []                         # 不串上一动作的 [7, 9]
