"""精炼熔断器 (Refinement Circuit Breaker) —— outcome-aware, 纯函数重放, 无第二事实源。

治"精炼贪心垄断": _pick_refine_candidate 纯按族 best MAE 排序, 最强家族恒候选 →
精炼预算长期砸单一族 (B7 v1 实证: ScaleGatedGCNExpertMixer 精炼到 v20)。

判据 ("父代相对操作性改善"):
  child_mae < parent_mae_at_proposal × (1 - min_rel_delta) 才算改善;
  adopted(=KEEP) 不作数 (KEEP 阈值宽松, 不改善也可能 KEEP)。
  注意: min_rel_delta=0.001 是暂定工程阈值 (PeMS04 ≈0.018), 单 seed 0.1% 不代表统计
  显著, 后续应按重复训练波动校准。

多候选按 action 聚合 (不把一次动作的多候选失败算成多次连败):
  - 任一 child 达标 → 动作成功;
  - 全部 child 终态 + ≥1 个可归因非改善 + 无改善 → 动作失败;
  - 任一 child 待评估 → 动作 pending;
  - 全部基础设施中断 → neutral (解 pending, 不增连败也不重置连败);
  - 合成阶段无 seed 产出: LLM 异常/服务超时/中断 (exception:*/llm/interrupt) → neutral
    (基建问题不罚族); 验证门/计划阶段失败 (validate/plan/synthesis) → 可归因失败;
  - 终态无有限 MAE 不得永久 pending (operational_failure/neutral 都是终态);
  - 候选自身 OOM 计 operational failure (架构过大是真实信号), 不归基建。

观察时刻 (异步回填防时间穿越): breaker 在第 N 个失败结果【被观察到时】打开
(outcome 事件的 resolved_after_action_seq), cooldown 只计此后完成的终态动作;
同族有 pending 精炼时禁止再发射; 半开试验须等结果回填再定关闭/重冷却。

状态机 (每族): closed →(连续 failures 次无改善) → cooldown (K 个终态动作) →
half_open (允许 1 次探针) → 改善关闭 / 再败重冷却 (opened_at 更新为探针观察时刻)。
全部状态由本模块从动作账本重放得出, 不落盘 —— 重启天然恢复, 离线可审计。
"""

from __future__ import annotations

__all__ = ["refine_action_result", "replay_refine_breakers", "family_selectable"]


def refine_action_result(action, outcomes: list[dict], min_rel_delta: float
                         ) -> tuple[str, int | None]:
    """判定一个精炼动作的聚合结局。返回 (result, observed_seq)。

    result ∈ success / failure / pending / neutral; observed_seq = 决定性结果被观察时
    已过最大 action_seq (pending 时为 None)。契约见模块 docstring。
    """
    if not action.complete:
        return ("pending", None)
    if action.candidate_count == 0:
        # 合成阶段无 seed 产出: 按失败阶段枚举归因 —— LLM 异常/服务超时/中断 (exception:*,
        # llm, interrupt) 是基础设施问题, 不罚族 (neutral); 验证门/计划阶段失败
        # (validate/plan/synthesis) 可归因于该族的可精炼性, 计一次失败。
        if action.terminal_status == "failed":
            stage = action.failure_stage or ""
            if stage.startswith("exception:") or stage in ("llm", "interrupt"):
                return ("neutral", action.action_seq)
            return ("failure", action.action_seq)
        return ("neutral", action.action_seq)      # interrupted/unavailable: 不归因于族
    if len(outcomes) < action.candidate_count:
        return ("pending", None)                   # 有 child 未消化 → 真 pending
    parent = action.parent_mae_at_proposal
    improved = False
    attributable_fail = False
    any_resolved = False
    observed = 0
    for o in outcomes:
        observed = max(observed, int(o.get("resolved_after_action_seq", 0)))
        kind = o.get("outcome_kind")
        if kind == "resolved":
            any_resolved = True
            mae = o.get("seed_val_mae")
            if (parent is not None and isinstance(mae, (int, float))
                    and mae < parent * (1 - min_rel_delta)):
                improved = True
            else:
                attributable_fail = True           # 有限 MAE 但未达相对阈值
        elif kind == "operational_failure":
            any_resolved = True
            attributable_fail = True               # NaN/模型错误/训练不可用: 可归因
        # neutral: 基础设施中断/人工取消 —— censored, 不归因
    if improved:
        return ("success", observed)
    if not any_resolved:
        return ("neutral", observed)               # 全部基建中断: 不罚也不奖
    if attributable_fail:
        return ("failure", observed)
    return ("neutral", observed)                   # 兜底: 无改善证据也不罚


def replay_refine_breakers(ledger, failures: int = 3, cooldown: int = 5,
                           min_rel_delta: float = 0.001) -> dict:
    """从动作账本纯函数重放每族熔断状态。

    返回 {family: {"state": "closed"|"cooldown"|"half_open",
                   "has_pending": bool, "opened_at_seq": int|None,
                   "cooldown_remaining": int, "consecutive_failures": int,
                   "last_refine_seq": int}}。不含任何精炼动作的族不出现在表里 (视为 closed)。
    """
    actions = ledger.replay()
    outcomes = ledger.outcomes_by_action()
    completed_seqs = sorted(a.action_seq for a in actions if a.complete)

    by_family: dict[str, list] = {}
    for a in actions:
        if a.effective_action_type != "operator_refine" or not a.target_refine_family:
            continue
        by_family.setdefault(a.target_refine_family, []).append(a)

    states: dict[str, dict] = {}
    for fam, acts in by_family.items():
        consec = 0
        opened_seq: int | None = None
        has_pending = False
        for a in acts:                             # replay() 已按 action_seq 升序
            res, observed = refine_action_result(a, outcomes.get(a.action_id, []),
                                                 min_rel_delta)
            if res == "pending":
                has_pending = True
                continue
            if res == "neutral":
                continue                           # 不增连败也不重置
            if opened_seq is not None:
                # 冷却/半开期间被解决的精炼视为探针: 改善 → 关闭; 再败 → 重新冷却
                if res == "success":
                    opened_seq = None
                    consec = 0
                else:
                    opened_seq = observed
                continue
            if res == "success":
                consec = 0
            else:
                consec += 1
                if consec >= failures:
                    opened_seq = observed          # 第 N 个失败被观察到时打开
                    consec = 0
        state = "closed"
        remaining = 0
        if opened_seq is not None:
            done_after = sum(1 for s in completed_seqs if s > opened_seq)
            remaining = max(0, cooldown - done_after)
            state = "cooldown" if remaining > 0 else "half_open"
        states[fam] = {"state": state, "has_pending": has_pending,
                       "opened_at_seq": opened_seq, "cooldown_remaining": remaining,
                       "consecutive_failures": consec,
                       "last_refine_seq": acts[-1].action_seq}
    return states


def family_selectable(states: dict, family: str) -> bool:
    """族当前是否可被精炼选中: 无记录 → 可选; pending/冷却 → 不可选; 半开可试一次。"""
    st = states.get(family)
    if st is None:
        return True
    if st["has_pending"]:
        return False
    return st["state"] in ("closed", "half_open")
