"""Tier-2 动作账本 (Action Ledger) —— 两事件生命周期的轻量 jsonl 账本。

为什么需要它: CreationRecord 只记"候选", 记不了"动作" —— LLM 异常/解析失败/无候选/
落盘前中断的动作在履历本里可能一行都没有, 配额与熔断若从候选行猜测会语义冲突。
账本为每次 Tier-2 动作写两行事件:

  event=started:   action_id, action_seq, run_tag, requested_action_type, action_type,
                   decision_reason, target_refine_family / selected_mechanism_ids,
                   parent_operator / parent_mae_at_proposal (refine 时), started_at
  event=terminal:  action_id, action_seq, terminal_status, failure_stage,
                   candidate_count, used_insight_ids, finished_at
                   (+ action_type: 实际执行类型与 started 不同时以 terminal 为准,
                    如检索自然命中 aux → natural_aux)

重放规则 (replay):
  - started + terminal = 完整动作; 只有 started = incomplete (不参与 quota/cooldown);
  - action_seq 从所有 started 事件的最大值续排 (防崩溃后重号);
  - 捕获到的中断写 interrupted; SIGKILL/掉电留 unmatched started, 不伪造 interrupted;
  - 单写者逐行 append + flush/fsync; 加载器丢弃唯一截断尾行, 中间行损坏 → fail closed (抛错)。

aux 配额计数表 (aux_due; 唯一语义):
  non-aux succeeded/failed              → non-aux 计数 +1
  aux 实际启动后 succeeded/failed       → 重置计数
  aux preflight unavailable → 回落      → 按最终执行的 non-aux 计数, 不重置
  caught interrupted / unmatched started → 不计数, 不重置
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = ["ActionLedger", "ActionLedgerError", "Tier2Action"]

ACTION_TYPES = ("operator_create", "operator_refine", "aux_create")
TERMINAL_STATUSES = ("succeeded", "failed", "unavailable", "interrupted")


class ActionLedgerError(Exception):
    """账本损坏 (中间行非法/结构校验失败) —— fail closed, 调用方不应静默继续。"""


LEDGER_SCHEMA_VERSION = 1
OUTCOME_KINDS = ("resolved", "operational_failure", "neutral")


def _validate_event(ev: dict, seen_started: set, finished: set) -> None:
    """最小结构校验 (fail closed): 非法直接 ActionLedgerError。

    覆盖: 事件类型/schema_version/action_id/action_seq/枚举/重复 started/孤立或重复
    terminal —— 这些污染会直接破坏 quota 与 breaker 的重放语义。
    """
    if not isinstance(ev, dict):
        raise ActionLedgerError(f"事件不是 JSON 对象: {ev!r:.80}")
    kind = ev.get("event")
    if kind not in ("started", "terminal", "outcome"):
        raise ActionLedgerError(f"非法事件类型: {kind!r}")
    sv = ev.get("schema_version")
    if sv is not None and sv != LEDGER_SCHEMA_VERSION:
        raise ActionLedgerError(f"不支持的 schema_version: {sv!r}")
    aid = ev.get("action_id")
    if not isinstance(aid, str) or not aid:
        raise ActionLedgerError(f"缺/非法 action_id: {aid!r}")
    seq = ev.get("action_seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise ActionLedgerError(f"缺/非法 action_seq: {seq!r}")
    if kind == "started":
        if aid in seen_started:
            raise ActionLedgerError(f"重复 started: {aid}")
        at = ev.get("action_type")
        if at is not None and at not in ACTION_TYPES:
            raise ActionLedgerError(f"非法 action_type: {at!r}")
    elif kind == "terminal":
        if aid not in seen_started:
            raise ActionLedgerError(f"孤立 terminal (无 started): {aid}")
        if aid in finished:
            raise ActionLedgerError(f"重复 terminal: {aid}")
        st = ev.get("terminal_status")
        if st not in TERMINAL_STATUSES:
            raise ActionLedgerError(f"非法 terminal_status: {st!r}")
    else:  # outcome
        ok = ev.get("outcome_kind")
        if ok not in OUTCOME_KINDS:
            raise ActionLedgerError(f"非法 outcome_kind: {ok!r}")


@dataclass
class Tier2Action:
    """重放后的一个动作 (started + 可选 terminal)。"""

    action_id: str
    action_seq: int
    run_tag: str = ""
    requested_action_type: str = ""
    action_type: str = ""                 # started 意图类型
    decision_reason: str = ""
    target_refine_family: str | None = None
    selected_mechanism_ids: list[str] = field(default_factory=list)
    parent_operator: str | None = None
    parent_mae_at_proposal: float | None = None
    started_at: str = ""
    # terminal (None = incomplete)
    terminal_status: str | None = None
    failure_stage: str | None = None
    candidate_count: int = 0
    used_insight_ids: list = field(default_factory=list)
    finished_at: str = ""
    final_action_type: str | None = None  # terminal 修正后的实际执行类型
    final_decision_reason: str | None = None  # terminal 修正后的实际决策原因

    @property
    def complete(self) -> bool:
        return self.terminal_status is not None

    @property
    def effective_action_type(self) -> str:
        """实际执行类型: terminal 修正优先, 否则 started 意图。"""
        return self.final_action_type or self.action_type

    @property
    def effective_decision_reason(self) -> str:
        return self.final_decision_reason or self.decision_reason


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActionLedger:
    """tier2_actions.jsonl 的单写者账本。小文件, 整体重放。"""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入 (单写者, 逐行 append + fsync)
    # ------------------------------------------------------------------

    def _append(self, rec: dict) -> None:
        rec = {"schema_version": LEDGER_SCHEMA_VERSION, **rec}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def start(self, action_seq: int, run_tag: str, requested_action_type: str,
              action_type: str, decision_reason: str,
              target_refine_family: str | None = None,
              selected_mechanism_ids: list[str] | None = None,
              parent_operator: str | None = None,
              parent_mae_at_proposal: float | None = None,
              action_id: str | None = None) -> str:
        """写 started 事件, 返回 action_id (默认 f"{run_tag}#a{seq}")。"""
        aid = action_id or f"{run_tag}#a{action_seq}"
        self._append({"event": "started", "action_id": aid, "action_seq": action_seq,
                      "run_tag": run_tag, "requested_action_type": requested_action_type,
                      "action_type": action_type, "decision_reason": decision_reason,
                      "target_refine_family": target_refine_family,
                      "selected_mechanism_ids": list(selected_mechanism_ids or []),
                      "parent_operator": parent_operator,
                      "parent_mae_at_proposal": parent_mae_at_proposal,
                      "started_at": _now_iso()})
        return aid

    def finish(self, action_id: str, action_seq: int, terminal_status: str,
               failure_stage: str | None = None, candidate_count: int = 0,
               used_insight_ids: list | None = None,
               final_action_type: str | None = None,
               final_decision_reason: str | None = None) -> None:
        """写 terminal 事件。final_* 用于修正实际执行类型/原因 (如检索自然命中 aux →
        final_action_type=aux_create, final_decision_reason=natural_aux)。"""
        self._append({"event": "terminal", "action_id": action_id, "action_seq": action_seq,
                      "terminal_status": terminal_status, "failure_stage": failure_stage,
                      "candidate_count": candidate_count,
                      "used_insight_ids": list(used_insight_ids or []),
                      "final_action_type": final_action_type,
                      "final_decision_reason": final_decision_reason,
                      "finished_at": _now_iso()})

    # ------------------------------------------------------------------
    # 重放
    # ------------------------------------------------------------------

    def _load_events(self) -> list[dict]:
        """读全部事件。容忍唯一截断尾行 (崩溃窗口); 中间行损坏 → fail closed。"""
        if not os.path.exists(self.path):
            return []
        events: list[dict] = []
        with open(self.path, encoding="utf-8") as f:
            lines = [l for l in (l.strip() for l in f) if l]
        for i, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except Exception:
                if i == len(lines) - 1:
                    break                      # 唯一截断尾行: 丢弃
                raise ActionLedgerError(f"账本中间行损坏 (第 {i + 1} 行): {self.path}")
        return events

    def replay(self) -> list[Tier2Action]:
        """事件 → 动作列表 (按 action_seq 升序)。started-only = incomplete。

        结构校验 (fail closed): 任何事件校验失败都抛 ActionLedgerError。
        (撕裂写容忍仅限不可解析的截断尾行 —— 见 _load_events; 能解析成 JSON 但结构
        非法的一律视为损坏, 不静默丢弃。)
        """
        by_id: dict[str, Tier2Action] = {}
        order: list[str] = []
        seen_started: set = set()
        finished: set = set()
        for ev in self._load_events():
            _validate_event(ev, seen_started, finished)
            aid = ev["action_id"]
            if ev.get("event") == "started":
                seen_started.add(aid)
                act = Tier2Action(
                    action_id=aid, action_seq=int(ev.get("action_seq", 0)),
                    run_tag=ev.get("run_tag", ""),
                    requested_action_type=ev.get("requested_action_type", ""),
                    action_type=ev.get("action_type", ""),
                    decision_reason=ev.get("decision_reason", ""),
                    target_refine_family=ev.get("target_refine_family"),
                    selected_mechanism_ids=list(ev.get("selected_mechanism_ids") or []),
                    parent_operator=ev.get("parent_operator"),
                    parent_mae_at_proposal=ev.get("parent_mae_at_proposal"),
                    started_at=ev.get("started_at", ""))
                by_id[aid] = act
                order.append(aid)
            elif ev.get("event") == "terminal" and aid in by_id:
                finished.add(aid)
                act = by_id[aid]
                act.terminal_status = ev.get("terminal_status")
                act.failure_stage = ev.get("failure_stage")
                act.candidate_count = int(ev.get("candidate_count", 0))
                act.used_insight_ids = list(ev.get("used_insight_ids") or [])
                act.finished_at = ev.get("finished_at", "")
                act.final_action_type = ev.get("final_action_type")
                act.final_decision_reason = ev.get("final_decision_reason")
        return sorted((by_id[a] for a in order), key=lambda a: a.action_seq)

    def next_seq(self) -> int:
        """下一个 action_seq = 所有 started 事件最大值 + 1 (崩溃后不重号)。"""
        seqs = [a.action_seq for a in self.replay()]
        return (max(seqs) + 1) if seqs else 0

    # ------------------------------------------------------------------
    # 结果观察事件 (F3 精炼熔断的"观察时刻"语义: 异步回填防时间穿越)
    # ------------------------------------------------------------------

    def outcome(self, action_id: str, action_seq: int, operator_name: str,
                seed_val_mae: float | None, outcome_kind: str,
                resolved_after_action_seq: int) -> None:
        """写 outcome 事件 (child 评测结果被消化/观察到的那一刻)。

        outcome_kind: resolved (有限 MAE) | operational_failure (NaN/模型错误/训练不可用)
        | neutral (基础设施中断/人工取消, censored 不罚族)。
        resolved_after_action_seq: 观察时已过最大 action_seq —— breaker 在第 N 个失败结果
        【被观察到时】打开, cooldown 只计此后的终态动作。
        """
        self._append({"event": "outcome", "action_id": action_id, "action_seq": action_seq,
                      "operator_name": operator_name, "seed_val_mae": seed_val_mae,
                      "outcome_kind": outcome_kind,
                      "resolved_after_action_seq": resolved_after_action_seq,
                      "resolved_at": _now_iso()})

    def outcomes_by_action(self) -> dict[str, list[dict]]:
        """outcome 事件按 action_id 分组 (重放用)。"""
        out: dict[str, list[dict]] = {}
        for ev in self._load_events():
            if ev.get("event") == "outcome" and ev.get("action_id"):
                out.setdefault(ev["action_id"], []).append(ev)
        return out

    # ------------------------------------------------------------------
    # aux 配额 (计数表见模块 docstring)
    # ------------------------------------------------------------------

    def non_aux_since_last_aux(self) -> int:
        """距上次 aux 尝试以来的 non-aux 终态动作数 (aux 实际启动过即重置)。"""
        n = 0
        for act in self.replay():
            if not act.complete:
                continue                       # incomplete: 不计数不重置
            st = act.terminal_status
            if st in ("interrupted",):
                continue                       # 捕获中断: 不计数不重置
            eff = act.effective_action_type
            if eff == "aux_create":
                if st in ("succeeded", "failed"):
                    n = 0                      # aux 实际启动过 (成败都算): 重置
                # aux unavailable (没真正启动): 不重置
            elif eff in ("operator_create", "operator_refine"):
                if st in ("succeeded", "failed"):
                    n += 1                     # 含 aux_unavailable_fallback 回落成的 non-aux
        return n

    def aux_due(self, force_every: int) -> bool:
        """force_every>0 且 non-aux 终态动作数 >= force_every → 该强制一次 aux。"""
        return force_every > 0 and self.non_aux_since_last_aux() >= force_every

    def aux_mechanism_use_counts(self) -> dict[str, int]:
        """aux 选卡历史: 机制名 → 实际启动的 aux 动作数 (按动作去重; 用于最少尝试轮转)。

        只统计 action_type=aux_create 的 started 事件 (实际启动), 不把普通检索履历里的
        机制提及算作 aux 尝试。
        """
        counts: dict[str, int] = {}
        for act in self.replay():
            if act.action_type != "aux_create":
                continue
            for name in act.selected_mechanism_ids:
                counts[name] = counts.get(name, 0) + 1
        return counts
