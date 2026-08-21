"""Tier-2 动作账本 (action_ledger) 测试。

契约锚点 (v2 方案):
  - 两事件生命周期: started+terminal=完整; started-only=incomplete (不参与计数);
  - action_seq 从所有 started 最大值续排 (崩溃后不重号);
  - 加载容忍唯一截断尾行, 中间行损坏 fail closed (ActionLedgerError);
  - aux 配额计数表: non-aux 终态 +1 / aux 启动过即重置 / unavailable 回落按 non-aux 计 /
    interrupted 与 unmatched started 不计数不重置。
"""

from __future__ import annotations

import json
import os

import pytest

from darwin_st.creation.action_ledger import ActionLedger, ActionLedgerError


def _mk(tmp_path) -> ActionLedger:
    return ActionLedger(str(tmp_path / "tier2_actions.jsonl"))


def _action(led, seq, action_type="operator_create", requested=None, reason="refine_first",
            status="failed", final_type=None, final_reason=None, mechs=None):
    """写一个完整动作 (started+terminal), 返回 action_id。"""
    aid = led.start(seq, "exp/test", requested_action_type=requested or action_type,
                    action_type=action_type, decision_reason=reason,
                    selected_mechanism_ids=mechs or [])
    led.finish(aid, seq, status, final_action_type=final_type,
               final_decision_reason=final_reason)
    return aid


# ---------------------------------------------------------------------------
# 生命周期与重放
# ---------------------------------------------------------------------------

def test_started_terminal_roundtrip(tmp_path):
    led = _mk(tmp_path)
    aid = led.start(0, "exp/t", requested_action_type="operator_refine",
                    action_type="operator_refine", decision_reason="refine_first",
                    target_refine_family="famA", parent_operator="synth_a_v2",
                    parent_mae_at_proposal=18.5)
    led.finish(aid, 0, "succeeded", candidate_count=1, used_insight_ids=[3, 5])
    acts = led.replay()
    assert len(acts) == 1 and acts[0].complete
    a = acts[0]
    assert a.action_seq == 0 and a.target_refine_family == "famA"
    assert a.parent_mae_at_proposal == 18.5 and a.used_insight_ids == [3, 5]
    assert a.effective_action_type == "operator_refine"


def test_unmatched_started_is_incomplete_and_seq_continues(tmp_path):
    """崩溃留 started: incomplete 不计数; next_seq 从其序号续排 (不重号)。"""
    led = _mk(tmp_path)
    _action(led, 0)
    led.start(1, "exp/t", requested_action_type="aux_create",
              action_type="aux_create", decision_reason="forced_aux_quota")
    acts = led.replay()
    assert len(acts) == 2 and not acts[1].complete
    assert led.next_seq() == 2                        # 含 unmatched started 的最大值续排


def test_truncated_tail_tolerated_middle_corrupt_fails(tmp_path):
    led = _mk(tmp_path)
    _action(led, 0)
    with open(led.path, "a") as f:                    # 崩溃截断的尾行
        f.write('{"event": "started", "action_id": "x", "action_seq": 1')
    acts = led.replay()
    assert len(acts) == 1                             # 截断尾行被丢弃
    with open(led.path, "w") as f:                    # 中间行损坏 → fail closed
        f.write('{"event":"started","action_id":"a","action_seq":0}\n')
        f.write("这不是json\n")
        f.write('{"event":"started","action_id":"b","action_seq":1}\n')
    with pytest.raises(ActionLedgerError):
        led.replay()


# ---------------------------------------------------------------------------
# aux 配额计数表
# ---------------------------------------------------------------------------

def test_quota_non_aux_counts_and_aux_resets(tmp_path):
    led = _mk(tmp_path)
    for i in range(3):
        _action(led, i, "operator_refine", status="failed")
    assert led.non_aux_since_last_aux() == 3
    _action(led, 3, "aux_create", reason="forced_aux_quota", status="failed")  # 败也重置
    assert led.non_aux_since_last_aux() == 0
    _action(led, 4, "operator_create", status="succeeded")
    assert led.non_aux_since_last_aux() == 1


def test_quota_unavailable_fallback_counts_non_aux_no_reset(tmp_path):
    """aux preflight 无卡回落: 按最终执行的 non-aux 计数, 不重置。"""
    led = _mk(tmp_path)
    _action(led, 0, "operator_refine", status="succeeded")
    _action(led, 1, "operator_refine", requested="aux_create",
            reason="aux_unavailable_fallback", status="failed")
    assert led.non_aux_since_last_aux() == 2          # 计 non-aux, 不当 aux 重置


def test_quota_interrupted_and_incomplete_not_counted(tmp_path):
    led = _mk(tmp_path)
    _action(led, 0, "operator_create", status="succeeded")
    led.start(1, "exp/t", requested_action_type="aux_create",     # unmatched started
              action_type="aux_create", decision_reason="forced_aux_quota")
    aid = led.start(2, "exp/t", requested_action_type="operator_refine",
                    action_type="operator_refine", decision_reason="refine_first")
    led.finish(aid, 2, "interrupted", failure_stage="interrupt")  # 捕获中断
    assert led.non_aux_since_last_aux() == 1          # 两者都不计数不重置


def test_quota_natural_aux_resets(tmp_path):
    """create 自然路由进 aux (final_action_type 修正) → 重置。"""
    led = _mk(tmp_path)
    _action(led, 0, "operator_create", status="succeeded")
    _action(led, 1, "operator_create", reason="create_fallback", status="succeeded",
            final_type="aux_create", final_reason="natural_aux")
    acts = led.replay()
    assert acts[1].effective_action_type == "aux_create"
    assert acts[1].effective_decision_reason == "natural_aux"
    assert led.non_aux_since_last_aux() == 0


def test_aux_due_threshold_and_disabled(tmp_path):
    led = _mk(tmp_path)
    assert not led.aux_due(0)                         # 默认 0 = 关闭
    for i in range(5):
        assert not led.aux_due(5)
        _action(led, i, "operator_create", status="failed")
    assert led.aux_due(5)


def test_aux_mechanism_use_counts_only_started_aux(tmp_path):
    """选卡统计只数实际启动的 aux 动作 (按动作去重); non-aux 提及不算。"""
    led = _mk(tmp_path)
    _action(led, 0, "aux_create", status="failed",
            mechs=["masked_autoencoding", "contrastive_learning"])
    _action(led, 1, "operator_create", status="succeeded",
            mechs=["masked_autoencoding"])            # 非 aux 动作: 不计
    led.start(2, "exp/t", requested_action_type="aux_create",   # 未启动完不计? started 即算
              action_type="aux_create", decision_reason="forced_aux_quota",
              selected_mechanism_ids=["contrastive_learning"])
    counts = led.aux_mechanism_use_counts()
    assert counts == {"masked_autoencoding": 1, "contrastive_learning": 2}


# ---------------------------------------------------------------------------
# 最小结构校验 (fail closed; 尾事件撕裂容忍)
# ---------------------------------------------------------------------------

def test_validate_duplicate_started_fails(tmp_path):
    led = _mk(tmp_path)
    aid = led.start(0, "exp/t", requested_action_type="operator_create",
                    action_type="operator_create", decision_reason="create_fallback")
    led.start(0, "exp/t", requested_action_type="operator_create",
              action_type="operator_create", decision_reason="create_fallback")
    led.finish(aid, 0, "failed")
    led.finish(aid, 0, "failed")
    with pytest.raises(ActionLedgerError, match="重复 started"):
        led.replay()


def test_validate_orphan_and_duplicate_terminal_fails(tmp_path):
    led = _mk(tmp_path)
    led.finish("ghost", 0, "failed")                      # 孤立 terminal
    with pytest.raises(ActionLedgerError, match="孤立 terminal"):
        led.replay()
    led2 = ActionLedger(str(tmp_path / "b.jsonl"))
    aid = led2.start(0, "exp/t", requested_action_type="operator_create",
                     action_type="operator_create", decision_reason="create_fallback")
    led2.finish(aid, 0, "failed")
    led2.finish(aid, 0, "failed")                         # 重复 terminal
    with pytest.raises(ActionLedgerError, match="重复 terminal"):
        led2.replay()


def test_validate_bad_enums_and_missing_fields(tmp_path):
    led = _mk(tmp_path)
    with open(led.path, "w") as f:
        f.write('{"event":"started","action_id":"a","action_seq":0,'
                '"action_type":"hack_create"}\n')
    with pytest.raises(ActionLedgerError, match="非法 action_type"):
        led.replay()
    with open(led.path, "w") as f:
        f.write('{"event":"started","action_seq":0}\n')    # 缺 action_id
    with pytest.raises(ActionLedgerError, match="action_id"):
        led.replay()
    with open(led.path, "w") as f:
        f.write('{"event":"flying","action_id":"a","action_seq":0}\n')
    with pytest.raises(ActionLedgerError, match="非法事件类型"):
        led.replay()


def test_torn_last_event_dropped_middle_fails(tmp_path):
    """截断尾行 (不可解析) 丢弃; 能解析但结构非法的一律 fail closed (含尾行)。"""
    led = _mk(tmp_path)
    _action(led, 0)
    with open(led.path, "a") as f:                        # 尾行: JSON 合法但缺字段
        f.write('{"event":"started","action_id":"x"}\n')
    with pytest.raises(ActionLedgerError):                # 结构非法, 不静默丢弃
        led.replay()
    with open(led.path, "a") as f:                        # 不可解析截断 → 容忍
        f.write('{"event": "started", "action_id": "x", "action_seq": 1')
    assert len(led._load_events()) == 3                   # 截断尾行被丢弃 (3 行合法保留)


def test_schema_version_written_and_checked(tmp_path):
    led = _mk(tmp_path)
    _action(led, 0)
    first = json.loads(open(led.path).readline())
    assert first["schema_version"] == 1
    with open(led.path, "w") as f:
        f.write('{"schema_version":99,"event":"started","action_id":"a","action_seq":0}\n')
    with pytest.raises(ActionLedgerError, match="schema_version"):
        led.replay()


def test_truncated_tail_repaired_on_reopen_and_continue(tmp_path):
    """掉电撕裂尾行: 重建账本 → 安全截断 → 继续写 → 完整 replay, next_seq 不重号。"""
    led = _mk(tmp_path)
    _action(led, 0, status="succeeded")
    with open(led.path, "a") as f:                        # 模拟掉电: 半行 started
        f.write('{"schema_version":1,"event":"started","action_id":"exp/t#a1","acti')
    led2 = ActionLedger(led.path)                         # 重建 (触发尾行修复)
    assert len(led2.replay()) == 1                        # 半行被截掉, 旧动作完整
    aid = led2.start(1, "exp/t", requested_action_type="operator_create",
                     action_type="operator_create", decision_reason="create_fallback")
    led2.finish(aid, 1, "failed")
    acts = led2.replay()                                  # 残缺行没有变成中间坏行
    assert len(acts) == 2 and all(a.complete for a in acts)
    assert led2.next_seq() == 2
    with open(led.path, "rb") as f:                       # 文件以整行结尾
        f.seek(-1, 2)
        assert f.read() == b"\n"
