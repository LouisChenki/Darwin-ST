"""memory/reflect.py 的正确性回归测试 (确定性, 无 LLM/网络)。

验证经验蒸馏与检索的契约:
  - summarize_graveyard 聚合失败模式
  - rank_by_relevance 三因子排序 (相似/重要/新近)
  - build_experience_context 汇总完整上下文 + SOTA 差距判定
  - propose_insight_candidates 规则式候选洞察
"""

from __future__ import annotations

import pytest

from darwin_st.memory.store import MemoryStore, Trial
from darwin_st.memory.reflect import (
    build_experience_context,
    propose_insight_candidates,
    rank_by_relevance,
    summarize_graveyard,
)

TS = "2026-06-17T10:00:00"


def _trial(**kw) -> Trial:
    base = dict(
        run_tag="exp/test", dataset="PeMS04", genotype={"depth": 2},
        status="KEEP", created_at=TS, val_mae=20.0,
    )
    base.update(kw)
    return Trial(**base)


@pytest.fixture
def store():
    s = MemoryStore(":memory:")
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Graveyard 聚合
# ---------------------------------------------------------------------------


def test_summarize_graveyard_counts(store):
    store.record_trial(_trial(status="CRASH", fail_reason="nan", genotype={"depth": 9}))
    store.record_trial(_trial(status="CRASH", fail_reason="nan", genotype={"depth": 10}))
    store.record_trial(_trial(status="DISCARD", fail_reason="regression", genotype={"depth": 3}))
    grave = summarize_graveyard(store, "PeMS04")
    assert grave["nan"] == 2
    assert grave["regression"] == 1
    # 降序: nan 最常见排第一
    assert list(grave.keys())[0] == "nan"


def test_summarize_graveyard_excludes_kept(store):
    store.record_trial(_trial(status="KEEP", val_mae=18.0))
    assert summarize_graveyard(store, "PeMS04") == {}


# ---------------------------------------------------------------------------
# 三因子排序
# ---------------------------------------------------------------------------


def test_rank_by_relevance_prefers_similar(store):
    """行为描述子更相似的 KEEP 经验应排更前 (其它因子相近时)。"""
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gcn", "depth": 2},
                              val_mae=19.0, genotype={"depth": 2}))
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gat", "depth": 8},
                              val_mae=19.0, genotype={"depth": 8}))
    ranked = rank_by_relevance(store, {"graph_conv": "gcn", "depth": 2}, dataset="PeMS04", k=2)
    assert ranked[0]["behavior_descriptor"]["graph_conv"] == "gcn"


def test_rank_by_relevance_empty(store):
    assert rank_by_relevance(store, {"depth": 2}, dataset="PeMS08") == []


def test_rank_excludes_non_kept(store):
    store.record_trial(_trial(status="DISCARD", behavior_descriptor={"depth": 2}, val_mae=30.0))
    assert rank_by_relevance(store, {"depth": 2}, dataset="PeMS04") == []


# ---------------------------------------------------------------------------
# 经验上下文
# ---------------------------------------------------------------------------


def test_build_context_basic(store):
    store.record_trial(_trial(val_mae=18.5))
    store.record_trial(_trial(status="CRASH", fail_reason="oom", genotype={"depth": 99}))
    ctx = build_experience_context(store, "PeMS04")
    assert ctx["best_mae"] == 18.5
    assert ctx["stats"]["KEEP"] == 1
    assert ctx["stats"]["CRASH"] == 1
    assert "oom" in ctx["graveyard_patterns"]


def test_build_context_sota_gap_not_beaten(store):
    store.record_trial(_trial(val_mae=18.5))
    ctx = build_experience_context(store, "PeMS04", sota_mae=17.8)
    assert ctx["sota_mae"] == 17.8
    assert abs(ctx["sota_gap"] - 0.7) < 1e-6
    assert ctx["beats_sota"] is False


def test_build_context_sota_beaten(store):
    store.record_trial(_trial(val_mae=17.5))
    ctx = build_experience_context(store, "PeMS04", sota_mae=17.8)
    assert ctx["beats_sota"] is True
    assert ctx["sota_gap"] < 0


def test_build_context_includes_similar_successes(store):
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gcn"}, val_mae=18.0))
    ctx = build_experience_context(store, "PeMS04", behavior_descriptor={"graph_conv": "gcn"})
    assert "similar_successes" in ctx
    assert len(ctx["similar_successes"]) >= 1


# ---------------------------------------------------------------------------
# 候选洞察
# ---------------------------------------------------------------------------


def test_propose_insight_recurring_failure(store):
    for d in (9, 10, 11):
        store.record_trial(_trial(status="CRASH", fail_reason="nan", genotype={"depth": d}))
    cands = propose_insight_candidates(store, "PeMS04")
    assert any("nan" in c for c in cands)


def test_propose_insight_low_keep_rate(store):
    store.record_trial(_trial(status="KEEP", val_mae=20.0))
    for i in range(8):
        store.record_trial(_trial(status="DISCARD", genotype={"depth": i + 3}, val_mae=25.0))
    cands = propose_insight_candidates(store, "PeMS04")
    assert any("保留率" in c for c in cands)


def test_propose_insight_quiet_when_little_data(store):
    """数据太少时不强行产出洞察。"""
    store.record_trial(_trial(status="KEEP", val_mae=20.0))
    assert propose_insight_candidates(store, "PeMS04") == []
