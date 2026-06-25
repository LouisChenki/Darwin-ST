"""memory/store.py 的正确性回归测试 (纯 stdlib sqlite, 无网络/torch)。

验证记忆系统的核心契约:
  - 落库/读取往返一致, JSON 字段正确解析
  - signature 内容相同则相同 (键序无关), 不同则不同
  - query_graveyard 能硬阻断已知失败、放行未见过的
  - best_so_far 取 KEEP 中最优
  - nearest_experiments 按行为描述子距离排序
  - 谱系/统计/洞察
"""

from __future__ import annotations

import pytest

from darwin_st.memory.store import MemoryStore, Trial, compute_signature


TS = "2026-06-17T10:00:00"


def _trial(**kw) -> Trial:
    """构造一个带默认值的 Trial, 便于按需覆盖字段。"""
    base = dict(
        run_tag="exp/test", dataset="PeMS04", genotype={"s_ops": ["gcn"], "depth": 2},
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
# 落库 / 读取往返
# ---------------------------------------------------------------------------


def test_record_and_get_roundtrip(store):
    tid = store.record_trial(_trial(genotype={"s_ops": ["gat"], "depth": 3}, val_mae=18.5))
    got = store.get_trial(tid)
    assert got["dataset"] == "PeMS04"
    assert got["status"] == "KEEP"
    assert got["val_mae"] == 18.5
    # JSON 字段解析回 dict
    assert got["genotype"] == {"s_ops": ["gat"], "depth": 3}
    assert isinstance(got["hp"], dict)


def test_invalid_status_rejected():
    with pytest.raises(ValueError):
        _trial(status="BOGUS")


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


def test_signature_key_order_invariant():
    """键序不同但内容相同 → 签名相同。"""
    a = compute_signature({"depth": 2, "s_ops": ["gcn"]}, {"lr": 0.01})
    b = compute_signature({"s_ops": ["gcn"], "depth": 2}, {"lr": 0.01})
    assert a == b


def test_signature_differs_on_content():
    a = compute_signature({"depth": 2}, {})
    b = compute_signature({"depth": 3}, {})
    assert a != b


def test_signature_includes_hp():
    """超参不同 → 签名不同 (同架构不同超参算不同配置)。"""
    a = compute_signature({"depth": 2}, {"lr": 0.01})
    b = compute_signature({"depth": 2}, {"lr": 0.05})
    assert a != b


# ---------------------------------------------------------------------------
# Graveyard 硬阻断
# ---------------------------------------------------------------------------


def test_graveyard_blocks_known_failure(store):
    geno = {"s_ops": ["cheb"], "depth": 9}
    store.record_trial(_trial(genotype=geno, status="CRASH", val_mae=None, fail_reason="nan"))
    hit = store.query_graveyard(geno)
    assert hit is not None
    assert hit["fail_reason"] == "nan"


def test_graveyard_passes_unseen(store):
    store.record_trial(_trial(genotype={"depth": 2}, status="CRASH"))
    # 没见过的配置 → 放行 (None)
    assert store.query_graveyard({"depth": 99}) is None


def test_graveyard_ignores_kept(store):
    """成功(KEEP)的配置不应出现在 Graveyard。"""
    geno = {"depth": 4}
    store.record_trial(_trial(genotype=geno, status="KEEP", val_mae=19.0))
    assert store.query_graveyard(geno) is None


def test_seen_signature(store):
    geno = {"depth": 5}
    assert not store.seen_signature(geno)
    store.record_trial(_trial(genotype=geno, status="DISCARD", val_mae=25.0))
    assert store.seen_signature(geno)


# ---------------------------------------------------------------------------
# best_so_far
# ---------------------------------------------------------------------------


def test_best_so_far_picks_min_kept(store):
    store.record_trial(_trial(val_mae=20.0))
    store.record_trial(_trial(val_mae=18.2, genotype={"depth": 3}))
    store.record_trial(_trial(val_mae=19.0, genotype={"depth": 4}))
    # DISCARD 的更低值不应被选 (只看 KEEP)
    store.record_trial(_trial(val_mae=10.0, status="DISCARD", genotype={"depth": 5}))
    best = store.best_so_far("PeMS04")
    assert best["val_mae"] == 18.2


def test_best_so_far_empty(store):
    assert store.best_so_far("PeMS08") is None


def test_best_so_far_dataset_scoped(store):
    store.record_trial(_trial(dataset="PeMS04", val_mae=18.0))
    store.record_trial(_trial(dataset="PeMS08", val_mae=14.0, genotype={"depth": 3}))
    assert store.best_so_far("PeMS04")["val_mae"] == 18.0
    assert store.best_so_far("PeMS08")["val_mae"] == 14.0


# ---------------------------------------------------------------------------
# list_trials (排行榜导出用: 返回全部匹配, 非最优)
# ---------------------------------------------------------------------------


def test_list_trials_filters_and_parses(store):
    store.record_trial(_trial(dataset="PeMS04", status="KEEP", val_mae=18.0,
                              genotype={"depth": 2}))
    store.record_trial(_trial(dataset="PeMS04", status="DISCARD", val_mae=99.0,
                              genotype={"depth": 3}))
    store.record_trial(_trial(dataset="PeMS08", status="KEEP", val_mae=14.0,
                              genotype={"depth": 4}))
    # 全部 (无过滤)
    assert len(store.list_trials()) == 3
    # 按 dataset+status 过滤, 且 genotype/hp 已解析回 dict
    keep04 = store.list_trials(dataset="PeMS04", status="KEEP")
    assert len(keep04) == 1
    assert keep04[0]["genotype"] == {"depth": 2}
    assert isinstance(keep04[0]["hp"], dict)


def test_list_trials_run_tag_isolation(store):
    store.record_trial(_trial(run_tag="exp/now", val_mae=18.0))
    store.record_trial(_trial(run_tag="exp/old", val_mae=19.0, genotype={"depth": 3}))
    rows = store.list_trials(dataset="PeMS04", run_tag="exp/now")
    assert len(rows) == 1 and rows[0]["run_tag"] == "exp/now"


# ---------------------------------------------------------------------------
# 最近邻检索
# ---------------------------------------------------------------------------


def test_nearest_experiments_orders_by_distance(store):
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gcn", "depth": 2}, val_mae=20.0))
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gat", "depth": 8},
                              val_mae=19.0, genotype={"depth": 8}))
    store.record_trial(_trial(behavior_descriptor={"graph_conv": "gcn", "depth": 3},
                              val_mae=18.0, genotype={"depth": 3}))
    # 查询 gcn/depth=2 → 最近的应是 gcn/depth=2 自身, 其次 gcn/depth=3, gat/depth=8 最远
    near = store.nearest_experiments({"graph_conv": "gcn", "depth": 2}, dataset="PeMS04", k=3)
    assert near[0]["behavior_descriptor"]["graph_conv"] == "gcn"
    assert near[-1]["behavior_descriptor"]["graph_conv"] == "gat"


def test_nearest_only_kept_by_default(store):
    store.record_trial(_trial(behavior_descriptor={"depth": 2}, status="DISCARD", val_mae=30.0))
    near = store.nearest_experiments({"depth": 2}, dataset="PeMS04", k=5)
    assert len(near) == 0  # 默认只检索 KEEP


# ---------------------------------------------------------------------------
# 谱系 / 统计 / 洞察
# ---------------------------------------------------------------------------


def test_lineage_auto_from_parent(store):
    pid = store.record_trial(_trial(val_mae=20.0))
    cid = store.record_trial(_trial(val_mae=19.0, genotype={"depth": 3}, parent_id=pid))
    children = store.get_children(pid)
    assert len(children) == 1
    assert children[0]["id"] == cid


def test_lineage_explicit_with_mutation(store):
    pid = store.record_trial(_trial(val_mae=20.0))
    cid = store.record_trial(_trial(val_mae=19.0, genotype={"depth": 3}))
    store.record_lineage(pid, cid, mutation_op="depth", mutation_detail="2→3")
    children = store.get_children(pid)
    assert children[0]["mutation_op"] == "depth"


def test_get_stats(store):
    store.record_trial(_trial(status="KEEP", val_mae=20.0))
    store.record_trial(_trial(status="DISCARD", genotype={"depth": 3}))
    store.record_trial(_trial(status="CRASH", genotype={"depth": 9}))
    stats = store.get_stats("PeMS04")
    assert stats["KEEP"] == 1
    assert stats["DISCARD"] == 1
    assert stats["CRASH"] == 1
    assert stats["total"] == 3


def test_insights_roundtrip(store):
    store.add_insight("深层 GCN 需残差连接抗过平滑", created_at=TS, dataset="PeMS04",
                      condition="depth>6", evidence_ids=[1, 2], confidence=0.8)
    store.add_insight("通用: LayerNorm 优于 BatchNorm 处理时序", created_at=TS, confidence=0.6)
    got = store.get_insights("PeMS04")
    assert len(got) == 2  # 专属 + 通用
    assert got[0]["confidence"] == 0.8  # 按置信度降序
    assert got[0]["evidence_ids"] == [1, 2]


def test_persistence_to_disk(tmp_path):
    """落盘库重开后数据仍在。"""
    db = str(tmp_path / "mem.db")
    with MemoryStore(db) as m:
        m.record_trial(_trial(val_mae=18.0))
    with MemoryStore(db) as m:
        assert m.best_so_far("PeMS04")["val_mae"] == 18.0
