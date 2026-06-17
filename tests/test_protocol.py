"""protocol.py 的正确性回归测试 (设备无关, 本地 CPU 即可全绿)。

重点验证三件**评测可比性**的命脉:
  1. 协议 profile 事实与论文/BasicTS 口径一致 (split、节点数、报告口径)
  2. 时序切分不打乱、不重叠、不丢样本 (防时间泄漏)
  3. scaler 仅由训练集拟合 (防数据泄漏)
"""

from __future__ import annotations

import numpy as np
import pytest

from darwin_st.data.protocol import (
    PROFILES,
    DatasetProfile,
    ZScoreScaler,
    chronological_split,
    get_profile,
)


# ---------------------------------------------------------------------------
# Profile 事实正确性
# ---------------------------------------------------------------------------


def test_all_four_datasets_registered():
    assert set(PROFILES) == {"PeMS04", "PeMS08", "METR-LA", "PEMS-BAY"}


@pytest.mark.parametrize(
    "name,nodes,split,mode",
    [
        ("PeMS04", 307, (0.6, 0.2, 0.2), "average"),
        ("PeMS08", 170, (0.6, 0.2, 0.2), "average"),
        ("METR-LA", 207, (0.7, 0.1, 0.2), "per_horizon"),
        ("PEMS-BAY", 325, (0.7, 0.1, 0.2), "per_horizon"),
    ],
)
def test_profile_facts(name, nodes, split, mode):
    """协议事实与文献/BasicTS 一致——这是数字可比的前提。"""
    p = get_profile(name)
    assert p.num_nodes == nodes
    assert p.split_ratios == split
    assert p.report_mode == mode


def test_pems_lineage_uses_distance_csv():
    for name in ("PeMS04", "PeMS08"):
        assert get_profile(name).graph_source == "distance_csv"
        assert get_profile(name).num_channels == 3  # flow/occupancy/speed


def test_speed_lineage_uses_dcrnn_csv_and_per_horizon():
    for name in ("METR-LA", "PEMS-BAY"):
        p = get_profile(name)
        assert p.graph_source == "dcrnn_csv"  # hazdzz 镜像: vel+wam csv
        assert p.num_channels == 1
        assert p.report_horizons == (3, 6, 12)  # DCRNN 协议 15/30/60 分钟


def test_split_ratios_sum_to_one():
    for p in PROFILES.values():
        assert np.isclose(sum(p.split_ratios), 1.0)


def test_get_profile_name_tolerance():
    """大小写/连字符容错。"""
    assert get_profile("pems04").name == "PeMS04"
    assert get_profile("metr_la").name == "METR-LA"
    assert get_profile("METRLA").name == "METR-LA"
    assert get_profile("pemsbay").name == "PEMS-BAY"


def test_get_profile_unknown_raises():
    with pytest.raises(KeyError):
        get_profile("NoSuchDataset")


def test_invalid_profile_construction_rejected():
    """split 不和为 1 应被拒 (防手滑登记错数据集)。"""
    with pytest.raises(ValueError):
        DatasetProfile(
            name="bad", num_nodes=10, num_channels=1, target_channel=0,
            split_ratios=(0.5, 0.2, 0.2),  # = 0.9, 非法
            seq_len_in=12, seq_len_out=12, report_mode="average",
            report_horizons=(12,), null_val=0.0, graph_source="adj_pkl",
        )


# ---------------------------------------------------------------------------
# 时序切分: 不打乱 / 不重叠 / 不丢样本
# ---------------------------------------------------------------------------


def test_split_is_contiguous_and_complete():
    """三段必须连续、无重叠、完整覆盖 [0, N)。"""
    p = get_profile("PeMS04")
    n = 1000
    tr, va, te = chronological_split(n, p)
    # 连续衔接
    assert tr.start == 0
    assert tr.stop == va.start
    assert va.stop == te.start
    assert te.stop == n
    # 无重叠 + 完整覆盖
    covered = list(range(tr.start, tr.stop)) + list(range(va.start, va.stop)) + list(range(te.start, te.stop))
    assert covered == list(range(n))


def test_split_ratio_proportions():
    """6/2/2 比例落在预期 (余数归 test)。"""
    p = get_profile("PeMS04")
    n = 1000
    tr, va, te = chronological_split(n, p)
    assert (tr.stop - tr.start) == 600
    assert (va.stop - va.start) == 200
    assert (te.stop - te.start) == 200


def test_split_remainder_goes_to_test():
    """非整除时余数进 test, 不丢样本。"""
    p = get_profile("METR-LA")  # 7/1/2
    n = 1003
    tr, va, te = chronological_split(n, p)
    assert (tr.stop - tr.start) == 702   # int(1003*0.7)=702
    assert (va.stop - va.start) == 100   # int(1003*0.1)=100
    assert (te.stop - te.start) == 201   # 余数 1003-702-100
    assert te.stop == n


def test_split_train_is_earliest():
    """train 必须是最早的时间段 (时序因果)。"""
    p = get_profile("PeMS08")
    tr, va, te = chronological_split(500, p)
    assert tr.start == 0
    assert tr.stop <= va.start <= te.start  # 时间单调推进


# ---------------------------------------------------------------------------
# Scaler: 无泄漏
# ---------------------------------------------------------------------------


def test_scaler_fit_uses_only_train_stats():
    """scaler 的 mean/std 必须只反映训练数据, 与 val/test 无关。"""
    train = np.array([0.0, 10.0, 20.0, 30.0])  # mean=15, std=sqrt(125)
    scaler = ZScoreScaler.fit(train)
    assert np.isclose(scaler.mean, 15.0)
    assert np.isclose(scaler.std, np.std(train))

    # 即使后续遇到分布迥异的 test 数据, scaler 统计量不变
    test = np.array([1000.0, 2000.0])
    transformed = scaler.transform(test)
    # 用训练统计量变换, 而非 test 自身统计量
    assert np.allclose(transformed, (test - 15.0) / np.std(train))


def test_scaler_inverse_roundtrip():
    """transform → inverse_transform 应还原原值 (评测前反归一化的正确性)。"""
    scaler = ZScoreScaler.fit(np.array([5.0, 15.0, 25.0]))
    x = np.array([7.0, 13.0, 40.0, -2.0])
    assert np.allclose(scaler.inverse_transform(scaler.transform(x)), x)


def test_scaler_constant_series_no_div_zero():
    """常数序列 std=0 应退化为不缩放, 不崩。"""
    scaler = ZScoreScaler.fit(np.array([7.0, 7.0, 7.0]))
    assert scaler.std == 1.0
    out = scaler.transform(np.array([7.0, 8.0]))
    assert np.all(np.isfinite(out))


def test_scaler_requires_fit_before_use():
    s = ZScoreScaler()
    with pytest.raises(RuntimeError):
        s.transform(np.array([1.0]))
    with pytest.raises(RuntimeError):
        s.inverse_transform(np.array([1.0]))
