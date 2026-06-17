"""baseline_registry.py 的正确性回归测试 (纯数据校验, 无 torch/网络)。

验证: 旧版臆造数字已被剔除、登记值落在合理区间、get_sota 取最小、排名升序、
协议标注齐全——确保「超越 SOTA」判定建立在可信锚点上。
"""

from __future__ import annotations

import pytest

from darwin_st.baseline_registry import (
    BASELINE_METRICS,
    PROTOCOL_NOTES,
    get_baseline_metric,
    get_sota,
    list_baselines,
)


DATASETS = ["PeMS04", "PeMS08", "METR-LA", "PEMS-BAY"]


def test_all_four_datasets_present():
    assert set(BASELINE_METRICS) == set(DATASETS)


def test_each_dataset_has_protocol_note():
    """每个数据集都必须有口径标注 (masked / split / 报告方式)。"""
    for ds in DATASETS:
        assert ds in PROTOCOL_NOTES
        assert "masked" in PROTOCOL_NOTES[ds]


def test_old_bogus_numbers_removed():
    """回归: 旧版偏差数字已被剔除。"""
    # 旧版 PeMS04 DCRNN=24.1 (真实≈19.6); 现值应明显更低且接近文献
    assert get_baseline_metric("PeMS04", "DCRNN", "mae") < 21.0
    # 旧版用 UniST 作条目, 非这些数据集公认基线, 应已移除
    for ds in DATASETS:
        assert "UniST" not in BASELINE_METRICS[ds]


@pytest.mark.parametrize(
    "ds,lo,hi",
    [
        ("PeMS04", 17.0, 24.0),   # 文献区间 ~17.8(SOTA) – 22.9(ASTGCN)
        ("PeMS08", 13.0, 19.0),   # ~13.4(SOTA) – 18.6(ASTGCN)
        ("METR-LA", 2.8, 3.3),    # ~2.9(SOTA) – 3.19(STID)
        ("PEMS-BAY", 1.4, 1.7),   # ~1.48 – 1.64
    ],
)
def test_mae_values_in_sane_range(ds, lo, hi):
    """所有登记 MAE 落在该数据集的文献合理区间 (防再次写错量级)。"""
    for name, entry in BASELINE_METRICS[ds].items():
        mae = entry.get("mae")
        if mae is not None:
            assert lo <= mae <= hi, f"{ds}/{name} MAE={mae} 超出合理区间 [{lo},{hi}]"


def test_rmse_greater_than_mae():
    """RMSE 恒 ≥ MAE (数学性质, 顺带抓录入错误)。"""
    for ds in DATASETS:
        for name, entry in BASELINE_METRICS[ds].items():
            mae, rmse = entry.get("mae"), entry.get("rmse")
            if mae is not None and rmse is not None:
                assert rmse >= mae, f"{ds}/{name}: RMSE {rmse} < MAE {mae}"


def test_get_sota_returns_minimum():
    """get_sota 必须返回该指标最小 (最强) 的模型。"""
    for ds in DATASETS:
        name, val = get_sota(ds, "mae")
        all_vals = [e["mae"] for e in BASELINE_METRICS[ds].values() if e.get("mae") is not None]
        assert val == min(all_vals)
        assert BASELINE_METRICS[ds][name]["mae"] == val


def test_known_sota_holders():
    """当前 SOTA 持有者与研究结论一致 (锚点正确性)。"""
    assert get_sota("PeMS04")[0] == "STD-MAE"
    assert get_sota("PeMS08")[0] == "STGformer"
    assert get_sota("METR-LA")[0] == "HimNet"


def test_list_baselines_ascending():
    """list_baselines 应按值升序 (最强在前)。"""
    ranked = list_baselines("PeMS04", "mae")
    vals = list(ranked.values())
    assert vals == sorted(vals)


def test_get_baseline_metric_unknown_returns_none():
    assert get_baseline_metric("NoSuchDS", "DCRNN") is None
    assert get_baseline_metric("PeMS04", "NoSuchModel") is None
    assert get_baseline_metric("PeMS04", "DCRNN", "no_such_metric") is None


def test_get_sota_unknown_dataset():
    assert get_sota("NoSuchDS") == (None, None)


def test_sota_beats_classic_baselines():
    """SOTA 必须强于经典基线 (如 ASTGCN), 否则锚点逻辑有误。"""
    for ds in ("PeMS04", "PeMS08"):
        _, sota_mae = get_sota(ds)
        astgcn = get_baseline_metric(ds, "ASTGCN")
        assert sota_mae < astgcn
