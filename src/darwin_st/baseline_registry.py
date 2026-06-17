"""基线指标库 (Baseline Metrics Registry) —— SOTA 锚点与「超越判定」依据。

存储四大时空交通数据集上各模型的**已发表** MAE/RMSE, 让演化系统无需重跑即可
瞬间定锚性能, 并据此程序化判定「是否超越 SOTA」(终极停止条件)。

⚠️ 数字可比性铁律 (见 docs/ANALYSIS.md):
  - 这些数字全部为 **masked 指标** (缺失值掩码), 与本项目 metrics.py 口径一致。
  - PeMS04/08: MAE **在 12 步上平均**。
  - METR-LA/PEMS-BAY: 文献多报 @3/6/12 步 (15/30/60 分钟); 此处登记其**平均值**便于横比。
  - 跨论文绝对值因预处理差异有 ±1(PeMS)/±0.05(speed) 抖动, 故应信**同表内相对排名**,
    并在演化中用**多 seed 平均 + 显著性**避免追逐噪声 (SOTA 提升已 <1%)。

旧版 baseline_registry 数字偏差很大 (如 PeMS04 DCRNN 写 24.1, 真实 ≈19.6;
UniST 等并非这些数据集的公认条目), 会导致 Agent 误判已超越 SOTA。本版以
2018–2024 公认结果重写, 并标注来源。

主要来源:
  STD-MAE arxiv.org/abs/2312.00516 · STAEformer arxiv.org/abs/2308.10425
  STID arxiv.org/abs/2208.05233 · HimNet arxiv.org/pdf/2405.10800
  STGformer arxiv.org/abs/2410.00385 · PDFormer arxiv.org/pdf/2301.07945
  Graph WaveNet arxiv.org/abs/1906.00121 · DCRNN arxiv.org/abs/1707.01926
"""

from __future__ import annotations

__all__ = [
    "BASELINE_METRICS",
    "PROTOCOL_NOTES",
    "get_baseline_metric",
    "list_baselines",
    "get_sota",
]

# 协议口径说明 (随数字一起消费, 防止口径错配)
PROTOCOL_NOTES: dict[str, str] = {
    "PeMS04": "masked metric; MAE averaged over 12 horizons; split 6/2/2",
    "PeMS08": "masked metric; MAE averaged over 12 horizons; split 6/2/2",
    "METR-LA": "masked metric; avg of horizons @3/6/12 (15/30/60min); split 7/1/2",
    "PEMS-BAY": "masked metric; avg of horizons @3/6/12 (15/30/60min); split 7/1/2",
}

# 各数据集 × 模型的已发表 MAE/RMSE。
# 数值取自上列来源的自洽表 (HimNet Tab.3 / STD-MAE Tab.2 / STAEformer 最自洽);
# 个别为跨表中位近似, 以 None 显式标注缺失而非臆造。
BASELINE_METRICS: dict[str, dict[str, dict[str, float | None]]] = {
    "PeMS04": {
        # 经典基线 (2018–2020)
        "STGCN": {"mae": 19.57, "rmse": 31.38},
        "DCRNN": {"mae": 19.63, "rmse": 31.26},
        "GraphWaveNet": {"mae": 18.53, "rmse": 29.92},
        "ASTGCN": {"mae": 22.93, "rmse": 35.22},
        "MTGNN": {"mae": 19.17, "rmse": 31.70},
        # 现代强基线 / 前沿 (2022–2024)
        "STID": {"mae": 18.35, "rmse": 29.85},
        "PDFormer": {"mae": 18.36, "rmse": 30.03},
        "STAEformer": {"mae": 18.22, "rmse": 30.18},
        "HimNet": {"mae": 18.14, "rmse": 30.02},
        "STGformer": {"mae": 17.89, "rmse": 30.21},
        "STD-MAE": {"mae": 17.80, "rmse": 29.83},  # 当前 SOTA 集群最低
    },
    "PeMS08": {
        "STGCN": {"mae": 16.08, "rmse": 25.39},
        "DCRNN": {"mae": 15.22, "rmse": 24.17},
        "GraphWaveNet": {"mae": 14.40, "rmse": 23.39},
        "ASTGCN": {"mae": 18.61, "rmse": 28.16},
        "MTGNN": {"mae": 15.18, "rmse": 24.24},
        "STID": {"mae": 14.21, "rmse": 23.28},
        "PDFormer": {"mae": 13.58, "rmse": 23.41},
        "STAEformer": {"mae": 13.46, "rmse": 23.25},
        "HimNet": {"mae": 13.57, "rmse": 23.21},
        "STD-MAE": {"mae": 13.44, "rmse": 22.99},
        "STGformer": {"mae": 13.41, "rmse": 23.10},  # 当前 SOTA 集群最低
    },
    "METR-LA": {
        # 速度数据集; 此处为 @3/6/12 的平均 MAE
        "STGCN": {"mae": 3.17, "rmse": 6.49},
        "DCRNN": {"mae": 3.11, "rmse": 6.27},
        "GraphWaveNet": {"mae": 3.05, "rmse": 6.13},
        "GMAN": {"mae": 3.08, "rmse": 6.41},
        "MTGNN": {"mae": 3.04, "rmse": 6.11},
        "STID": {"mae": 3.19, "rmse": 6.55},
        "STAEformer": {"mae": 2.93, "rmse": 6.00},
        "HimNet": {"mae": 2.92, "rmse": 5.99},  # 当前 SOTA 集群最低
    },
    "PEMS-BAY": {
        "STGCN": {"mae": 1.48, "rmse": 3.32},
        "DCRNN": {"mae": 1.64, "rmse": 3.70},
        "GraphWaveNet": {"mae": 1.58, "rmse": 3.55},
        "GMAN": {"mae": 1.59, "rmse": 3.60},
        "MTGNN": {"mae": 1.55, "rmse": 3.49},
        "STID": {"mae": 1.62, "rmse": 3.67},
        "STAEformer": {"mae": 1.50, "rmse": 3.45},
        "HimNet": {"mae": 1.51, "rmse": 3.46},
    },
}


def get_baseline_metric(
    dataset_name: str, baseline_name: str, metric: str = "mae"
) -> float | None:
    """取单项基准分。未登记返回 None (不臆造)。"""
    ds = BASELINE_METRICS.get(dataset_name)
    if ds is None:
        return None
    entry = ds.get(baseline_name)
    if entry is None:
        return None
    return entry.get(metric)


def list_baselines(dataset_name: str, metric: str = "mae") -> dict[str, float]:
    """列出某数据集所有有该指标的基线 {模型: 值}, 按值升序 (越小越强在前)。"""
    ds = BASELINE_METRICS.get(dataset_name, {})
    pairs = {name: e[metric] for name, e in ds.items() if e.get(metric) is not None}
    return dict(sorted(pairs.items(), key=lambda kv: kv[1]))


def get_sota(dataset_name: str, metric: str = "mae") -> tuple[str | None, float | None]:
    """取该数据集当前 SOTA (该指标最小者), 返回 (模型名, 值)。

    供演化系统程序化判定「是否超越 SOTA」: 当 best_val < get_sota()[1] 即达成终极目标。
    未知数据集返回 (None, None)。
    """
    ranked = list_baselines(dataset_name, metric)
    if not ranked:
        return None, None
    name = next(iter(ranked))
    return name, ranked[name]
