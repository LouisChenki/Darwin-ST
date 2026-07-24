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

存疑声称 (仅登记展示, 不进入主表、不参与 get_sota 判定, 见 CLAIMED_UNVERIFIED):
  SSL-STMFormer (AAAI'25) 声称 PeMS04 MAE 17.06; 但其文中 STD-MAE 基线数字与
  官方系统性不符、无独立复现、社区未采纳, 口径存疑, 不作对标锚点。
"""

from __future__ import annotations

__all__ = [
    "BASELINE_METRICS",
    "CLAIMED_UNVERIFIED",
    "PROTOCOL_NOTES",
    "get_baseline_metric",
    "get_claimed",
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
        "STGCN": {"mae": 19.57, "rmse": 31.38},  # 原论文来源待核 (数值取自 HimNet/STD-MAE 等对照表)
        "DCRNN": {"mae": 19.63, "rmse": 31.26},  # DCRNN, arXiv:1707.01926
        "GraphWaveNet": {"mae": 18.53, "rmse": 29.92},  # Graph WaveNet, arXiv:1906.00121
        "ASTGCN": {"mae": 22.93, "rmse": 35.22},  # 原论文来源待核 (数值取自 HimNet/STD-MAE 等对照表)
        "MTGNN": {"mae": 19.17, "rmse": 31.70},  # 原论文来源待核 (数值取自 HimNet/STD-MAE 等对照表)
        # 现代强基线 / 前沿 (2022–2024); 以下 6 条 MAE 经 2026-07 一手文献核实
        "STID": {"mae": 18.35, "rmse": 29.85},  # STID, CIKM'22, arXiv:2208.05233
        "PDFormer": {"mae": 18.36, "rmse": 30.03},  # PDFormer, AAAI'23, arXiv:2301.07945
        "STAEformer": {"mae": 18.22, "rmse": 30.18},  # STAEformer, CIKM'23, arXiv:2308.10425
        "HimNet": {"mae": 18.14, "rmse": 30.02},  # HimNet, KDD'24, arXiv:2405.10800
        "STGformer": {"mae": 17.89, "rmse": 30.21},  # STGformer, arXiv:2410.00385
        "STD-MAE": {"mae": 17.80, "rmse": 29.83},  # STD-MAE, IJCAI'24, arXiv:2312.00516 —— 当前公认 SOTA
    },
    "PeMS08": {
        "STGCN": {"mae": 16.08, "rmse": 25.39},  # 原论文来源待核
        "DCRNN": {"mae": 15.22, "rmse": 24.17},  # DCRNN, arXiv:1707.01926
        "GraphWaveNet": {"mae": 14.40, "rmse": 23.39},  # Graph WaveNet, arXiv:1906.00121
        "ASTGCN": {"mae": 18.61, "rmse": 28.16},  # 原论文来源待核
        "MTGNN": {"mae": 15.18, "rmse": 24.24},  # 原论文来源待核
        "STID": {"mae": 14.21, "rmse": 23.28},  # STID, CIKM'22, arXiv:2208.05233
        "PDFormer": {"mae": 13.58, "rmse": 23.41},  # PDFormer, AAAI'23, arXiv:2301.07945
        "STAEformer": {"mae": 13.46, "rmse": 23.25},  # STAEformer, CIKM'23, arXiv:2308.10425
        "HimNet": {"mae": 13.57, "rmse": 23.21},  # HimNet, KDD'24, arXiv:2405.10800
        "STD-MAE": {"mae": 13.44, "rmse": 22.99},  # STD-MAE, IJCAI'24, arXiv:2312.00516
        "STGformer": {"mae": 13.41, "rmse": 23.10},  # STGformer, arXiv:2410.00385 —— 当前 SOTA 集群最低
    },
    "METR-LA": {
        # 速度数据集; 此处为 @3/6/12 的平均 MAE
        "STGCN": {"mae": 3.17, "rmse": 6.49},  # 原论文来源待核
        "DCRNN": {"mae": 3.11, "rmse": 6.27},  # DCRNN, arXiv:1707.01926
        "GraphWaveNet": {"mae": 3.05, "rmse": 6.13},  # Graph WaveNet, arXiv:1906.00121
        "GMAN": {"mae": 3.08, "rmse": 6.41},  # 原论文来源待核
        "MTGNN": {"mae": 3.04, "rmse": 6.11},  # 原论文来源待核
        "STID": {"mae": 3.19, "rmse": 6.55},  # STID, CIKM'22, arXiv:2208.05233
        "STAEformer": {"mae": 2.93, "rmse": 6.00},  # STAEformer, CIKM'23, arXiv:2308.10425
        "HimNet": {"mae": 2.92, "rmse": 5.99},  # HimNet, KDD'24, arXiv:2405.10800 —— 当前 SOTA 集群最低
    },
    "PEMS-BAY": {
        "STGCN": {"mae": 1.48, "rmse": 3.32},  # 原论文来源待核
        "DCRNN": {"mae": 1.64, "rmse": 3.70},  # DCRNN, arXiv:1707.01926
        "GraphWaveNet": {"mae": 1.58, "rmse": 3.55},  # Graph WaveNet, arXiv:1906.00121
        "GMAN": {"mae": 1.59, "rmse": 3.60},  # 原论文来源待核
        "MTGNN": {"mae": 1.55, "rmse": 3.49},  # 原论文来源待核
        "STID": {"mae": 1.62, "rmse": 3.67},  # STID, CIKM'22, arXiv:2208.05233
        "STAEformer": {"mae": 1.50, "rmse": 3.45},  # STAEformer, CIKM'23, arXiv:2308.10425
        "HimNet": {"mae": 1.51, "rmse": 3.46},  # HimNet, KDD'24, arXiv:2405.10800
    },
}


# 存疑声称登记处 —— 仅展示用, 非对标锚点。
# 规则: 声称优于主表 SOTA 但口径/复现存疑的条目登记在此, 绝不并入 BASELINE_METRICS,
# 亦不得被 get_sota() / list_baselines() / leaderboard 导出消费。
CLAIMED_UNVERIFIED: dict[str, dict[str, dict[str, float | None]]] = {
    "PeMS04": {
        # SSL-STMFormer (AAAI'25) 声称 PeMS04 MAE 17.06, 优于公认 SOTA STD-MAE 17.80。
        # 存疑依据: 其文中 STD-MAE 基线数字与官方系统性不符; 无独立复现; 社区未采纳。
        "SSL-STMFormer": {"mae": 17.06},
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


def get_claimed(dataset_name: str) -> dict[str, dict[str, float | None]]:
    """取该数据集的「存疑声称」副本 {模型: 指标}。

    ⚠️ 仅展示用, 非对标锚点: 这些数字口径/复现存疑, 绝不可用于
    「是否超越 SOTA」的程序化判定 (get_sota / list_baselines 均不消费它们)。
    未知数据集返回 {}。
    """
    return {name: dict(entry) for name, entry in CLAIMED_UNVERIFIED.get(dataset_name, {}).items()}
