"""数据集协议分流 (Dataset Protocol Profiles) —— 评测可比性的制度保证。

时空预测有**两条互不兼容的数据集谱系**, 协议细节几乎处处不同。若用单一硬编码
协议处理所有数据集, 一半 benchmark 的数字会与论文静默不可比。本模块把这些差异
收敛为「按数据集 profile 分流」的一等设计:

  ┌──────────────┬───────┬─────────┬──────────────────┬────────────────┐
  │ 数据集        │ 节点  │  split  │ 报告口径          │ 图来源          │
  ├──────────────┼───────┼─────────┼──────────────────┼────────────────┤
  │ PeMS04       │ 307   │ 6/2/2   │ 12 步平均 MAE     │ distance.csv   │
  │ PeMS08       │ 170   │ 6/2/2   │ 12 步平均 MAE     │ distance.csv   │
  │ METR-LA      │ 207   │ 7/1/2   │ 逐 horizon @3/6/12│ adj_mx.pkl     │
  │ PEMS-BAY     │ 325   │ 7/1/2   │ 逐 horizon @3/6/12│ adj_mx.pkl     │
  └──────────────┴───────┴─────────┴──────────────────┴────────────────┘

新增数据集 = 往 PROFILES 加一条 + 配一个下载/邻接适配器, 不改核心逻辑。
对齐 BasicTS (github.com/GestaltCogTeam/BasicTS) 口径。

设计分工:
- 本模块 (numpy, 数据准备时, CPU): profile 分流 / 时序切分 / 无泄漏 scaler
- metrics.py (torch, 评测时): masked 指标

详见 docs/BLUEPRINT.md §2。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DatasetProfile",
    "PROFILES",
    "get_profile",
    "chronological_split",
    "ZScoreScaler",
]


# ---------------------------------------------------------------------------
# 数据集 profile 定义与注册表
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetProfile:
    """单个数据集的标准协议事实 (不可变)。"""

    name: str
    num_nodes: int
    num_channels: int           # 原始通道数 (PeMS=3: flow/occupancy/speed; speed 集=1)
    target_channel: int         # 预测目标通道 (恒为 0 = flow 或 speed)
    split_ratios: tuple[float, float, float]  # (train, val, test), 时序不打乱
    seq_len_in: int             # 历史输入步
    seq_len_out: int            # 预测步
    report_mode: str            # "average" (全 12 步平均) | "per_horizon" (逐 horizon)
    report_horizons: tuple[int, ...]  # 1-indexed 报告步; average 模式下为全部 1..T_out
    null_val: float             # masked 指标的缺失值 (统一 0.0; speed 集 0=缺失)
    graph_source: str           # "distance_csv"(PeMS .npz+csv) | "dcrnn_csv"(vel+wam csv) | "adj_pkl"
    freq_minutes: int = 5       # 采样频率

    def __post_init__(self) -> None:
        s = sum(self.split_ratios)
        if not np.isclose(s, 1.0):
            raise ValueError(f"{self.name}: split_ratios 必须和为 1.0, 实为 {s}")
        if self.report_mode not in ("average", "per_horizon"):
            raise ValueError(f"{self.name}: report_mode 非法: {self.report_mode}")
        if self.graph_source not in ("distance_csv", "dcrnn_csv", "adj_pkl"):
            raise ValueError(f"{self.name}: graph_source 非法: {self.graph_source}")
        if max(self.report_horizons) > self.seq_len_out:
            raise ValueError(f"{self.name}: report_horizons 超出 seq_len_out")


# 四大标准数据集的协议事实。新增数据集在此登记即可。
PROFILES: dict[str, DatasetProfile] = {
    "PeMS04": DatasetProfile(
        name="PeMS04", num_nodes=307, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    ),
    "PeMS08": DatasetProfile(
        name="PeMS08", num_nodes=170, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    ),
    "METR-LA": DatasetProfile(
        name="METR-LA", num_nodes=207, num_channels=1, target_channel=0,
        split_ratios=(0.7, 0.1, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="per_horizon", report_horizons=(3, 6, 12),
        null_val=0.0, graph_source="dcrnn_csv",
    ),
    "PEMS-BAY": DatasetProfile(
        name="PEMS-BAY", num_nodes=325, num_channels=1, target_channel=0,
        split_ratios=(0.7, 0.1, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="per_horizon", report_horizons=(3, 6, 12),
        null_val=0.0, graph_source="dcrnn_csv",
    ),
}


def get_profile(dataset_name: str) -> DatasetProfile:
    """按名取 profile (大小写/连字符容错)。未知数据集抛 KeyError 并提示可选项。"""
    if dataset_name in PROFILES:
        return PROFILES[dataset_name]
    # 容错: 归一化 key (去连字符/下划线、大写) 后匹配
    norm = dataset_name.upper().replace("_", "").replace("-", "")
    for key, prof in PROFILES.items():
        if key.upper().replace("_", "").replace("-", "") == norm:
            return prof
    raise KeyError(
        f"未知数据集 '{dataset_name}'. 可选: {sorted(PROFILES)}. "
        f"新增请在 protocol.PROFILES 登记一条 DatasetProfile。"
    )


# ---------------------------------------------------------------------------
# 时序切分 (Chronological Split) —— 绝不打乱, 防止时间泄漏
# ---------------------------------------------------------------------------


def chronological_split(
    num_samples: int, profile: DatasetProfile
) -> tuple[slice, slice, slice]:
    """按 profile 比例做**时序**三切分, 返回 (train, val, test) 的 slice。

    时序数据严禁随机打乱:train 取最早一段、test 取最晚一段, 模拟「用过去预测未来」。
    余数归入 test (最晚), 保证三段无重叠、无空隙、完整覆盖 [0, num_samples)。
    """
    if num_samples <= 0:
        raise ValueError(f"num_samples 必须为正, 实为 {num_samples}")

    train_ratio, val_ratio, _ = profile.split_ratios
    n_train = int(num_samples * train_ratio)
    n_val = int(num_samples * val_ratio)
    # test 吃掉余数 (而非 int(n*test_ratio)), 确保覆盖到末尾不丢样本
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, num_samples)
    return train_slice, val_slice, test_slice


# ---------------------------------------------------------------------------
# Z-Score 归一化 (无泄漏: 仅用训练集统计量)
# ---------------------------------------------------------------------------


@dataclass
class ZScoreScaler:
    """单一全局标量 z-score (对齐 BasicTS norm_each_channel=False)。

    **无泄漏铁律**: 统计量 (mean/std) 只能由训练集拟合, 再用于 val/test。
    只对目标通道 (target_channel, 通常是 flow/speed) 归一化, 其余通道
    (如 time-of-day 协变量) 不动。评测前须 inverse_transform 回真实尺度。
    """

    mean: float = field(default=0.0)
    std: float = field(default=1.0)
    fitted: bool = field(default=False)

    @classmethod
    def fit(cls, train_values: np.ndarray) -> "ZScoreScaler":
        """仅用训练集目标通道的值拟合。train_values 应为目标通道切片 (任意形状)。"""
        mean = float(np.mean(train_values))
        std = float(np.std(train_values))
        if std == 0.0:
            std = 1.0  # 退化保护: 常数序列不缩放
        return cls(mean=mean, std=std, fitted=True)

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ZScoreScaler 未拟合, 先调用 fit()")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ZScoreScaler 未拟合, 先调用 fit()")
        return x * self.std + self.mean
