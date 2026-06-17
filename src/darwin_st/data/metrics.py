"""masked 评测指标 (Masked Evaluation Metrics) —— 演化系统的唯一标尺。

时空交通数据中,缺失/故障传感器读数被编码为 0(METR-LA/PEMS-BAY 的 speed 尤其如此)。
若把这些 0 当真值算进误差,会注入虚假误差并使 MAPE 除零崩溃。学术界标准做法是
**masked metric**:只在 `label != null_val` 的有效位置计误差。

本实现严格对齐社区规范实现(Graph WaveNet `util.py`、DCRNN、BasicTS),核心是
`mask /= mask.mean()` 重加权技巧:

    mask = (labels != null_val)              # 有效位置为 1
    mask = mask / mask.mean()                # 每个有效权重变成 1/p, p=有效占比
    loss = |preds - labels| * mask
    return loss.mean()                       # 对【全部】元素求均值, 数学上等于【仅有效位】的均值

载重的一行是 `mask /= mask.mean()`:因 `mask.mean() = #有效/#总数`,对整个张量做
普通 `mean` 恰好还原为「仅对有效位求均值」。这样可微、形状稳定、且与论文口径一致。

⚠️ 重要:传入的 preds/labels 必须已 **inverse-transform 回真实交通流尺度**(非归一化值),
否则得到的是归一化尺度下的假指标。反归一化由调用方(评测循环)负责。

参考:
- Graph WaveNet util.py  https://github.com/nnzhan/Graph-WaveNet/blob/master/util.py
- BasicTS               https://github.com/GestaltCogTeam/BasicTS
"""

from __future__ import annotations

import torch

__all__ = ["masked_mae", "masked_rmse", "masked_mape", "compute_all_metrics"]


def _build_mask(labels: torch.Tensor, null_val: float) -> torch.Tensor:
    """构造归一化后的掩码张量 (与 labels 同形)。

    返回的 mask 已做 `mask /= mask.mean()` 重加权:有效位为 1/p、无效位为 0。
    对加权后的逐元素误差求**整体均值**即得「仅有效位的均值」。

    null_val 为 NaN 时按 isnan 判定缺失;否则按数值相等(带浮点容差)判定。
    """
    if null_val != null_val:  # null_val is NaN
        mask = ~torch.isnan(labels)
    else:
        # 浮点容差比较,避免 0.0 因尺度变换产生的微小偏差被误判为有效
        mask = ~torch.isclose(labels, torch.tensor(null_val, dtype=labels.dtype, device=labels.device))

    mask = mask.float()
    denom = mask.mean()
    # 全部缺失的极端情形:denom=0 会产生 nan/inf,用 nan_to_num 兜底为 0 权重
    mask = mask / denom
    mask = torch.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
    return mask


def masked_mae(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """masked 平均绝对误差 (Mean Absolute Error)。preds/labels 须为真实尺度。"""
    mask = _build_mask(labels, null_val)
    loss = torch.abs(preds - labels) * mask
    loss = torch.nan_to_num(loss, nan=0.0)  # 防 preds 中的 nan 污染(应在评测层提前熔断)
    return loss.mean()


def masked_rmse(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """masked 均方根误差 (Root Mean Squared Error) = sqrt(masked MSE)。"""
    mask = _build_mask(labels, null_val)
    loss = (preds - labels) ** 2 * mask
    loss = torch.nan_to_num(loss, nan=0.0)
    return torch.sqrt(loss.mean())


def masked_mape(preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    """masked 平均绝对百分比误差 (Mean Absolute Percentage Error), 返回比例值(非百分数)。

    在 masked 基础上再施加一层除零保护:即使某有效位 label 极小, 也按其自身掩码处理。
    与 BasicTS 口径一致——分母用 labels 本身, 缺失位权重为 0。
    """
    mask = _build_mask(labels, null_val)
    loss = torch.abs((preds - labels) / labels) * mask
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)  # 二次防除零
    return loss.mean()


def compute_all_metrics(
    preds: torch.Tensor, labels: torch.Tensor, null_val: float = 0.0
) -> dict[str, float]:
    """一次性计算三指标, 返回 Python float 字典 (便于落库/打印)。"""
    return {
        "mae": float(masked_mae(preds, labels, null_val).item()),
        "rmse": float(masked_rmse(preds, labels, null_val).item()),
        "mape": float(masked_mape(preds, labels, null_val).item()),
    }
