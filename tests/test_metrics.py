"""metrics.py 的正确性回归测试 (设备无关, 本地 CPU 即可全绿)。

验证 masked MAE/RMSE/MAPE 与社区规范口径一致, 重点验证 `mask/=mask.mean()`
重加权确实等价于「仅对有效位求均值」, 且 0(缺失)被正确忽略。
"""

from __future__ import annotations

import math

import pytest
import torch

from darwin_st.data.metrics import (
    masked_huber,
    masked_mae,
    masked_mape,
    masked_rmse,
    compute_all_metrics,
)


def test_all_valid_equals_plain_mae():
    """无缺失值时, masked MAE 退化为普通 MAE。"""
    preds = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = torch.tensor([1.5, 2.5, 2.0, 5.0])
    expected = torch.abs(preds - labels).mean()
    assert torch.isclose(masked_mae(preds, labels), expected, atol=1e-6)


def test_zeros_are_masked_out():
    """label=0 的位置应被完全忽略, 不贡献误差。"""
    # 两个有效位误差各为 2.0; 两个 0 位无论 pred 多离谱都不该算进去
    preds = torch.tensor([3.0, 999.0, 5.0, -999.0])
    labels = torch.tensor([1.0, 0.0, 3.0, 0.0])
    # 仅有效位 [3-1, 5-3] = [2, 2] → MAE=2.0
    assert torch.isclose(masked_mae(preds, labels), torch.tensor(2.0), atol=1e-6)


def test_mask_reweight_equals_mean_over_valid():
    """核心性质: mask/=mask.mean() 后整体 mean == 仅有效位的 mean。"""
    torch.manual_seed(0)
    preds = torch.randn(4, 6, 10)
    labels = torch.randn(4, 6, 10)
    # 随机把 30% 位置置 0 (模拟缺失)
    drop = torch.rand_like(labels) < 0.3
    labels = labels.masked_fill(drop, 0.0)

    got = masked_mae(preds, labels, null_val=0.0)

    valid = labels != 0.0
    ref = torch.abs(preds - labels)[valid].mean()  # 直接对有效位求均值
    assert torch.isclose(got, ref, atol=1e-6)


def test_masked_huber_quadratic_and_linear_branch():
    """δ=1: |err|≤δ 走二次支路 0.5·err², |err|>δ 走线性支路 δ·(|err|−0.5δ); 缺失位被 mask。"""
    preds = torch.tensor([1.5, 5.0, 999.0])
    labels = torch.tensor([1.0, 2.0, 0.0])   # 第三位缺失, 不计
    # 有效误差: 0.5 → 0.5·0.5²=0.125; 3.0 → 3.0−0.5=2.5 → mean = 1.3125
    out = masked_huber(preds, labels, null_val=0.0, delta=1.0)
    assert torch.isclose(out, torch.tensor(1.3125), atol=1e-6)


def test_masked_huber_matches_torch_elementwise():
    """无缺失时与 torch F.huber_loss (mean) 一致; 大误差下小于同尺度 MSE (抗离群)。"""
    torch.manual_seed(0)
    preds = torch.randn(4, 6, 10)
    labels = torch.randn(4, 6, 10)
    out = masked_huber(preds, labels, delta=2.0)
    ref = torch.nn.functional.huber_loss(preds, labels, reduction="mean", delta=2.0)
    assert torch.isclose(out, ref, atol=1e-6)


def test_masked_huber_differentiable():
    """训练 loss 必须可微: 反向传播后 preds 拿到有限梯度。"""
    preds = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    labels = torch.tensor([0.5, 5.0, 1.0])
    loss = masked_huber(preds, labels, null_val=0.0, delta=1.0)
    loss.backward()
    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()


def test_rmse_is_sqrt_of_masked_mse():
    preds = torch.tensor([2.0, 0.0, 6.0])
    labels = torch.tensor([0.0, 0.0, 3.0])  # 仅第三位有效, 误差 3 → RMSE=3
    assert torch.isclose(masked_rmse(preds, labels), torch.tensor(3.0), atol=1e-6)


def test_rmse_known_value():
    """有效位误差 [3, 4] → MSE=(9+16)/2=12.5 → RMSE=sqrt(12.5)。"""
    preds = torch.tensor([4.0, 7.0, 100.0])
    labels = torch.tensor([1.0, 3.0, 0.0])  # 第三位缺失被 mask; 有效误差 [3, 4]
    expected = math.sqrt((9 + 16) / 2)
    assert torch.isclose(masked_rmse(preds, labels), torch.tensor(expected), atol=1e-5)


def test_mape_masks_zeros_no_divide_by_zero():
    """MAPE 不应因 label=0 而出现 inf/nan。"""
    preds = torch.tensor([1.1, 5.0, 2.0])
    labels = torch.tensor([1.0, 0.0, 4.0])
    out = masked_mape(preds, labels, null_val=0.0)
    assert torch.isfinite(out)
    # 有效位: |1.1-1|/1=0.1, |2-4|/4=0.5 → mean=0.3
    assert torch.isclose(out, torch.tensor(0.3), atol=1e-6)


def test_nan_null_val():
    """null_val=NaN 时按 isnan 判缺失。"""
    preds = torch.tensor([2.0, 100.0, 4.0])
    labels = torch.tensor([1.0, float("nan"), 1.0])
    # 有效位误差 [1, 3] → MAE=2.0
    out = masked_mae(preds, labels, null_val=float("nan"))
    assert torch.isclose(out, torch.tensor(2.0), atol=1e-6)


def test_all_missing_returns_zero_not_nan():
    """全部缺失的极端情形应兜底为 0, 不崩成 nan/inf。"""
    preds = torch.tensor([5.0, 6.0])
    labels = torch.tensor([0.0, 0.0])
    out = masked_mae(preds, labels, null_val=0.0)
    assert torch.isfinite(out) and float(out) == 0.0


def test_device_agnostic_cpu_consistency():
    """同一份数据多次计算结果一致 (确定性 / 无随机性泄漏)。"""
    preds = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    labels = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) * 0.9
    a = compute_all_metrics(preds, labels)
    b = compute_all_metrics(preds, labels)
    assert a == b
    assert set(a.keys()) == {"mae", "rmse", "mape"}


def test_compute_all_metrics_types():
    preds = torch.tensor([1.0, 2.0, 3.0])
    labels = torch.tensor([1.0, 2.0, 6.0])
    out = compute_all_metrics(preds, labels)
    assert all(isinstance(v, float) for v in out.values())
    assert math.isclose(out["mae"], 1.0, abs_tol=1e-6)  # 仅第三位误差 3, 但三位全有效 → (0+0+3)/3=1
