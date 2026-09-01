"""统计基线 (statistical.py) 测试。"""

from __future__ import annotations

import torch

from darwin_st.baselines import BASELINES, build_baseline
from darwin_st.baselines.statistical import AutoregressiveBaseline, FCLSTM, HistoricalAverage

B, T, N, C, T_OUT = 2, 12, 8, 3, 12


def _x():
    return torch.randn(B, T, N, C)


def test_ha_copies_last_window():
    m = HistoricalAverage(num_nodes=N, in_channels=C, seq_len_in=T, seq_len_out=T_OUT)
    x = _x()
    y = m(x)
    assert y.shape == (B, T_OUT, N)
    assert torch.equal(y, x[:, -T_OUT:, :, 0])      # 就是最后 T_out 步的目标通道


def test_ar_closed_form_fit_and_rollout():
    m = AutoregressiveBaseline(num_nodes=N, in_channels=C, seq_len_in=T, seq_len_out=T_OUT)
    # 构造强周期序列: s[t] = 0.8*s[t-1] - 0.2*s[t-2] + 噪声 → AR(2) 主导
    S = 500
    s = torch.zeros(S, N)
    for t in range(2, S):
        s[t] = 0.8 * s[t - 1] - 0.2 * s[t - 2] + 0.01 * torch.randn(N)
    m.fit_closed_form(s)
    # 系数应学到近似 [0.8, -0.2, ~0, ...] 结构 (p=12)
    assert abs(m.coef_[0].item() - 0.8) < 0.1
    assert abs(m.coef_[1].item() + 0.2) < 0.1
    y = m(_x())
    assert y.shape == (B, T_OUT, N) and not torch.isnan(y).any()


def test_ar_unfit_defaults_to_window_mean():
    m = AutoregressiveBaseline(num_nodes=N, in_channels=C, seq_len_in=T, seq_len_out=T_OUT)
    x = torch.full((B, T, N, C), 2.0)
    y = m(x)
    assert torch.allclose(y, torch.full_like(y, 2.0), atol=1e-5)   # 均值系数 → 常数序列


def test_fc_lstm_shape_and_grad():
    m = FCLSTM(num_nodes=N, in_channels=C, seq_len_in=T, seq_len_out=T_OUT)
    x = _x()
    y = m(x)
    assert y.shape == (B, T_OUT, N) and not torch.isnan(y).any()
    y.mean().backward()                              # 可训练 (有梯度回路)
    assert any(p.grad is not None for p in m.parameters())


def test_statistical_registered_in_registry():
    for name in ("ha", "ar", "fc_lstm"):
        assert name in BASELINES
        m = build_baseline(name, num_nodes=N, in_channels=C, seq_len_in=T, seq_len_out=T_OUT)
        assert m(_x()).shape == (B, T_OUT, N)
