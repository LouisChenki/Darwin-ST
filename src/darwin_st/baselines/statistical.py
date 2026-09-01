"""统计基线 (Statistical Baselines) —— HA / AR / FC-LSTM。

审稿人默认要看的简单基线三件套, 同协议评测 (masked MAE, 12 horizons 平均):
  - HA (Historical Average): 直接复制最近 12 个观测步 (零参数, 免训练);
  - AR(p=12): 闭式最小二乘自回归 (构造时已一次性拟合 train split, 存 buffer;
    无 statsmodels 依赖 —— 与全量 ARIMA 的差别: 无差分/滑动平均项, 见 docstring 备注);
  - FC-LSTM: 2 层 LSTM (hidden 64) + 线性直接头, 用 train_baseline 训练。

统一契约: forward(x: [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]。
只消费目标通道 x[..., 0] (与其他 baseline 口径一致)。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["HistoricalAverage", "AutoregressiveBaseline", "FCLSTM"]


class HistoricalAverage(nn.Module):
    """HA: y_pred[:, h] = x[:, -T_out + h, :, 0] (复制最近 T_out 个观测)。"""

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj=None, **kw):
        super().__init__()
        self.seq_len_out = seq_len_out

    def forward(self, x: torch.Tensor, tod_idx=None, dow_idx=None) -> torch.Tensor:
        return x[:, -self.seq_len_out:, :, 0]


class AutoregressiveBaseline(nn.Module):
    """AR(p): 每步线性自回归, 系数在构造时由 train split 闭式最小二乘一次性拟合 (存 buffer)。

    与 statsmodels ARIMA 的差别 (如实声明): 无差分 (I) 与滑动平均 (MA) 项,
    是纯 AR(p) 自回归; 对 5 分钟交通流这类强周期序列是合理的轻量统计基线。
    coef_ 形状 [p] (各节点共享系数; 逐节点系数在数据量上是足够的, 但共享更稳)。
    预测方式: 自回归滚动 T_out 步。
    """

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj=None, p: int = 12, **kw):
        super().__init__()
        self.p = p
        self.seq_len_out = seq_len_out
        self.register_buffer("coef_", torch.ones(p) / p)   # 未 fit 前退化为窗口均值

    @torch.no_grad()
    def fit_closed_form(self, series: torch.Tensor) -> None:
        """series: [S, N] 归一化尺度目标通道训练序列。闭式解 coef = (X^T X)^-1 X^T y。"""
        S = series.shape[0]
        if S <= self.p + 1:
            return
        X = torch.stack([series[self.p - k - 1: S - k - 1] for k in range(self.p)], dim=-1)
        y = series[self.p:]
        Xf = X.reshape(-1, self.p)
        yf = y.reshape(-1)
        # 岭正则防病态 (S 大时闭式解稳定)
        reg = 1e-4 * torch.eye(self.p)
        coef = torch.linalg.solve(Xf.T @ Xf + reg, Xf.T @ yf)
        self.coef_.copy_(coef)

    def forward(self, x: torch.Tensor, tod_idx=None, dow_idx=None) -> torch.Tensor:
        s = x[:, :, :, 0]                              # [B, T_in, N]
        hist = s[:, -self.p:, :]                       # [B, p, N]
        outs = []
        for _ in range(self.seq_len_out):
            nxt = (hist * self.coef_.view(1, self.p, 1)).sum(dim=1)   # [B, N]
            outs.append(nxt)
            hist = torch.cat([hist[:, 1:, :], nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)                # [B, T_out, N]


class FCLSTM(nn.Module):
    """FC-LSTM: 2 层 LSTM (hidden 64) 编码输入序列, 末步隐状态线性直接出 T_out 步。"""

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj=None, hidden: int = 64, layers: int = 2, **kw):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers,
                            batch_first=True)
        self.seq_len_out = seq_len_out
        self.head = nn.Linear(hidden, seq_len_out)

    def forward(self, x: torch.Tensor, tod_idx=None, dow_idx=None) -> torch.Tensor:
        B, T, N, _ = x.shape
        s = x[:, :, :, 0].permute(0, 2, 1).reshape(B * N, T, 1)   # [B*N, T, 1]
        out, _ = self.lstm(s)
        h_last = out[:, -1, :]                                    # [B*N, hidden]
        y = self.head(h_last)                                     # [B*N, T_out]
        return y.reshape(B, N, self.seq_len_out).permute(0, 2, 1)
