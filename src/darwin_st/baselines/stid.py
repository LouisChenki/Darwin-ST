"""STID 基线同炉复现 (Spatial-Temporal Identity)。

出处: Shao et al., "Spatial-Temporal Identity: A Simple yet Effective Baseline for
Multivariate Time Series Forecasting", CIKM'22, arXiv:2208.05233。
官方实现 github.com/zezhishao/STID; BasicTS 同构配置。

结构 (论文默认配置):
  - 时序嵌入: 目标通道历史展平 Linear(T_in×1 → embed_dim=32) (逐节点共享权重)。
  - 身份嵌入拼接: NodeEmbedding [N,32] + TodEmbedding [288,32] + DowEmbedding [7,32],
    tod/dow 取输入窗口**最后一步**的时间索引 (官方口径)。拼接后 hidden = 32×4 = 128。
  - 主干: num_layer=3 个残差 MLP 块 (Linear→ReLU→Linear + 残差, hidden=128)。
  - 回归头: Linear(128 → T_out) 直接多步。

纯 MLP 无图模型: 契约参数 adj 不消费。tod_idx/dow_idx 为 None 时退化查 slot 0
(嵌入参数仍有梯度, 与 search/embeddings.py STEmbedding 的优雅降级约定一致)。
输入只消费目标通道 (官方 input_dim=1); in_channels 保留作契约参数。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["STID", "TodEmbedding", "TiwEmbedding", "DowEmbedding"]

STEPS_PER_DAY = 288   # 5 分钟采样 → 一天 288 槽 (与 data/prepare.make_time_indices 对齐)
DAYS_PER_WEEK = 7


class TodEmbedding(nn.Module):
    """time-of-day 身份嵌入 [288, dim]: 同一时刻 (如早高峰) 跨天共享语义。"""

    def __init__(self, dim: int, steps_per_day: int = STEPS_PER_DAY):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(steps_per_day, dim))
        nn.init.xavier_uniform_(self.emb)

    def forward(self, tod_idx: torch.Tensor) -> torch.Tensor:
        return self.emb[tod_idx.long()]


class DowEmbedding(nn.Module):
    """day-of-week 身份嵌入 [7, dim]: 工作日/周末模式差异。"""

    def __init__(self, dim: int, days_per_week: int = DAYS_PER_WEEK):
        super().__init__()
        self.emb = nn.Parameter(torch.empty(days_per_week, dim))
        nn.init.xavier_uniform_(self.emb)

    def forward(self, dow_idx: torch.Tensor) -> torch.Tensor:
        return self.emb[dow_idx.long()]


class TiwEmbedding(DowEmbedding):
    """time-in-week 身份嵌入: 5 分钟采样协议下官方 STID 的 tiw 按天粒度,

    即 day-of-week (7 槽), 与 DowEmbedding 同构 (此处作命名兼容别名)。
    """


class _MLPBlock(nn.Module):
    """残差 MLP 块 (官方 MultiLayerPerceptron): Linear→ReLU→Linear + 残差相加。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.fc2(F.relu(self.fc1(z)))


class STID(nn.Module):
    """STID (纯 MLP + 身份嵌入)。

    契约: forward(x [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]。
    tod_idx/dow_idx: [B, T_in] 整数 (0..287 / 0..6), 由 data 管线落盘提供。
    """

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj=None,
                 embed_dim: int = 32, node_dim: int = 32, tod_dim: int = 32,
                 dow_dim: int = 32, num_layer: int = 3,
                 steps_per_day: int = STEPS_PER_DAY,
                 days_per_week: int = DAYS_PER_WEEK, **_kw):
        super().__init__()
        del adj                                          # 纯 MLP 无图模型
        self.num_nodes = num_nodes
        # 输入只消费目标通道 (官方 input_dim=1), 展平时间 → 时序嵌入
        self.ts_emb = nn.Linear(seq_len_in * 1, embed_dim)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.tod_emb = TodEmbedding(tod_dim, steps_per_day)
        self.dow_emb = DowEmbedding(dow_dim, days_per_week)
        hidden_dim = embed_dim + node_dim + tod_dim + dow_dim   # 默认 32×4 = 128
        self.encoder = nn.Sequential(*[_MLPBlock(hidden_dim) for _ in range(num_layer)])
        self.regression = nn.Linear(hidden_dim, seq_len_out)

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"
        # 时序嵌入: [B,T,N] → [B,N,T] → Linear(T → embed_dim)
        h_ts = self.ts_emb(x[..., 0].permute(0, 2, 1))          # [B,N,embed]
        h_node = self.node_emb.unsqueeze(0).expand(B, -1, -1)   # [B,N,node_dim]
        # 时间身份取输入窗口最后一步索引 (官方口径); None → 查 slot 0 优雅降级
        # (嵌入参数仍参与计算、有梯度, 同 STEmbedding 约定)
        tod_last = tod_idx[:, -1] if tod_idx is not None else x.new_zeros(B, dtype=torch.long)
        dow_last = dow_idx[:, -1] if dow_idx is not None else x.new_zeros(B, dtype=torch.long)
        h_tod = self.tod_emb(tod_last).unsqueeze(1).expand(-1, N, -1)
        h_dow = self.dow_emb(dow_last).unsqueeze(1).expand(-1, N, -1)
        z = torch.cat([h_ts, h_node, h_tod, h_dow], dim=-1)     # [B,N,hidden]
        z = self.encoder(z)
        out = self.regression(z)                                # [B,N,T_out]
        return out.permute(0, 2, 1)                             # [B,T_out,N]
