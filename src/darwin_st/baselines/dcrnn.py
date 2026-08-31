"""DCRNN 基线同炉复现 (Diffusion Convolutional Recurrent Neural Network)。

出处: Li et al., "Diffusion Convolutional Recurrent Neural Network: Data-Driven
Traffic Forecasting", ICLR'18, arXiv:1707.01926。
官方实现 github.com/liyaguang/DCRNN (TF) 与社区 PyTorch 移植 (chnsh/DCRNN_PyTorch)。

结构 (论文默认配置):
  - 扩散卷积: 双向随机游走转移矩阵 P_f = D^-1 W 与 P_b = D^-1 W^T, 由传入的原始
    加权邻接 W 计算 (prepare.py 落盘的 adj.npy 即原始 W)。扩散步 K=2; k=0 项即
    恒等 (节点自身信息), 故 P 不加自环 —— 与官方一致。
  - DCGRU: GRU 的全连接门 (z/r 更新门与候选态) 全部换成扩散卷积。
  - seq2seq: num_layers=2 层 DCGRU encoder 读 T_in 步, 2 层 DCGRU decoder 自回归
    直接出 T_out=12 步 (上一步预测作下一步输入, GO 符号为零向量)。hidden=64。

与同炉协议对齐的偏差 (均在此声明):
  1. 输入只消费目标通道 (x[..., 0:1]): 官方喂 speed+tod 双特征; 本协议仅目标通道
     被归一化 (prepare.py 只 transform target_channel), 其余通道原始尺度混入会
     破坏尺度一致性, 且四个基线统一单通道口径更公平。in_channels 保留作契约参数。
  2. decoder 无 scheduled sampling: 同炉接口 forward(x) 不接收标签, 训练/推理同
     为全自回归。官方训练按概率用真值替换输入, 收敛更快但该接口放不下。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from darwin_st.data.adjacency import random_walk_normalize

__all__ = ["DCRNN"]


def _transition_supports(adj: np.ndarray | None, num_nodes: int) -> np.ndarray:
    """由原始加权邻接 W 构造双向随机游走转移矩阵, 返回 [2, N, N] (P_f, P_b)。

    不加自环 (扩散卷积的 k=0 项已含节点自身, 同官方); adj=None 时退化为双自环
    (等价于普通 GRU 堆叠, 供无图数据集/冒烟测试兜底)。
    """
    if adj is None:
        eye = np.eye(num_nodes, dtype=np.float32)
        return np.stack([eye, eye])
    w = np.asarray(adj, dtype=np.float32)
    return np.stack([
        random_walk_normalize(w, add_self_loop=False),      # P_f = D^-1 W
        random_walk_normalize(w.T, add_self_loop=False),    # P_b = D^-1 W^T
    ])


class _DiffusionConv(nn.Module):
    """扩散卷积: concat[x, P_f^k x, P_b^k x (k=1..K)] @ W + b (双向 K 阶)。"""

    def __init__(self, c_in: int, c_out: int, K: int = 2, n_supports: int = 2):
        super().__init__()
        self.K = K
        self.weight = nn.Parameter(torch.empty((K * n_supports + 1) * c_in, c_out))
        self.bias = nn.Parameter(torch.zeros(c_out))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c_in]; supports: [n_supports, N, N]
        out = [x]
        for p in supports:
            xk = x
            for _ in range(self.K):
                xk = torch.matmul(p, xk)          # [N,N] @ [B,N,C] → [B,N,C] (P^k x)
                out.append(xk)
        h = torch.cat(out, dim=-1)                # [B, N, (K*n_supports+1)*c_in]
        return h @ self.weight + self.bias        # [B, N, c_out]


class _DCGRUCell(nn.Module):
    """DCGRU 单元: GRU 的 z/r 门与候选态全替换为扩散卷积。"""

    def __init__(self, input_dim: int, hidden_dim: int, K: int = 2):
        super().__init__()
        self.gates = _DiffusionConv(input_dim + hidden_dim, 2 * hidden_dim, K)
        self.candidate = _DiffusionConv(input_dim + hidden_dim, hidden_dim, K)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.gates(torch.cat([x, h], dim=-1), supports))
        z, r = zr.chunk(2, dim=-1)
        cand = torch.tanh(self.candidate(torch.cat([x, r * h], dim=-1), supports))
        return z * h + (1.0 - z) * cand


class DCRNN(nn.Module):
    """DCRNN seq2seq 直接多步预测模型。

    契约: forward(x [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]。
    tod_idx/dow_idx 为同炉接口契约参数, DCRNN 不消费 (官方用 one-hot 时槽特征,
    此处省略, 见模块 docstring 偏差 1)。
    """

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj: np.ndarray | None = None,
                 hidden_dim: int = 64, num_layers: int = 2,
                 max_diffusion_step: int = 2, **_kw):
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len_out = seq_len_out
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # 输入只消费目标通道 (通道 0, 见模块 docstring 偏差 1), 故 input_dim=1
        self.encoder = nn.ModuleList(
            [_DCGRUCell(1 if i == 0 else hidden_dim, hidden_dim, max_diffusion_step)
             for i in range(num_layers)]
        )
        self.decoder = nn.ModuleList(
            [_DCGRUCell(1 if i == 0 else hidden_dim, hidden_dim, max_diffusion_step)
             for i in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, 1)     # 每步隐状态 → 目标通道预测
        sup = torch.from_numpy(_transition_supports(adj, num_nodes)).float()
        self.register_buffer("supports", sup, persistent=False)   # 非参数, 随设备迁移

    def _encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        """encoder: 逐步读入 [B,T_in,N,1], 返回各层末态。"""
        B = x.shape[0]
        h = [x.new_zeros(B, self.num_nodes, self.hidden_dim) for _ in range(self.num_layers)]
        for t in range(x.shape[1]):
            inp = x[:, t]
            for i, cell in enumerate(self.encoder):
                h[i] = cell(inp, h[i], self.supports)
                inp = h[i]
        return h

    def _decode(self, h: list[torch.Tensor]) -> torch.Tensor:
        """decoder: 以 encoder 末态初始化, 全自回归出 T_out 步 (无 scheduled sampling)。"""
        B = h[0].shape[0]
        inp = h[0].new_zeros(B, self.num_nodes, 1)   # GO 符号 = 零向量
        outs = []
        for _ in range(self.seq_len_out):
            x = inp
            for i, cell in enumerate(self.decoder):
                h[i] = cell(x, h[i], self.supports)
                x = h[i]
            pred = self.out_proj(h[-1])              # [B, N, 1]
            outs.append(pred)
            inp = pred                               # 上一步预测作下一步输入
        return torch.stack(outs, dim=1).squeeze(-1)  # [B, T_out, N]

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        del tod_idx, dow_idx                          # 契约参数, 不消费
        B, T, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"
        return self._decode(self._encode(x[..., 0:1]))
