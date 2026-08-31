"""AGCRN 基线同炉复现 (Adaptive Graph Convolutional Recurrent Network)。

出处: Bai et al., "Adaptive Graph Convolutional Recurrent Network for Traffic
Forecasting", NeurIPS'20, arXiv:2007.02842。
官方实现 github.com/LeiBAI/AGCRN; BasicTS 同构配置。

结构 (论文默认配置):
  - 节点嵌入 E [N, embed_dim=10] 一物两用:
    a) 自适应图 adp = softmax(relu(E E^T)), 与恒等矩阵组成 cheb_k=2 支持集;
    b) NAPL (Node Adaptive Parameter Learning): 每节点专属卷积参数
       W_n = E_n @ W_pool, b_n = E_n @ b_pool (节点自适应参数学习)。
  - AGCRU: GRU 的 z/r 门与候选态全替换为 AVWGCN; num_layers=2 层堆叠 encoder
    读 T_in 步, hidden=64。
  - 输出: 末层最后隐状态 [B,N,64] → Linear(64 → T_out) 直接多步 (与项目
    search/builder.py STModel 的展平线性头同一精神)。

偏差声明: 官方 decoder 为与 DCRNN 同款的自回归迭代解码; 任务书指定「GRU 堆叠
encoder + 线性输出 T_out」, 直接头免 teacher forcing、与同炉接口天然契合, 在此
声明。输入只消费目标通道 (理由同 DCRNN 模块 docstring 偏差 1); AGCRN 是自学习
图模型, 契约参数 adj 不消费 (见 data/adjacency.py 模块说明)。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AGCRN"]


class _AVWGCN(nn.Module):
    """Node Adaptive Parameter Learning 图卷积 (官方 AVWGCR 内的 AVWGCN 部分)。

    支持集 = [I, adp] (cheb_k=2); 卷积权重/偏置由节点嵌入经 pool 生成 (每节点专属)。
    """

    def __init__(self, c_in: int, c_out: int, embed_dim: int, cheb_k: int = 2):
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.empty(embed_dim, c_in * cheb_k, c_out))
        self.bias_pool = nn.Parameter(torch.empty(embed_dim, c_out))
        nn.init.xavier_uniform_(self.weights_pool)
        nn.init.xavier_uniform_(self.bias_pool)

    def forward(self, x: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        # x: [B, N, c_in]; node_emb: [N, embed_dim]
        n = node_emb.shape[0]
        adp = F.softmax(F.relu(node_emb @ node_emb.t()), dim=1)
        supports = torch.stack([
            torch.eye(n, device=node_emb.device, dtype=node_emb.dtype), adp,
        ])                                               # [cheb_k, N, N]
        weights = torch.einsum("nd,dio->nio", node_emb, self.weights_pool)  # [N,k*c_in,c_out]
        bias = node_emb @ self.bias_pool                 # [N, c_out]
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # [k, B, N, c_in]
        x_g = x_g.permute(1, 2, 0, 3).reshape(x.shape[0], n, -1)  # [B, N, k*c_in]
        return torch.einsum("bni,nio->bno", x_g, weights) + bias  # [B, N, c_out]


class _AGCRUCell(nn.Module):
    """AGCRU 单元: GRU 的 z/r 门与候选态全替换为 AVWGCN。"""

    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.gate = _AVWGCN(input_dim + hidden_dim, 2 * hidden_dim, embed_dim)
        self.update = _AVWGCN(input_dim + hidden_dim, hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        zr = torch.sigmoid(self.gate(torch.cat([x, h], dim=-1), node_emb))
        z, r = zr.chunk(2, dim=-1)
        cand = torch.tanh(self.update(torch.cat([x, r * h], dim=-1), node_emb))
        return z * h + (1.0 - z) * cand


class AGCRN(nn.Module):
    """AGCRN (encoder + 线性直接多步头)。

    契约: forward(x [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]。
    tod_idx/dow_idx 与 adj 均为契约参数, AGCRN 不消费 (自适应图来自节点嵌入)。
    """

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj=None,
                 embed_dim: int = 10, hidden_dim: int = 64, num_layers: int = 2, **_kw):
        super().__init__()
        del adj                                          # 自学习图模型, 不用预定义邻接
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_emb = nn.Parameter(torch.empty(num_nodes, embed_dim))
        nn.init.xavier_uniform_(self.node_emb)
        # 输入只消费目标通道 (见模块 docstring), 首层 input_dim=1
        self.cells = nn.ModuleList(
            [_AGCRUCell(1 if i == 0 else hidden_dim, hidden_dim, embed_dim)
             for i in range(num_layers)]
        )
        self.out_proj = nn.Linear(hidden_dim, seq_len_out)  # 线性直接多步头

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        del tod_idx, dow_idx                             # 契约参数, 不消费
        B, T, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"
        xs = x[..., 0:1]                                 # 目标通道 [B,T,N,1]
        h = [xs.new_zeros(B, N, self.hidden_dim) for _ in range(self.num_layers)]
        for t in range(T):
            inp = xs[:, t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i], self.node_emb)
                inp = h[i]
        out = self.out_proj(h[-1])                       # [B,N,T_out]
        return out.permute(0, 2, 1)                      # [B,T_out,N]
