"""Graph WaveNet 基线同炉复现 (Wu et al., IJCAI'19, arXiv:1906.00121)。

官方实现 github.com/nnzhan/Graph-WaveNet; BasicTS 同构配置。

结构 (论文/BasicTS 默认配置):
  - 自适应邻接 adp = softmax(relu(E1 @ E2)), E1 [N,10]/E2 [10,N] 可学习节点向量
    (官方 randn 初始化, apt_dim=10), 与预定义双向转移矩阵 P_f/P_b (随机游走归一化,
    不加自环, 同 DCRNN 口径) 一起作为图卷积 supports。
  - 时序: dilated causal conv (kernel=2), num_blocks=4 × layers_per_block=2,
    dilation 每块内 [1,2] → 感受野 13 ≥ T_in=12 (不足时输入左侧补零, 同官方 scope)。
  - 门控 tanh ⊙ sigmoid; 每层扩散图卷积 (order=2: k=0 自身 + 每 support 两跳,
    拼接后 1x1 混合, 内部带 dropout) 替代 residual 1x1; BatchNorm; skip 1x1 跨层
    聚合 → ReLU → end 两层 1x1 → 一次出 T_out (直接多步)。
  - 宽度: residual=32, dilation=32, skip=256, end=512, dropout=0.3 (官方默认)。

偏差声明: 任务书「隐藏维 32/32/32」按官方口径理解为残差/门控/图卷积的处理宽均
为 32 (与官方一致); 若把第三项理解为 skip 聚合宽并压到 32, 输出端容量会被砍掉
大半、显著偏离论文配置, 故 skip/end 保留官方 256/512。输入只消费目标通道
(x[..., 0]), 理由同 DCRNN 模块 docstring 偏差 1; in_channels 保留作契约参数。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darwin_st.data.adjacency import random_walk_normalize

__all__ = ["GWNet"]


def _nconv(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """图聚合 (官方 nconv 口径): y[b,c,m,t] = sum_n x[b,c,n,t] * a[n,m]。"""
    return torch.einsum("bcnt,nm->bcmt", x, a)


class _GCN(nn.Module):
    """扩散图卷积 (官方 gcn 模块): k=0 自身 + 每个 support 的 order 跳, 拼接后 1x1 混合。"""

    def __init__(self, c_in: int, c_out: int, dropout: float, support_len: int, order: int = 2):
        super().__init__()
        self.order = order
        self.dropout = dropout
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor, supports: list[torch.Tensor]) -> torch.Tensor:
        out = [x]
        for a in supports:
            x1 = _nconv(x, a)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x2 = _nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)                 # [B, (order*S+1)*c_in, N, T]
        h = self.mlp(h)
        return F.dropout(h, self.dropout, training=self.training)


class GWNet(nn.Module):
    """Graph WaveNet。

    契约: forward(x [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]。
    tod_idx/dow_idx 为契约参数, GWNet 不消费。
    """

    def __init__(self, num_nodes: int, in_channels: int, seq_len_in: int,
                 seq_len_out: int, adj: np.ndarray | None = None,
                 dropout: float = 0.3, residual_channels: int = 32,
                 dilation_channels: int = 32, skip_channels: int = 256,
                 end_channels: int = 512, kernel_size: int = 2,
                 num_blocks: int = 4, layers_per_block: int = 2,
                 gcn_order: int = 2, apt_dim: int = 10, **_kw):
        super().__init__()
        self.num_nodes = num_nodes

        # 输入只消费目标通道 (见模块 docstring), start_conv 输入维 = 1
        self.start_conv = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))

        # 预定义 supports: 双向随机游走转移矩阵 (不加自环, 同 DCRNN); 无图则仅靠自适应
        if adj is not None:
            w = np.asarray(adj, dtype=np.float32)
            sup = np.stack([
                random_walk_normalize(w, add_self_loop=False),
                random_walk_normalize(w.T, add_self_loop=False),
            ])
            self.register_buffer("predef_supports", torch.from_numpy(sup).float(),
                                 persistent=False)
            n_predef = 2
        else:
            self.register_buffer("predef_supports", None, persistent=False)
            n_predef = 0
        support_len = n_predef + 1                  # + 自适应邻接

        # 自适应邻接的节点向量 (官方 randn 初始化)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, apt_dim))
        self.nodevec2 = nn.Parameter(torch.randn(apt_dim, num_nodes))

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # dilation 每块内 [1,2,4,...] 翻倍 (layers_per_block=2 → [1,2]), 感受野累计
        receptive_field = 1
        for _ in range(num_blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers_per_block):
                self.filter_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                   kernel_size=(1, kernel_size),
                                                   dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(residual_channels, dilation_channels,
                                                 kernel_size=(1, kernel_size),
                                                 dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels,
                                                 kernel_size=(1, 1)))
                self.gconvs.append(_GCN(dilation_channels, residual_channels, dropout,
                                        support_len=support_len, order=gcn_order))
                self.bns.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv2 = nn.Conv2d(end_channels, seq_len_out, kernel_size=(1, 1))
        # 感受野 (4 块 × [1,2] → 13) 大于 T_in 时左侧补零 (官方 scope 机制)
        self.scope = max(0, receptive_field - seq_len_in)

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        del tod_idx, dow_idx                        # 契约参数, 不消费
        B, T, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"

        x = x[..., 0].permute(0, 2, 1).unsqueeze(1)  # [B,1,N,T] 目标通道
        if self.scope:
            x = F.pad(x, (self.scope, 0, 0, 0))      # 时间维左侧补零至感受野
        x = self.start_conv(x)                       # [B,residual,N,T+scope]

        adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
        supports = ([s for s in self.predef_supports.unbind(0)]
                    if self.predef_supports is not None else []) + [adp]

        skip = None
        for i in range(len(self.filter_convs)):
            residual = x
            f = torch.tanh(self.filter_convs[i](x))
            g = torch.sigmoid(self.gate_convs[i](x))
            x = f * g                                # 门控, 时间维随 dilation 缩短
            s = self.skip_convs[i](x)                # [B,skip,N,T']
            skip = s if skip is None else skip[..., -s.size(3):] + s
            x = self.gconvs[i](x, supports)          # 图卷积 (替代 residual 1x1)
            x = x + residual[..., -x.size(3):]       # 残差对齐时间维
            x = self.bns[i](x)

        h = F.relu(skip)
        h = F.relu(self.end_conv1(h))
        h = self.end_conv2(h)                        # [B,T_out,N,1] (末步位置承载全部 horizon)
        return h.squeeze(-1)                         # [B,T_out,N]
