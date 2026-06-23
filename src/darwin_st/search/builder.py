"""genotype → nn.Module 编译器 (Builder)。

把离散 genotype 解码 (编译) 成真实可训练的 PyTorch 模型 (phenotype)。

模型结构 (STID/STAEformer 式, 见 docs/P2_ALGORITHM_DESIGN.md):
  输入 [B,T_in,N,C]
    → STEmbedding (输入投影 + 拼接身份嵌入)        [B,T_in,N,hidden]
    → 堆叠 ST-block (每块: 空间算子 + 时序算子, 按 fusion 接线)
    → 直接多步线性头 (一次出全部 T_out 步)          [B,T_out,N]

设计要点:
  - 张量全程 [B,T,N,F], 节点维 N 不丢 (架构铁律, 见 docs/P2_ALGORITHM_DESIGN.md)。
  - 解码器固定为直接多步 (非迭代), 研究表明严格更优。
  - 邻接矩阵按 genotype.adj_mode 归一化后作为 buffer; 自学习/无图算子忽略它。
  - 时间索引 (tod/dow) 可选; 模型 forward 接受可选 tod_idx/dow_idx。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from darwin_st.data.adjacency import random_walk_normalize, symmetric_normalize
from darwin_st.search.embeddings import STEmbedding
from darwin_st.search.genotype import Genotype, STBlock
from darwin_st.search.operators import build_spatial_op, build_temporal_op

__all__ = ["STBlockModule", "STModel", "build_model", "count_params"]


class STBlockModule(nn.Module):
    """单个 ST-block: 空间算子 + 时序算子, 按 fusion 方式组合。保持 [B,T,N,hidden]。

    fusion:
      - sequential: x → 空间 → 时序 (串联)
      - parallel:   空间(x) + 时序(x) (并联相加)
      - residual:   x + sequential(x) (残差, 抗过平滑/利于深层)
    每块后接 LayerNorm (稳定时序训练, 不用 BatchNorm)。
    """

    def __init__(self, block: STBlock, hidden: int, num_nodes: int):
        super().__init__()
        self.fusion = block.fusion
        self.spatial = build_spatial_op(block.spatial_op, dim=hidden, num_nodes=num_nodes)
        self.temporal = build_temporal_op(block.temporal_op, dim=hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if self.fusion == "sequential":
            h = self.spatial(x, adj)            # 空间聚合
            h = self.temporal(h, adj)           # 再时序建模
        elif self.fusion == "parallel":
            h = self.spatial(x, adj) + self.temporal(x, adj)  # 并联相加
        elif self.fusion == "residual":
            h = self.spatial(x, adj)
            h = self.temporal(h, adj)
            h = x + h                           # 残差连接
        else:
            raise ValueError(f"未知 fusion: {self.fusion}")
        return self.norm(h)                     # LayerNorm 稳定


class STModel(nn.Module):
    """由 genotype 编译出的完整时空预测模型。

    forward(x, tod_idx=None, dow_idx=None): x [B,T_in,N,C] → [B,T_out,N]。
    """

    def __init__(self, genotype: Genotype, num_nodes: int, in_channels: int,
                 seq_len_in: int, seq_len_out: int, adj: np.ndarray | None = None):
        super().__init__()
        genotype.validate()
        self.genotype = genotype
        self.num_nodes = num_nodes
        self.seq_len_out = seq_len_out
        hidden = genotype.hidden

        # 输入嵌入 (投影 + 身份嵌入拼接)
        ec = genotype.embedding
        self.embed = STEmbedding(
            in_channels=in_channels, hidden=hidden, num_nodes=num_nodes,
            use_node=ec.use_node, node_dim=ec.node_dim,
            use_tod=ec.use_tod, tod_dim=ec.tod_dim,
            use_dow=ec.use_dow, dow_dim=ec.dow_dim,
        )

        # ST-block 堆叠
        self.blocks = nn.ModuleList(
            [STBlockModule(b, hidden, num_nodes) for b in genotype.blocks]
        )

        # 直接多步预测头: 把 T_in 步的 hidden 展平到时间, 一次输出 T_out 步
        # [B,N,T_in*hidden] -> [B,N,T_out]
        self.head = nn.Linear(seq_len_in * hidden, seq_len_out)

        # 邻接矩阵 (按 adj_mode 归一化) 作为 buffer; 无图则为 None
        self._register_adj(adj, genotype.adj_mode, num_nodes)

    def _register_adj(self, adj: np.ndarray | None, adj_mode: str, num_nodes: int) -> None:
        if adj is None:
            self.register_buffer("adj", None, persistent=False)
            return
        a = np.asarray(adj, dtype=np.float32)
        if adj_mode == "sym":
            a = symmetric_normalize(a)
        elif adj_mode == "rw":
            a = random_walk_normalize(a)
        # adj_mode == "none": 原始加权图不归一化
        self.register_buffer("adj", torch.from_numpy(a).float(), persistent=False)

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        B, T_in, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"

        adj = self.adj  # buffer (可能 None)
        h = self.embed(x, tod_idx=tod_idx, dow_idx=dow_idx)   # [B,T_in,N,hidden]
        for block in self.blocks:
            h = block(h, adj)                                  # [B,T_in,N,hidden]

        # 直接多步头: [B,T_in,N,hidden] -> [B,N,T_in*hidden] -> [B,N,T_out] -> [B,T_out,N]
        h = h.permute(0, 2, 1, 3).reshape(B, N, -1)           # 张量变换: 时间×特征 展平
        out = self.head(h)                                     # [B,N,T_out]
        return out.permute(0, 2, 1)                            # [B,T_out,N]


def build_model(
    genotype: Genotype, num_nodes: int, in_channels: int,
    seq_len_in: int, seq_len_out: int, adj: np.ndarray | None = None,
) -> STModel:
    """编译入口: genotype + 数据规格 → 可训练 STModel。"""
    return STModel(genotype, num_nodes, in_channels, seq_len_in, seq_len_out, adj)


def count_params(model: nn.Module) -> int:
    """可训练参数量 (供 MAP-Elites 行为描述子的参数量档使用)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
