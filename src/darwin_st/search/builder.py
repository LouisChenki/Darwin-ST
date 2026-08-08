"""genotype → nn.Module 编译器 (Builder)。

把离散 genotype 解码 (编译) 成真实可训练的 PyTorch 模型 (phenotype)。

模型结构 (STID/STAEformer 式, 见 docs/P2_ALGORITHM_DESIGN.md):
  输入 [B,T_in,N,C]
    → STEmbedding (输入投影 + 拼接身份嵌入)        [B,T_in,N,hidden]
    → 堆叠 ST-block (每块: 空间算子 + 时序算子, 按 fusion 接线)
    → 直接多步线性头 (一次出全部 T_out 步)          [B,T_out,N]

B7 扩展: forward 拆为 forward_features (主干表征 h, [B,T_in,N,hidden]) + head (h→pred),
forward = head(forward_features(...)) —— 无 aux 时输出与拆分前**逐点一致** (行为不变铁律)。
辅助任务模块 (genotype.aux_op 非 None 时挂 model.aux_module) 吃 forward_features 产出的 h
计算自监督辅助损失 (train_one 侧 λ 加权叠加)。

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
from darwin_st.search.operators import (
    accepts_num_heads, build_aux_op, build_op, build_spatial_op, build_temporal_op,
)

__all__ = ["STBlockModule", "STModel", "build_model", "count_params"]


class STBlockModule(nn.Module):
    """单个 ST-block。两种模式:
      - 一体 (joint_op 非空): 单个时空算子一条边建模时空。
      - 分离 (joint_op=None): 空间算子 + 时序算子, 按 fusion 接线。

    fusion (分离模式):
      - sequential:    x → 空间 → 时序 (先S后T)
      - sequential_ts: x → 时序 → 空间 (先T后S)
      - parallel:      空间(x) + 时序(x) (并联相加)
      - residual:      x + (空间→时序) (残差, 抗过平滑/利于深层)
      - cross:         时序(x) * sigmoid(空间(x)) (门控交叉交互, 时空互相调制)
      - iterative:     S→T→S→T (双向迭代两轮, 加残差)
    每块后接 LayerNorm。全程 [B,T,N,hidden]。

    num_heads: HPO 调的多头注意力头数, 只传给头数可配的算子 (accepts_num_heads 判定:
    attn/st_graph_attn/series_decomp_attn); 单头写死的算子 (gat/dynamic_gat) 与其余算子
    一律不传 (它们的 __init__ 没有该参数, 传了也会被 **kw 静默吞掉, 不如不传语义干净)。
    None = 各算子用自带默认 (现有行为不变)。
    """

    def __init__(self, block: STBlock, hidden: int, num_nodes: int,
                 num_heads: int | None = None):
        super().__init__()
        self.fusion = block.fusion
        self.joint = None
        if block.joint_op is not None:
            kw = {"num_heads": num_heads} if (num_heads is not None and accepts_num_heads(block.joint_op)) else {}
            self.joint = build_op(block.joint_op, dim=hidden, num_nodes=num_nodes, **kw)
        else:
            s_kw = {"num_heads": num_heads} if (num_heads is not None and accepts_num_heads(block.spatial_op)) else {}
            t_kw = {"num_heads": num_heads} if (num_heads is not None and accepts_num_heads(block.temporal_op)) else {}
            self.spatial = build_spatial_op(block.spatial_op, dim=hidden, num_nodes=num_nodes, **s_kw)
            self.temporal = build_temporal_op(block.temporal_op, dim=hidden, num_nodes=num_nodes, **t_kw)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None) -> torch.Tensor:
        if self.joint is not None:
            return self.norm(self.joint(x, adj))    # 一体模式: 一条边
        f = self.fusion
        if f == "sequential":
            h = self.spatial(x, adj)                 # 先空间
            h = self.temporal(h, adj)                # 再时序
        elif f == "sequential_ts":
            h = self.temporal(x, adj)                # 先时序
            h = self.spatial(h, adj)                 # 再空间
        elif f == "parallel":
            h = self.spatial(x, adj) + self.temporal(x, adj)  # 并联相加
        elif f == "residual":
            h = self.spatial(x, adj)
            h = self.temporal(h, adj)
            h = x + h                                # 残差
        elif f == "cross":
            s = self.spatial(x, adj)                 # 空间作门
            t = self.temporal(x, adj)                # 时序作值
            h = t * torch.sigmoid(s) + x             # 门控交叉交互 + 残差
        elif f == "iterative":
            h = self.temporal(self.spatial(x, adj), adj)      # 第一轮 S→T
            h = self.temporal(self.spatial(h, adj), adj)      # 第二轮 S→T
            h = x + h                                # 双向迭代 + 残差
        else:
            raise ValueError(f"未知 fusion: {f}")
        return self.norm(h)                          # LayerNorm 稳定


class STModel(nn.Module):
    """由 genotype 编译出的完整时空预测模型。

    forward(x, tod_idx=None, dow_idx=None): x [B,T_in,N,C] → [B,T_out,N],
    恒等于 head(forward_features(...)) (B7 拆分):
      - forward_features: 主干 (嵌入 + block 堆叠 + 头前 dropout) → h [B,T_in,N,hidden]
      - head:             h → [B,T_out,N] 直接多步预测
    aux_module: 可选辅助任务模块 (B7, genotype.aux_op 非 None 时按名从 AUX_OPS 实例化;
    None = 无辅助任务, 默认)。它消费 forward_features 的 h 与原始输入 x 算 aux_loss,
    不进 forward 主路径 → 主任务前向/评测行为零变化。

    dropout: HPO 调的正则强度, 只设在【block 堆叠后、输出头前】一处 (最小侵入;
    嵌入处不加 —— 身份嵌入是最高杠杆组件, 丢它伤精度)。默认 0.0 → nn.Dropout(0.0)
    恒等映射, 现有行为完全不变。
    num_heads: HPO 调的多头注意力头数, 透传给各 ST-block (仅头数可配算子消费)。
    """

    def __init__(self, genotype: Genotype, num_nodes: int, in_channels: int,
                 seq_len_in: int, seq_len_out: int, adj: np.ndarray | None = None,
                 dropout: float = 0.0, num_heads: int | None = None):
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

        # ST-block 堆叠 (num_heads 仅头数可配的注意力算子消费, 见 STBlockModule)
        self.blocks = nn.ModuleList(
            [STBlockModule(b, hidden, num_nodes, num_heads=num_heads) for b in genotype.blocks]
        )
        # 输出头前的正则 (HPO dropout 旋钮的唯一作用点; 0.0 恒等)
        self.dropout = nn.Dropout(dropout)

        # 直接多步预测头: 把 T_in 步的 hidden 展平到时间, 一次输出 T_out 步
        # [B,N,T_in*hidden] -> [B,N,T_out]
        self.out_proj = nn.Linear(seq_len_in * hidden, seq_len_out)

        # B7 辅助任务模块: genotype.aux_op 非 None → 从 AUX_OPS 查名实例化挂载
        # (channels=hidden 主干表征维, num_nodes 节点数, seq_in/seq_out 按数据规格);
        # 未注册 → build_aux_op 抛 ValueError (带已注册名清单)。None = 无辅助任务 (默认)。
        self.aux_module: nn.Module | None = None
        if genotype.aux_op is not None:
            self.aux_module = build_aux_op(genotype.aux_op, channels=hidden,
                                           num_nodes=num_nodes,
                                           seq_in=seq_len_in, seq_out=seq_len_out)

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

    def forward_features(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                         dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        """主干表征: x [B,T_in,N,C] → h [B,T_in,N,hidden] (B7: 供 aux_loss 消费, 带梯度)。"""
        B, T_in, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: {N} vs {self.num_nodes}"

        adj = self.adj  # buffer (可能 None)
        h = self.embed(x, tod_idx=tod_idx, dow_idx=dow_idx)   # [B,T_in,N,hidden]
        for block in self.blocks:
            h = block(h, adj)                                  # [B,T_in,N,hidden]
        h = self.dropout(h)                                    # 头前正则 (p=0 恒等)
        return h

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """直接多步预测头: h [B,T_in,N,hidden] → [B,T_out,N] (张量变换与拆分前逐字节一致)。"""
        B, T_in, N, _ = h.shape
        # [B,T_in,N,hidden] -> [B,N,T_in*hidden] -> [B,N,T_out] -> [B,T_out,N]
        h = h.permute(0, 2, 1, 3).reshape(B, N, -1)           # 张量变换: 时间×特征 展平
        out = self.out_proj(h)                                 # [B,N,T_out]
        return out.permute(0, 2, 1)                            # [B,T_out,N]

    def forward(self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
                dow_idx: torch.Tensor | None = None) -> torch.Tensor:
        # forward = head(forward_features(...)): 与拆分前**逐点一致** (行为不变铁律, 有测试守)
        return self.head(self.forward_features(x, tod_idx=tod_idx, dow_idx=dow_idx))


def build_model(
    genotype: Genotype, num_nodes: int, in_channels: int,
    seq_len_in: int, seq_len_out: int, adj: np.ndarray | None = None,
    dropout: float = 0.0, num_heads: int | None = None,
) -> STModel:
    """编译入口: genotype + 数据规格 → 可训练 STModel。

    dropout/num_heads 是 HPO 超参的接线口 (train_one 从 hps 读出传入); 默认 0.0/None
    时与旧行为完全一致 (正则恒等, 各算子用自带默认头数)。

    B7: genotype.aux_op 非 None 时按名从 AUX_OPS 实例化辅助任务模块挂 model.aux_module
    (channels=hidden / num_nodes / seq_in / seq_out 按 genotype 与数据规格; 具体挂载在
    STModel.__init__)。未注册 → ValueError (报"辅助任务未注册"及当前已注册名清单)。
    """
    return STModel(genotype, num_nodes, in_channels, seq_len_in, seq_len_out, adj,
                   dropout=dropout, num_heads=num_heads)


def count_params(model: nn.Module) -> int:
    """可训练参数量 (供 MAP-Elites 行为描述子的参数量档使用)。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
