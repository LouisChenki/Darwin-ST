"""时空算子库 (Spatio-Temporal Operator Library) —— NAS 搜索空间的原子积木。

每个算子是一个 nn.Module 工厂, 是 genotype 解码 (builder) 时实例化的基本单元。
分两类:
  - 空间算子 (Spatial): 在节点维 N 上按图结构混合 —— ChebGCN / GCN / GAT /
    DiffusionConv / AdaptiveAdj / Identity。理解空间邻接关系。
  - 时序算子 (Temporal): 在时间维 T 上混合 —— DilatedTCN / GRU / TemporalAttn / Identity。

⚠️ 张量约定 (全库统一, 见 docs/P2_ALGORITHM_DESIGN.md 架构铁律):
    内部一律 [B, T, N, F] (Batch, Time, Nodes, Features)。
    **严禁把空间节点维 N 暴力展平** —— 那会摧毁交通节点的物理空间拓扑。
    空间算子在 N 维按邻接混合; 时序算子在 T 维混合。两类都保持 [B,T,N,F] 进出。

每个张量变换带中文维度注释, 并用 assert 防御 shape 幻觉。
"""

from __future__ import annotations

import hashlib
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SPATIAL_OPS",
    "TEMPORAL_OPS",
    "SPATIOTEMPORAL_OPS",
    "OP_CATEGORY",
    "SYNTH_PREFIX",
    "op_category",
    "builtin_op_signature",
    "build_op",
    "build_spatial_op",
    "build_temporal_op",
    "Identity",
    "ChebGCN",
    "GCNConv",
    "GATConv",
    "DiffusionConv",
    "AdaptiveAdjConv",
    "DilatedTCN",
    "GRUTemporal",
    "TemporalAttention",
    "STSeparableConv",
    "STGraphAttn",
    "MixHop",
    "GWNetAdaptive",
    "MultiScaleTCN",
    "SeriesDecompAttn",
    "DynamicGAT",
    "STJointConv",
    "DiagonalSSM",
]


# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------


class Identity(nn.Module):
    """恒等算子 (占位/可被搜索选择跳过某一层)。保持 [B,T,N,F] 不变。"""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        return x


def _normalize_adj_sym(adj: torch.Tensor) -> torch.Tensor:
    """对称归一化 D^{-1/2}(A+I)D^{-1/2} (GCN 用)。adj: [N,N] → [N,N]。"""
    n = adj.shape[0]
    a = adj + torch.eye(n, device=adj.device, dtype=adj.dtype)  # 加自环: A+I
    deg = a.sum(dim=1)                                          # 度向量 [N]
    d_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)         # D^{-1/2}
    return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)  # [N,N]


# ---------------------------------------------------------------------------
# 空间算子 (在 N 维按图混合, 保持 [B,T,N,F])
# ---------------------------------------------------------------------------


class GCNConv(nn.Module):
    """一阶图卷积 (Kipf GCN)。X' = Â X W, Â=对称归一化邻接。需要外部邻接 adj。"""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert adj is not None, "GCNConv 需要邻接矩阵 adj"
        a_hat = _normalize_adj_sym(adj)                  # [N,N]
        # 张量变换: [B,T,N,F] 在 N 维按 Â 聚合 -> [B,T,N,F]
        out = torch.einsum("nm,btmf->btnf", a_hat, x)    # 邻居特征加权聚合
        return self.lin(out)                             # 线性变换特征维 F


class ChebGCN(nn.Module):
    """切比雪夫图卷积 (ChebNet, K 阶)。用 K 阶多项式逼近谱图卷积, 捕获 K-hop 邻域。"""

    def __init__(self, dim: int, K: int = 3, **kw):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(dim * K, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert adj is not None, "ChebGCN 需要邻接矩阵 adj"
        n = adj.shape[0]
        a_hat = _normalize_adj_sym(adj)                  # [N,N] 作为 ~L 的近似基
        # 切比雪夫递推: T_0=I·x, T_1=Â·x, T_k=2Â T_{k-1} - T_{k-2}
        out = [x]                                        # T_0 (x): [B,T,N,F]
        if self.K > 1:
            out.append(torch.einsum("nm,btmf->btnf", a_hat, x))  # T_1
        for _ in range(2, self.K):
            tk = 2 * torch.einsum("nm,btmf->btnf", a_hat, out[-1]) - out[-2]
            out.append(tk)
        # 张量变换: 拼接 K 阶 -> [B,T,N,F*K] -> 线性回 F
        cat = torch.cat(out, dim=-1)                     # [B,T,N,F*K]
        return self.lin(cat)


class GATConv(nn.Module):
    """图注意力 (单头简化版)。按节点对注意力权重在 N 维聚合, 仅在有边处注意。"""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.att_src = nn.Linear(dim, 1)
        self.att_dst = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert adj is not None, "GATConv 需要邻接矩阵 adj (定义注意力可达边)"
        h = self.lin(x)                                  # [B,T,N,F]
        # 注意力打分 e[n,m] = att_src(h_n) + att_dst(h_m): 显式广播成 [B,T,N,N]
        src = self.att_src(h)                            # [B,T,N,1] 作为目标节点 n
        dst = self.att_dst(h)                            # [B,T,N,1] 作为邻居节点 m
        e = src + dst.transpose(-1, -2)                  # [B,T,N,1]+[B,T,1,N] -> [B,T,N,N]
        e = F.leaky_relu(e, 0.2)
        mask = (adj > 0).unsqueeze(0).unsqueeze(0)        # 只在有边处注意 [1,1,N,N]
        e = e.masked_fill(~mask, float("-inf"))
        att = torch.softmax(e, dim=-1)                    # [B,T,N,N] 行归一 (对邻居 m)
        att = torch.nan_to_num(att, nan=0.0)              # 孤立节点全 -inf → 0
        # 张量变换: [B,T,N,N]×[B,T,N,F] 在 N(邻居 m) 维聚合 -> [B,T,N,F]
        return torch.einsum("btnm,btmf->btnf", att, h)


class DiffusionConv(nn.Module):
    """扩散图卷积 (DCRNN)。双向随机游走 K 步扩散, 适合有向交通流。"""

    def __init__(self, dim: int, K: int = 2, **kw):
        super().__init__()
        self.K = K
        self.lin = nn.Linear(dim * (2 * K + 1), dim)

    def _rw(self, adj: torch.Tensor) -> torch.Tensor:
        """随机游走归一化 D^{-1}A → 转移概率矩阵 [N,N]。"""
        deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return adj / deg

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert adj is not None, "DiffusionConv 需要邻接矩阵 adj"
        p_fwd = self._rw(adj)                             # 前向转移 [N,N]
        p_bwd = self._rw(adj.t())                         # 反向转移 [N,N]
        feats = [x]                                       # 0 步
        z_f, z_b = x, x
        for _ in range(self.K):
            z_f = torch.einsum("nm,btmf->btnf", p_fwd, z_f)  # 前向扩散一步
            z_b = torch.einsum("nm,btmf->btnf", p_bwd, z_b)  # 反向扩散一步
            feats.extend([z_f, z_b])
        cat = torch.cat(feats, dim=-1)                    # [B,T,N,F*(2K+1)]
        return self.lin(cat)


class AdaptiveAdjConv(nn.Module):
    """自适应图卷积 (Graph WaveNet 式)。从可学习节点嵌入自学邻接, 不需外部图。"""

    def __init__(self, dim: int, num_nodes: int, emb_dim: int = 10, **kw):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)  # 源嵌入
        self.e2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)  # 汇嵌入
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        # 自适应邻接: softmax(ReLU(E1 E2^T)), 无需外部 adj
        a_adp = torch.softmax(F.relu(self.e1 @ self.e2.t()), dim=1)  # [N,N]
        out = torch.einsum("nm,btmf->btnf", a_adp, x)               # N 维聚合
        return self.lin(out)


# ---------------------------------------------------------------------------
# 时序算子 (在 T 维混合, 保持 [B,T,N,F])
# ---------------------------------------------------------------------------


class DilatedTCN(nn.Module):
    """膨胀因果时间卷积 (WaveNet/TCN)。沿 T 维 1D 卷积, 保持时间长度不变 (padding)。"""

    def __init__(self, dim: int, kernel_size: int = 2, dilation: int = 1, **kw):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(dim, dim, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        # 张量变换: [B,T,N,F] -> [B*N, F, T] 让 Conv1d 在 T 维卷积
        h = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)   # [B*N, F, T]
        h = F.pad(h, (self.pad, 0))                       # 因果左填充 (不看未来)
        h = self.conv(h)                                  # [B*N, F, T]
        # 张量变换: 还原 -> [B,T,N,F]
        h = h.reshape(B, N, Fd, T).permute(0, 3, 1, 2)
        assert h.shape == x.shape, f"DilatedTCN 形状漂移: {h.shape} vs {x.shape}"
        return h


class GRUTemporal(nn.Module):
    """GRU 时序建模。每个节点独立过 GRU, 取全序列隐藏态, 保持 T。"""

    def __init__(self, dim: int, **kw):
        super().__init__()
        self.gru = nn.GRU(dim, dim, batch_first=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        # 张量变换: [B,T,N,F] -> [B*N, T, F] 让 GRU 在 T 维循环
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, Fd)   # [B*N, T, F]
        out, _ = self.gru(h)                              # [B*N, T, F]
        return out.reshape(B, N, T, Fd).permute(0, 2, 1, 3)  # 还原 [B,T,N,F]


class TemporalAttention(nn.Module):
    """时序自注意力。每个节点在 T 维做多头自注意力, 捕获长程时间依赖。"""

    def __init__(self, dim: int, num_heads: int = 4, **kw):
        super().__init__()
        # dim 须能被 heads 整除; 否则退化为单头
        heads = num_heads if dim % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * N, T, Fd)   # [B*N, T, F]
        out, _ = self.attn(h, h, h)                       # T 维自注意力
        return out.reshape(B, N, T, Fd).permute(0, 2, 1, 3)  # 还原 [B,T,N,F]


# ---------------------------------------------------------------------------
# 时空一体算子 (Spatio-Temporal Joint): 单个算子同时在 N 维与 T 维交互。
# 一条边即完成时空建模, 取代"空间算子+时序算子"对 (genotype 的 joint block 模式)。
# 仍保持 [B,T,N,F] 进出 + 节点维 N 不丢 (架构铁律)。
# ---------------------------------------------------------------------------


class STSeparableConv(nn.Module):
    """时空可分离卷积 (深度可分离式): 空间图卷积 (N 维聚合) + 时序因果卷积 (T 维) 打包为一个算子。

    类比 depthwise-separable: 先在 N 维按邻接做轻量聚合 (空间 depthwise),
    再在 T 维做因果卷积 (时序), 最后线性融合特征。一条边完成时空建模。
    无图 (adj=None) 时退化为纯时序卷积 (空间聚合跳过)。
    """

    def __init__(self, dim: int, num_nodes: int | None = None, kernel_size: int = 2, **kw):
        super().__init__()
        self.pad = kernel_size - 1
        self.tconv = nn.Conv1d(dim, dim, kernel_size)       # 时序因果卷积
        self.lin = nn.Linear(dim, dim)                      # 空间聚合后特征变换

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        # 空间 depthwise: N 维按对称归一化邻接聚合 (有图才做)
        if adj is not None:
            a_hat = _normalize_adj_sym(adj)                 # [N,N]
            h = torch.einsum("nm,btmf->btnf", a_hat, x)     # [B,T,N,F]
            h = self.lin(h)
        else:
            h = self.lin(x)
        # 时序: [B,T,N,F] -> [B*N,F,T] 因果卷积 -> 还原
        ht = h.permute(0, 2, 3, 1).reshape(B * N, Fd, T)    # [B*N,F,T]
        ht = F.pad(ht, (self.pad, 0))                       # 因果左填充
        ht = self.tconv(ht)                                 # [B*N,F,T]
        ht = ht.reshape(B, N, Fd, T).permute(0, 3, 1, 2)    # [B,T,N,F]
        assert ht.shape == x.shape, f"STSeparableConv 形状漂移: {ht.shape} vs {x.shape}"
        return ht


class STGraphAttn(nn.Module):
    """图卷积引导的时序注意力 (复杂时空交互范例): 空间邻接调制时序注意力。

    每个节点在 T 维做自注意力, 但 value 先经空间图卷积聚合邻居信息 ——
    即"时序注意力 + 空间引导"的耦合: 注意力决定看哪些时刻, 图卷积决定融合哪些节点。
    一条边完成"时序注意力引导的动态空间交互"(用户举例的复杂时空算子)。
    """

    def __init__(self, dim: int, num_nodes: int | None = None, num_heads: int = 4, **kw):
        super().__init__()
        h = num_heads if dim % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(dim, h, batch_first=True)
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        # 空间引导: value 先经图卷积聚合邻居 (有图才做)
        if adj is not None:
            a_hat = _normalize_adj_sym(adj)                 # [N,N]
            v = torch.einsum("nm,btmf->btnf", a_hat, x)     # 空间聚合后的 value
            v = self.lin(v)
        else:
            v = self.lin(x)
        # 时序注意力: 每节点在 T 维, query/key=x, value=空间聚合后
        xq = x.permute(0, 2, 1, 3).reshape(B * N, T, Fd)    # [B*N,T,F]
        vv = v.permute(0, 2, 1, 3).reshape(B * N, T, Fd)    # [B*N,T,F]
        out, _ = self.attn(xq, xq, vv)                      # [B*N,T,F]
        out = out.reshape(B, N, T, Fd).permute(0, 2, 1, 3)  # [B,T,N,F]
        return self.norm(out + x)                           # 残差 + 稳定


# ---------------------------------------------------------------------------
# Stage 2: 扩充原子算子集 (更丰富的搜索空间, 破线性天花板)
# 全 [B,T,N,F]→同形, 节点维 N 不丢, CPU 安全。
# ---------------------------------------------------------------------------


class MixHop(nn.Module):
    """MixHop 多跳图卷积: 拼接 A^0/A^1/A^2 x 再线性。多跳邻域混合, 抗过平滑 (不同跳分开保留)。"""

    def __init__(self, dim: int, hops: int = 3, **kw):
        super().__init__()
        self.hops = hops
        self.lin = nn.Linear(dim * hops, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        assert adj is not None, "MixHop 需要邻接矩阵 adj"
        a_hat = _normalize_adj_sym(adj)                  # [N,N]
        outs = [x]                                       # A^0 x = x
        z = x
        for _ in range(self.hops - 1):
            z = torch.einsum("nm,btmf->btnf", a_hat, z)  # 再传播一跳
            outs.append(z)
        cat = torch.cat(outs, dim=-1)                    # [B,T,N,F*hops]
        return self.lin(cat)


class GWNetAdaptive(nn.Module):
    """Graph WaveNet 双邻接: 外部图 + 可学习自适应图 E1E2^T 联合聚合 (泛化 adaptive)。

    无外部图 (adj=None) 时退化为纯自适应图 (等价 AdaptiveAdjConv)。
    """

    def __init__(self, dim: int, num_nodes: int, emb_dim: int = 10, **kw):
        super().__init__()
        self.e1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)
        self.e2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.01)
        self.lin = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        a_adp = torch.softmax(F.relu(self.e1 @ self.e2.t()), dim=1)  # 自适应图 [N,N]
        out = torch.einsum("nm,btmf->btnf", a_adp, x)               # 自适应聚合
        if adj is not None:
            a_hat = _normalize_adj_sym(adj)
            out = out + torch.einsum("nm,btmf->btnf", a_hat, x)     # 叠外部图 (双邻接)
        return self.lin(out)


class MultiScaleTCN(nn.Module):
    """多尺度膨胀 TCN: 并行 dilation 1/2/4 因果卷积相加, 捕获多时间粒度周期。"""

    def __init__(self, dim: int, kernel_size: int = 2, dilations=(1, 2, 4), **kw):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size, dilation=d) for d in dilations
        ])
        self.pads = [(kernel_size - 1) * d for d in dilations]

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)   # [B*N,F,T]
        out = 0
        for conv, pad in zip(self.branches, self.pads):
            out = out + conv(F.pad(h, (pad, 0)))          # 各膨胀分支因果卷积相加
        out = out.reshape(B, N, Fd, T).permute(0, 3, 1, 2)  # [B,T,N,F]
        assert out.shape == x.shape, f"MultiScaleTCN 形漂移: {out.shape}"
        return out


class SeriesDecompAttn(nn.Module):
    """Autoformer 式序列分解注意力: 分解 trend(移动平均)+seasonal(残差), seasonal 走时序注意力。

    交通流有强 trend+周期; 分解后对 seasonal 建注意力比对原序列更干净。
    """

    def __init__(self, dim: int, num_heads: int = 4, kernel: int = 5, **kw):
        super().__init__()
        h = num_heads if dim % num_heads == 0 else 1
        self.attn = nn.MultiheadAttention(dim, h, batch_first=True)
        self.kernel = kernel
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        # 移动平均取 trend (在 T 维), seasonal = x - trend
        h = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)   # [B*N,F,T]
        pad = self.kernel // 2
        trend = F.avg_pool1d(F.pad(h, (pad, pad), mode="replicate"), self.kernel, stride=1)
        trend = trend[..., :T]
        seasonal = h - trend                              # [B*N,F,T]
        # seasonal 走时序注意力
        s = seasonal.permute(0, 2, 1)                     # [B*N,T,F]
        s_attn, _ = self.attn(s, s, s)                    # [B*N,T,F]
        out = s_attn + trend.permute(0, 2, 1)             # 注意力 seasonal + trend
        out = out.reshape(B, N, T, Fd).permute(0, 2, 1, 3)  # [B,T,N,F]
        return self.norm(out)


class DynamicGAT(nn.Module):
    """注意力引导动态图卷积 (时空一体): 每时刻按特征相似度动态重加权图边, 再 N 维聚合。

    区别于静态 GAT: 注意力权重随时间步动态变化 (捕获时变空间依赖)。用户举例的复杂时空算子。
    """

    def __init__(self, dim: int, num_nodes: int | None = None, **kw):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        q, k, v = self.q(x), self.k(x), self.v(x)         # [B,T,N,F]
        # 每时刻动态注意力打分 [B,T,N,N] (节点对相似度)
        e = torch.einsum("btnf,btmf->btnm", q, k) * self.scale
        if adj is not None:
            mask = (adj > 0).unsqueeze(0).unsqueeze(0)     # 限静态图可达边
            e = e.masked_fill(~mask, float("-inf"))
        att = torch.softmax(e, dim=-1)                    # 行归一
        att = torch.nan_to_num(att, nan=0.0)
        out = torch.einsum("btnm,btmf->btnf", att, v)     # 动态图聚合
        return self.norm(out + x)                         # 残差


class STJointConv(nn.Module):
    """时空联合卷积 (时空一体): 图混合 + 时序卷积在一个算子内因子化融合。

    与 STSeparableConv 区别: 这里空间/时序交替两轮 (S→T→S→T 内部), 交互更深。
    """

    def __init__(self, dim: int, num_nodes: int | None = None, kernel_size: int = 2, **kw):
        super().__init__()
        self.pad = kernel_size - 1
        self.tconv1 = nn.Conv1d(dim, dim, kernel_size)
        self.tconv2 = nn.Conv1d(dim, dim, kernel_size)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def _tconv(self, x, conv):
        B, T, N, Fd = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, Fd, T)
        h = conv(F.pad(h, (self.pad, 0)))
        return h.reshape(B, N, Fd, T).permute(0, 3, 1, 2)

    def _sconv(self, x, lin, adj):
        if adj is not None:
            a_hat = _normalize_adj_sym(adj)
            return lin(torch.einsum("nm,btmf->btnf", a_hat, x))
        return lin(x)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        h = self._sconv(x, self.lin1, adj)                # 空间1
        h = self._tconv(h, self.tconv1)                   # 时序1
        h = self._sconv(h, self.lin2, adj)                # 空间2
        h = self._tconv(h, self.tconv2)                   # 时序2
        return self.norm(h + x)                           # 残差


class DiagonalSSM(nn.Module):
    """对角状态空间模型 (S4/Mamba-lite): h_t = a·h_{t-1} + b·x_t, y = c·h + d·x。

    O(T) 顺序扫描 (非 O(T²) 注意力), 长程线性复杂度。a=sigmoid(_a)∈(0,1) 保稳定 (谱半径<1)。
    压轴算子: 单独测试门, 因扫描循环需验证数值稳定 + 梯度不爆。逐通道对角 (无跨通道混合矩阵)。
    """

    def __init__(self, dim: int, **kw):
        super().__init__()
        # 对角参数: 逐通道独立 SSM。_a 经 sigmoid 落 (0,1) → 保证 |a|<1 稳定
        self._a = nn.Parameter(torch.randn(dim) * 0.1 - 1.0)  # sigmoid(-1)≈0.27 起步 (温和衰减)
        self.b = nn.Parameter(torch.ones(dim) * 0.1)
        self.c = nn.Parameter(torch.ones(dim) * 0.1)
        self.d = nn.Parameter(torch.ones(dim))               # 直通项 (跳连, 类零初始残差稳定)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        B, T, N, Fd = x.shape
        a = torch.sigmoid(self._a)                           # (0,1) 稳定衰减 [F]
        h = torch.zeros(B, N, Fd, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):                                   # O(T) 扫描
            h = a * h + self.b * x[:, t]                      # 状态递推 [B,N,F]
            ys.append(self.c * h + self.d * x[:, t])          # 输出 = 状态读出 + 直通
        out = torch.stack(ys, dim=1)                         # [B,T,N,F]
        assert out.shape == x.shape, f"DiagonalSSM 形漂移: {out.shape}"
        return self.norm(out)


# ---------------------------------------------------------------------------
# 注册表 + 工厂 (genotype 用算子名引用; 新增创新点算子在此登记)
# ---------------------------------------------------------------------------

SPATIAL_OPS = {
    "identity": Identity,
    "gcn": GCNConv,
    "cheb": ChebGCN,
    "gat": GATConv,
    "diffusion": DiffusionConv,
    "adaptive": AdaptiveAdjConv,
    "mixhop": MixHop,                # Stage2: 多跳抗过平滑
    "gwnet_adp": GWNetAdaptive,      # Stage2: 双邻接 (外部+自适应)
}

TEMPORAL_OPS = {
    "identity": Identity,
    "tcn": DilatedTCN,
    "gru": GRUTemporal,
    "attn": TemporalAttention,
    "multiscale_tcn": MultiScaleTCN,     # Stage2: 多尺度膨胀
    "series_decomp_attn": SeriesDecompAttn,  # Stage2: 序列分解注意力
    "ssm": DiagonalSSM,                  # Stage2: 对角状态空间 (O(T) 长程扫描)
}

# 时空一体算子 (joint block 用): 单算子同时建模时空。Tier-2 合成的时空算子也归此类。
SPATIOTEMPORAL_OPS = {
    "st_separable": STSeparableConv,
    "st_graph_attn": STGraphAttn,
    "dynamic_gat": DynamicGAT,       # Stage2: 注意力引导动态图卷积
    "stjoint_conv": STJointConv,     # Stage2: 时空交替深度融合
}

# 算子类别 (spatial/temporal/spatiotemporal): 让 genotype/builder 正确放槽。
# synth 算子的类别由 registry 在注入时登记 (默认 spatiotemporal), 见 op_category()。
OP_CATEGORY: dict[str, str] = (
    {name: "spatial" for name in SPATIAL_OPS}
    | {name: "temporal" for name in TEMPORAL_OPS}
    | {name: "spatiotemporal" for name in SPATIOTEMPORAL_OPS}
)
# identity 同时在三表, 归为通用 (任何槽可用); 不强制类别
OP_CATEGORY["identity"] = "any"

# 需要 num_nodes 构造参数的空间/时序算子 (自学邻接类)。时空一体类 + synth_ 单独判。
NEEDS_NODES: set[str] = {"adaptive", "gwnet_adp"}

# 合成算子注册名前缀 (Tier-2 创造注入 SPATIAL_OPS 时加此前缀区分内置算子)。
# 定义在此 (而非 registry) 是为了让 builtin_op_signature 能排除 synth_ 算子且不引入
# operators→registry 的反向依赖; registry.py re-export 保持向后兼容。
SYNTH_PREFIX = "synth_"


def op_category(name: str) -> str:
    """查算子类别。内置查 OP_CATEGORY; synth_ 前缀查 registry 登记的类别, 兜底 spatiotemporal。"""
    if name in OP_CATEGORY:
        return OP_CATEGORY[name]
    if name.startswith(SYNTH_PREFIX):
        # registry 注入时把类别写进 OP_CATEGORY; 若没写 (旧算子) 默认时空一体 (诚实兜底)
        return "spatiotemporal"
    return "spatiotemporal"


def builtin_op_signature() -> str:
    """内置搜索空间的稳定短哈希 (排除 synth_ 运行时注入算子)。

    作用: 作为 ExperimentScope.space_version 的自动来源。**内置算子集或 fusion 集一变
    (Stage1→2→3), 哈希就变 → 新代实验自动落到独立作用域**, 从根上杜绝跨代记忆污染
    (Stage2 稀释 bug 的结构性修复)。

    稳定性铁律 (改这里前务必想清楚, 否则旧作用域会"漂移"):
      - 只哈希算子名 (keys), sorted, 排除 synth_ 前缀 (运行时注入不算搜索空间代际)。
      - 不哈希算子代码/dict 插入顺序/超参。
      - genotype schema 本身变化 (非算子名) 时, 手动 bump 下面的 "schema" 标记。
    """
    from darwin_st.search.genotype import VALID_FUSION  # 函数内 import 避免顶层环
    builtin = lambda d: sorted(n for n in d if not n.startswith(SYNTH_PREFIX))
    payload = {
        "spatial": builtin(SPATIAL_OPS),
        "temporal": builtin(TEMPORAL_OPS),
        "joint": builtin(SPATIOTEMPORAL_OPS),
        "fusion": sorted(VALID_FUSION),
        "schema": "v1",   # genotype schema 变了 (非算子名) 手动 bump
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def _all_ops() -> dict:
    """合并视图: 三类算子 + 已注入的 synth (synth 注册进 SPATIAL_OPS dict)。"""
    return {**SPATIAL_OPS, **TEMPORAL_OPS, **SPATIOTEMPORAL_OPS}


def build_op(name: str, dim: int, num_nodes: int | None = None, **kw) -> nn.Module:
    """统一算子工厂 (不分槽): 按名从合并视图实例化任意类别算子。

    adaptive / synth_ / spatiotemporal 类算子可能需要 num_nodes; 缺则报错。
    joint block 与未来 DAG 边用它; build_spatial_op/build_temporal_op 是它的分类 wrapper。
    """
    ops = _all_ops()
    if name not in ops:
        raise KeyError(f"未知算子 '{name}'. 可选: {sorted(ops)}")
    cls = ops[name]
    # 需 num_nodes 的算子: adaptive/gwnet_adp (自学邻接) + synth_ (合成契约) + 时空一体类
    needs_nodes = (name in NEEDS_NODES or name.startswith("synth_")
                   or name in SPATIOTEMPORAL_OPS)
    if needs_nodes:
        if num_nodes is None:
            raise ValueError(f"算子 '{name}' 需要 num_nodes")
        return cls(dim, num_nodes=num_nodes, **kw)
    return cls(dim, **kw)


def build_spatial_op(name: str, dim: int, num_nodes: int | None = None, **kw) -> nn.Module:
    """按名实例化空间算子。adaptive/gwnet_adp 与合成算子(synth_前缀)需要 num_nodes。

    放宽: 也接受时空一体算子坐空间槽 (类别兼容, 签名相同)。全部委托 build_op 统一处理
    (NEEDS_NODES / synth_ / 时空一体 的 num_nodes 判定是单一真源, 避免此处漂移)。
    """
    return build_op(name, dim, num_nodes=num_nodes, **kw)


def build_temporal_op(name: str, dim: int, num_nodes: int | None = None, **kw) -> nn.Module:
    """按名实例化时序算子。放宽: 也接受时空一体算子坐时序槽 (类别兼容)。委托 build_op 统一处理。"""
    return build_op(name, dim, num_nodes=num_nodes, **kw)
