"""身份嵌入 (Identity Embeddings) —— 接近 SOTA 的最高杠杆组件。

研究结论 (docs/P2_ALGORITHM_DESIGN.md §0): 节点/时间身份嵌入比图算子的选择更能提升精度。
  - 去掉 STID 的空间节点嵌入 → MAE +18% (18.29→21.65)
  - 去掉 STAEformer 的自适应嵌入 → MAE +19% (18.22→21.63)
不补这些, 搜出的架构无论图算子多好都到不了 SOTA 档。

三类嵌入 (STID arXiv:2208.05233 / STAEformer arXiv:2308.10425):
  - SpatialNodeEmbedding:  每个传感器一个可学习向量 E[N,D]。**最关键且免费**——
    仅需节点位置索引 0..N-1, 始终可用, 无需改数据管道。
  - TimeOfDayEmbedding:    一天 288 个 5 分钟槽位的查找表 [288,D]。需时间索引。
  - DayOfWeekEmbedding:    一周 7 天的查找表 [7,D]。需时间索引。

注入方式 = **拼接 (concatenation)**, 非相加 (STID/STAEformer 均拼接), 接在输入投影处。
张量约定: 与 operators 一致 [B,T,N,F]。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "SpatialNodeEmbedding",
    "TimeOfDayEmbedding",
    "DayOfWeekEmbedding",
    "STEmbedding",
    "STEPS_PER_DAY",
    "DAYS_PER_WEEK",
]

STEPS_PER_DAY = 288   # 5 分钟采样 → 一天 24*60/5 = 288 个时间槽
DAYS_PER_WEEK = 7


class SpatialNodeEmbedding(nn.Module):
    """空间节点嵌入 E[N,D]: 每个传感器一个可学习向量, 解决"空间不可区分"瓶颈。

    **免费的最大杠杆**——只需节点位置(0..N-1), 始终可用。广播到 [B,T,N,D]。
    """

    def __init__(self, num_nodes: int, dim: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.dim = dim
        self.emb = nn.Parameter(torch.empty(num_nodes, dim))
        nn.init.xavier_uniform_(self.emb)

    def forward(self, B: int, T: int) -> torch.Tensor:
        # 张量变换: [N,D] -> 广播 [B,T,N,D]
        return self.emb.unsqueeze(0).unsqueeze(0).expand(B, T, self.num_nodes, self.dim)


class TimeOfDayEmbedding(nn.Module):
    """一天内时刻嵌入 [288,D]: 同一时刻(如早高峰)跨天共享语义。需时间索引。"""

    def __init__(self, dim: int, steps_per_day: int = STEPS_PER_DAY):
        super().__init__()
        self.steps_per_day = steps_per_day
        self.emb = nn.Embedding(steps_per_day, dim)

    def forward(self, tod_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """tod_idx: [B,T] 整数 (0..287)。返回 [B,T,N,D] (跨节点共享)。"""
        e = self.emb(tod_idx.long())                          # [B,T,D]
        return e.unsqueeze(2).expand(-1, -1, num_nodes, -1)   # 广播到 N


class DayOfWeekEmbedding(nn.Module):
    """星期几嵌入 [7,D]: 工作日/周末模式差异。需时间索引。"""

    def __init__(self, dim: int, days_per_week: int = DAYS_PER_WEEK):
        super().__init__()
        self.days_per_week = days_per_week
        self.emb = nn.Embedding(days_per_week, dim)

    def forward(self, dow_idx: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """dow_idx: [B,T] 整数 (0..6)。返回 [B,T,N,D]。"""
        e = self.emb(dow_idx.long())                          # [B,T,D]
        return e.unsqueeze(2).expand(-1, -1, num_nodes, -1)   # 广播到 N


class STEmbedding(nn.Module):
    """复合身份嵌入: 把启用的嵌入与输入投影【拼接】, 产出统一隐藏表示。

    流程 (STID 式):
      H = Linear(x)                       # 输入投影到 hidden  [B,T,N,hidden]
      Z = concat(H, E_node?, E_tod?, E_dow?)  沿特征维拼接
      out = Linear(Z) -> [B,T,N,hidden]   # 投影回 hidden 供后续算子

    可配置开关: use_node / use_tod / use_dow。各嵌入维度独立。
    时间索引 (tod_idx/dow_idx) 缺失时, 对应时间嵌入自动跳过 (优雅降级)。
    """

    def __init__(
        self, in_channels: int, hidden: int, num_nodes: int,
        use_node: bool = True, node_dim: int = 32,
        use_tod: bool = False, tod_dim: int = 32,
        use_dow: bool = False, dow_dim: int = 32,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.use_node, self.use_tod, self.use_dow = use_node, use_tod, use_dow

        self.input_proj = nn.Linear(in_channels, hidden)
        cat_dim = hidden
        if use_node:
            self.node_emb = SpatialNodeEmbedding(num_nodes, node_dim)
            cat_dim += node_dim
        if use_tod:
            self.tod_emb = TimeOfDayEmbedding(tod_dim)
            cat_dim += tod_dim
        if use_dow:
            self.dow_emb = DayOfWeekEmbedding(dow_dim)
            cat_dim += dow_dim
        # 拼接后投影回 hidden (融合各身份信号)
        self.fuse = nn.Linear(cat_dim, hidden)

    def forward(
        self, x: torch.Tensor, tod_idx: torch.Tensor | None = None,
        dow_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, N, _ = x.shape
        assert N == self.num_nodes, f"节点数不符: x 有 {N}, 嵌入按 {self.num_nodes} 建"
        parts = [self.input_proj(x)]                          # [B,T,N,hidden]
        if self.use_node:
            parts.append(self.node_emb(B, T))                 # [B,T,N,node_dim]
        if self.use_tod:
            assert tod_idx is not None, "use_tod=True 但未提供 tod_idx"
            parts.append(self.tod_emb(tod_idx, N))            # [B,T,N,tod_dim]
        if self.use_dow:
            assert dow_idx is not None, "use_dow=True 但未提供 dow_idx"
            parts.append(self.dow_emb(dow_idx, N))            # [B,T,N,dow_dim]
        z = torch.cat(parts, dim=-1)                          # 沿特征维拼接
        return self.fuse(z)                                   # [B,T,N,hidden]
