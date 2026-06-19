"""零初始化残差融合 (Zero-Init Residual Fusion) —— 组合式创造的最强防退化守卫。

研究结论 (docs/TIER2_RESEARCH_FINDINGS.md §7, 证据强度最高): 把融合算子写成
    y = baseline(x) + α · new_branch(x),   α = nn.Parameter(0)
init 时 y = baseline(x) **精确相等** → 坏融合此刻是 no-op, 伤不了基线;
但 branch 梯度 ≠ 0 → 网络能从 0 学起(融合好), 或保持 0 忽略(融合差)。
证据: ReZero(arXiv:2003.04887) / ControlNet / LoRA / LayerScale。

诚实边界: "init 不退化"是定理; "训练后不退化"无全局界 —— 只能说"模型能学会
忽略坏分支", 不能说"保证 ≥ baseline"。最终裁判仍是真实 masked-MAE 评测。

张量约定: 与 operators 一致 [B,T,N,C]。baseline 与 branch 都须保持该形状进出。
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["ZeroInitResidualFusion", "GatedFusion"]


class ZeroInitResidualFusion(nn.Module):
    """y = baseline(x) + α·branch(x), α 初始为 0 (标量, ReZero 式)。

    baseline: 已知有效的基线算子 (如某个 KEEP 的算子)。
    branch:   LLM 融合出的新分支 (待验证的创造)。
    两者都接受 (x, adj) 并返回同形状 [B,T,N,C]。
    """

    def __init__(self, baseline: nn.Module, branch: nn.Module, per_channel: bool = False,
                 channels: int | None = None):
        super().__init__()
        self.baseline = baseline
        self.branch = branch
        if per_channel:
            # LayerScale 式: 每通道一个 α (略优于纯标量, 需知 channels)
            assert channels is not None, "per_channel=True 需提供 channels"
            self.alpha = nn.Parameter(torch.zeros(channels))
        else:
            self.alpha = nn.Parameter(torch.zeros(1))  # ReZero 式标量
        self.per_channel = per_channel

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        base = self.baseline(x, adj)
        br = self.branch(x, adj)
        if self.per_channel:
            # alpha: [C] 广播到 [B,T,N,C]
            a = self.alpha.view(1, 1, 1, -1)
        else:
            a = self.alpha
        out = base + a * br
        assert out.shape == base.shape, f"融合形状漂移: {out.shape} vs {base.shape}"
        return out

    @torch.no_grad()
    def is_noop_at_init(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> bool:
        """验证 init 时 y == baseline(x) (α=0 的 no-op 性质)。"""
        return torch.allclose(self.forward(x, adj), self.baseline(x, adj), atol=1e-6)


class GatedFusion(nn.Module):
    """数据相关门控融合: y = baseline(x) + g(x)·branch(x), 门 g 由输入决定。

    比标量 α 更灵活(Highway 式), 门初始偏置设负 → sigmoid(g)≈0 → init 近 no-op。
    用于"是否启用新分支取决于输入"的场景。
    """

    def __init__(self, baseline: nn.Module, branch: nn.Module, channels: int,
                 init_bias: float = -3.0):
        super().__init__()
        self.baseline = baseline
        self.branch = branch
        self.gate = nn.Linear(channels, channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, init_bias)  # sigmoid(-3)≈0.047, init 近 no-op

    def forward(self, x: torch.Tensor, adj: torch.Tensor | None = None) -> torch.Tensor:
        base = self.baseline(x, adj)
        br = self.branch(x, adj)
        g = torch.sigmoid(self.gate(x))   # [B,T,N,C] 门控
        return base + g * br
