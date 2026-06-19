"""creation 层的正确性回归测试 (设备无关, CPU)。

核心契约:
  - ZeroInitResidualFusion: init 时是 no-op (y==baseline), 但 branch 可学 (最强守卫)
  - validate_operator: 放行好算子, 拒绝坏算子(不可微/错形状/NaN/平凡/超大)
  - 契约校验 (FusionPlan/4算子文法)
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from darwin_st.creation import (
    COMPOSITION_OPS,
    FusionPlan,
    FusionRequest,
    GatedFusion,
    ValidationConfig,
    ZeroInitResidualFusion,
    validate_operator,
)
from darwin_st.knowledge import all_seed_mechanisms

B, T, N, C = 2, 6, 5, 16


# 一些测试用算子 (forward(x, adj) -> [B,T,N,C])
class _GoodOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()
        self.lin = nn.Linear(channels, channels)

    def forward(self, x, adj=None):
        return self.lin(x)


class _IdentityOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()

    def forward(self, x, adj=None):
        return x  # 平凡: 恒等


class _ConstantOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()
        self.channels = channels

    def forward(self, x, adj=None):
        return torch.ones_like(x) * 3.0  # 平凡: 常数


class _WrongShapeOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()
        self.lin = nn.Linear(channels, channels)

    def forward(self, x, adj=None):
        return self.lin(x).mean(dim=2)  # 丢了节点维 N → [B,T,C]


class _NonDifferentiableOp(nn.Module):
    """破图算子: detach 切断梯度 → 输入收不到梯度 (退化/不可学)。"""
    def __init__(self, channels, num_nodes):
        super().__init__()
        self.lin = nn.Linear(channels, channels)

    def forward(self, x, adj=None):
        return self.lin(x.detach())  # detach 切断对输入的梯度


class _NaNOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()

    def forward(self, x, adj=None):
        return x / 0.0  # inf/nan


class _HugeOp(nn.Module):
    def __init__(self, channels, num_nodes):
        super().__init__()
        self.big = nn.Linear(channels, 5_000_000 // channels + channels)
        self.back = nn.Linear(5_000_000 // channels + channels, channels)

    def forward(self, x, adj=None):
        return self.back(torch.relu(self.big(x)))


# ---------------------------------------------------------------------------
# 零初始化残差融合 (最强守卫)
# ---------------------------------------------------------------------------


def test_zero_init_is_noop():
    """init 时 y == baseline(x) 精确相等 (坏融合此刻无害)。"""
    base = _GoodOp(C, N)
    branch = _GoodOp(C, N)
    fuse = ZeroInitResidualFusion(base, branch)
    x = torch.randn(B, T, N, C)
    assert fuse.is_noop_at_init(x)
    assert torch.allclose(fuse(x), base(x), atol=1e-6)


def test_zero_init_branch_learnable():
    """ReZero 正确动力学: init 时 α=0 → branch 梯度为 0 (数学必然), 但 α 自身有梯度,
    α 一旦被更新成非 0, branch 随后即可学。这正是'安全启动'性质: 坏分支不会一上来就捣乱。
    """
    base, branch = _GoodOp(C, N), _GoodOp(C, N)
    fuse = ZeroInitResidualFusion(base, branch)
    x = torch.randn(B, T, N, C)
    fuse(x).sum().backward()
    # α 收到梯度 (= branch(x) 的和), 会被优化器推离 0
    assert fuse.alpha.grad is not None
    assert fuse.alpha.grad.abs().sum() > 0
    # 手动把 α 推离 0, 再验证 branch 此后能收到梯度
    with torch.no_grad():
        fuse.alpha += 0.1
    fuse.zero_grad()
    fuse(x).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()), \
        "α 非 0 后 branch 仍无梯度 (融合断链)"


def test_zero_init_per_channel():
    fuse = ZeroInitResidualFusion(_GoodOp(C, N), _GoodOp(C, N), per_channel=True, channels=C)
    x = torch.randn(B, T, N, C)
    assert fuse.is_noop_at_init(x)
    assert fuse.alpha.shape == (C,)


def test_gated_fusion_near_noop_at_init():
    """门控融合 init 时门近 0 → 近似 no-op。"""
    base, branch = _GoodOp(C, N), _GoodOp(C, N)
    fuse = GatedFusion(base, branch, channels=C)
    x = torch.randn(B, T, N, C)
    # 门 sigmoid(-3)≈0.047, 输出接近 baseline (不要求精确)
    diff = (fuse(x) - base(x)).abs().mean().item()
    base_scale = base(x).abs().mean().item()
    assert diff < 0.1 * base_scale + 1e-3  # 偏离基线很小


def test_fusion_preserves_shape():
    fuse = ZeroInitResidualFusion(_GoodOp(C, N), _GoodOp(C, N))
    out = fuse(torch.randn(B, T, N, C))
    assert out.shape == (B, T, N, C)


# ---------------------------------------------------------------------------
# 验证 harness (放行好的, 拒绝坏的)
# ---------------------------------------------------------------------------


def test_validate_accepts_good_op():
    rep = validate_operator(_GoodOp)
    assert rep.passed, f"好算子被拒: {rep.gate} {rep.reason}"
    assert rep.gate == "all"
    assert rep.num_params > 0


def test_validate_rejects_identity():
    rep = validate_operator(_IdentityOp)
    assert not rep.passed
    assert rep.gate == "trivial_identity"


def test_validate_rejects_constant():
    rep = validate_operator(_ConstantOp)
    assert not rep.passed
    assert rep.gate in ("trivial_constant", "trivial_identity")


def test_validate_rejects_wrong_shape():
    rep = validate_operator(_WrongShapeOp)
    assert not rep.passed
    assert rep.gate == "shape"  # 丢了节点维 N


def test_validate_rejects_non_differentiable():
    rep = validate_operator(_NonDifferentiableOp)
    assert not rep.passed
    assert rep.gate in ("disconnected", "gradcheck")  # detach 破图 → 输入无梯度


def test_validate_rejects_nan():
    rep = validate_operator(_NaNOp)
    assert not rep.passed
    assert rep.gate in ("nan_forward", "forward", "trivial_constant")


def test_validate_rejects_huge():
    rep = validate_operator(_HugeOp, ValidationConfig(max_params=1_000_000))
    assert not rep.passed
    assert rep.gate == "param_ceiling"


def test_validate_fusion_passes():
    """零初始化融合算子本身应通过验证 (init no-op 但非平凡 —— α 会被训练)。

    注: init 时融合输出 = baseline, 非平凡(baseline 是线性变换), 故应过门。
    """
    def fusion_factory(channels, num_nodes):
        return ZeroInitResidualFusion(_GoodOp(channels, num_nodes), _GoodOp(channels, num_nodes))
    rep = validate_operator(fusion_factory)
    assert rep.passed, f"{rep.gate}: {rep.reason}"


# ---------------------------------------------------------------------------
# 契约
# ---------------------------------------------------------------------------


def test_fusion_request_to_prompt_context():
    ms = all_seed_mechanisms()[:2]
    req = FusionRequest(bottleneck="长程依赖+标签稀缺",
                        target_preconditions=["long_range_dependency", "label_scarcity"],
                        mechanisms=ms)
    ctx = req.to_prompt_context()
    assert ctx["bottleneck"]
    assert len(ctx["mechanisms"]) == 2
    # 每个机制陈述含因果 (按"为何有效"融合)
    assert "causal_behavior" in ctx["mechanisms"][0]
    assert "addresses_preconditions" in ctx["mechanisms"][0]


def test_fusion_plan_valid():
    plan = FusionPlan(operator_name="masked_adaptive_op", rationale="...",
                      shared_structure="...", composition="additive_residual",
                      source_mechanisms=["masked_autoencoding", "adaptive_adjacency"],
                      expected_effect="...")
    plan.validate()  # 不抛即合法


def test_fusion_plan_bad_composition():
    with pytest.raises(ValueError):
        FusionPlan(operator_name="x", rationale="", shared_structure="",
                   composition="frankenstein",  # 不在 4 算子文法
                   source_mechanisms=["a"], expected_effect="").validate()


def test_fusion_plan_bad_name():
    with pytest.raises(ValueError):
        FusionPlan(operator_name="123 not valid", rationale="", shared_structure="",
                   composition="sequential", source_mechanisms=["a"], expected_effect="").validate()


def test_composition_grammar_is_four():
    assert COMPOSITION_OPS == {"sequential", "parallel", "additive_residual", "gated_routed"}
