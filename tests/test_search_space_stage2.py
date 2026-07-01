"""Stage 2 搜索空间扩充测试: 新原子算子 + 变异池覆盖 (纳入 joint + 新算子)。

锁住四件事 + 向后兼容铁律:
  - 7 个新算子 (mixhop/gwnet_adp/multiscale_tcn/series_decomp_attn/ssm/dynamic_gat/stjoint_conv)
    保形 [B,T,N,F] + 反向有梯度 + 正确类别 + 无图退化 (不需 adj 的能 adj=None 跑)。
  - SSM 压轴: 长 T 扫描数值稳定 (谱半径<1) + 梯度不爆不消失。
  - 变异池覆盖: swap 采到类别兼容全算子 (含时空一体坐空间/时序槽); swap_joint/toggle_joint 生效。
  - add_block 可产 joint block; 大量变异全合法。
  - **golden-hash**: 旧线性 genotype signature 字节不变 (纯加法, 不破 graveyard dedup)。
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from darwin_st.search.builder import build_model
from darwin_st.search.evolution import (
    _joint_pool,
    _spatial_slot_pool,
    _temporal_slot_pool,
    random_mutation,
    seed_genotypes,
)
from darwin_st.search.genotype import Genotype, STBlock, _block_to_dict, mutate
from darwin_st.search.operators import (
    SPATIAL_OPS,
    SPATIOTEMPORAL_OPS,
    TEMPORAL_OPS,
    build_op,
    op_category,
)

B, T, N, Fd = 2, 12, 10, 32
ADJ = np.eye(N, dtype=np.float32)
X3 = torch.randn(2, 12, N, 3)

# Stage 2 新增的 7 个算子及其期望类别
NEW_OPS = {
    "mixhop": "spatial",
    "gwnet_adp": "spatial",
    "multiscale_tcn": "temporal",
    "series_decomp_attn": "temporal",
    "ssm": "temporal",
    "dynamic_gat": "spatiotemporal",
    "stjoint_conv": "spatiotemporal",
}
# 这些算子不依赖外部图, 应能 adj=None 跑
NO_GRAPH_OK = ["gwnet_adp", "multiscale_tcn", "series_decomp_attn", "ssm"]


# ---------------------------------------------------------------------------
# 新算子: 注册 + 类别
# ---------------------------------------------------------------------------


def test_new_ops_registered_in_correct_tables():
    """7 个新算子进对应注册表 (决定坐哪个槽)。"""
    assert "mixhop" in SPATIAL_OPS and "gwnet_adp" in SPATIAL_OPS
    for op in ("multiscale_tcn", "series_decomp_attn", "ssm"):
        assert op in TEMPORAL_OPS, f"{op} 不在 TEMPORAL_OPS"
    assert "dynamic_gat" in SPATIOTEMPORAL_OPS and "stjoint_conv" in SPATIOTEMPORAL_OPS


@pytest.mark.parametrize("name,cat", list(NEW_OPS.items()))
def test_new_op_category(name, cat):
    """每个新算子类别正确 (让 genotype/builder 放对槽)。"""
    assert op_category(name) == cat


# ---------------------------------------------------------------------------
# 新算子: 保形 + 可微 + 节点维 N 不丢 (架构铁律)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(NEW_OPS))
def test_new_op_preserves_shape_and_grad(name):
    """[B,T,N,F]→同形 (N 不丢) + 反向有有限梯度。"""
    x = torch.randn(B, T, N, Fd, requires_grad=True)
    adj = torch.eye(N) + torch.rand(N, N) * 0.3
    op = build_op(name, dim=Fd, num_nodes=N)
    out = op(x, adj)
    assert out.shape == (B, T, N, Fd), f"{name} 改变形状 (节点维 N 必须保持!)"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), f"{name} 梯度含 nan/inf"


@pytest.mark.parametrize("name", NO_GRAPH_OK)
def test_new_op_no_graph_degradation(name):
    """不依赖外部图的算子: adj=None 也能正常前向 (退化不崩)。"""
    x = torch.randn(B, T, N, Fd)
    op = build_op(name, dim=Fd, num_nodes=N)
    assert op(x, None).shape == (B, T, N, Fd)


# ---------------------------------------------------------------------------
# SSM 压轴: 数值稳定 + 梯度健康 (顺序扫描需专门验证)
# ---------------------------------------------------------------------------


def test_ssm_long_horizon_stable():
    """长 T=48 扫描输出有限 (a=sigmoid∈(0,1) 保谱半径<1, 不发散)。"""
    x = torch.randn(2, 48, N, Fd)
    op = build_op("ssm", dim=Fd, num_nodes=N)
    out = op(x, None)
    assert out.shape == x.shape
    assert torch.isfinite(out).all(), "SSM 长程输出发散"


def test_ssm_gradient_healthy_through_scan():
    """T=24 扫描的梯度不爆不消失 (真 MSE loss, 非退化 sum)。"""
    torch.manual_seed(0)
    x = torch.randn(4, 24, N, Fd, requires_grad=True)
    tgt = torch.randn(4, 24, N, Fd)
    op = build_op("ssm", dim=Fd, num_nodes=N)
    loss = ((op(x, None) - tgt) ** 2).mean()
    loss.backward()
    pgn = torch.cat([p.grad.flatten() for p in op.parameters()]).norm().item()
    xgn = x.grad.norm().item()
    assert 1e-5 < pgn < 100, f"SSM 参数梯度不健康: {pgn}"
    assert 1e-5 < xgn < 100, f"SSM 输入梯度不健康: {xgn}"


# ---------------------------------------------------------------------------
# 端到端: 新算子在正确槽内编译 + 前向
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["mixhop", "gwnet_adp"])
def test_new_spatial_op_builds_model(op):
    g = Genotype(blocks=[STBlock(op, "tcn")], hidden=32)
    out = build_model(g, num_nodes=N, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)(X3)
    assert out.shape == (2, 12, N)


@pytest.mark.parametrize("op", ["multiscale_tcn", "series_decomp_attn", "ssm"])
def test_new_temporal_op_builds_model(op):
    g = Genotype(blocks=[STBlock("gcn", op)], hidden=32)
    out = build_model(g, num_nodes=N, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)(X3)
    assert out.shape == (2, 12, N)


@pytest.mark.parametrize("op", ["dynamic_gat", "stjoint_conv"])
def test_new_joint_op_builds_model(op):
    g = Genotype(blocks=[STBlock("identity", "identity", joint_op=op)], hidden=32)
    out = build_model(g, num_nodes=N, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)(X3)
    assert out.shape == (2, 12, N)


# ---------------------------------------------------------------------------
# 变异池覆盖 (关键 gap: 否则新算子白加)
# ---------------------------------------------------------------------------


def test_slot_pools_include_new_ops():
    """类别兼容池: 新原子算子 + 时空一体算子都进得了对应槽。"""
    sp, tp, jp = _spatial_slot_pool(), _temporal_slot_pool(), _joint_pool()
    assert "mixhop" in sp and "gwnet_adp" in sp
    assert {"multiscale_tcn", "series_decomp_attn", "ssm"} <= set(tp)
    assert {"dynamic_gat", "stjoint_conv"} <= set(jp)
    # 放宽: 时空一体算子可坐空间/时序槽 (类别兼容)
    assert "st_separable" in sp and "st_separable" in tp
    assert "dynamic_gat" in sp and "dynamic_gat" in tp


def test_mutation_reaches_new_ops_and_joint():
    """3000 次变异: 采到 ≥5 个新算子 + 产出 joint block + 全部合法。"""
    rng = random.Random(0)
    g = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)
    seen_ops, joint_count = set(), 0
    for _ in range(3000):
        child = random_mutation(g, rng)
        child.validate()  # 每个子代必须合法 (否则抛错)
        for b in child.blocks:
            seen_ops.update((b.spatial_op, b.temporal_op))
            if b.joint_op is not None:
                joint_count += 1
                seen_ops.add(b.joint_op)
    hit = set(NEW_OPS) & seen_ops
    assert len(hit) >= 5, f"变异只采到 {sorted(hit)}, 新算子覆盖不足"
    assert joint_count >= 1, "变异从未产出 joint block"


def test_swap_joint_mutation():
    """swap_joint: 一体模式块能换时空算子; 分离模式块拒绝 (无 joint 可换)。"""
    gj = Genotype(blocks=[STBlock("identity", "identity", joint_op="st_separable")], hidden=64)
    g2 = mutate(gj, "swap_joint", index=0, new_op="dynamic_gat")
    assert g2.blocks[0].joint_op == "dynamic_gat"
    # 分离模式块无 joint 算子可换
    gl = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)
    with pytest.raises(ValueError):
        mutate(gl, "swap_joint", index=0, new_op="dynamic_gat")


def test_toggle_joint_mutation_round_trip():
    """toggle_joint: 分离→一体→分离 往返 (进出一体模式)。"""
    gl = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)
    gj = mutate(gl, "toggle_joint", index=0, new_op="stjoint_conv")
    assert gj.blocks[0].joint_op == "stjoint_conv"
    gback = mutate(gj, "toggle_joint", index=0)
    assert gback.blocks[0].joint_op is None   # 退回分离模式


def test_seed_genotypes_valid_with_new_pool():
    """bootstrap 种子用扩充池仍全部合法编译。"""
    rng = random.Random(1)
    for g in seed_genotypes(20, rng):
        g.validate()
        # 能编译前向 (个别可能需 adj, 给一个)
        out = build_model(g, num_nodes=N, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)(X3)
        assert out.shape == (2, 12, N)


# ---------------------------------------------------------------------------
# 向后兼容 (golden-hash): 纯加法不破旧签名
# ---------------------------------------------------------------------------


def test_old_linear_signature_unchanged_after_stage2():
    """铁律: Stage 2 纯加法, 旧线性 genotype signature 字节不变 (graveyard dedup 安全)。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn", "sequential")], hidden=64)
    assert _block_to_dict(g.blocks[0]) == {
        "spatial_op": "gcn", "temporal_op": "tcn", "fusion": "sequential"}
    assert "joint_op" not in str(g.to_dict())
    # 这是升级前 gcn+tcn+sequential+默认embedding+hidden64 的固定签名 (与 Stage1 测试同源)
    assert g.signature() == "17104fc6b3688f33a018c72f4a40ce310ffbc0ab"


def test_protected_scans_joint_slot():
    """保护区扫描含 joint 槽: joint 位的保护算子不可被 swap/toggle 掉。"""
    g = Genotype(blocks=[STBlock("identity", "identity", joint_op="stjoint_conv")],
                 hidden=64, protected=["stjoint_conv"])
    g.validate()  # protected 在 joint 槽应被 _uses_op 认到 (不报"未使用")
    with pytest.raises(ValueError):
        mutate(g, "swap_joint", index=0, new_op="dynamic_gat")  # 保护区拒换
    with pytest.raises(ValueError):
        mutate(g, "toggle_joint", index=0)                      # 保护区拒退出一体
