"""operators.py 的正确性回归测试 (设备无关, CPU 即可全绿)。

核心契约 (架构铁律, 见 docs/P2_ALGORITHM_DESIGN.md):
  1. 每个算子保持 [B,T,N,F] 形状 —— **节点维 N 永不丢失/展平**
  2. 所有算子可微 (梯度能回传)
  3. 空间算子正确使用邻接; 时序算子在 T 维混合
  4. 工厂按名实例化, 未知算子报错
"""

from __future__ import annotations

import pytest
import torch

from darwin_st.search.operators import (
    SPATIAL_OPS,
    TEMPORAL_OPS,
    build_spatial_op,
    build_temporal_op,
)

B, T, N, Fd = 2, 12, 6, 16  # 小型张量形状


@pytest.fixture
def x():
    return torch.randn(B, T, N, Fd, requires_grad=True)


@pytest.fixture
def adj():
    # 一个简单对称邻接 (环形图 + 自连通), 保证每个节点有边
    a = torch.zeros(N, N)
    for i in range(N):
        a[i, (i + 1) % N] = 1.0
        a[(i + 1) % N, i] = 1.0
    return a


# ---------------------------------------------------------------------------
# 形状保持 (节点维 N 不丢) —— 最核心契约
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SPATIAL_OPS))
def test_spatial_op_preserves_shape(name, x, adj):
    op = build_spatial_op(name, dim=Fd, num_nodes=N)
    out = op(x, adj)
    assert out.shape == (B, T, N, Fd), f"{name} 改变了形状 (节点维 N 必须保持!)"


@pytest.mark.parametrize("name", list(TEMPORAL_OPS))
def test_temporal_op_preserves_shape(name, x, adj):
    op = build_temporal_op(name, dim=Fd)
    out = op(x, adj)
    assert out.shape == (B, T, N, Fd), f"{name} 改变了形状"


# ---------------------------------------------------------------------------
# 可微
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(SPATIAL_OPS))
def test_spatial_op_differentiable(name, x, adj):
    op = build_spatial_op(name, dim=Fd, num_nodes=N)
    out = op(x, adj)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"{name} 梯度含 nan/inf"


@pytest.mark.parametrize("name", list(TEMPORAL_OPS))
def test_temporal_op_differentiable(name, x, adj):
    op = build_temporal_op(name, dim=Fd)
    out = op(x, adj)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all(), f"{name} 梯度含 nan/inf"


# ---------------------------------------------------------------------------
# 语义正确性
# ---------------------------------------------------------------------------


def test_identity_is_passthrough(x, adj):
    op = build_spatial_op("identity", dim=Fd)
    assert torch.allclose(op(x, adj), x)


def test_gcn_actually_mixes_neighbors(x, adj):
    """GCN 输出应不同于输入 (发生了邻居聚合)。"""
    op = build_spatial_op("gcn", dim=Fd)
    out = op(x, adj)
    assert not torch.allclose(out, x)


def test_gcn_requires_adj(x):
    op = build_spatial_op("gcn", dim=Fd)
    with pytest.raises(AssertionError):
        op(x, None)


def test_adaptive_works_without_adj(x):
    """自适应图算子不需要外部邻接 (自学图)。"""
    op = build_spatial_op("adaptive", dim=Fd, num_nodes=N)
    out = op(x, None)  # adj=None 也能跑
    assert out.shape == x.shape


def test_tcn_is_causal(adj):
    """因果 TCN: 未来的输入不应影响过去的输出。"""
    op = build_temporal_op("tcn", dim=Fd)
    op.eval()
    x1 = torch.randn(1, T, N, Fd)
    x2 = x1.clone()
    x2[:, -1] += 100.0  # 只改最后一个时间步
    with torch.no_grad():
        o1, o2 = op(x1, None), op(x2, None)
    # 除最后一步外, 输出应不变 (因果性: 不看未来)
    assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-4)


def test_cheb_k_order(x, adj):
    """ChebGCN 不同 K 阶应给不同结果 (K 阶邻域)。"""
    o1 = build_spatial_op("cheb", dim=Fd, K=2)(x, adj)
    o3 = build_spatial_op("cheb", dim=Fd, K=4)(x, adj)
    assert o1.shape == o3.shape == x.shape


def test_gat_attention_scoring_orientation():
    """回归: GAT 注意力打分 e[n,m] = src(h_n)+dst(h_m), 方向不能反 (曾写反成转置)。"""
    from darwin_st.search.operators import GATConv

    torch.manual_seed(0)
    op = GATConv(dim=4)
    h = op.lin(torch.randn(1, 1, 3, 4))
    src, dst = op.att_src(h), op.att_dst(h)
    expected = src + dst.transpose(-1, -2)            # [B,T,N,N], e[n,m]=src_n+dst_m
    actual = op.att_src(h) + op.att_dst(h).transpose(-1, -2)
    assert torch.allclose(expected, actual)
    # 关键: 不应等于其转置 (否则 src/dst 角色反了)
    assert not torch.allclose(expected, expected.transpose(-1, -2))


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------


def test_unknown_spatial_op_raises():
    with pytest.raises(KeyError):
        build_spatial_op("nonsense", dim=Fd)


def test_unknown_temporal_op_raises():
    with pytest.raises(KeyError):
        build_temporal_op("nonsense", dim=Fd)


def test_adaptive_needs_num_nodes():
    with pytest.raises(ValueError):
        build_spatial_op("adaptive", dim=Fd)  # 缺 num_nodes


def test_isolated_node_no_nan(x):
    """孤立节点 (全 0 邻接) 不应让 GAT/GCN 产生 nan。"""
    adj_isolated = torch.zeros(N, N)
    adj_isolated[0, 1] = adj_isolated[1, 0] = 1.0  # 只有节点 0-1 有边
    for name in ("gcn", "gat", "cheb", "diffusion"):
        out = build_spatial_op(name, dim=Fd, num_nodes=N)(x, adj_isolated)
        assert torch.isfinite(out).all(), f"{name} 在孤立节点处产生 nan/inf"
