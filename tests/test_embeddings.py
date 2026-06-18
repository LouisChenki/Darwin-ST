"""embeddings.py 的正确性回归测试 (设备无关)。

核心契约:
  - 各嵌入输出形状正确, 广播到 [B,T,N,D]
  - STEmbedding 拼接逻辑正确, 输出 [B,T,N,hidden], 可微
  - 开关组合 (node/tod/dow) 都工作; 时间嵌入缺索引时报错
  - 空间节点嵌入"免费"(无需时间索引), 不同节点学到不同向量
"""

from __future__ import annotations

import pytest
import torch

from darwin_st.search.embeddings import (
    DayOfWeekEmbedding,
    STEmbedding,
    SpatialNodeEmbedding,
    TimeOfDayEmbedding,
)

B, T, N, C, H = 2, 12, 6, 3, 32


def test_spatial_node_embedding_shape():
    emb = SpatialNodeEmbedding(num_nodes=N, dim=16)
    out = emb(B, T)
    assert out.shape == (B, T, N, 16)


def test_spatial_node_embedding_distinct_per_node():
    """不同节点应有不同嵌入 (打破空间不可区分)。"""
    emb = SpatialNodeEmbedding(num_nodes=N, dim=16)
    out = emb(1, 1)[0, 0]  # [N,16]
    # 至少存在两个节点的嵌入不同
    assert not torch.allclose(out[0], out[1])


def test_spatial_node_embedding_broadcast_consistent():
    """同一节点跨 batch/time 的嵌入一致 (只依赖节点身份)。"""
    emb = SpatialNodeEmbedding(num_nodes=N, dim=8)
    out = emb(B, T)
    assert torch.allclose(out[0, 0, 2], out[1, 5, 2])  # 节点2 处处相同


def test_tod_embedding_shape():
    emb = TimeOfDayEmbedding(dim=16)
    tod = torch.randint(0, 288, (B, T))
    out = emb(tod, num_nodes=N)
    assert out.shape == (B, T, N, 16)


def test_dow_embedding_shape():
    emb = DayOfWeekEmbedding(dim=8)
    dow = torch.randint(0, 7, (B, T))
    out = emb(dow, num_nodes=N)
    assert out.shape == (B, T, N, 8)


def test_tod_shared_across_nodes():
    """同一时刻的嵌入跨节点共享 (time-of-day 不依赖节点)。"""
    emb = TimeOfDayEmbedding(dim=8)
    tod = torch.zeros(1, 1, dtype=torch.long)  # 全是时刻 0
    out = emb(tod, num_nodes=N)[0, 0]  # [N,8]
    assert torch.allclose(out[0], out[3])  # 各节点相同


# ---------------------------------------------------------------------------
# STEmbedding 复合
# ---------------------------------------------------------------------------


def test_st_embedding_node_only():
    """默认只开节点嵌入(免费), 输出 [B,T,N,hidden], 不需时间索引。"""
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N, use_node=True)
    x = torch.randn(B, T, N, C)
    out = st(x)  # 无 tod/dow 也能跑
    assert out.shape == (B, T, N, H)


def test_st_embedding_all_on():
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N,
                     use_node=True, use_tod=True, use_dow=True)
    x = torch.randn(B, T, N, C)
    tod = torch.randint(0, 288, (B, T))
    dow = torch.randint(0, 7, (B, T))
    out = st(x, tod_idx=tod, dow_idx=dow)
    assert out.shape == (B, T, N, H)


def test_st_embedding_differentiable():
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N, use_node=True)
    x = torch.randn(B, T, N, C, requires_grad=True)
    out = st(x)
    out.sum().backward()
    assert x.grad is not None
    # 节点嵌入参数也应收到梯度
    assert st.node_emb.emb.grad is not None
    assert torch.isfinite(st.node_emb.emb.grad).all()


def test_st_embedding_tod_requires_index():
    """开了 use_tod 却不给 tod_idx 应报错。"""
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N, use_tod=True)
    x = torch.randn(B, T, N, C)
    with pytest.raises(AssertionError):
        st(x)  # 缺 tod_idx


def test_st_embedding_node_count_mismatch():
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N, use_node=True)
    x = torch.randn(B, T, N + 1, C)  # 节点数不符
    with pytest.raises(AssertionError):
        st(x)


def test_st_embedding_no_embeddings():
    """全关时退化为纯输入投影 (仍输出 hidden)。"""
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N,
                     use_node=False, use_tod=False, use_dow=False)
    x = torch.randn(B, T, N, C)
    out = st(x)
    assert out.shape == (B, T, N, H)


def test_st_embedding_improves_node_distinguishability():
    """开节点嵌入后, 相同输入下不同节点的表示应可区分 (这正是 +18% 的来源)。"""
    torch.manual_seed(0)
    st = STEmbedding(in_channels=C, hidden=H, num_nodes=N, use_node=True)
    # 所有节点喂完全相同的输入
    x = torch.randn(B, T, 1, C).expand(B, T, N, C).contiguous()
    out = st(x)  # [B,T,N,H]
    # 尽管输入相同, 节点嵌入应让输出在节点间不同
    assert not torch.allclose(out[0, 0, 0], out[0, 0, 1])
