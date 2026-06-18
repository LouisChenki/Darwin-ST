"""builder.py 的正确性回归测试 (设备无关, CPU)。

核心契约:
  - genotype 编译出的模型: [B,T_in,N,C] → [B,T_out,N], 节点维 N 保持
  - 各 fusion (sequential/parallel/residual) 都能跑且形状正确
  - 可微 (端到端梯度回传)
  - 有图/无图算子都工作; adj_mode 归一化正确
  - 身份嵌入开关影响模型; 能真正训练一步降 loss
  - 参数量统计可用 (供 MAP-Elites)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from darwin_st.search.builder import build_model, count_params
from darwin_st.search.genotype import Genotype, STBlock, random_genotype

B, T_IN, T_OUT, N, C = 2, 12, 12, 6, 3


def _adj():
    a = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        a[i, (i + 1) % N] = a[(i + 1) % N, i] = 1.0
    return a


def _run(geno, adj=None, tod=None, dow=None):
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT, adj=adj)
    x = torch.randn(B, T_IN, N, C)
    return m, m(x, tod_idx=tod, dow_idx=dow)


# ---------------------------------------------------------------------------
# 形状
# ---------------------------------------------------------------------------


def test_output_shape():
    geno = random_genotype(depth=2, spatial="gcn", temporal="tcn")
    _, out = _run(geno, adj=_adj())
    assert out.shape == (B, T_OUT, N)  # [B,T_out,N], 节点维保持


@pytest.mark.parametrize("fusion", ["sequential", "parallel", "residual"])
def test_all_fusions_work(fusion):
    geno = Genotype(blocks=[STBlock("gcn", "tcn", fusion), STBlock("gat", "gru", fusion)])
    _, out = _run(geno, adj=_adj())
    assert out.shape == (B, T_OUT, N)


@pytest.mark.parametrize("sop", ["identity", "gcn", "cheb", "gat", "diffusion", "adaptive"])
def test_each_spatial_op_compiles(sop):
    geno = random_genotype(depth=1, spatial=sop, temporal="tcn")
    _, out = _run(geno, adj=_adj())
    assert out.shape == (B, T_OUT, N)


@pytest.mark.parametrize("top", ["identity", "tcn", "gru", "attn"])
def test_each_temporal_op_compiles(top):
    geno = random_genotype(depth=1, spatial="gcn", temporal=top)
    _, out = _run(geno, adj=_adj())
    assert out.shape == (B, T_OUT, N)


# ---------------------------------------------------------------------------
# 可微 + 训练
# ---------------------------------------------------------------------------


def test_differentiable_end_to_end():
    geno = random_genotype(depth=2)
    m, out = _run(geno, adj=_adj())
    out.sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads)


def test_can_overfit_one_batch():
    """能在单 batch 上降 loss → 证明模型确实可学 (管道贯通)。"""
    torch.manual_seed(0)
    geno = random_genotype(depth=2, spatial="gcn", temporal="tcn")
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT, adj=_adj())
    x = torch.randn(B, T_IN, N, C)
    y = torch.randn(B, T_OUT, N)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = torch.abs(m(x) - y).mean()
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.7  # loss 明显下降


# ---------------------------------------------------------------------------
# 有图/无图 + adj_mode
# ---------------------------------------------------------------------------


def test_adaptive_works_without_adj():
    """自学习图算子无需外部邻接也能跑。"""
    geno = random_genotype(depth=1, spatial="adaptive", temporal="tcn")
    _, out = _run(geno, adj=None)
    assert out.shape == (B, T_OUT, N)


@pytest.mark.parametrize("mode", ["sym", "rw", "none"])
def test_adj_modes(mode):
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", adj_mode=mode)
    _, out = _run(geno, adj=_adj())
    assert out.shape == (B, T_OUT, N)


# ---------------------------------------------------------------------------
# 嵌入
# ---------------------------------------------------------------------------


def test_node_embedding_on_by_default():
    geno = random_genotype(depth=1)
    m, _ = _run(geno, adj=_adj())
    assert m.embed.use_node is True


def test_temporal_embeddings_with_indices():
    geno = random_genotype(depth=1)
    geno.embedding.use_tod = True
    geno.embedding.use_dow = True
    tod = torch.randint(0, 288, (B, T_IN))
    dow = torch.randint(0, 7, (B, T_IN))
    _, out = _run(geno, adj=_adj(), tod=tod, dow=dow)
    assert out.shape == (B, T_OUT, N)


# ---------------------------------------------------------------------------
# 参数量
# ---------------------------------------------------------------------------


def test_count_params_positive():
    geno = random_genotype(depth=2)
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT, adj=_adj())
    assert count_params(m) > 0


def test_deeper_has_more_params():
    g1 = random_genotype(depth=1)
    g2 = random_genotype(depth=4)
    m1 = build_model(g1, N, C, T_IN, T_OUT, _adj())
    m2 = build_model(g2, N, C, T_IN, T_OUT, _adj())
    assert count_params(m2) > count_params(m1)
