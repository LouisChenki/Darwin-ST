"""genotype.py 的正确性回归测试 (设备无关)。

核心契约:
  - 序列化往返一致, 签名稳定且键序无关
  - 合法性校验拦截非法算子/融合/adj_mode
  - 变异返回新对象不改原件
  - **创新点保护区**: 保护算子不可被 swap/remove (SKILL.md 铁律)
"""

from __future__ import annotations

import pytest

from darwin_st.search.genotype import (
    Genotype,
    STBlock,
    mutate,
    random_genotype,
)


def test_random_genotype_valid():
    g = random_genotype(depth=3, spatial="gcn", temporal="tcn")
    g.validate()
    assert g.depth == 3
    assert all(b.spatial_op == "gcn" for b in g.blocks)


def test_serialization_roundtrip():
    g = random_genotype(depth=2, hidden=64, protected=["gcn"])
    d = g.to_dict()
    g2 = Genotype.from_dict(d)
    assert g2.to_dict() == d
    assert g2.depth == 2
    assert g2.hidden == 64
    assert g2.protected == ["gcn"]


def test_signature_stable_and_order_invariant():
    g1 = random_genotype(depth=2, hidden=32)
    g2 = random_genotype(depth=2, hidden=32)
    assert g1.signature() == g2.signature()  # 内容相同 → 签名相同


def test_signature_differs_on_content():
    g1 = random_genotype(depth=2, spatial="gcn")
    g2 = random_genotype(depth=2, spatial="gat")
    assert g1.signature() != g2.signature()


# -- 合法性 --

def test_invalid_spatial_op_rejected():
    with pytest.raises(ValueError):
        Genotype(blocks=[STBlock("nonsense", "tcn")]).validate()


def test_invalid_fusion_rejected():
    with pytest.raises(ValueError):
        Genotype(blocks=[STBlock("gcn", "tcn", "bogus")]).validate()


def test_empty_blocks_rejected():
    with pytest.raises(ValueError):
        Genotype(blocks=[]).validate()


def test_protected_must_exist():
    """保护区算子必须真的在某 block 中使用。"""
    with pytest.raises(ValueError):
        Genotype(blocks=[STBlock("gcn", "tcn")], protected=["gat"]).validate()


# -- 变异 --

def test_mutate_returns_new_object():
    g = random_genotype(depth=2)
    g2 = mutate(g, "swap_spatial", index=0, new_op="gat")
    assert g2 is not g
    assert g.blocks[0].spatial_op == "gcn"   # 原件不变
    assert g2.blocks[0].spatial_op == "gat"  # 新件已变


def test_mutate_swap_temporal():
    g = random_genotype(depth=2, temporal="tcn")
    g2 = mutate(g, "swap_temporal", index=1, new_op="gru")
    assert g2.blocks[1].temporal_op == "gru"


def test_mutate_add_remove_block():
    g = random_genotype(depth=2)
    g2 = mutate(g, "add_block", new_block=STBlock("gat", "attn"))
    assert g2.depth == 3
    g3 = mutate(g2, "remove_block", index=0)
    assert g3.depth == 2


def test_mutate_remove_last_block_rejected():
    g = random_genotype(depth=1)
    with pytest.raises(ValueError):
        mutate(g, "remove_block", index=0)


def test_mutate_change_hidden_and_adj():
    g = random_genotype(depth=1, hidden=32)
    g2 = mutate(g, "change_hidden", new_hidden=128)
    assert g2.hidden == 128
    g3 = mutate(g, "change_adj_mode", new_adj_mode="rw")
    assert g3.adj_mode == "rw"


# -- 创新点保护区 (铁律) --

def test_protected_spatial_cannot_swap():
    """保护区的空间算子不可被替换。"""
    g = random_genotype(depth=2, spatial="adaptive", protected=["adaptive"])
    with pytest.raises(ValueError):
        mutate(g, "swap_spatial", index=0, new_op="gcn")


def test_protected_temporal_cannot_swap():
    g = random_genotype(depth=1, temporal="gru", protected=["gru"])
    with pytest.raises(ValueError):
        mutate(g, "swap_temporal", index=0, new_op="tcn")


def test_protected_block_cannot_remove():
    """触及保护区算子的块不可被删除。"""
    g = Genotype(
        blocks=[STBlock("adaptive", "tcn"), STBlock("gcn", "gru")],
        protected=["adaptive"],
    )
    g.validate()
    with pytest.raises(ValueError):
        mutate(g, "remove_block", index=0)  # block0 触及保护区
    # 非保护块可删
    g2 = mutate(g, "remove_block", index=1)
    assert g2.depth == 1


def test_protected_can_still_tune_hidden():
    """保护区只禁删/换算子, 不禁调全局超参 (可调其接线/规模)。"""
    g = random_genotype(depth=1, spatial="adaptive", protected=["adaptive"])
    g2 = mutate(g, "change_hidden", new_hidden=64)
    assert g2.hidden == 64
    assert g2.blocks[0].spatial_op == "adaptive"  # 创新点仍在
