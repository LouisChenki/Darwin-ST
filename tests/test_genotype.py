"""genotype.py 的正确性回归测试 (设备无关)。

核心契约:
  - 序列化往返一致, 签名稳定且键序无关
  - 合法性校验拦截非法算子/融合/adj_mode
  - 变异返回新对象不改原件
  - **创新点保护区**: 保护算子不可被 swap/remove (架构铁律)
"""

from __future__ import annotations

import pytest

from darwin_st.search.genotype import (
    EmbeddingConfig,
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


# -- 身份嵌入基因 (研究结论: 比图算子更提精度) --

def test_default_genotype_enables_node_embedding():
    """默认应开节点嵌入(免费的最大杠杆), tod/dow 默认关。"""
    g = random_genotype(depth=2)
    assert g.embedding.use_node is True
    assert g.embedding.use_tod is False
    assert g.embedding.use_dow is False


def test_embedding_in_serialization():
    g = random_genotype(depth=1)
    g.embedding.node_dim = 64
    d = g.to_dict()
    assert "embedding" in d
    g2 = Genotype.from_dict(d)
    assert g2.embedding.node_dim == 64
    assert g2.embedding.use_node is True


def test_embedding_affects_signature():
    """嵌入配置不同 → 签名不同 (它是一等基因)。"""
    g1 = random_genotype(depth=1)
    g2 = g1.copy()
    g2.embedding.use_tod = True
    assert g1.signature() != g2.signature()


def test_backward_compat_no_embedding_field():
    """旧 genotype dict (无 embedding 字段) 应用默认配置, 不报错。"""
    old = {"blocks": [{"spatial_op": "gcn", "temporal_op": "tcn", "fusion": "sequential"}],
           "hidden": 32, "adj_mode": "sym", "protected": []}
    g = Genotype.from_dict(old)
    g.validate()
    assert g.embedding.use_node is True  # 默认


def test_mutate_toggle_embedding_enable():
    g = random_genotype(depth=1)  # tod 默认关
    g2 = mutate(g, "toggle_embedding", which="tod", enable=True)
    assert g2.embedding.use_tod is True
    assert g.embedding.use_tod is False  # 原件不变


def test_mutate_toggle_embedding_dim():
    g = random_genotype(depth=1)
    g2 = mutate(g, "toggle_embedding", which="node", dim=64)
    assert g2.embedding.node_dim == 64


def test_mutate_toggle_embedding_disable_node():
    """可关掉节点嵌入 (消融实验需要)。"""
    g = random_genotype(depth=1)
    g2 = mutate(g, "toggle_embedding", which="node", enable=False)
    assert g2.embedding.use_node is False


def test_mutate_toggle_embedding_bad_which():
    g = random_genotype(depth=1)
    with pytest.raises(ValueError):
        mutate(g, "toggle_embedding", which="bogus", enable=True)


def test_embedding_invalid_dim_rejected():
    g = random_genotype(depth=1)
    g.embedding.node_dim = -5
    with pytest.raises(ValueError):
        g.validate()
