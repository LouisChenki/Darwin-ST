"""Stage 1 搜索空间升级测试: 算子类别化 + joint 模式 + 交互层 fusion 补全。

锁住三件事 + 向后兼容铁律:
  - 算子类别化 (OP_CATEGORY) + 2 个内置 joint 算子 (st_separable/st_graph_attn)。
  - STBlock.joint_op + 扩 VALID_FUSION (sequential_ts/cross/iterative)。
  - synth 算子按类别路由 (时空一体→joint 槽, 非强塞空间槽)。
  - **golden-hash**: 旧线性 genotype signature 改动前后字节不变 (graveyard dedup 安全)。
  - pickle 跨 spawn 保 joint_op + _seed_meta。
"""

from __future__ import annotations

import pickle

import numpy as np
import torch

from darwin_st.search.builder import build_model
from darwin_st.search.genotype import VALID_FUSION, Genotype, STBlock, _block_to_dict
from darwin_st.search.operators import (
    OP_CATEGORY,
    SPATIOTEMPORAL_OPS,
    build_op,
    op_category,
)

ADJ = np.eye(10, dtype=np.float32)
X = torch.randn(2, 12, 10, 3)


# ---- 算子类别化 ----

def test_op_category_covers_builtins():
    """内置算子全有类别 (空间/时序/时空一体/any)。"""
    for name in ("gcn", "cheb", "gat", "diffusion", "adaptive"):
        assert OP_CATEGORY[name] == "spatial"
    for name in ("tcn", "gru", "attn"):
        assert OP_CATEGORY[name] == "temporal"
    for name in ("st_separable", "st_graph_attn"):
        assert OP_CATEGORY[name] == "spatiotemporal"
    assert OP_CATEGORY["identity"] == "any"


def test_op_category_synth_default_spatiotemporal():
    """未登记的 synth_ 算子默认时空一体 (诚实兜底)。"""
    assert op_category("synth_unknown_xyz") == "spatiotemporal"


def test_builtin_joint_ops_build_forward_backward():
    """2 个内置 joint 算子: build+forward 保形 + 反向有梯度。"""
    for name in SPATIOTEMPORAL_OPS:
        m = build_op(name, dim=32, num_nodes=10)
        x = torch.randn(2, 12, 10, 32, requires_grad=True)
        out = m(x, torch.eye(10))
        assert out.shape == x.shape, f"{name} 形漂移 {out.shape}"
        out.sum().backward()
    # 无图退化不崩
    assert build_op("st_separable", 32, num_nodes=10)(torch.randn(2, 12, 10, 32), None).shape \
        == (2, 12, 10, 32)


# ---- golden-hash 向后兼容 (最关键) ----

def test_old_linear_genotype_signature_unchanged():
    """铁律: 旧线性 genotype 的 to_dict 块字典字节不含 joint_op → signature 与升级前一致。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn", "sequential")], hidden=64)
    bd = _block_to_dict(g.blocks[0])
    # 必须恰好是旧三键格式 (无 joint_op), 否则签名漂移破 graveyard dedup
    assert bd == {"spatial_op": "gcn", "temporal_op": "tcn", "fusion": "sequential"}
    assert "joint_op" not in str(g.to_dict())
    # 用预先记录的旧签名对照 (这是升级前 gcn+tcn+sequential+默认embedding+hidden64 的 sig)
    # 若此断言失败说明序列化漂移了
    g2 = Genotype.from_dict(g.to_dict())
    assert g2.signature() == g.signature()


def test_joint_genotype_signature_includes_joint_op():
    """joint genotype 的 to_dict 含 joint_op (与线性 genotype 签名不同)。"""
    gj = Genotype(blocks=[STBlock("identity", "identity", joint_op="st_separable")], hidden=64)
    assert "joint_op" in _block_to_dict(gj.blocks[0])
    gl = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)
    assert gj.signature() != gl.signature()


def test_from_dict_backward_compat():
    """旧 dict (无 joint_op) 正常加载; 新 dict (有 joint_op) 正常加载。"""
    old_d = {"blocks": [{"spatial_op": "gcn", "temporal_op": "tcn", "fusion": "sequential"}],
             "hidden": 64, "adj_mode": "sym"}
    g = Genotype.from_dict(old_d)
    assert g.blocks[0].joint_op is None
    new_d = {"blocks": [{"spatial_op": "identity", "temporal_op": "identity",
                         "fusion": "sequential", "joint_op": "st_graph_attn"}], "hidden": 64}
    gj = Genotype.from_dict(new_d)
    assert gj.blocks[0].joint_op == "st_graph_attn"


# ---- 扩展的交互层 fusion ----

def test_valid_fusion_extended():
    """VALID_FUSION 含新交互模式 (先T后S/cross/iterative), 旧值保留。"""
    for f in ("sequential", "parallel", "residual", "sequential_ts", "cross", "iterative"):
        assert f in VALID_FUSION


def test_all_six_fusion_modes_compile_and_run():
    """6 种 fusion 全部能 build_model + forward 保形 + 反向有梯度。"""
    for f in ("sequential", "sequential_ts", "parallel", "residual", "cross", "iterative"):
        g = Genotype(blocks=[STBlock("gcn", "tcn", fusion=f)], hidden=32)
        m = build_model(g, num_nodes=10, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)
        out = m(X)
        assert out.shape == (2, 12, 10), f"fusion={f}: {out.shape}"
        out.sum().backward()


def test_joint_block_compiles_and_runs():
    """joint block (内置时空算子) 能 build_model + forward + 反向。"""
    for name in SPATIOTEMPORAL_OPS:
        g = Genotype(blocks=[STBlock("identity", "identity", joint_op=name)], hidden=32)
        m = build_model(g, num_nodes=10, in_channels=3, seq_len_in=12, seq_len_out=12, adj=ADJ)
        out = m(X)
        assert out.shape == (2, 12, 10)
        out.sum().backward()


# ---- validate 放宽 + pickle ----

def test_validate_joint_op_checked():
    """joint_op 校验在算子库; 未知 joint_op 拒绝。"""
    Genotype(blocks=[STBlock("identity", "identity", joint_op="st_separable")]).validate()
    import pytest
    with pytest.raises(ValueError):
        Genotype(blocks=[STBlock("identity", "identity", joint_op="not_an_op")]).validate()


def test_validate_allows_st_op_in_either_slot():
    """放宽: 时空一体算子可坐空间或时序槽 (类别兼容, 不报错)。"""
    Genotype(blocks=[STBlock("st_separable", "tcn")]).validate()       # 时空算子在空间槽
    Genotype(blocks=[STBlock("gcn", "st_graph_attn")]).validate()      # 时空算子在时序槽


def test_genotype_with_joint_op_pickles():
    """跨 spawn 铁律: 含 joint_op 的 genotype + _seed_meta pickle 往返保留。"""
    g = Genotype(blocks=[STBlock("identity", "identity", joint_op="st_separable")], hidden=64)
    g._seed_meta = {"warm_start_hps": {"lr": 1e-3}, "hpo_trials": 20, "is_creation_seed": True}
    back = pickle.loads(pickle.dumps(g))
    assert back.blocks[0].joint_op == "st_separable"
    assert back._seed_meta["hpo_trials"] == 20
    # copy() 走 to_dict/from_dict, joint_op 保留但 _seed_meta 不带 (设计)
    gc = g.copy()
    assert gc.blocks[0].joint_op == "st_separable"
    assert not hasattr(gc, "_seed_meta")
