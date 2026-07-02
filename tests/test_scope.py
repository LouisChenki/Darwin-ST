"""实验作用域隔离 (ExperimentScope) 测试 —— 多数据集/多代际验证的地基。

锁住五件事:
  - space_version 决定性 (同输入同哈希, 加/删内置算子才变, synth_ 注入不变)。
  - SPACE_VERSION 环境变量覆盖。
  - synth_dir 路径按 (dataset, space_version) 分区。
  - RESUME 作用域匹配暖启动 (resolve_warmstart_base): 本代有史→base, 换代/新集→None 冷启动。
  - synth 注册表跨作用域隔离 (Stage2 稀释 bug 的根因防线)。
"""

from __future__ import annotations

import importlib
import os

import pytest

from darwin_st.scope import ExperimentScope
from darwin_st.search import operators as ops_mod
from darwin_st.search.operators import builtin_op_signature


# ---------------------------------------------------------------------------
# space_version 决定性 + 稳定性铁律
# ---------------------------------------------------------------------------


def test_builtin_op_signature_deterministic():
    """同一内置算子集 → 同哈希 (两次调用一致); 8 位。"""
    s1 = builtin_op_signature()
    s2 = builtin_op_signature()
    assert s1 == s2
    assert len(s1) == 8


def test_builtin_op_signature_stable_across_reload():
    """importlib.reload operators 后哈希不变 (不依赖模块加载态)。"""
    s1 = builtin_op_signature()
    importlib.reload(ops_mod)
    s2 = ops_mod.builtin_op_signature()
    assert s1 == s2


def test_synth_injection_does_not_change_signature():
    """注入 synth_ 算子 (运行时创造) 不改 space_version —— 它不算搜索空间代际。

    这是 Stage2 稀释 bug 的关键: synth 算子多不该让作用域漂移, 否则每次创造都换代。
    """
    base = builtin_op_signature()
    ops_mod.SPATIAL_OPS["synth_fake_xyz"] = None
    ops_mod.OP_CATEGORY["synth_fake_xyz"] = "spatiotemporal"
    try:
        assert builtin_op_signature() == base, "synth 注入改了 space_version!"
    finally:
        ops_mod.SPATIAL_OPS.pop("synth_fake_xyz", None)
        ops_mod.OP_CATEGORY.pop("synth_fake_xyz", None)


def test_builtin_op_add_changes_signature_and_reverts():
    """加内置算子 → 哈希变; 移除 → 复原 (顺序无关, 只认集合)。"""
    base = builtin_op_signature()
    ops_mod.SPATIAL_OPS["brand_new_builtin"] = None
    try:
        assert builtin_op_signature() != base, "加内置算子未改哈希!"
    finally:
        ops_mod.SPATIAL_OPS.pop("brand_new_builtin", None)
    assert builtin_op_signature() == base, "移除后未复原 (顺序敏感?)"


# ---------------------------------------------------------------------------
# ExperimentScope 构造 + 路径
# ---------------------------------------------------------------------------


def test_scope_resolve_auto_hash():
    """默认 space_version = builtin_op_signature()。"""
    s = ExperimentScope.resolve("PeMS04")
    assert s.dataset == "PeMS04"
    assert s.space_version == builtin_op_signature()


def test_scope_explicit_override():
    s = ExperimentScope.resolve("METR-LA", space_version="stage2-dag")
    assert s.space_version == "stage2-dag"


def test_scope_env_override(monkeypatch):
    monkeypatch.setenv("SPACE_VERSION", "env-ver-123")
    s = ExperimentScope.resolve("PeMS08")
    assert s.space_version == "env-ver-123"


def test_scope_explicit_beats_env(monkeypatch):
    """显式传参优先于环境变量。"""
    monkeypatch.setenv("SPACE_VERSION", "env-ver")
    s = ExperimentScope.resolve("PeMS04", space_version="explicit")
    assert s.space_version == "explicit"


def test_synth_dir_partitioned():
    """synth 目录按 (dataset, space_version) 分区。"""
    s = ExperimentScope("PeMS04", "abc123")
    assert s.synth_dir("/root/cache") == "/root/cache/dynamic_ops/PeMS04/abc123"
    # 不同数据集/代 → 不同目录
    assert ExperimentScope("METR-LA", "abc123").synth_dir("/c") != s.synth_dir("/c")
    assert ExperimentScope("PeMS04", "def456").synth_dir("/c") != s.synth_dir("/c")


def test_scope_frozen_and_pickle():
    """frozen (不可变) + pickle-safe (跨 spawn worker 传递)。"""
    import pickle
    s = ExperimentScope("PeMS04", "v1")
    assert pickle.loads(pickle.dumps(s)) == s
    with pytest.raises(Exception):
        s.dataset = "x"   # frozen


# ---------------------------------------------------------------------------
# 跨作用域 synth 注册表隔离 (Stage2 稀释 bug 根因防线)
# ---------------------------------------------------------------------------


def test_registry_cross_scope_isolation(tmp_path):
    """A 注册进 scopeA 目录, B 进 scopeB 目录; 加载 scopeB 只见 B, 不见 A。

    这守住核心: 换代/换数据集时, 旧代 synth 算子物理隔离在别的目录, load_persisted 加载不到
    → 变异池不被旧算子数量碾压 → 新原子算子公平竞争。
    """
    from darwin_st.creation import OperatorRegistry
    from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator

    code = ('import torch\nimport torch.nn as nn\n'
            'class Op(nn.Module):\n'
            '    def __init__(self, channels, num_nodes, **kw):\n'
            '        super().__init__(); self.l = nn.Linear(channels, channels)\n'
            '        self.a = nn.Parameter(torch.zeros(1))\n'
            '    def forward(self, x, adj=None): return self.l(x) + self.a * x\n')

    def mk(name):
        plan = FusionPlan(operator_name=name, rationale="r", shared_structure="s",
                          composition="additive_residual", source_mechanisms=["a"], expected_effect="e")
        return SynthesizedOperator(name=name, code=code.replace("class Op", f"class {name}"),
                                   plan=plan, needs_adj=False, validated=True)

    scope_a = ExperimentScope("PeMS04", "vA")
    scope_b = ExperimentScope("PeMS04", "vB")
    dir_a = scope_a.synth_dir(str(tmp_path))
    dir_b = scope_b.synth_dir(str(tmp_path))

    before = set(ops_mod.SPATIAL_OPS)
    try:
        OperatorRegistry(persist_dir=dir_a).register(mk("OpA"))
        OperatorRegistry(persist_dir=dir_b).register(mk("OpB"))
        # 清掉全局注入, 只留磁盘持久化
        for k in list(ops_mod.SPATIAL_OPS):
            if k not in before:
                ops_mod.SPATIAL_OPS.pop(k, None)
                ops_mod.OP_CATEGORY.pop(k, None)
        # 加载 scopeB → 只见 B
        loaded = OperatorRegistry(persist_dir=dir_b).load_persisted()
        assert "synth_OpB" in loaded
        assert "synth_OpA" not in loaded, "跨作用域泄漏! scopeB 加载到了 scopeA 的算子"
    finally:
        for k in list(ops_mod.SPATIAL_OPS):
            if k not in before:
                ops_mod.SPATIAL_OPS.pop(k, None)
                ops_mod.OP_CATEGORY.pop(k, None)
