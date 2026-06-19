"""registry.py 的正确性回归测试 (设备无关)。

核心契约:
  - register: 把合成算子注入 SPATIAL_OPS, genotype/builder 能用它
  - 未验证算子拒绝注入
  - 持久化 + 重载存活
  - 端到端: 注入算子 → genotype 引用 → builder 编译 → 可训练
  - unregister 移除
"""

from __future__ import annotations

import pytest
import torch

from darwin_st.creation import OperatorRegistry, SYNTH_PREFIX
from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator
from darwin_st.search import operators as ops_mod
from darwin_st.search.builder import build_model, count_params
from darwin_st.search.genotype import Genotype, STBlock


# 一个自包含的"合成算子"代码 (融合: 膨胀卷积思想 + 零初始化残差)
_SYNTH_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=0)
        self.branch = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        B, T, N, C = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        h = F.pad(h, (2, 0))
        h = self.conv(h)
        main = h.view(B, N, C, T).permute(0, 3, 1, 2)
        br = self.branch(x)
        return main + self.alpha * br
'''


def _synth_op(name="FusedOp", validated=True):
    plan = FusionPlan(operator_name=name, rationale="r", shared_structure="s",
                      composition="additive_residual", source_mechanisms=["a", "b"],
                      expected_effect="e")
    return SynthesizedOperator(name=name, code=_SYNTH_CODE.replace("FusedOp", name),
                               plan=plan, needs_adj=False, validated=validated)


@pytest.fixture(autouse=True)
def _clean_spatial_ops():
    """每个测试后清理注入的 synth 算子, 不污染全局 SPATIAL_OPS。"""
    before = set(ops_mod.SPATIAL_OPS)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before:
            ops_mod.SPATIAL_OPS.pop(k, None)


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------


def test_register_injects_into_spatial_ops():
    reg = OperatorRegistry()
    name = reg.register(_synth_op("MyFusedOp"))
    assert name == SYNTH_PREFIX + "MyFusedOp"
    assert name in ops_mod.SPATIAL_OPS
    assert reg.is_registered(name)


def test_register_rejects_unvalidated():
    reg = OperatorRegistry()
    with pytest.raises(ValueError):
        reg.register(_synth_op(validated=False))


def test_unregister_removes():
    reg = OperatorRegistry()
    name = reg.register(_synth_op("TempOp"))
    assert name in ops_mod.SPATIAL_OPS
    reg.unregister(name)
    assert name not in ops_mod.SPATIAL_OPS
    assert not reg.is_registered(name)


# ---------------------------------------------------------------------------
# 端到端: 注入 → genotype → builder → 训练
# ---------------------------------------------------------------------------


def test_synth_op_usable_in_genotype_and_builder():
    """核心: 注入的合成算子能被 genotype 引用、builder 编译、跑前向。"""
    reg = OperatorRegistry()
    name = reg.register(_synth_op("E2EOp"))

    N, C, T_in, T_out = 5, 16, 12, 12
    geno = Genotype(blocks=[STBlock(spatial_op=name, temporal_op="identity")])
    geno.validate()  # genotype 接受合成算子名
    model = build_model(geno, num_nodes=N, in_channels=3, seq_len_in=T_in, seq_len_out=T_out)
    x = torch.randn(2, T_in, N, 3)
    out = model(x)
    assert out.shape == (2, T_out, N)  # 编译并跑通


def test_synth_op_trainable():
    """注入的合成算子参与的模型能训练降 loss。"""
    reg = OperatorRegistry()
    name = reg.register(_synth_op("TrainOp"))
    N, C = 5, 16
    geno = Genotype(blocks=[STBlock(spatial_op=name, temporal_op="tcn")], hidden=16)
    model = build_model(geno, num_nodes=N, in_channels=3, seq_len_in=12, seq_len_out=12)
    x = torch.randn(2, 12, N, 3)
    y = torch.randn(2, 12, N)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(15):
        opt.zero_grad()
        loss = torch.abs(model(x) - y).mean()
        loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]  # 在学


# ---------------------------------------------------------------------------
# 持久化
# ---------------------------------------------------------------------------


def test_persist_and_reload(tmp_path):
    """注册→持久化→新 registry 重载→算子仍可用。"""
    reg1 = OperatorRegistry(persist_dir=str(tmp_path))
    name = reg1.register(_synth_op("PersistOp"))
    import os
    assert os.path.exists(os.path.join(tmp_path, name + ".py"))
    assert os.path.exists(os.path.join(tmp_path, name + ".json"))

    # 移除全局注册, 模拟重启
    ops_mod.SPATIAL_OPS.pop(name, None)

    reg2 = OperatorRegistry(persist_dir=str(tmp_path))
    loaded = reg2.load_persisted()
    assert name in loaded
    assert name in ops_mod.SPATIAL_OPS  # 重载后又可用


def test_load_persisted_empty(tmp_path):
    reg = OperatorRegistry(persist_dir=str(tmp_path))
    assert reg.load_persisted() == []
