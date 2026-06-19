"""creation_loop.py 的正确性回归测试 (mock 全链, 无 LLM/GPU/网络)。

核心契约:
  - diagnose_bottleneck: 从最优架构诊断瓶颈
  - maybe_create: 诊断→跨域检索→合成→注入→产出待评 genotype→记 insight
  - 合成失败 → 记失败 insight, 不崩
  - 注入的算子能被产出的 genotype 引用 + builder 编译
"""

from __future__ import annotations

import pytest
import torch

from darwin_st.creation import (
    CreationConfig,
    CreationLoop,
    MockLLM,
    OperatorRegistry,
    OperatorSynthesizer,
    SynthesisConfig,
    diagnose_bottleneck,
)
from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
from darwin_st.memory.store import MemoryStore
from darwin_st.search import operators as ops_mod
from darwin_st.search.builder import build_model
from darwin_st.search.genotype import Genotype, STBlock


_GOOD_PLAN = (
    '{"operator_name": "FusedCreationOp", "rationale": "融合长程与自监督",'
    '"shared_structure": "时序混合", "composition": "additive_residual",'
    '"source_mechanisms": ["state_space_model", "contrastive_learning"],'
    '"expected_effect": "改善长程"}'
)

_GOOD_CODE = '''```python
import torch
import torch.nn as nn
class FusedCreationOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.branch = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        return self.proj(x) + self.alpha * self.branch(x)
```'''


@pytest.fixture(autouse=True)
def _clean_ops():
    before = set(ops_mod.SPATIAL_OPS)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before:
            ops_mod.SPATIAL_OPS.pop(k, None)


@pytest.fixture
def store():
    s = InMemoryGraphStore(HashEmbedder(dim=256))
    for m in all_seed_mechanisms():
        s.add_mechanism(m)
    return s


def _loop(store, llm_responses, memory=None):
    resp = iter(llm_responses)
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    reg = OperatorRegistry()
    return CreationLoop(store, store.embedder, synth, reg, memory=memory)


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def test_diagnose_no_genotype():
    assert "精度" in diagnose_bottleneck(None, None)


def test_diagnose_missing_long_range():
    """只用 gcn+tcn 的架构应诊断出'长程不足'。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    b = diagnose_bottleneck(g, sota_gap=1.0)
    assert "长程" in b


def test_diagnose_sota_gap():
    g = Genotype(blocks=[STBlock("gat", "attn")])
    g.embedding.use_node = True
    b = diagnose_bottleneck(g, sota_gap=5.0)
    assert "SOTA" in b or "差距" in b


# ---------------------------------------------------------------------------
# 创造 (端到端 mock)
# ---------------------------------------------------------------------------


def test_maybe_create_success(store):
    loop = _loop(store, [_GOOD_PLAN, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create(best, sota_gap=3.0)
    assert outcome.success, outcome.reason
    assert outcome.operator_name.startswith("synth_")
    assert outcome.seed_genotype is not None
    assert outcome.retrieved_mechanisms  # 检索到了跨域机制
    # 注入的算子在 SPATIAL_OPS
    assert outcome.operator_name in ops_mod.SPATIAL_OPS


def test_maybe_create_seed_genotype_uses_new_op(store):
    loop = _loop(store, [_GOOD_PLAN, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create(best, sota_gap=3.0)
    # 产出的 genotype 第一块用了新算子
    assert outcome.seed_genotype.blocks[0].spatial_op == outcome.operator_name


def test_maybe_create_seed_genotype_compiles(store):
    """产出的待评 genotype 能 builder 编译 + 前向 (闭环到进化)。"""
    loop = _loop(store, [_GOOD_PLAN, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=16)
    outcome = loop.maybe_create(best, sota_gap=3.0)
    model = build_model(outcome.seed_genotype, num_nodes=5, in_channels=3,
                        seq_len_in=12, seq_len_out=12)
    out = model(torch.randn(2, 12, 5, 3))
    assert out.shape == (2, 12, 5)


def test_maybe_create_records_insight(store):
    """成功创造写 insight 到 memory(融合经验, 不进机制库)。"""
    mem = MemoryStore(":memory:")
    loop = _loop(store, [_GOOD_PLAN, _GOOD_CODE], memory=mem)
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0, dataset="PeMS04")
    insights = mem.get_insights("PeMS04")
    assert len(insights) >= 1
    assert "融合" in insights[0]["insight_text"]
    mem.close()


def test_maybe_create_synth_failure_records_and_no_crash(store):
    """合成失败 → 记失败 insight, 优雅返回(不崩, 不中止优化)。"""
    mem = MemoryStore(":memory:")
    # 计划 OK 但代码一直坏(丢节点维)
    bad_code = '```python\nimport torch.nn as nn\nclass FusedCreationOp(nn.Module):\n    def __init__(self,channels,num_nodes,**kw):\n        super().__init__(); self.l=nn.Linear(channels,channels)\n    def forward(self,x,adj=None): return self.l(x).mean(dim=2)\n```'
    loop = _loop(store, [_GOOD_PLAN, bad_code, bad_code, bad_code], memory=mem)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert not outcome.success
    assert outcome.insight  # 记了失败经验
    insights = mem.get_insights("PeMS04")
    assert any("未通过" in i["insight_text"] for i in insights)
    mem.close()
