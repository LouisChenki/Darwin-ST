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

# 多假设路径: 计划阶段返回 JSON 数组
_GOOD_PLAN_ARRAY = "[" + _GOOD_PLAN + "]"

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
# LLM 诊断接线 (层2/3): _diagnose 优先 LLM, 缺则规则兜底; override 流进检索
# ---------------------------------------------------------------------------


def _trace_dict():
    """构造一个收敛平台的训练轨迹 (供 summarize_trace)。"""
    return {
        "train_loss": [1.0, 0.6, 0.4, 0.35, 0.33], "val_mae": [30, 22, 19, 18.7, 18.69],
        "grad_norm_mean": [1.0] * 5, "grad_norm_max": [2.0] * 5,
        "n_epochs_run": 5, "max_epochs": 5, "best_epoch": 4, "best_mae": 18.69,
        "stopped_early": False, "nan_hit": False, "lr_schedule": "cosine", "lr": 1e-3,
    }


def test_diagnose_uses_llm_preconditions(store):
    """有 llm + trace → _diagnose 返回 LLM 的受控前提词 (override)。"""
    import json as _json
    diag_llm = MockLLM(lambda msgs: _json.dumps(
        {"reasoning": "r", "evidence": "e", "diagnosis": "缺多尺度建模",
         "preconditions": ["multi_scale_structure"], "direction": "多尺度分解"}))
    synth = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), llm=diag_llm)
    bottleneck, override = loop._diagnose(
        Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=0.89, best_trace=_trace_dict())
    assert override == ["multi_scale_structure"]
    assert "多尺度" in bottleneck


def test_diagnose_falls_back_to_rules_without_llm(store):
    """无 llm → 退回规则版 diagnose_bottleneck, override=None。"""
    synth = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), llm=None)
    bottleneck, override = loop._diagnose(
        Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=1.0, best_trace=_trace_dict())
    assert override is None
    assert isinstance(bottleneck, str) and bottleneck


def test_diagnose_falls_back_without_trace(store):
    """有 llm 但无 trace → 仍退规则版 (无训练信号无从诊断)。"""
    diag_llm = MockLLM(lambda msgs: '{"diagnosis":"x","preconditions":["heterogeneity"]}')
    synth = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), llm=diag_llm)
    _, override = loop._diagnose(Genotype(blocks=[STBlock("gcn", "tcn")]),
                                 sota_gap=1.0, best_trace=None)
    assert override is None


def test_maybe_create_records_opro_history(store):
    """maybe_create 累积 OPRO 历史轨迹 (跨轮去重用)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0,
                      best_trace=_trace_dict())
    assert len(loop._history) == 1
    assert loop._round_idx == 1
    assert "bottleneck" in loop._history[0]


# ---------------------------------------------------------------------------
# 创造 (端到端 mock)
# ---------------------------------------------------------------------------


def test_maybe_create_success(store):
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create(best, sota_gap=3.0)
    assert outcome.success, outcome.reason
    assert outcome.operator_name.startswith("synth_")
    assert outcome.seed_genotype is not None
    assert outcome.retrieved_mechanisms  # 检索到了跨域机制
    # 注入的算子在 SPATIAL_OPS
    assert outcome.operator_name in ops_mod.SPATIAL_OPS


def test_maybe_create_seed_genotype_uses_new_op(store):
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create(best, sota_gap=3.0)
    # 产出的 genotype 第一块用了新算子
    assert outcome.seed_genotype.blocks[0].spatial_op == outcome.operator_name


def test_seed_genotype_carries_seed_meta(store):
    """A: 创造 seed 挂 _seed_meta (warm_start_hps + hpo_trials), 标记走专项大 HPO。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    parent_hps = {"lr": 1e-3, "dropout": 0.1}
    outcome = loop.maybe_create(best, sota_gap=3.0, best_hps=parent_hps)
    seed = outcome.seed_genotype
    meta = getattr(seed, "_seed_meta", None)
    assert meta is not None
    assert meta["warm_start_hps"] == parent_hps     # 继承父最优超参 (warm-start)
    assert meta["hpo_trials"] == loop.cfg.seed_hpo_trials   # 专项大 HPO 预算 (默认 20)
    assert meta["is_creation_seed"] is True


def test_seed_meta_survives_pickle_and_not_in_to_dict(store):
    """A 铁律: _seed_meta 跨 spawn pickle 保留 + 不进 to_dict/signature (graveyard dedup 安全)。"""
    import pickle
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    seed = loop.maybe_create(best, sota_gap=3.0, best_hps={"lr": 1e-3}).seed_genotype
    # to_dict 不含 _seed_meta
    assert "_seed_meta" not in str(seed.to_dict())
    # pickle 往返保留 (进程后端 worker 靠它判大 HPO)
    back = pickle.loads(pickle.dumps(seed))
    assert getattr(back, "_seed_meta", None) is not None
    assert back._seed_meta["hpo_trials"] == loop.cfg.seed_hpo_trials


def test_seed_meta_default_warm_start_none_without_best_hps(store):
    """不传 best_hps → warm_start_hps=None (首轮无父超参时不崩, 走纯大 HPO)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    seed = loop.maybe_create(best, sota_gap=3.0).seed_genotype   # best_hps 默认 None
    assert seed._seed_meta["warm_start_hps"] is None
    assert seed._seed_meta["hpo_trials"] == loop.cfg.seed_hpo_trials


def test_maybe_create_seed_genotype_compiles(store):
    """产出的待评 genotype 能 builder 编译 + 前向 (闭环到进化)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    best = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=16)
    outcome = loop.maybe_create(best, sota_gap=3.0)
    model = build_model(outcome.seed_genotype, num_nodes=5, in_channels=3,
                        seq_len_in=12, seq_len_out=12)
    out = model(torch.randn(2, 12, 5, 3))
    assert out.shape == (2, 12, 5)


def test_maybe_create_records_insight(store):
    """成功创造写 insight 到 memory(融合经验, 不进机制库)。"""
    mem = MemoryStore(":memory:")
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE], memory=mem)
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
    loop = _loop(store, [_GOOD_PLAN_ARRAY, bad_code, bad_code], memory=mem)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert not outcome.success
    assert outcome.insight  # 记了失败经验
    insights = mem.get_insights("PeMS04")
    assert any("未过验证" in i["insight_text"] for i in insights)
    mem.close()
