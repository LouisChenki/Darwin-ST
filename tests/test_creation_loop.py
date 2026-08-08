"""creation_loop.py 的正确性回归测试 (mock 全链, 无 LLM/GPU/网络)。

核心契约:
  - diagnose_bottleneck: 从最优架构诊断瓶颈
  - maybe_create: 诊断→跨域检索→合成→注入→产出待评 genotype (insight 文本随 outcome 返回)
  - 合成失败 → 优雅返回不崩 (失败教训进 B1 履历本; 不再双写 memory insights 表)
  - 注入的算子能被产出的 genotype 引用 + builder 编译
"""

from __future__ import annotations

import json

import pytest
import torch

from darwin_st.creation import (
    CreationConfig,
    CreationLoop,
    CreationArchive,
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
    # 本组用例的响应序列按 synthesize_many (单次调用产 N) 编排 → 固定走旧路径;
    # 独立采样 (默认开) 的用例单独构造, 见 B5 段
    return CreationLoop(store, store.embedder, synth, reg, memory=memory,
                        config=CreationConfig(independent_sampling=False))


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
    # 产出的 genotype 第一块用了新算子 (Stage 1: 时空一体算子默认进 joint 槽, 非空间槽)
    b0 = outcome.seed_genotype.blocks[0]
    used = {b0.spatial_op, b0.temporal_op, b0.joint_op}
    assert outcome.operator_name in used, f"新算子未被用到: {b0}"
    # 默认 spatiotemporal → joint 槽 (修了"强塞空间槽"的语义错位)
    assert b0.joint_op == outcome.operator_name


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


def test_maybe_create_no_longer_writes_insight(store):
    """双写停止: 成功创造【不再】写 memory insights 表 (流水账是 B1 履历本子集且无人读取;
    insights 表今后只由反思环节写入)。insight 文本仍随 outcome 返回 (供调用方/日志)。"""
    mem = MemoryStore(":memory:")
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE], memory=mem)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0,
                                dataset="PeMS04")
    assert outcome.success and "融合" in outcome.insight   # 文本仍随 outcome 返回
    assert mem.get_insights("PeMS04") == []                # 但不再落 insights 表
    mem.close()


def test_maybe_create_synth_failure_records_and_no_crash(store):
    """合成失败 → 优雅返回(不崩, 不中止优化); 失败教训进 B1 履历本, 同样不写 insights 表。"""
    mem = MemoryStore(":memory:")
    # 计划 OK 但代码一直坏(丢节点维)
    bad_code = '```python\nimport torch.nn as nn\nclass FusedCreationOp(nn.Module):\n    def __init__(self,channels,num_nodes,**kw):\n        super().__init__(); self.l=nn.Linear(channels,channels)\n    def forward(self,x,adj=None): return self.l(x).mean(dim=2)\n```'
    loop = _loop(store, [_GOOD_PLAN_ARRAY, bad_code, bad_code], memory=mem)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert not outcome.success
    assert outcome.insight  # 失败经验文本仍随 outcome 返回
    assert mem.get_insights("PeMS04") == []   # 停双写: 失败流水账也不落 insights 表
    mem.close()


# ---------------------------------------------------------------------------
# B4 方向信用分 + 温度自适应 (LLMatic curiosity)
# ---------------------------------------------------------------------------

_BAD_CODE_DROP_N = ('```python\nimport torch.nn as nn\nclass FusedCreationOp(nn.Module):\n'
                    '    def __init__(self,channels,num_nodes,**kw):\n'
                    '        super().__init__(); self.l=nn.Linear(channels,channels)\n'
                    '    def forward(self,x,adj=None): return self.l(x).mean(dim=2)\n```')


def _failing_loop(store):
    """全部假设挂门的 loop (代码丢节点维, 挂 shape 门; max_retries=2 → 2 次坏代码)。"""
    return _loop(store, [_GOOD_PLAN_ARRAY, _BAD_CODE_DROP_N, _BAD_CODE_DROP_N])


def test_history_records_have_score_field(store):
    """_history 每条带 score 字段, 初始 0.0 (旧记录无此字段时按 0 渲染, 见 diagnosis)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert loop._history[0]["score"] == 0.0


def test_all_hypotheses_fail_scores_minus_one(store):
    """全部假设挂门 (无 seed 产出, 立即可知) → 本轮方向同步记 −1。"""
    loop = _failing_loop(store)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert not outcome.success
    assert loop._history[0]["score"] == -1.0


def test_all_fail_raises_temperature(store):
    """零假设过门 → 温度 +0.05 (升温探索)。"""
    loop = _failing_loop(store)
    assert loop._temperature == pytest.approx(0.8)
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert loop._temperature == pytest.approx(0.85)


def test_update_direction_outcome_accumulates(store):
    """三种结局记分累加: adopted +1 / 罚 −0.5; 未知 round_idx 静默跳过。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    loop.update_direction_outcome(0, 1.0)
    assert loop._history[0]["score"] == pytest.approx(1.0)
    loop.update_direction_outcome(0, -0.5)
    assert loop._history[0]["score"] == pytest.approx(0.5)
    loop.update_direction_outcome(99, 1.0)          # 无此轮 → 静默跳过, 不崩
    assert loop._history[0]["score"] == pytest.approx(0.5)


def test_adopted_lowers_temperature_penalty_does_not(store):
    """+1 (adopted) → 温度 −0.05 (收敛利用); 罚分不动温度 (全灭升温在轮末做)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    loop.update_direction_outcome(0, 1.0)
    assert loop._temperature == pytest.approx(0.75)
    loop.update_direction_outcome(0, -0.5)
    assert loop._temperature == pytest.approx(0.75)


def test_temperature_clamp_upper(store):
    loop = _failing_loop(store)
    loop._temperature = 1.0
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert loop._temperature == 1.0                              # 上界 clamp [_, 1.0]


def test_temperature_clamp_lower(store):
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    loop._temperature = 0.5
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    loop.update_direction_outcome(0, 1.0)
    assert loop._temperature == 0.5                              # 下界 clamp [0.5, _]


def test_constructor_temperature_override(store):
    """构造参数覆盖初始温度; 越界初始值也 clamp 到 [0.5, 1.0]。"""
    synth = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), temperature=0.65)
    assert loop._temperature == pytest.approx(0.65)
    synth2 = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    loop2 = CreationLoop(store, store.embedder, synth2, OperatorRegistry(), temperature=1.5)
    assert loop2._temperature == 1.0


def test_seed_meta_carries_creation_round(store):
    """seed._seed_meta 带 creation_round = 产出它的诊断轮次 (round_idx 递增时机: 记 history 后+1)。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE, _GOOD_PLAN_ARRAY, _GOOD_CODE])
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    o1 = loop.maybe_create(g, sota_gap=3.0)
    o2 = loop.maybe_create(o1.seed_genotype, sota_gap=3.0)
    assert o1.seed_genotype._seed_meta["creation_round"] == 0
    assert o2.seed_genotype._seed_meta["creation_round"] == 1


def test_maybe_create_passes_adaptive_temperature_to_synthesizer(store):
    """maybe_create 调合成时透传 self._temperature (B4 温度自适应接线)。"""
    temps = []

    class _TempLLM(MockLLM):
        def chat(self, messages, temperature=0.7, max_tokens=4096):
            temps.append(temperature)
            return self.responder(messages)

    resp = iter([_GOOD_PLAN_ARRAY, _GOOD_CODE])
    synth = OperatorSynthesizer(_TempLLM(lambda msgs: next(resp)), SynthesisConfig(max_retries=1))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), temperature=0.66,
                        config=CreationConfig(independent_sampling=False))
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert temps and all(t == pytest.approx(0.66) for t in temps)


# ---------------------------------------------------------------------------
# B4 方向状态持久化 (direction_state.json: OPRO 轨迹 + 温度, 重启回放不归零)
# ---------------------------------------------------------------------------


def _arch_loop(store, llm_responses, archive, **kw):
    """带 B1 履历本的 loop (state_path 缺省 → 派生为履历本同目录 direction_state.json)。"""
    resp = iter(llm_responses)
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    return CreationLoop(store, store.embedder, synth, OperatorRegistry(),
                        config=CreationConfig(independent_sampling=False),
                        archive=archive, **kw)


def test_direction_state_roundtrip_restart(store, tmp_path):
    """写读往返: 轨迹+温度落盘, 重启 (新建 loop 同 archive) 全保真回放, 轮次计数续上。"""
    arch = CreationArchive(str(tmp_path / "creation_archive.jsonl"))
    loop = _arch_loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE], arch)
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0,
                      best_trace=_trace_dict())
    loop.update_direction_outcome(0, 1.0)          # adopted: score +1, 温度 0.8→0.75
    assert (tmp_path / "direction_state.json").exists()
    # 重启: 同 archive → 同目录 direction_state.json 自动回放
    loop2 = _arch_loop(store, [], arch)
    assert loop2._history == loop._history         # score/result_mae/preconditions 全保真
    assert loop2._temperature == pytest.approx(0.75)
    assert loop2._round_idx == 1                   # 轮次续上不归零 (防与旧轮次撞号)


def test_direction_state_persisted_on_append_and_update(store, tmp_path):
    """落盘时机: maybe_create append 轨迹后即落盘; update_direction_outcome 改分调温后再落。"""
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    loop = _arch_loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE], arch)
    state_file = tmp_path / "direction_state.json"
    assert not state_file.exists()                 # 构造只回放, 不落盘
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert len(data["history"]) == 1 and data["history"][0]["score"] == 0.0
    assert data["temperature"] == pytest.approx(0.8)
    loop.update_direction_outcome(0, 1.0)
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["history"][0]["score"] == pytest.approx(1.0)
    assert data["temperature"] == pytest.approx(0.75)


def test_direction_state_bad_file_tolerated(store, tmp_path):
    """坏文件容错: direction_state.json 内容损坏 → 空启动 (不崩)。"""
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    (tmp_path / "direction_state.json").write_text("{not json!!", encoding="utf-8")
    loop = _arch_loop(store, [], arch)
    assert loop._history == [] and loop._round_idx == 0
    assert loop._temperature == pytest.approx(0.8)


def test_direction_state_missing_file_empty_start(store, tmp_path):
    """文件不存在 → 空启动, 且构造不创建文件 (首次 append 才落盘)。"""
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    loop = _arch_loop(store, [], arch)
    assert loop._history == []
    assert not (tmp_path / "direction_state.json").exists()


def test_direction_state_explicit_path_without_archive(store, tmp_path):
    """state_path 显式注入: 无 archive 也持久化到指定路径。"""
    p = tmp_path / "custom_state.json"
    loop = _arch_loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE], archive=None, state_path=str(p))
    assert loop._state_path == str(p)
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    data = json.loads(p.read_text(encoding="utf-8"))
    assert len(data["history"]) == 1


def test_direction_state_disabled_without_archive(store):
    """无 archive 且未显式给 state_path → 不持久化 (_state_path=None), 落盘调用静默 no-op。"""
    loop = _loop(store, [_GOOD_PLAN_ARRAY, _GOOD_CODE])
    assert loop._state_path is None
    loop._persist_direction_state()                # no-op, 不崩
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert loop._state_path is None


# ---------------------------------------------------------------------------
# B5 假设独立采样接线 (independent_sampling 开关两态)
# ---------------------------------------------------------------------------

_IND_PLAN_A = ('{"operator_name": "IndepA", "rationale": "r", "shared_structure": "s",'
               ' "composition": "additive_residual", "source_mechanisms": ["state_space_model"],'
               ' "expected_effect": "e"}')
_IND_PLAN_G = ('{"operator_name": "IndepG", "rationale": "r", "shared_structure": "s",'
               ' "composition": "gated_routed", "source_mechanisms": ["state_space_model"],'
               ' "expected_effect": "e"}')


def _ind_code(name):
    return (f'```python\nimport torch\nimport torch.nn as nn\nclass {name}(nn.Module):\n'
            f'    def __init__(self, channels, num_nodes, **kw):\n'
            f'        super().__init__(); self.l = nn.Linear(channels, channels)\n'
            f'        self.a = nn.Parameter(torch.zeros(1))\n'
            f'    def forward(self, x, adj=None): return self.l(x) + self.a * x\n```')


def _independent_loop(store, responses, **cfg_kw):
    resp = iter(responses)
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    cfg = CreationConfig(n_hypotheses=2, **cfg_kw)
    return CreationLoop(store, store.embedder, synth, OperatorRegistry(), config=cfg)


def test_independent_sampling_default_on(store):
    """开关 True (默认): 走 synthesize_independent —— N 次独立 plan 调用 (组合文法轮转)。"""
    loop = _independent_loop(store, [_IND_PLAN_A, _ind_code("IndepA"),
                                     _IND_PLAN_G, _ind_code("IndepG")])
    assert loop.cfg.independent_sampling is True
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success and outcome.n_success == 2 and outcome.n_hypotheses == 2
    assert len(outcome.seed_genotypes) == 2
    assert loop.synthesizer.llm.call_count == 4          # 2 次独立 plan + 2 次 code
    # 两个注入算子都进算子库
    assert all(n in ops_mod.SPATIAL_OPS for n in outcome.operator_names)


def test_independent_sampling_off_uses_synthesize_many(store):
    """开关 False: 走 synthesize_many —— 单次 plan 调用产 N (旧行为, 向后兼容)。"""
    array = "[" + _IND_PLAN_A + ", " + _IND_PLAN_G + "]"
    loop = _independent_loop(store, [array, _ind_code("IndepA"), _ind_code("IndepG")],
                             independent_sampling=False)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success and outcome.n_success == 2
    assert loop.synthesizer.llm.call_count == 3          # 1 次 plan (数组) + 2 次 code
