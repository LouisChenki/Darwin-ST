"""B7 第三块: 辅助任务合成 + 创造通道路由的回归测试 (MockLLM/mock 检索, 无 LLM/GPU/网络)。

契约:
  - build_aux_plan_prompt/build_aux_code_prompt: AuxTaskPlan schema 字段 + 设计硬约束进 prompt,
    exemplars/insights=None 不注入 (与算子通道同纪律)
  - synthesize_aux_task: plan(设计表)→code(模块)→exec→validate_aux_operator 防泄漏门, bounded 重试
    (合法 plan→合法代码→过门成功; plan 非法→失败; 代码挂门→重试反馈→成功; 重试耗尽→失败带 gate)
  - 创造通道路由: 检索命中自监督/掩码族 (AUX_MECHANISM_FAMILIES) → 辅助任务通道
    (register_aux + seed.aux_op + 履历 composition=aux:<mask_pattern>); 不命中/开关关闭 → 原算子通道
    (行为同 B7 前); 冷启动 best=None → 默认骨架含 aux_op
"""

from __future__ import annotations

import json

import pytest

from darwin_st.creation import (
    AUX_MECHANISM_FAMILIES,
    AuxTaskPlan,
    CreationArchive,
    CreationConfig,
    CreationLoop,
    FusionRequest,
    MockLLM,
    OperatorRegistry,
    OperatorSynthesizer,
    SynthesisConfig,
)
from darwin_st.creation.creation_loop import _should_create_aux, parse_failure_gate
from darwin_st.creation.synthesizer import build_aux_code_prompt, build_aux_plan_prompt
from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
from darwin_st.knowledge.retrieval import RetrievalResult
from darwin_st.search import operators as ops_mod
from darwin_st.search.genotype import Genotype, STBlock


@pytest.fixture(autouse=True)
def _clean_ops():
    """合成注册表全局态清理 (synth 算子 + aux 任务都不跨测试泄漏)。"""
    before_spatial = set(ops_mod.SPATIAL_OPS)
    before_aux = set(ops_mod.AUX_OPS)
    before_cat = set(ops_mod.OP_CATEGORY)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before_spatial:
            ops_mod.SPATIAL_OPS.pop(k, None)
    for k in list(ops_mod.AUX_OPS):
        if k not in before_aux:
            ops_mod.AUX_OPS.pop(k, None)
    for k in list(ops_mod.OP_CATEGORY):
        if k not in before_cat:
            ops_mod.OP_CATEGORY.pop(k, None)


def _aux_req():
    ms = [m for m in all_seed_mechanisms() if m.name in ("masked_autoencoding", "contrastive_learning")]
    return FusionRequest(
        bottleneck="标签稀缺, 表征能力不足",
        target_preconditions=["label_scarcity"],
        mechanisms=ms,
    )


# 合法设计表 (MockLLM 排队响应用)
_AUX_PLAN = json.dumps({
    "task_name": "aux_sensor_mask", "rationale": "掩码重建塑造时空表征",
    "mask_pattern": "sensor", "mask_ratio": 0.25, "recon_target": "raw_input",
    "loss_form": "masked_mae", "decoder_form": "linear",
    "source_mechanisms": ["masked_autoencoding"], "expected_effect": "标签稀缺下提升表征",
}, ensure_ascii=False)

# 合法模块代码: sensor 掩码 + 线性解码 + masked MAE (过全部防泄漏门; 无禁词)
_AUX_GOOD_CODE = '''```python
import torch
import torch.nn as nn

class aux_sensor_mask(nn.Module):
    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, in_channels=3, mask_ratio=0.25, **kw):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.dec = nn.Linear(channels, in_channels)
    def aux_loss(self, h, x):
        mask = torch.rand_like(x[..., 0]) < self.mask_ratio
        pred = self.dec(h)
        err = (pred - x).abs()
        if mask.any():
            return err[mask].mean()
        return err.mean()
```'''

# 泄漏代码: 出现禁词标识符 target (静态扫描门 static_leak 应拒; 其余门本可过)
_AUX_LEAK_CODE = '''```python
import torch
import torch.nn as nn

class aux_sensor_mask(nn.Module):
    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, **kw):
        super().__init__()
        self.dec = nn.Linear(channels, 3)
    def aux_loss(self, h, x):
        target = x
        return (self.dec(h) - target).abs().mean()
```'''


# ---------------------------------------------------------------------------
# prompt 构造: schema 字段 + 设计约束 + None 注入兼容
# ---------------------------------------------------------------------------


def test_build_aux_plan_prompt_schema_fields_and_constraints():
    """system prompt 含 AuxTaskPlan 全字段 + 文献设计硬约束 (掩码率/时空解耦/轻解码器/防泄漏)。"""
    msgs = build_aux_plan_prompt(_aux_req())
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    sys_p = msgs[0]["content"]
    for field in ("task_name", "rationale", "mask_pattern", "mask_ratio", "recon_target",
                  "loss_form", "decoder_form", "source_mechanisms", "expected_effect"):
        assert field in sys_p, f"schema 字段 {field} 不在 plan prompt"
    # 设计硬约束写进 prompt (来自文献调研的硬知识)
    assert "0.25" in sys_p                       # 掩码率默认 (STD-MAE 实证最优)
    assert "时空解耦" in sys_p                    # 时空解耦优于混合
    assert "linear" in sys_p and "mlp" in sys_p  # 解码器要轻
    assert "没有 y" in sys_p                      # 签名无 y 防泄漏
    # 文法枚举也在
    for token in ("sensor", "temporal_patch", "random", "none",
                  "raw_input", "hidden", "masked_mae", "mse", "huber",
                  "light_transformer"):
        assert token in sys_p, f"文法枚举 {token} 不在 plan prompt"
    # user 段与算子通道同构: 机制上下文开头, None → 无履历/经验段
    user = msgs[1]["content"]
    assert user.startswith("瓶颈与跨域机制集:")
    assert "成功先例" not in user and "历史经验" not in user


def test_build_aux_plan_prompt_exemplars_injection_and_none_compat():
    """exemplars 注入方式与算子通道一致: None → 不注入; 非空 → 成功先例+失败教训段。"""
    exemplars = {"successes": [{"operator_name": "aux_old", "composition": "aux:sensor",
                                "seed_val_mae": 18.5, "rationale": "r", "code": "class X: pass"}],
                 "failures": [{"operator_name": "aux_bad", "composition": "aux:random",
                               "rationale": "r2", "gate": "static_leak", "error": "禁词"}]}
    user = build_aux_plan_prompt(_aux_req(), exemplars=exemplars)[1]["content"]
    assert "成功先例" in user and "aux_old" in user
    assert "失败教训" in user and "aux_bad" in user
    # None → 与不注入逐字节一致 (同字段无履历段)
    plain = build_aux_plan_prompt(_aux_req())[1]["content"]
    assert plain == build_aux_plan_prompt(_aux_req(), exemplars=None)[1]["content"]
    assert "成功先例" not in plain


def test_build_aux_plan_prompt_insights_injection_and_none_compat():
    """insights 注入方式与算子通道一致: None/空 → 不注入; 非空 → 反思经验块。"""
    insights = [{"insight_text": "掩码率超 0.4 的方案全挂对抗门", "condition": "fusion:masked",
                 "confidence": 0.85, "id": 7, "evidence_ids": [1, 2]}]
    user = build_aux_plan_prompt(_aux_req(), insights=insights)[1]["content"]
    assert "历史经验" in user and "掩码率超 0.4" in user
    plain = build_aux_plan_prompt(_aux_req())[1]["content"]
    assert plain == build_aux_plan_prompt(_aux_req(), insights=None)[1]["content"]
    assert plain == build_aux_plan_prompt(_aux_req(), insights=[])[1]["content"]


def test_build_aux_code_prompt_contract_and_plan_summary():
    """code prompt: 硬契约 (签名/无 y) + 设计表摘要 + 机制 math_structure; prev_error 反馈。"""
    plan = AuxTaskPlan(task_name="aux_sensor_mask", rationale="r", mask_pattern="sensor",
                       mask_ratio=0.25, recon_target="raw_input", loss_form="masked_mae",
                       decoder_form="linear", source_mechanisms=["masked_autoencoding"],
                       expected_effect="e")
    msgs = build_aux_code_prompt(_aux_req(), plan)
    sys_p = msgs[0]["content"]
    assert "__init__(self, channels, num_nodes, seq_in=12, seq_out=12, **kw)" in sys_p
    assert "aux_loss(self, h, x)" in sys_p
    assert "没有 y" in sys_p
    user = msgs[1]["content"]
    assert "aux_sensor_mask" in user and "sensor" in user          # 设计表摘要
    assert "math_structure" in user                                # 机制数学结构
    assert "验证失败" not in user                                   # 无 prev_error → 无反馈段
    user2 = build_aux_code_prompt(_aux_req(), plan, prev_error="门=static_leak: 禁词")[1]["content"]
    assert "验证失败" in user2 and "static_leak" in user2          # 上轮错误进 prompt


# ---------------------------------------------------------------------------
# synthesize_aux_task 全流程 (MockLLM 排队响应)
# ---------------------------------------------------------------------------


def test_synthesize_aux_task_success():
    """合法 plan → 合法代码 → 过防泄漏门, 一次成功。"""
    responses = iter([_AUX_PLAN, _AUX_GOOD_CODE])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    res = synth.synthesize_aux_task(_aux_req())
    assert res.success, res.last_error
    assert res.operator is not None and res.operator.validated
    assert res.operator.name == "aux_sensor_mask"
    assert res.operator.validation_gate == "all"
    assert res.attempts == 1
    assert res.plan is not None and res.plan.mask_pattern == "sensor" and res.plan.mask_ratio == 0.25


def test_synthesize_aux_task_bad_plan_fails():
    """plan 非法 (mask_ratio 超界) → 计划阶段失败, 不进代码阶段。"""
    bad_plan = json.dumps({
        "task_name": "aux_sensor_mask", "rationale": "r", "mask_pattern": "sensor",
        "mask_ratio": 0.9, "recon_target": "raw_input", "loss_form": "masked_mae",
        "decoder_form": "linear", "source_mechanisms": ["masked_autoencoding"],
        "expected_effect": "e"}, ensure_ascii=False)
    llm = MockLLM(lambda msgs: bad_plan)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    res = synth.synthesize_aux_task(_aux_req())
    assert not res.success
    assert res.attempts == 0
    assert "计划" in res.last_error and "mask_ratio" in res.last_error
    assert llm.call_count == 1                        # 只调了 plan, 未进代码阶段


def test_synthesize_aux_task_retry_feedback_then_success():
    """代码挂静态泄漏门 → 错误反馈进下轮 prompt → 第二轮给好代码 → 成功。"""
    seen_prompts = []

    def responder(msgs):
        seen_prompts.append(msgs[-1]["content"])
        if len(seen_prompts) == 1:
            return _AUX_PLAN
        if len(seen_prompts) == 2:
            return _AUX_LEAK_CODE                     # 第一轮挂 static_leak 门
        return _AUX_GOOD_CODE

    llm = MockLLM(responder)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3))
    res = synth.synthesize_aux_task(_aux_req())
    assert res.success, res.last_error
    assert res.attempts == 2
    # 第三轮 (第二轮代码) 的 prompt 应含上一轮验证错误反馈
    assert "验证失败" in seen_prompts[2] and "static_leak" in seen_prompts[2]


def test_synthesize_aux_task_fails_after_max_retries_with_gate():
    """一直挂门 → 重试耗尽 → 失败带 gate (parse_failure_gate 可解析)。"""
    responses = iter([_AUX_PLAN] + [_AUX_LEAK_CODE] * 5)
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    res = synth.synthesize_aux_task(_aux_req())
    assert not res.success
    assert res.attempts == 2
    assert "验证" in res.last_error
    assert parse_failure_gate(res.last_error) == "static_leak"
    assert res.plan is not None                       # 失败结果仍带 plan (履历用)


def test_synthesize_aux_task_exemplars_insights_passthrough():
    """exemplars/insights 透传进 plan prompt (非空注入); 不影响合成主流程。"""
    prompts = []

    def responder(msgs):
        prompts.append(msgs[-1]["content"])
        return _AUX_PLAN if len(prompts) == 1 else _AUX_GOOD_CODE

    llm = MockLLM(responder)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    exemplars = {"successes": [{"operator_name": "aux_old", "composition": "aux:sensor",
                                "seed_val_mae": 18.5, "rationale": "r", "code": "class X: pass"}],
                 "failures": []}
    insights = [{"insight_text": "轻解码器方案全部过门", "condition": "fusion:masked",
                 "confidence": 0.8, "id": 3, "evidence_ids": []}]
    res = synth.synthesize_aux_task(_aux_req(), exemplars=exemplars, insights=insights)
    assert res.success
    assert "成功先例" in prompts[0] and "历史经验" in prompts[0]


# ---------------------------------------------------------------------------
# synthesize_aux_task: code_backend (Aider 路径, mock backend)
# ---------------------------------------------------------------------------


class _MockBackend:
    """mock 代码后端: 按队列返回 (code, error)。"""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def write_operator(self, instruction, target_file="op.py"):
        self.calls.append(instruction)
        return next(self.outputs)


_GOOD_AUX_RAW = (
    "import torch\nimport torch.nn as nn\n"
    "class aux_sensor_mask(nn.Module):\n"
    "    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, in_channels=3, mask_ratio=0.25, **kw):\n"
    "        super().__init__()\n"
    "        self.mask_ratio = mask_ratio\n"
    "        self.dec = nn.Linear(channels, in_channels)\n"
    "    def aux_loss(self, h, x):\n"
    "        mask = torch.rand_like(x[..., 0]) < self.mask_ratio\n"
    "        pred = self.dec(h)\n"
    "        err = (pred - x).abs()\n"
    "        if mask.any():\n"
    "            return err[mask].mean()\n"
    "        return err.mean()\n"
)

_LEAK_AUX_RAW = (
    "import torch\nimport torch.nn as nn\n"
    "class aux_sensor_mask(nn.Module):\n"
    "    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, **kw):\n"
    "        super().__init__()\n"
    "        self.dec = nn.Linear(channels, 3)\n"
    "    def aux_loss(self, h, x):\n"
    "        target = x\n"
    "        return (self.dec(h) - target).abs().mean()\n"
)


def test_synthesize_aux_task_with_code_backend():
    """code_backend 路径: plan 用 llm, 代码用后端; 挂门后错误反馈进下轮指令。"""
    llm = MockLLM(lambda msgs: _AUX_PLAN)             # 只需出设计表
    backend = _MockBackend([(_LEAK_AUX_RAW, ""), (_GOOD_AUX_RAW, "")])
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3), code_backend=backend)
    res = synth.synthesize_aux_task(_aux_req())
    assert res.success, res.last_error
    assert res.attempts == 2
    assert len(backend.calls) == 2
    assert "aux_loss(self, h, x)" in backend.calls[0]            # 指令含硬契约
    assert "static_leak" in backend.calls[1]                     # 上轮错误进下轮指令


# ---------------------------------------------------------------------------
# 创造通道路由 (_should_create_aux + maybe_create 接线; mock 检索)
# ---------------------------------------------------------------------------


def test_should_create_aux_hit_and_miss():
    """路由判定: 命中自监督/掩码族任一机制 → True; 全不命中/空集 → False。"""
    by_name = {m.name: m for m in all_seed_mechanisms()}
    assert _should_create_aux([by_name["masked_autoencoding"]])
    assert _should_create_aux([by_name["state_space_model"], by_name["contrastive_learning"]])
    assert not _should_create_aux([by_name["state_space_model"], by_name["self_attention"]])
    assert not _should_create_aux([])
    # 可扩展集合: 615 卡的掩码族卡名在册 (新卡名加入集合即路由, 无需改代码)
    assert "decoupled_st_masked_pretraining" in AUX_MECHANISM_FAMILIES
    assert "multi_strategy_patch_pretraining" in AUX_MECHANISM_FAMILIES
    assert "correlation_adaptive_augmentation" in AUX_MECHANISM_FAMILIES


def _retrieval_result(names):
    mechs = [m for m in all_seed_mechanisms() if m.name in names]
    assert len(mechs) == len(names), f"种子库缺机制: {names}"
    return RetrievalResult(mechanisms=mechs, covered_preconditions=set(),
                           target_preconditions=[], target_domain="ST", rationale=[])


def _routed_loop(monkeypatch, mech_names, llm_responses, archive=None, **cfg_kw):
    """检索被 mock 的 CreationLoop: find_cross_domain_analogy 固定返回指定机制集。"""
    resp = iter(llm_responses)
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    reg = OperatorRegistry()
    store = InMemoryGraphStore(HashEmbedder(dim=256))   # 检索已 patch, store 仅为构造参数
    monkeypatch.setattr("darwin_st.creation.creation_loop.find_cross_domain_analogy",
                        lambda *a, **kw: _retrieval_result(mech_names))
    return CreationLoop(store, store.embedder, synth, reg,
                        config=CreationConfig(**cfg_kw), archive=archive), reg


def test_route_hit_selfsupervised_goes_aux_channel(monkeypatch, tmp_path):
    """命中自监督族 → 辅助任务通道: register_aux 被调 + seed.aux_op 正确 + 履历 aux:<pattern>。"""
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    loop, reg = _routed_loop(monkeypatch, ["masked_autoencoding", "state_space_model"],
                             [_AUX_PLAN, _AUX_GOOD_CODE], archive=arch)
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create(best, sota_gap=3.0, best_hps={"lr": 1e-3})
    assert outcome.success, outcome.reason
    assert outcome.n_hypotheses == 1 and outcome.n_success == 1
    # register_aux 被调: 注册名 aux_ 前缀, 进 AUX_OPS 而非 SPATIAL_OPS
    assert outcome.operator_name == "aux_sensor_mask"
    assert outcome.operator_name in ops_mod.AUX_OPS
    assert reg.is_registered_aux("aux_sensor_mask")
    assert "aux_sensor_mask" not in ops_mod.SPATIAL_OPS
    # seed: 架构不动 (best.copy), 只挂 aux_op 槽; _seed_meta 机制同算子通道
    seed = outcome.seed_genotype
    assert seed.aux_op == "aux_sensor_mask"
    b0 = seed.blocks[0]
    assert (b0.spatial_op, b0.temporal_op, b0.joint_op) == ("gcn", "tcn", None)
    assert seed.signature() != best.signature()        # 带 aux 是不同 trial (签名含 aux_op)
    meta = seed._seed_meta
    assert meta["warm_start_hps"] == {"lr": 1e-3}
    assert meta["hpo_trials"] == loop.cfg.seed_hpo_trials
    assert meta["is_creation_seed"] is True
    assert meta["operator_name"] == "aux_sensor_mask"
    assert meta["creation_round"] == 0
    # 履历: composition=aux:<mask_pattern>, operator_name=注册名, mechanisms 照记
    recs = arch._load()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.gate == "all"
    assert rec.composition == "aux:sensor"
    assert rec.operator_name == "aux_sensor_mask"
    assert set(rec.mechanisms) == {"masked_autoencoding", "state_space_model"}
    assert rec.seed_signature == seed.signature()
    # 算子注册表无新算子 (辅助通道不碰 SPATIAL_OPS)
    assert reg.registered_names() == []


_OP_PLAN_ARRAY = ('[{"operator_name": "RoutedFusionOp", "rationale": "r",'
                  ' "shared_structure": "s", "composition": "additive_residual",'
                  ' "source_mechanisms": ["state_space_model"], "expected_effect": "e"}]')

_OP_GOOD_CODE = '''```python
import torch
import torch.nn as nn
class RoutedFusionOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.l = nn.Linear(channels, channels)
        self.a = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        return self.l(x) + self.a * x
```'''


def test_route_miss_goes_operator_channel(monkeypatch):
    """不命中自监督族 → 原算子通道 (行为同 B7 前): synth_ 注册 + seed 无 aux_op。"""
    loop, reg = _routed_loop(monkeypatch, ["state_space_model", "self_attention"],
                             [_OP_PLAN_ARRAY, _OP_GOOD_CODE], independent_sampling=False)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success, outcome.reason
    assert outcome.operator_name == "synth_RoutedFusionOp"
    assert outcome.operator_name in ops_mod.SPATIAL_OPS
    assert outcome.seed_genotype.aux_op is None        # 算子通道不挂 aux
    assert reg.registered_aux_names() == []            # 不碰 AUX_OPS


def test_route_aux_disabled_goes_operator_channel(monkeypatch):
    """enable_aux_creation=False: 即使命中自监督族也全走原算子通道。"""
    loop, reg = _routed_loop(monkeypatch, ["masked_autoencoding"],
                             [_OP_PLAN_ARRAY, _OP_GOOD_CODE],
                             independent_sampling=False, enable_aux_creation=False)
    assert loop.cfg.enable_aux_creation is False
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success, outcome.reason
    assert outcome.operator_name.startswith("synth_")
    assert outcome.seed_genotype.aux_op is None
    assert reg.registered_aux_names() == []


def test_route_aux_cold_start_best_none(monkeypatch):
    """冷启动 (best=None): seed 用默认骨架 (gcn+tcn) 且挂 aux_op。"""
    loop, reg = _routed_loop(monkeypatch, ["contrastive_learning"],
                             [_AUX_PLAN, _AUX_GOOD_CODE])
    outcome = loop.maybe_create(None, sota_gap=None)
    assert outcome.success, outcome.reason
    seed = outcome.seed_genotype
    assert seed.aux_op == "aux_sensor_mask"
    b0 = seed.blocks[0]
    assert (b0.spatial_op, b0.temporal_op, b0.joint_op) == ("gcn", "tcn", None)
    assert seed.hidden == 64                           # 默认骨架 (参照 _make_seed_genotype best=None 分支)
    assert seed._seed_meta["warm_start_hps"] is None   # 无父超参
    assert seed._seed_meta["is_creation_seed"] is True


def test_route_aux_failure_discipline(monkeypatch, tmp_path):
    """辅助通道失败: 与算子通道同纪律 —— 不崩 + 履历记失败门 + 方向信用分 −1 + 升温。"""
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    loop, reg = _routed_loop(monkeypatch, ["masked_autoencoding"],
                             [_AUX_PLAN, _AUX_LEAK_CODE, _AUX_LEAK_CODE], archive=arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert not outcome.success                          # 优雅失败, 不中止闭环
    assert outcome.n_success == 0 and outcome.insight
    assert reg.registered_aux_names() == []             # 未过门不注册
    assert loop._history[0]["score"] == -1.0            # B4 信用分照常
    assert loop._temperature == pytest.approx(0.85)     # 升温探索照常
    recs = arch._load()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.gate == "static_leak"                    # 履历记失败门
    assert rec.composition == "aux:sensor"              # 失败也记设计表掩码模式
    assert rec.operator_name == "aux_sensor_mask"       # 无注册名 → 记 plan 的 task_name
    assert rec.error and "static_leak" in rec.error


# ---------------------------------------------------------------------------
# v2 强制 aux 通道 (配额驱动, 绕过检索; 见 action_ledger/orchestrator 三段调度)
# ---------------------------------------------------------------------------


def _store_with_aux_cards(names=("masked_autoencoding", "contrastive_learning")):
    store = InMemoryGraphStore(HashEmbedder(dim=256))
    by_name = {m.name: m for m in all_seed_mechanisms()}
    for n in names:
        store.add_mechanism(by_name[n])
    return store


def test_aux_cards_available_preflight():
    """preflight: 库里有/无 aux 卡。"""
    loop_ok = CreationLoop(_store_with_aux_cards(), None, None, OperatorRegistry())
    assert loop_ok.aux_cards_available() is True
    store_empty = InMemoryGraphStore(HashEmbedder(dim=256))
    by_name = {m.name: m for m in all_seed_mechanisms()}
    store_empty.add_mechanism(by_name["state_space_model"])     # 非 aux 卡
    loop_no = CreationLoop(store_empty, None, None, OperatorRegistry())
    assert loop_no.aux_cards_available() is False


def test_pick_aux_mechanisms_least_tried_rotation(tmp_path):
    """强制选卡: 历史尝试最少优先 (账本统计), 不重复砸同一张卡。"""
    from darwin_st.creation.action_ledger import ActionLedger
    store = _store_with_aux_cards()
    ledger = ActionLedger(str(tmp_path / "tier2_actions.jsonl"))
    loop = CreationLoop(store, None, None, OperatorRegistry(), action_ledger=ledger)
    first = loop.pick_aux_mechanisms(max_pick=1)[0].name
    # 记一次实际启动的 aux 动作 (选了 first)
    aid = ledger.start(0, "exp/t", requested_action_type="aux_create",
                       action_type="aux_create", decision_reason="forced_aux_quota",
                       selected_mechanism_ids=[first])
    ledger.finish(aid, 0, "failed")
    second = loop.pick_aux_mechanisms(max_pick=1)[0].name
    assert second != first                                      # 轮到尝试次数更少的卡


def test_maybe_create_aux_direct_end_to_end(tmp_path):
    """强制 aux 全链: 绕过检索直接合成 → 七门 → register_aux → seed 挂 aux_op → 履历带动作上下文。"""
    from darwin_st.creation.action_ledger import ActionLedger
    resp = iter([_AUX_PLAN, _AUX_GOOD_CODE])
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    reg = OperatorRegistry()
    store = _store_with_aux_cards()
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    ledger = ActionLedger(str(tmp_path / "tier2_actions.jsonl"))
    loop = CreationLoop(store, None, synth, reg, config=CreationConfig(),
                        archive=arch, action_ledger=ledger)
    loop.set_action_ctx("exp/t#a0", 0, "exp/t")
    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    outcome = loop.maybe_create_aux_direct(best, run_tag="exp/t", best_hps={"lr": 1e-3})
    assert outcome.success, outcome.reason
    assert outcome.action_type == "aux_create"
    assert outcome.operator_name.startswith("aux_")
    assert outcome.seed_genotypes[0].aux_op == outcome.operator_name
    recs = arch._load()
    assert len(recs) == 1
    assert recs[0].composition.startswith("aux:")                # aux:<mask_pattern>
    assert recs[0].action_id == "exp/t#a0" and recs[0].action_seq == 0
    assert recs[0].run_tag == "exp/t"


def test_maybe_create_aux_direct_no_cards(tmp_path):
    """无可用 aux 卡 → 失败 outcome (不抛异常; preflight 本该拦住, 这里是防御)。"""
    store = InMemoryGraphStore(HashEmbedder(dim=256))           # 空库
    loop = CreationLoop(store, None, None, OperatorRegistry())
    outcome = loop.maybe_create_aux_direct(None)
    assert not outcome.success and "无可用 aux 机制卡" in outcome.reason
    assert outcome.action_type == "aux_create"


def test_set_action_ctx_clears_insight_collector(tmp_path):
    """set_action_ctx 初始化动作级 insight collector (防跨动作串值)。"""
    loop = CreationLoop(InMemoryGraphStore(HashEmbedder(dim=256)), None, None,
                        OperatorRegistry())
    loop._last_used_insight_ids = [1, 2, 3]
    loop.set_action_ctx("a0", 0, "exp/t")
    assert loop._last_used_insight_ids == []
    assert loop._action_ctx["action_id"] == "a0"
