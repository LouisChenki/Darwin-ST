"""B2 算子谱系 + 微进化的正确性回归测试 (mock 全链, 无 LLM/GPU/网络)。

核心契约:
  - 家谱 (registry): register_variant 版本递增 + lineage 持久化/旧目录兼容 +
    update_real_mae 回填 + best_in_family (None mae 排除)
  - refiner: 三种 mode 的 prompt 各含对应指令 + 父代码/MAE; best_shot 双版本对比续写;
    验证失败 bounded 重试 + 错误反馈; 族名/版本命名
  - creation_loop.maybe_refine: 无候选 → None; 有候选 → 精炼 + 注册变体 + seed meta + 履历
  - orchestrator: refinement seed 的实测 MAE 回填 registry 家谱 (CRASH 不回填);
    _try_creation 精炼优先、无候选回落从零创造
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from darwin_st.creation import (
    CreationArchive,
    CreationConfig,
    CreationLoop,
    MockLLM,
    OperatorRefiner,
    OperatorRegistry,
    RefineConfig,
    SynthesisResult,
    family_of,
    version_of,
)
from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator
from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
from darwin_st.search import operators as ops_mod
from darwin_st.search.genotype import Genotype, STBlock


# 自包含的合法算子代码模板 (过 6 门验证: 非平凡线性主路径 + 零初始化残差)
_CODE = '''
import torch
import torch.nn as nn
class Foo(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.l = nn.Linear(channels, channels)
        self.a = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        return self.l(x) + self.a * x
'''

# 坏代码: 丢掉节点维 N (挂在 shape 门)
_BAD_CODE = '''```python
import torch.nn as nn
class Foo_v2(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.l = nn.Linear(channels, channels)
    def forward(self, x, adj=None):
        return self.l(x).mean(dim=2)
```'''


def _mkop(name, validated=True, real_mae=None):
    plan = FusionPlan(operator_name=name, rationale="r", shared_structure="s",
                      composition="additive_residual", source_mechanisms=["m1", "m2"],
                      expected_effect="e")
    return SynthesizedOperator(name=name, code=_CODE.replace("Foo", name), plan=plan,
                               needs_adj=False, validated=validated, real_mae=real_mae)


def _fenced_code(name):
    return "```python\n" + _CODE.replace("Foo", name) + "\n```"


def _refined_result(name="Foo_v2"):
    """伪造一次成功的精炼结果 (spy refiner 用)。"""
    plan = FusionPlan(operator_name=name, rationale="refine(small): x", shared_structure="s",
                      composition="additive_residual", source_mechanisms=["m1", "m2"],
                      expected_effect="e")
    op = SynthesizedOperator(name=name, code=_CODE.replace("Foo", name), plan=plan,
                             validated=True)
    return SynthesisResult(True, operator=op, attempts=1, plan=plan)


class _SpyRefiner:
    """记录 refine/best_shot 调用并返回预设结果 (CreationLoop 测试用)。"""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def refine(self, parent, mode, round_idx):
        self.calls.append(("refine", dict(parent), mode, round_idx))
        return self.result

    def best_shot(self, versions, round_idx):
        self.calls.append(("best_shot", list(versions), round_idx))
        return self.result


@pytest.fixture(autouse=True)
def _clean_spatial_ops():
    """每个测试后清理注入的 synth 算子, 不污染全局 SPATIAL_OPS。"""
    before = set(ops_mod.SPATIAL_OPS)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before:
            ops_mod.SPATIAL_OPS.pop(k, None)


@pytest.fixture
def store():
    s = InMemoryGraphStore(HashEmbedder(dim=128))
    for m in all_seed_mechanisms():
        s.add_mechanism(m)
    return s


def _loop(store, registry, refiner=None, archive=None, cfg=None):
    from darwin_st.creation import OperatorSynthesizer, SynthesisConfig
    synth = OperatorSynthesizer(MockLLM(lambda m: ""), SynthesisConfig(max_retries=1))
    return CreationLoop(store, store.embedder, synth, registry,
                        config=cfg or CreationConfig(), archive=archive, refiner=refiner)


# ---------------------------------------------------------------------------
# 家谱: 命名 / 注册变体 / 版本递增
# ---------------------------------------------------------------------------


def test_family_of_and_version_of():
    assert family_of("synth_foo") == "synth_foo"
    assert family_of("synth_foo_v2") == "synth_foo"
    assert family_of("synth_foo_v12") == "synth_foo"
    assert version_of("synth_foo") == 1
    assert version_of("synth_foo_v3") == 3


def test_register_variant_version_increments():
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1, round_idx=3)
    r3 = reg.register_variant(_mkop("Foo_v3"), r1, round_idx=4)
    assert (r1, r2, r3) == ("synth_Foo", "synth_Foo_v2", "synth_Foo_v3")
    assert r2 in ops_mod.SPATIAL_OPS and r3 in ops_mod.SPATIAL_OPS
    assert reg.families() == ["synth_Foo"]
    vs = reg.family_versions("synth_Foo")
    assert [v["reg_name"] for v in vs] == [r1, r2, r3]          # 按版本序
    assert [v["version"] for v in vs] == [1, 2, 3]
    assert vs[1]["parent_reg_name"] == r1                        # 变体带父指针
    assert vs[1]["round_idx"] == 3
    assert vs[0]["parent_reg_name"] is None                      # 首版无父


def test_register_variant_family_mismatch_raises():
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    with pytest.raises(ValueError, match="不同族"):
        reg.register_variant(_mkop("Bar_v2"), r1)                # 别族名拒绝


def test_register_variant_duplicate_version_raises():
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    reg.register_variant(_mkop("Foo_v2"), r1)
    with pytest.raises(ValueError, match="已存在"):
        reg.register_variant(_mkop("Foo_v2"), r1)                # 版本不重复


def test_register_variant_unknown_parent_raises():
    reg = OperatorRegistry()
    with pytest.raises(ValueError, match="未注册"):
        reg.register_variant(_mkop("Foo_v2"), "synth_Foo")


def test_register_variant_rejects_unvalidated():
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    with pytest.raises(ValueError):
        reg.register_variant(_mkop("Foo_v2", validated=False), r1)


# ---------------------------------------------------------------------------
# 家谱: 持久化 + 重载 (含旧目录兼容)
# ---------------------------------------------------------------------------


def test_lineage_persisted_and_reloaded(tmp_path):
    reg = OperatorRegistry(persist_dir=str(tmp_path))
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1, round_idx=5)
    reg.update_real_mae(r1, 20.5)
    reg.update_real_mae(r2, 19.8)

    lineage_path = tmp_path / "lineage_synth_Foo.json"
    assert lineage_path.exists()
    data = json.loads(lineage_path.read_text())
    assert data["family"] == "synth_Foo"
    assert [v["reg_name"] for v in data["versions"]] == [r1, r2]
    assert data["versions"][0]["real_mae"] == 20.5

    # 模拟重启: 清全局注册后重载
    for k in (r1, r2):
        ops_mod.SPATIAL_OPS.pop(k, None)
    reg2 = OperatorRegistry(persist_dir=str(tmp_path))
    loaded = reg2.load_persisted()
    assert set(loaded) == {r1, r2}
    assert reg2.families() == ["synth_Foo"]
    vs = reg2.family_versions("synth_Foo")
    assert [v["version"] for v in vs] == [1, 2]
    best = reg2.best_in_family("synth_Foo")
    assert best["reg_name"] == r2 and best["real_mae"] == 19.8
    assert best["code"] and "class Foo_v2" in best["code"]       # enrich 带代码
    assert best["composition"] == "additive_residual"


def test_load_persisted_legacy_dir_without_lineage(tmp_path):
    """旧目录 (无 lineage 文件, 算子 json 无新字段) 正常加载: 单版本族隐式成立。"""
    # 手写旧式持久化产物 (B2 之前的格式)
    (tmp_path / "synth_Old.py").write_text(_CODE.replace("Foo", "Old"))
    (tmp_path / "synth_Old.json").write_text(json.dumps({
        "reg_name": "synth_Old", "class_name": "Old", "needs_adj": False,
        "plan_name": "Old", "composition": "sequential", "source_mechanisms": ["x"],
        "category": "spatiotemporal", "real_mae": 21.3}))

    reg = OperatorRegistry(persist_dir=str(tmp_path))
    loaded = reg.load_persisted()
    assert loaded == ["synth_Old"]
    assert reg.families() == ["synth_Old"]                        # 隐式单版本族
    vs = reg.family_versions("synth_Old")
    assert len(vs) == 1 and vs[0]["version"] == 1
    assert vs[0]["parent_reg_name"] is None
    assert vs[0]["real_mae"] == 21.3                              # 旧 json 的 real_mae 钩子读回
    assert reg.best_in_family("synth_Old")["reg_name"] == "synth_Old"


def test_update_real_mae_syncs_operator_json(tmp_path):
    reg = OperatorRegistry(persist_dir=str(tmp_path))
    r1 = reg.register(_mkop("Foo"))
    assert reg.best_in_family("synth_Foo") is None                # 无 mae → None
    reg.update_real_mae(r1, 20.5)
    meta = json.loads((tmp_path / "synth_Foo.json").read_text())
    assert meta["real_mae"] == 20.5                               # 算子 json 同步
    reg.update_real_mae("synth_ghost", 1.0)                       # 未知名静默跳过, 不崩


def test_best_in_family_picks_min_finite_mae():
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1)
    r3 = reg.register_variant(_mkop("Foo_v3"), r1)
    assert reg.best_in_family("synth_Foo") is None                # 全无 mae
    reg.update_real_mae(r2, 19.8)
    reg.update_real_mae(r3, 21.0)
    # r1 无 mae (None) 不参与比较
    best = reg.best_in_family("synth_Foo")
    assert best["reg_name"] == r2 and best["real_mae"] == 19.8
    reg.update_real_mae(r1, 19.5)
    assert reg.best_in_family("synth_Foo")["reg_name"] == r1      # 更小者胜出 (查询端天然替换)


# ---------------------------------------------------------------------------
# refiner: refine 三种 mode 的 prompt
# ---------------------------------------------------------------------------


def _parent(reg_name="synth_Foo", mae=19.8, next_version=2):
    return {"reg_name": reg_name, "code": _CODE.replace("Foo", "Foo"),
            "real_mae": mae, "composition": "sequential", "source_mechanisms": ["m1", "m2"],
            "needs_adj": False, "next_version": next_version}


@pytest.mark.parametrize("mode,keyword", [
    ("small", "微调内部系数"),
    ("param", "只改数值超参"),
    ("struct", "只改一处结构"),
])
def test_refine_prompt_contains_mode_instruction_parent_code_and_mae(mode, keyword):
    captured = []
    llm = MockLLM(lambda msgs: (captured.append(msgs), _fenced_code("Foo_v2"))[1])
    ref = OperatorRefiner(llm, RefineConfig(max_retries=1))
    res = ref.refine(_parent(), mode, round_idx=2)
    assert res.success, res.last_error
    assert len(captured) == 1
    user = captured[0][1]["content"]
    assert keyword in user                                       # mode 对应中文指令
    assert f"mode={mode}" in user
    assert "class Foo(" in user                                  # 父代码全文
    assert "19.8000" in user                                     # 父实测 MAE
    assert "synth_Foo" in user
    assert "Foo_v2" in user                                      # 指定新类名
    contract = captured[0][0]["content"]
    for kw in ("[B, T, N, C]", "绝不能丢节点维 N", "可微", "标签", "参数量"):
        assert kw in contract                                    # 与 synthesizer 一致的硬契约
    # 返回值语义: SynthesisResult + validated + plan 沿用父版
    assert res.operator.validated and res.operator.name == "Foo_v2"
    assert res.plan.composition == "sequential"                  # 沿用父版
    assert res.plan.source_mechanisms == ["m1", "m2"]            # 沿用父版
    assert res.plan.rationale.startswith(f"refine({mode}):")     # rationale 记指令


def test_refine_rejects_unknown_mode():
    ref = OperatorRefiner(MockLLM(lambda m: ""))
    with pytest.raises(ValueError, match="未知 refine mode"):
        ref.refine(_parent(), "wild", round_idx=0)


def test_refine_accepts_object_parent():
    """parent 也可以是同名属性对象 (spec: dict 或对象)。"""
    llm = MockLLM(lambda msgs: _fenced_code("Foo_v2"))
    ref = OperatorRefiner(llm, RefineConfig(max_retries=1))
    obj = SimpleNamespace(**_parent())
    res = ref.refine(obj, "small", round_idx=0)
    assert res.success and res.operator.name == "Foo_v2"


def test_refine_names_next_version_from_parent_reg():
    """缺省 next_version → 父版版本号 + 1 (synth_Foo_v2 → Foo_v3)。"""
    llm = MockLLM(lambda msgs: _fenced_code("Foo_v3"))
    ref = OperatorRefiner(llm, RefineConfig(max_retries=1))
    res = ref.refine(_parent(reg_name="synth_Foo_v2", next_version=None), "small", round_idx=0)
    assert res.success and res.operator.name == "Foo_v3"


def test_refine_retry_feedback_and_success():
    """第一次坏代码 (shape 门) → 第二次 prompt 带错误反馈 → 成功, attempts=2。"""
    captured = []
    responses = iter([_BAD_CODE, _fenced_code("Foo_v2")])
    llm = MockLLM(lambda msgs: (captured.append(msgs), next(responses))[1])
    ref = OperatorRefiner(llm, RefineConfig(max_retries=2))
    res = ref.refine(_parent(), "small", round_idx=0)
    assert res.success, res.last_error
    assert res.attempts == 2
    assert len(captured) == 2
    second_user = captured[1][1]["content"]
    assert "上一版代码验证失败" in second_user                     # 错误反馈进下一轮 prompt
    assert "门=shape" in second_user


def test_refine_all_retries_fail():
    llm = MockLLM(lambda msgs: _BAD_CODE)
    ref = OperatorRefiner(llm, RefineConfig(max_retries=2))
    res = ref.refine(_parent(), "small", round_idx=0)
    assert not res.success
    assert res.attempts == 2
    assert "重试 2 次" in res.last_error
    assert res.operator is None


def test_refine_uses_lower_temperature():
    seen = {}

    class _TempSpyLLM:
        def chat(self, messages, temperature=0.7, max_tokens=4096):
            seen["t"] = temperature
            return _fenced_code("Foo_v2")

    ref = OperatorRefiner(_TempSpyLLM(), RefineConfig(max_retries=1))
    assert ref.cfg.temperature == 0.6                              # 比创造 (0.8) 低
    ref.refine(_parent(), "param", round_idx=0)
    assert seen["t"] == 0.6


# ---------------------------------------------------------------------------
# refiner: best_shot
# ---------------------------------------------------------------------------


def _versions_for_best_shot():
    return [
        {"reg_name": "synth_Foo", "version": 1, "code": _CODE.replace("Foo", "Foo"),
         "real_mae": 20.0, "composition": "additive_residual", "source_mechanisms": ["m1"]},
        {"reg_name": "synth_Foo_v2", "version": 2, "code": _CODE.replace("Foo", "Foo_v2"),
         "real_mae": 19.0, "composition": "additive_residual", "source_mechanisms": ["m1"]},
    ]


def test_best_shot_prompt_two_codes_mae_order_and_improved_header():
    captured = []
    llm = MockLLM(lambda msgs: (captured.append(msgs), _fenced_code("Foo_v3"))[1])
    ref = OperatorRefiner(llm, RefineConfig(max_retries=1))
    res = ref.best_shot(_versions_for_best_shot(), round_idx=3)
    assert res.success, res.last_error
    assert res.operator.name == "Foo_v3"                           # max 版本 + 1
    user = captured[0][1]["content"]
    assert "class Foo(" in user and "class Foo_v2(" in user        # 两份代码
    assert "20.0000" in user and "19.0000" in user                 # 两份 MAE
    i_worse, i_better = user.index("== 较差版"), user.index("== 较好版")
    assert i_worse < i_better
    assert "较差版 synth_Foo (实测 MAE=20.0000)" in user           # 差=MAE 高
    assert "较好版 synth_Foo_v2 (实测 MAE=19.0000)" in user        # 好=MAE 低
    assert "Improved version of synth_Foo_v2 and synth_Foo" in user  # 空版头 docstring
    assert f"class Foo_v3(nn.Module):" in user                     # 新版类头
    assert res.plan.rationale.startswith("refine(best_shot):")
    assert res.plan.source_mechanisms == ["m1"]                    # 沿用较好版


def test_best_shot_requires_two_scored_versions():
    ref = OperatorRefiner(MockLLM(lambda m: _fenced_code("Foo_v3")), RefineConfig(max_retries=1))
    one = _versions_for_best_shot()[:1]
    res = ref.best_shot(one, round_idx=0)
    assert not res.success and res.attempts == 0
    assert "≥2" in res.last_error


# ---------------------------------------------------------------------------
# creation_loop.maybe_refine
# ---------------------------------------------------------------------------


def test_maybe_refine_no_candidate_returns_none(store):
    """空家谱 / 有族但无实测 MAE / 无精炼器 → None (调用方回落从零创造)。"""
    reg = OperatorRegistry()
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    assert loop.maybe_refine(round_idx=0) is None                  # 空家谱

    r1 = reg.register(_mkop("Foo"))
    assert loop.maybe_refine(round_idx=0) is None                  # 有族无 MAE

    reg.update_real_mae(r1, 20.0)
    loop_no_refiner = _loop(store, reg, refiner=None)
    loop_no_refiner.refiner = None                                 # 显式无精炼器
    assert loop_no_refiner.maybe_refine(round_idx=0) is None


def test_maybe_refine_success_registers_variant_and_seeds(store, tmp_path):
    """有候选 → 走 refine → register_variant + seed meta (is_refinement/family) + 履历。"""
    reg = OperatorRegistry(persist_dir=str(tmp_path))
    r1 = reg.register(_mkop("Foo"))
    reg.update_real_mae(r1, 20.0)
    spy = _SpyRefiner(_refined_result("Foo_v2"))
    arch = CreationArchive(str(tmp_path / "archive.jsonl"))
    loop = _loop(store, reg, refiner=spy, archive=arch)

    best = Genotype(blocks=[STBlock("gcn", "tcn")])
    out = loop.maybe_refine(round_idx=7, best_genotype=best, best_hps={"lr": 1e-3},
                            current_best_mae=20.1)
    assert out is not None and out.success, out.reason
    kind, parent, mode, ridx = spy.calls[0]
    assert kind == "refine" and ridx == 7
    assert mode in ("small", "param", "struct")
    assert parent["reg_name"] == r1 and parent["next_version"] == 2  # 族 best 版 + 版本号算好

    reg_name = out.operator_name
    assert reg_name == "synth_Foo_v2"
    assert reg.is_registered(reg_name)
    vs = reg.family_versions("synth_Foo")
    assert [v["reg_name"] for v in vs] == [r1, reg_name]             # 家谱只增
    assert vs[1]["round_idx"] == 7

    seed = out.seed_genotype
    assert seed._seed_meta["is_creation_seed"] is True
    assert seed._seed_meta["is_refinement"] is True
    assert seed._seed_meta["family"] == "synth_Foo"
    assert seed._seed_meta["operator_name"] == reg_name
    assert seed._seed_meta["warm_start_hps"] == {"lr": 1e-3}         # 复用 warm-start
    assert seed.blocks[0].joint_op == reg_name                       # 时空算子进 joint 槽

    # B1 履历也记: composition=refine:<mode>, mechanisms 沿用父版 plan
    ctx = arch.exemplar_context()
    assert len(ctx["successes"]) == 1
    rec = ctx["successes"][0]
    assert rec["operator_name"] == reg_name
    assert rec["composition"].startswith("refine:")
    assert rec["seed_val_mae"] is None                               # 待评测, 回填前 None


def test_maybe_refine_prefers_best_shot_with_two_scored_versions(store):
    """族内 ≥2 版本有实测 MAE → best_shot (FunSearch 双版本), 不再走单父版 refine。"""
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1)
    reg.update_real_mae(r1, 20.0)
    reg.update_real_mae(r2, 19.5)
    spy = _SpyRefiner(_refined_result("Foo_v3"))
    loop = _loop(store, reg, refiner=spy)
    out = loop.maybe_refine(round_idx=1, best_genotype=Genotype(blocks=[STBlock("gcn", "tcn")]),
                            current_best_mae=19.5)
    assert out is not None and out.success
    assert spy.calls[0][0] == "best_shot"
    assert out.operator_name == "synth_Foo_v3"
    assert out.seed_genotype._seed_meta["family"] == "synth_Foo"


def test_maybe_refine_mode_weighted_random_reproducible(store):
    """40/30/30 加权随机选 mode: 同 seed 两个 loop 选到同一 mode (可复现)。"""
    def _mk():
        reg = OperatorRegistry()
        r1 = reg.register(_mkop("Foo"))
        reg.update_real_mae(r1, 20.0)
        spy = _SpyRefiner(_refined_result("Foo_v2"))
        return _loop(store, reg, refiner=spy, cfg=CreationConfig(seed=42)), spy

    loop1, spy1 = _mk()
    loop2, spy2 = _mk()
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    m1 = loop1.maybe_refine(round_idx=0, best_genotype=g, current_best_mae=20.0)
    m2 = loop2.maybe_refine(round_idx=0, best_genotype=g, current_best_mae=20.0)
    assert m1.success and m2.success
    assert spy1.calls[0][2] == spy2.calls[0][2]                      # 同 seed 同 mode
    assert spy1.calls[0][2] in ("small", "param", "struct")


def test_maybe_refine_candidate_window_rules(store):
    """候选窗口: 版本数 <3 的族总是候选; ≥3 版本的族须距当前 best ≤ refine_mae_window。"""
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1)
    r3 = reg.register_variant(_mkop("Foo_v3"), r1)                   # 3 版本族
    for r, m in ((r1, 20.0), (r2, 19.9), (r3, 19.8)):
        reg.update_real_mae(r, m)
    spy = _SpyRefiner(_refined_result("Foo_v4"))
    loop = _loop(store, reg, refiner=spy, cfg=CreationConfig(refine_mae_window=0.5))
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    # 族 best=19.8, 当前 best=21.0 → 差 1.2 > 0.5 且版本数=3 → 无候选
    assert loop.maybe_refine(round_idx=0, best_genotype=g, current_best_mae=21.0) is None
    # 当前 best=20.0 → 差 0.2 ≤ 0.5 → 候选, 走 best_shot
    out = loop.maybe_refine(round_idx=0, best_genotype=g, current_best_mae=20.0)
    assert out is not None and out.success


def test_maybe_refine_failure_records_archive_and_returns_failed(store, tmp_path):
    """精炼未过验证 → CreationOutcome(success=False) + 失败履历 (反例素材)。"""
    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    reg.update_real_mae(r1, 20.0)
    fail = SynthesisResult(False, attempts=2, last_error="重试 2 次仍未过验证: 验证未过, 门=shape: x",
                           plan=FusionPlan(operator_name="Foo_v2", rationale="r", shared_structure="s",
                                           composition="additive_residual", source_mechanisms=["m1"],
                                           expected_effect="e"))
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    loop = _loop(store, reg, refiner=_SpyRefiner(fail), archive=arch)
    out = loop.maybe_refine(round_idx=1, best_genotype=Genotype(blocks=[STBlock("gcn", "tcn")]),
                            current_best_mae=20.0)
    assert out is not None and not out.success
    assert not reg.is_registered("synth_Foo_v2")                     # 失败不注册
    ctx = arch.exemplar_context()
    assert len(ctx["failures"]) == 1
    assert ctx["failures"][0]["gate"] == "shape"
    assert ctx["failures"][0]["composition"].startswith("refine:")


# ---------------------------------------------------------------------------
# orchestrator: 回填联动 + 精炼优先节奏
# ---------------------------------------------------------------------------


def _orch(loop, **cfg_kw):
    from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
    from darwin_st.optim.scheduler import EvalResult

    cfg = OrchestratorConfig(max_rounds=1, **cfg_kw)
    return Orchestrator(cfg, lambda g, d: EvalResult(genotype=g, status="OK", mae=1.0, device=d),
                        devices=1, creation_loop=loop)


def _refinement_seed(reg_name="synth_Foo_v2"):
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"is_creation_seed": True, "is_refinement": True,
                    "family": "synth_Foo", "operator_name": reg_name}
    return g


def test_digest_backfills_refinement_real_mae_keep_and_discard(store):
    """带 is_refinement meta 的结果: KEEP 与 DISCARD (有限 MAE) 都回填 registry 家谱。"""
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1)
    reg.update_real_mae(r1, 20.0)
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    orch = _orch(loop, discard_above=25.0)

    # KEEP: mae=19.5 ≤ 阈值
    orch._digest(EvalResult(genotype=_refinement_seed(), status="OK", mae=19.5, rmse=29.0,
                            device="cpu", extra={"num_params": 1000}))
    assert reg.best_in_family("synth_Foo")["reg_name"] == r2
    assert reg.best_in_family("synth_Foo")["real_mae"] == 19.5

    # DISCARD: mae=30 > 阈值 25, 有限 MAE 仍回填 (族内比较靠真实信号)
    orch._digest(EvalResult(genotype=_refinement_seed(), status="OK", mae=30.0, rmse=45.0,
                            device="cpu", extra={"num_params": 1000}))
    maes = {v["reg_name"]: v["real_mae"] for v in reg.family_versions("synth_Foo")}
    assert maes[r2] == 30.0


def test_digest_crash_does_not_backfill(store):
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    r2 = reg.register_variant(_mkop("Foo_v2"), r1)
    reg.update_real_mae(r1, 20.0)
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    orch = _orch(loop)
    crash = EvalResult(genotype=_refinement_seed(), status="CRASH", mae=float("nan"),
                       device="cpu", fail_reason="mock_nan")
    orch._digest(crash)
    maes = {v["reg_name"]: v["real_mae"] for v in reg.family_versions("synth_Foo")}
    assert maes[r2] is None                                          # CRASH 不回填


def _creation_seed(reg_name="synth_Foo"):
    """普通创造 v1 的 seed (无 is_refinement) —— 任务0回填修正的回归对象。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"is_creation_seed": True, "operator_name": reg_name}
    return g


def test_digest_backfills_plain_creation_v1_real_mae(store):
    """任务0修正: 普通创造 v1 (无 is_refinement) 的实测 MAE 也要回填家谱 ——
    否则 v1 族 real_mae 恒 None, B2 候选族选择 best_in_family 拿不到 v1 成绩。"""
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    orch = _orch(loop, discard_above=25.0)

    # KEEP: mae=19.5 ≤ 阈值 → 回填
    orch._digest(EvalResult(genotype=_creation_seed(), status="OK", mae=19.5, rmse=29.0,
                            device="cpu", extra={"num_params": 1000}))
    best = reg.best_in_family("synth_Foo")
    assert best is not None and best["reg_name"] == r1 and best["real_mae"] == 19.5

    # DISCARD: mae=30 > 阈值 25, 有限 MAE 仍回填
    orch._digest(EvalResult(genotype=_creation_seed(), status="OK", mae=30.0, rmse=45.0,
                            device="cpu", extra={"num_params": 1000}))
    maes = {v["reg_name"]: v["real_mae"] for v in reg.family_versions("synth_Foo")}
    assert maes[r1] == 30.0


def test_digest_plain_creation_crash_does_not_backfill(store):
    """普通创造 seed CRASH (无效 MAE) 不回填 (与精炼路径一致)。"""
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    r1 = reg.register(_mkop("Foo"))
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    orch = _orch(loop)
    crash = EvalResult(genotype=_creation_seed(), status="CRASH", mae=float("nan"),
                       device="cpu", fail_reason="mock_nan")
    orch._digest(crash)
    maes = {v["reg_name"]: v["real_mae"] for v in reg.family_versions("synth_Foo")}
    assert maes[r1] is None


# ---------------------------------------------------------------------------
# B4 方向信用分: orchestrator digest 结局回授 (KEEP +1 / DISCARD −0.5 / CRASH −0.5)
# ---------------------------------------------------------------------------


def _scored_seed(reg_name="synth_Foo", rnd=0):
    """带 creation_round 的创造 seed (B4 回授归属轮次)。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"is_creation_seed": True, "operator_name": reg_name, "creation_round": rnd}
    return g


def test_digest_reports_outcome_to_direction_score(store):
    """digest 回填时: KEEP → +1 (并触发温度 −0.05); DISCARD/CRASH → −0.5, 累加到归属轮次。"""
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    reg.register(_mkop("Foo"))
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    loop._history.append({"round": 0, "bottleneck": "b", "preconditions": ["x"],
                          "result_mae": None, "score": 0.0})
    orch = _orch(loop, discard_above=25.0)

    # KEEP: mae=19.5 ≤ 阈值 → +1, adopted 触发降温
    orch._digest(EvalResult(genotype=_scored_seed(), status="OK", mae=19.5, rmse=29.0,
                            device="cpu", extra={"num_params": 1000}))
    assert loop._history[0]["score"] == pytest.approx(1.0)
    assert loop._temperature == pytest.approx(0.75)

    # DISCARD: mae=30 > 阈值 25 → −0.5 (罚分不动温度)
    orch._digest(EvalResult(genotype=_scored_seed(), status="OK", mae=30.0, rmse=45.0,
                            device="cpu", extra={"num_params": 1000}))
    assert loop._history[0]["score"] == pytest.approx(0.5)
    assert loop._temperature == pytest.approx(0.75)

    # CRASH → −0.5
    orch._digest(EvalResult(genotype=_scored_seed(), status="CRASH", mae=float("nan"),
                            device="cpu", fail_reason="mock_nan"))
    assert loop._history[0]["score"] == pytest.approx(0.0)


def test_digest_seed_without_creation_round_skips_scoring(store):
    """无 creation_round 的旧 seed (改动前产出) → 跳过信用分回授, 行为不变不崩。"""
    from darwin_st.optim.scheduler import EvalResult

    reg = OperatorRegistry()
    reg.register(_mkop("Foo"))
    loop = _loop(store, reg, refiner=_SpyRefiner(_refined_result()))
    orch = _orch(loop)
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"is_creation_seed": True, "operator_name": "synth_Foo"}   # 无 creation_round
    orch._digest(EvalResult(genotype=g, status="OK", mae=19.5, rmse=29.0,
                            device="cpu", extra={"num_params": 1000}))
    assert loop._history == []                               # 无轨迹可记, 不崩
    assert loop._temperature == pytest.approx(0.8)           # 温度未被误触发


def test_try_creation_refine_first_when_candidate(store):
    """refine_first (默认): maybe_refine 有产出 → 用其 seed, 不再调 maybe_create。"""
    from darwin_st.creation.creation_loop import CreationOutcome

    class _RefineFirstLoop:
        def __init__(self):
            self.refine_calls = 0
            self.create_calls = 0

        def maybe_refine(self, round_idx=0, best_genotype=None, best_hps=None,
                         current_best_mae=None, dataset=""):
            self.refine_calls += 1
            return CreationOutcome(True, operator_names=["synth_Foo_v2"],
                                   seed_genotypes=[Genotype(blocks=[STBlock("gcn", "tcn")])],
                                   bottleneck="refine:synth_Foo", n_success=1)

        def maybe_create(self, *a, **kw):
            self.create_calls += 1
            return CreationOutcome(False, reason="不应被调")

    loop = _RefineFirstLoop()
    orch = _orch(loop)
    assert orch._try_creation() is True
    assert loop.refine_calls == 1 and loop.create_calls == 0
    events = [h for h in orch.state.history if h.get("event") == "creation"]
    assert events and events[0]["kind"] == "refine"
    assert len(orch._pending_seed_genotypes) == 1


def test_try_creation_falls_back_to_create_without_candidate(store):
    """maybe_refine 返回 None (无候选族) → 回落原 maybe_create。"""
    from darwin_st.creation.creation_loop import CreationOutcome

    class _NoCandidateLoop:
        def __init__(self):
            self.refine_calls = 0
            self.create_calls = 0

        def maybe_refine(self, round_idx=0, **kw):
            self.refine_calls += 1
            return None                                            # 无候选族

        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            self.create_calls += 1
            return CreationOutcome(True, operator_names=["synth_New"],
                                   seed_genotypes=[Genotype(blocks=[STBlock("gcn", "tcn")])],
                                   bottleneck="b", n_success=1)

    loop = _NoCandidateLoop()
    orch = _orch(loop)
    assert orch._try_creation() is True
    assert loop.refine_calls == 1 and loop.create_calls == 1
    events = [h for h in orch.state.history if h.get("event") == "creation"]
    assert events and events[0]["kind"] == "create"


def test_try_creation_refine_first_disabled(store):
    """refine_first=False → 跳过精炼直接从零创造 (旧行为)。"""
    from darwin_st.creation.creation_loop import CreationOutcome

    class _SpyLoop:
        def __init__(self):
            self.refine_calls = 0

        def maybe_refine(self, round_idx=0, **kw):
            self.refine_calls += 1
            return None

        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            return CreationOutcome(True, operator_names=["synth_New"],
                                   seed_genotypes=[Genotype(blocks=[STBlock("gcn", "tcn")])],
                                   bottleneck="b", n_success=1)

    loop = _SpyLoop()
    orch = _orch(loop, refine_first=False)
    assert orch._try_creation() is True
    assert loop.refine_calls == 0


def test_try_creation_spy_loop_without_maybe_refine(store):
    """老式 creation_loop (无 maybe_refine 方法) 兼容: 直接走 maybe_create。"""
    from darwin_st.creation.creation_loop import CreationOutcome

    class _LegacyLoop:
        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            return CreationOutcome(True, operator_names=["synth_New"],
                                   seed_genotypes=[Genotype(blocks=[STBlock("gcn", "tcn")])],
                                   bottleneck="b", n_success=1)

    orch = _orch(_LegacyLoop())
    assert orch._try_creation() is True
