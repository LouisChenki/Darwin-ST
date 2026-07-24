"""creation_archive.py (B1 创造档案回读) 的测试 (mock 全链, 无 LLM/GPU/网络)。

核心契约:
  - 档案: record / top_successes 排序 (含无 mae 记录) / recent_failures /
    update_outcome 命中与未命中 / 坏行容错 / exemplar_context 截断与字段齐全
  - prompt: exemplars 注入后 user 含先例代码+MAE+失败门; exemplars=None 与注入前逐字节一致
  - creation_loop 集成: 成败假设都记录; seed_meta 含 operator_name; 下一轮合成收到 exemplars
  - orchestrator 回填: 创造 seed 结局回写履历本 (adopted / seed_val_mae)
"""

from __future__ import annotations

import json

import pytest

from darwin_st.creation import (
    CreationConfig,
    CreationLoop,
    CreationRecord,
    CreationArchive,
    MockLLM,
    OperatorRegistry,
    OperatorSynthesizer,
    SynthesisConfig,
)
from darwin_st.creation.creation_loop import parse_failure_gate
from darwin_st.creation.synthesizer import build_plan_many_prompt
from darwin_st.creation.contracts import FusionRequest
from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
from darwin_st.search import operators as ops_mod
from darwin_st.search.genotype import Genotype, STBlock


def _rec(op="op_a", gate="all", mae=None, created="2026-01-01T00:00:00+00:00",
         code="class A: pass", err=None, rnd=0):
    return CreationRecord(round_idx=rnd, created_at=created, bottleneck="瓶颈b",
                          preconditions=["p1"], mechanisms=["m1", "m2"],
                          operator_name=op, composition="additive_residual",
                          rationale="理由r", code=code, gate=gate, error=err,
                          seed_val_mae=mae)


# ---------------------------------------------------------------------------
# 档案本体
# ---------------------------------------------------------------------------


def test_empty_archive_when_file_missing(tmp_path):
    arch = CreationArchive(str(tmp_path / "nope.jsonl"))
    assert arch.is_empty()
    assert arch.top_successes(3) == []
    assert arch.recent_failures(3) == []
    ctx = arch.exemplar_context()
    assert ctx == {"successes": [], "failures": []}


def test_record_and_top_successes_ordering(tmp_path):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    arch.record(_rec(op="synth_b", mae=20.0, created="2026-01-02T00:00:00+00:00"))
    arch.record(_rec(op="synth_a", mae=18.0, created="2026-01-01T00:00:00+00:00"))
    # 无 mae 的成功记录: 按时间近者优先, 且排有 mae 者之后
    arch.record(_rec(op="synth_old", mae=None, created="2026-01-01T00:00:00+00:00"))
    arch.record(_rec(op="synth_new", mae=None, created="2026-01-03T00:00:00+00:00"))
    tops = arch.top_successes(4)
    assert [r.operator_name for r in tops] == ["synth_a", "synth_b", "synth_new", "synth_old"]
    # k 截断: 有 mae 的优先
    assert [r.operator_name for r in arch.top_successes(1)] == ["synth_a"]
    # 非有限 mae (nan/inf) 视同无 mae
    arch.record(_rec(op="synth_nan", mae=float("nan"), created="2026-01-04T00:00:00+00:00"))
    assert "synth_nan" not in [r.operator_name for r in arch.top_successes(2)]


def test_top_successes_excludes_failures(tmp_path):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    arch.record(_rec(op="ok", gate="all", mae=19.0))
    arch.record(_rec(op="bad", gate="shape", err="门=shape"))
    assert [r.operator_name for r in arch.top_successes(5)] == ["ok"]


def test_recent_failures_ordering(tmp_path):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    arch.record(_rec(op="f1", gate="shape", created="2026-01-01T00:00:00+00:00"))
    arch.record(_rec(op="ok", gate="all", created="2026-01-03T00:00:00+00:00"))
    arch.record(_rec(op="f2", gate="plan", created="2026-01-02T00:00:00+00:00"))
    fails = arch.recent_failures(5)
    assert [r.operator_name for r in fails] == ["f2", "f1"]   # 时间倒序, 成功排除
    assert [r.operator_name for r in arch.recent_failures(1)] == ["f2"]


def test_update_outcome_hit_updates_most_recent(tmp_path):
    p = str(tmp_path / "a.jsonl")
    arch = CreationArchive(p)
    arch.record(_rec(op="synth_x", created="2026-01-01T00:00:00+00:00", rnd=0))
    arch.record(_rec(op="synth_x", created="2026-01-02T00:00:00+00:00", rnd=1))
    ok = arch.update_outcome("synth_x", "sig123", 19.5, True)
    assert ok is True
    recs = arch._load()
    assert recs[0].seed_val_mae is None and recs[0].adopted is None   # 旧记录不动
    assert recs[1].seed_val_mae == 19.5
    assert recs[1].adopted is True
    assert recs[1].seed_signature == "sig123"


def test_update_outcome_miss_returns_false_and_keeps_file(tmp_path):
    p = str(tmp_path / "a.jsonl")
    arch = CreationArchive(p)
    arch.record(_rec(op="synth_x"))
    arch.record(_rec(op="failed_op", gate="shape"))
    before = open(p, encoding="utf-8").read()
    assert arch.update_outcome("synth_nope", "sig", 1.0, True) is False
    assert arch.update_outcome("failed_op", "sig", 1.0, True) is False   # 失败记录不算成功
    assert open(p, encoding="utf-8").read() == before                    # 未命中不改文件


def test_bad_lines_are_skipped(tmp_path):
    p = tmp_path / "a.jsonl"
    good = json.dumps({"round_idx": 0, "created_at": "t0", "operator_name": "ok",
                       "gate": "all", "seed_val_mae": 20.0}, ensure_ascii=False)
    p.write_text(good + "\n" + "{not json\n" + json.dumps({"bogus": 1}) + "\n\n",
                 encoding="utf-8")
    arch = CreationArchive(str(p))
    recs = arch._load()
    assert len(recs) == 1 and recs[0].operator_name == "ok"
    # {"bogus": 1} 缺必填字段 → 跳过; 未知字段也应被过滤
    p.write_text(good[:-1] + ', "unknown_field": 1}\n', encoding="utf-8")
    assert arch._load()[0].operator_name == "ok"


def test_exemplar_context_fields_and_truncation(tmp_path):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    long_code = "\n".join(f"# line {i}" for i in range(100))
    arch.record(_rec(op="synth_good", mae=18.5, code=long_code))
    arch.record(_rec(op="bad_op", gate="gradcheck", err="E" * 500))
    ctx = arch.exemplar_context(k_success=4, k_failure=3, max_code_lines=10)
    s = ctx["successes"][0]
    assert set(s) == {"operator_name", "composition", "rationale", "seed_val_mae", "code"}
    assert s["operator_name"] == "synth_good" and s["seed_val_mae"] == 18.5
    assert len(s["code"].splitlines()) == 11            # 10 行代码 + 1 行截断标注
    assert "截断" in s["code"] and "# line 9" in s["code"] and "# line 10" not in s["code"]
    f = ctx["failures"][0]
    assert set(f) == {"operator_name", "composition", "rationale", "gate", "error"}
    assert f["gate"] == "gradcheck" and len(f["error"]) == 300   # 错误截 300 字


def test_exemplar_context_short_code_not_annotated(tmp_path):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    arch.record(_rec(op="s", mae=1.0, code="line1\nline2"))
    ctx = arch.exemplar_context(max_code_lines=60)
    assert ctx["successes"][0]["code"] == "line1\nline2"


# ---------------------------------------------------------------------------
# 失败门解析
# ---------------------------------------------------------------------------


def test_parse_failure_gate():
    assert parse_failure_gate("重试 2 次仍未过验证: 验证未过, 门=shape: 维度错") == "shape"
    assert parse_failure_gate("计划非法: ValueError: x") == "plan"
    assert parse_failure_gate("多假设计划阶段失败: ValueError: y") == "plan"
    assert parse_failure_gate("别的错误") == "unknown"
    assert parse_failure_gate("") == "unknown"


# ---------------------------------------------------------------------------
# prompt 注入 (synthesizer)
# ---------------------------------------------------------------------------


def _req():
    ms = [m for m in all_seed_mechanisms() if m.name in ("state_space_model", "masked_autoencoding")]
    return FusionRequest(bottleneck="长程依赖 + 标签稀缺",
                         target_preconditions=["long_range_dependency", "label_scarcity"],
                         mechanisms=ms)


def _exemplars():
    return {
        "successes": [{"operator_name": "synth_prev", "composition": "additive_residual",
                       "rationale": "残差融合安全", "seed_val_mae": 18.42,
                       "code": "class PrevOp(nn.Module): pass"}],
        "failures": [{"operator_name": "bad_prev", "composition": "sequential",
                      "rationale": "串接两机制", "gate": "shape", "error": "门=shape: 丢节点维"}],
    }


def test_prompt_exemplars_injected():
    msgs = build_plan_many_prompt(_req(), 2, exemplars=_exemplars())
    user = msgs[-1]["content"]
    assert "成功先例" in user and "失败教训" in user
    assert "class PrevOp(nn.Module): pass" in user      # 先例代码片段
    assert "18.42" in user                              # 实测 MAE
    assert "shape" in user and "丢节点维" in user        # 失败门名 + 错误摘要
    assert "实现级差异" in user                          # 差异化指令


def test_prompt_exemplars_none_byte_identical():
    req = _req()
    plain = build_plan_many_prompt(req, 2)
    assert build_plan_many_prompt(req, 2, exemplars=None) == plain
    assert build_plan_many_prompt(req, 2, exemplars={}) == plain
    assert build_plan_many_prompt(req, 2, exemplars={"successes": [], "failures": []}) == plain
    # 与现状格式逐字节一致 (锁住回归)
    import json as _json
    expected_user = ("请提出 2 个不同的融合方案。\n瓶颈与跨域机制集:\n"
                     + _json.dumps(req.to_prompt_context(), ensure_ascii=False, indent=2))
    assert plain[-1]["content"] == expected_user
    assert "成功先例" not in plain[-1]["content"]


def test_synthesize_many_passes_exemplars_to_prompt():
    captured = []

    def cb(msgs):
        captured.append(msgs)
        return resp.pop(0)

    plan = json.dumps([{"operator_name": "SomeOp", "rationale": "r", "shared_structure": "s",
                        "composition": "additive_residual", "source_mechanisms": ["a"],
                        "expected_effect": "e"}], ensure_ascii=False)
    resp = [plan]   # 计划阶段即被捕获; 后续代码阶段不需走到 (计划用坏代码让它失败亦可)
    resp.append('```python\nimport torch.nn as nn\nclass SomeOp(nn.Module):\n'
                '    def __init__(self, channels, num_nodes, **kw):\n'
                '        super().__init__(); self.l = nn.Linear(channels, channels)\n'
                '    def forward(self, x, adj=None): return self.l(x).mean(dim=2)\n```')
    resp.append(resp[-1])
    synth = OperatorSynthesizer(MockLLM(cb), SynthesisConfig(max_retries=2))
    synth.synthesize_many(_req(), n_hypotheses=1, exemplars=_exemplars())
    assert captured, "LLM 未被调用"
    assert "成功先例" in captured[0][-1]["content"]     # 第一条消息 = plan prompt


# ---------------------------------------------------------------------------
# creation_loop 集成 (档案写入 + 下一轮回读)
# ---------------------------------------------------------------------------


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


_PLAN_ARRAY_2 = (
    '[{"operator_name": "GoodArchOp", "rationale": "好融合", "shared_structure": "s",'
    ' "composition": "additive_residual", "source_mechanisms": ["a"], "expected_effect": "e"},'
    '{"operator_name": "BadArchOp", "rationale": "坏融合", "shared_structure": "s",'
    ' "composition": "sequential", "source_mechanisms": ["a"], "expected_effect": "e"}]'
)

_GOOD_CODE = '''```python
import torch
import torch.nn as nn
class GoodArchOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.branch = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        return self.proj(x) + self.alpha * self.branch(x)
```'''

_BAD_CODE = '''```python
import torch
import torch.nn as nn
class BadArchOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.l = nn.Linear(channels, channels)
    def forward(self, x, adj=None):
        return self.l(x).mean(dim=2)
```'''

_PLAN_ARRAY_1 = (
    '[{"operator_name": "Round2Op", "rationale": "r2", "shared_structure": "s",'
    ' "composition": "gated_routed", "source_mechanisms": ["a"], "expected_effect": "e"}]'
)

_ROUND2_CODE = '''```python
import torch
import torch.nn as nn
class Round2Op(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        return self.proj(x) + self.alpha * x
```'''


def _loop_with_archive(store, responses, archive):
    captured = []

    def cb(msgs):
        captured.append(msgs)
        return next(resp_iter)

    resp_iter = iter(responses)
    llm = MockLLM(cb)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    # 响应序列按 synthesize_many (单次调用产 N) 编排 → 固定走旧路径 (独立采样默认开)
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(), archive=archive,
                        config=CreationConfig(independent_sampling=False))
    return loop, captured


def test_maybe_create_records_success_and_failure(store, tmp_path):
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    loop, _ = _loop_with_archive(store, [_PLAN_ARRAY_2, _GOOD_CODE, _BAD_CODE, _BAD_CODE], arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success and outcome.n_success == 1
    recs = arch._load()
    assert len(recs) == 2                                # 成败各一条
    succ = [r for r in recs if r.gate == "all"]
    fail = [r for r in recs if r.gate != "all"]
    assert len(succ) == 1 and len(fail) == 1
    s = succ[0]
    assert s.operator_name == outcome.operator_name      # 成功记注册名 (synth_ 前缀)
    assert s.code and "class GoodArchOp" in s.code
    assert s.seed_signature == outcome.seed_genotype.signature()
    assert s.mechanisms and s.bottleneck and s.round_idx == 0
    f = fail[0]
    assert f.operator_name == "BadArchOp"                # 失败记计划名
    assert f.gate == "shape"                             # 从 last_error 解析出的失败门
    assert f.error and "门=shape" in f.error
    assert f.composition == "sequential" and f.rationale == "坏融合"


def test_seed_meta_contains_operator_name(store, tmp_path):
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    loop, _ = _loop_with_archive(store, [_PLAN_ARRAY_2, _GOOD_CODE, _BAD_CODE, _BAD_CODE], arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    meta = outcome.seed_genotype._seed_meta
    assert meta["operator_name"] == outcome.operator_name   # 回填识别键
    assert meta["is_creation_seed"] is True


def test_next_round_synthesis_receives_exemplars(store, tmp_path):
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    responses = [_PLAN_ARRAY_2, _GOOD_CODE, _BAD_CODE, _BAD_CODE,   # 轮 1: 一成一败
                 _PLAN_ARRAY_1, _ROUND2_CODE]                        # 轮 2
    loop, captured = _loop_with_archive(store, responses, arch)
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    n_calls_round1 = len(captured)
    out2 = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert out2.success
    plan_prompt_round2 = captured[n_calls_round1][-1]["content"]   # 轮 2 第一条 = plan prompt
    assert "成功先例" in plan_prompt_round2
    assert "class GoodArchOp" in plan_prompt_round2                # 先例代码片段喂回去了
    assert "失败教训" in plan_prompt_round2 and "shape" in plan_prompt_round2


def test_archive_none_keeps_prompt_unchanged(store):
    captured = []

    def cb(msgs):
        captured.append(msgs)
        return next(resp_iter)

    resp_iter = iter([_PLAN_ARRAY_1, _ROUND2_CODE])
    synth = OperatorSynthesizer(MockLLM(cb), SynthesisConfig(max_retries=2))
    loop = CreationLoop(store, store.embedder, synth, OperatorRegistry(),   # 无 archive
                        config=CreationConfig(independent_sampling=False))
    loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert "成功先例" not in captured[0][-1]["content"]


# ---------------------------------------------------------------------------
# orchestrator 回填
# ---------------------------------------------------------------------------


def _make_orch_with_archive(tmp_path):
    from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig

    arch = CreationArchive(str(tmp_path / "ca.jsonl"))

    class _StubCreationLoop:
        def __init__(self, archive):
            self.archive = archive

    cfg = OrchestratorConfig(population_size=4, tournament_size=2, warmup_keep=16)
    orch = Orchestrator(cfg, lambda g, d: None, devices=1,
                        creation_loop=_StubCreationLoop(arch))
    return orch, arch


def _seed_result(status="OK", mae=19.5, op="synth_x"):
    from darwin_st.optim.scheduler import EvalResult

    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"warm_start_hps": None, "hpo_trials": 20,
                    "is_creation_seed": True, "operator_name": op}
    return EvalResult(genotype=g, status=status, mae=mae, rmse=mae * 1.5,
                      device="cpu", extra={"num_params": 1000})


def test_orchestrator_backfills_keep_outcome(tmp_path):
    orch, arch = _make_orch_with_archive(tmp_path)
    arch.record(_rec(op="synth_x", gate="all"))
    res = _seed_result(status="OK", mae=19.5)
    orch._digest(res)
    rec = arch._load()[0]
    assert rec.adopted is True
    assert rec.seed_val_mae == 19.5
    assert rec.seed_signature == res.genotype.signature()


def test_orchestrator_backfills_crash_as_not_adopted(tmp_path):
    orch, arch = _make_orch_with_archive(tmp_path)
    arch.record(_rec(op="synth_x", gate="all"))
    res = _seed_result(status="CRASH", mae=float("inf"))
    res.fail_reason = "mock_nan"
    orch._digest(res)
    rec = arch._load()[0]
    assert rec.adopted is False
    assert rec.seed_val_mae is None                       # CRASH 记 None
    assert rec.seed_signature == res.genotype.signature()


def test_orchestrator_backfills_discard_with_actual_mae(tmp_path):
    from darwin_st.optim.scheduler import EvalResult

    orch, arch = _make_orch_with_archive(tmp_path)
    arch.record(_rec(op="synth_x", gate="all"))
    orch._n_valid = 100                                   # 跳过 warmup → 触发相对 DISCARD 判定
    orch.state.best_mae = 10.0
    g = Genotype(blocks=[STBlock("gcn", "tcn")])
    g._seed_meta = {"is_creation_seed": True, "operator_name": "synth_x"}
    res = EvalResult(genotype=g, status="OK", mae=99.0, device="cpu", extra={})
    orch._digest(res)
    rec = arch._load()[0]
    assert rec.adopted is False                           # DISCARD → adopted=False
    assert rec.seed_val_mae == 99.0                       # 有限 MAE 按实际回填


def test_orchestrator_ignores_non_creation_seed(tmp_path):
    orch, arch = _make_orch_with_archive(tmp_path)
    arch.record(_rec(op="synth_x", gate="all"))
    from darwin_st.optim.scheduler import EvalResult

    g = Genotype(blocks=[STBlock("gcn", "tcn")])          # 无 _seed_meta
    orch._digest(EvalResult(genotype=g, status="OK", mae=19.5, device="cpu", extra={}))
    g2 = Genotype(blocks=[STBlock("gcn", "tcn")])
    g2._seed_meta = {"hpo_trials": 5}                     # 自适应 HPO meta (非创造 seed)
    orch._digest(EvalResult(genotype=g2, status="OK", mae=19.0, device="cpu", extra={}))
    rec = arch._load()[0]
    assert rec.adopted is None and rec.seed_val_mae is None   # 均未回填


def test_orchestrator_no_archive_no_crash(tmp_path):
    from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig

    class _NoArchiveLoop:
        archive = None

    cfg = OrchestratorConfig(population_size=4, tournament_size=2)
    orch = Orchestrator(cfg, lambda g, d: None, devices=1, creation_loop=_NoArchiveLoop())
    orch._digest(_seed_result())                          # archive=None → 跳过, 不崩
    assert orch.state.evals == 1
