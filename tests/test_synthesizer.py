"""synthesizer.py 的正确性回归测试 (MockLLM, 本地无网络/无真LLM)。

验证 plan-then-code 合成 + 验证守卫 + 重试的契约:
  - 好 LLM 响应 → 合成出过验证的算子
  - 坏代码 → 触发重试 → 最终失败(不崩)
  - 错误反馈进入下一轮 prompt
  - 代码抽取 / 沙箱 exec
"""

from __future__ import annotations

import json
import re

import pytest

from darwin_st.creation import (
    COMPOSITION_OPS,
    FusionRequest,
    MockLLM,
    OperatorSynthesizer,
    SynthesisConfig,
    exec_operator_code,
    extract_code,
)
from darwin_st.creation.synthesizer import build_plan_prompt
from darwin_st.creation.validation import ValidationConfig
from darwin_st.knowledge import all_seed_mechanisms


def _req():
    ms = [m for m in all_seed_mechanisms() if m.name in ("state_space_model", "masked_autoencoding")]
    return FusionRequest(
        bottleneck="长程依赖 + 标签稀缺",
        target_preconditions=["long_range_dependency", "label_scarcity"],
        mechanisms=ms,
    )


# 一个会通过验证的好算子代码 (零初始化残差)
_GOOD_PLAN = json.dumps({
    "operator_name": "FusedLongRangeOp", "rationale": "融合长程与自监督",
    "shared_structure": "时序混合", "composition": "additive_residual",
    "source_mechanisms": ["state_space_model", "masked_autoencoding"],
    "expected_effect": "改善长程",
}, ensure_ascii=False)

_GOOD_CODE = '''```python
import torch
import torch.nn as nn

class FusedLongRangeOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.proj = nn.Linear(channels, channels)
        self.branch = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        base = self.proj(x)
        return base + self.alpha * self.branch(x)
```'''

# 一个丢节点维 N 的坏算子 (验证应拒)
_BAD_CODE = '''```python
import torch
import torch.nn as nn

class FusedLongRangeOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
    def forward(self, x, adj=None):
        return self.lin(x).mean(dim=2)  # 丢了节点维 N
```'''


def test_extract_code():
    assert "class Foo" in extract_code("```python\nclass Foo: pass\n```")
    assert extract_code("no fences here") == "no fences here"


def test_exec_operator_code():
    code = "import torch.nn as nn\nclass MyOp(nn.Module):\n    def forward(self,x,adj=None): return x"
    cls = exec_operator_code(code, "MyOp")
    import torch.nn as nn
    assert issubclass(cls, nn.Module)


def test_exec_rejects_non_module():
    with pytest.raises(ValueError):
        exec_operator_code("class NotAModule: pass", "NotAModule")


def test_synthesize_success():
    """好计划+好代码 → 合成出过验证的算子。"""
    responses = iter([_GOOD_PLAN, _GOOD_CODE])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    res = synth.synthesize(_req())
    assert res.success, res.last_error
    assert res.operator is not None
    assert res.operator.name == "FusedLongRangeOp"
    assert res.operator.validated
    assert res.attempts == 1


def test_synthesize_retries_on_bad_code():
    """坏代码(丢N) → 重试 → 第二次给好代码 → 成功。"""
    responses = iter([_GOOD_PLAN, _BAD_CODE, _GOOD_CODE])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3))
    res = synth.synthesize(_req())
    assert res.success
    assert res.attempts == 2  # 第一次坏, 第二次好


def test_synthesize_fails_after_max_retries():
    """一直坏代码 → 重试耗尽 → 失败但不崩。"""
    responses = iter([_GOOD_PLAN] + [_BAD_CODE] * 10)
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3))
    res = synth.synthesize(_req())
    assert not res.success
    assert res.attempts == 3
    assert "验证" in res.last_error or "门" in res.last_error


def test_synthesize_error_feedback_in_prompt():
    """验证失败的错误应进入下一轮 code prompt(让 LLM 知道哪错了)。"""
    seen_prompts = []

    def responder(msgs):
        seen_prompts.append(msgs[-1]["content"])
        if len(seen_prompts) == 1:
            return _GOOD_PLAN
        if len(seen_prompts) == 2:
            return _BAD_CODE
        return _GOOD_CODE

    llm = MockLLM(responder)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3))
    res = synth.synthesize(_req())
    assert res.success
    # 第三次(第二轮代码)的 prompt 应含上一轮错误反馈
    assert "验证失败" in seen_prompts[2] or "shape" in seen_prompts[2]


def test_synthesize_bad_plan_fails_gracefully():
    """计划阶段返回非法 JSON → 优雅失败。"""
    llm = MockLLM(lambda msgs: "这不是 JSON")
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    res = synth.synthesize(_req())
    assert not res.success
    assert "计划" in res.last_error


# ---------------------------------------------------------------------------
# code_backend (Aider 路径) —— 用 mock backend 测逻辑, 不需真 aider
# ---------------------------------------------------------------------------


class _MockBackend:
    """mock 代码后端: 按队列返回 (code, error)。"""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    def write_operator(self, instruction, target_file="op.py"):
        self.calls.append(instruction)
        return next(self.outputs)


_GOOD_OP_CODE = (
    "import torch\nimport torch.nn as nn\n"
    "class FusedLongRangeOp(nn.Module):\n"
    "    def __init__(self, channels, num_nodes, **kw):\n"
    "        super().__init__()\n"
    "        self.proj = nn.Linear(channels, channels)\n"
    "        self.branch = nn.Linear(channels, channels)\n"
    "        self.alpha = nn.Parameter(torch.zeros(1))\n"
    "    def forward(self, x, adj=None):\n"
    "        return self.proj(x) + self.alpha * self.branch(x)\n"
)

_BAD_OP_CODE = (
    "import torch.nn as nn\n"
    "class FusedLongRangeOp(nn.Module):\n"
    "    def __init__(self, channels, num_nodes, **kw):\n"
    "        super().__init__()\n"
    "        self.lin = nn.Linear(channels, channels)\n"
    "    def forward(self, x, adj=None):\n"
    "        return self.lin(x).mean(dim=2)\n"  # 丢节点维 N
)


def test_synthesize_with_code_backend_success():
    """code_backend(Aider 路径): plan 用 llm, 代码用后端。"""
    llm = MockLLM(lambda msgs: _GOOD_PLAN)  # 只需出计划
    backend = _MockBackend([(_GOOD_OP_CODE, "")])
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2), code_backend=backend)
    res = synth.synthesize(_req())
    assert res.success, res.last_error
    assert res.operator.name == "FusedLongRangeOp"
    assert len(backend.calls) == 1  # 后端被调一次


def test_synthesize_code_backend_retries():
    """后端先给坏代码 → 重试 → 第二次好代码 → 成功; 错误反馈进指令。"""
    llm = MockLLM(lambda msgs: _GOOD_PLAN)
    backend = _MockBackend([(_BAD_OP_CODE, ""), (_GOOD_OP_CODE, "")])
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3), code_backend=backend)
    res = synth.synthesize(_req())
    assert res.success
    assert res.attempts == 2
    # 第二次调用的指令应含上轮验证错误
    assert "验证失败" in backend.calls[1] or "shape" in backend.calls[1]


def test_synthesize_code_backend_failure():
    """后端一直返回 None(如 aider 失败) → 优雅失败。"""
    llm = MockLLM(lambda msgs: _GOOD_PLAN)
    backend = _MockBackend([(None, "aider 超时")] * 5)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2), code_backend=backend)
    res = synth.synthesize(_req())
    assert not res.success
    assert "后端" in res.last_error or "验证" in res.last_error


# ---------------------------------------------------------------------------
# 多假设合成 (synthesize_many)
# ---------------------------------------------------------------------------


_PLAN_ARRAY = (
    '[{"operator_name": "HypoA", "rationale": "r", "shared_structure": "s",'
    ' "composition": "additive_residual", "source_mechanisms": ["a"], "expected_effect": "e"},'
    ' {"operator_name": "HypoB", "rationale": "r2", "shared_structure": "s2",'
    ' "composition": "parallel", "source_mechanisms": ["b"], "expected_effect": "e2"}]'
)

def _op_code(name):
    return (f'```python\nimport torch\nimport torch.nn as nn\n'
            f'class {name}(nn.Module):\n'
            f'    def __init__(self, channels, num_nodes, **kw):\n'
            f'        super().__init__(); self.l = nn.Linear(channels, channels)\n'
            f'        self.a = nn.Parameter(torch.zeros(1))\n'
            f'    def forward(self, x, adj=None): return self.l(x) + self.a * x\n```')


def test_synthesize_many_multiple_success():
    """一次生成 2 个假设, 各自合成 → 2 个成功结果。"""
    responses = iter([_PLAN_ARRAY, _op_code("HypoA"), _op_code("HypoB")])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_many(_req(), n_hypotheses=2)
    successes = [r for r in results if r.success]
    assert len(successes) == 2
    assert {r.operator.name for r in successes} == {"HypoA", "HypoB"}


def test_synthesize_many_partial_success():
    """2 假设, 一个好一个坏 → 1 成功 1 失败, 不崩。"""
    bad = ('```python\nimport torch.nn as nn\nclass HypoB(nn.Module):\n'
           '    def __init__(self,channels,num_nodes,**kw):\n        super().__init__(); self.l=nn.Linear(channels,channels)\n'
           '    def forward(self,x,adj=None): return self.l(x).mean(dim=2)\n```')  # 丢N
    responses = iter([_PLAN_ARRAY, _op_code("HypoA"), bad])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_many(_req(), n_hypotheses=2)
    assert sum(1 for r in results if r.success) == 1
    assert sum(1 for r in results if not r.success) == 1


def test_synthesize_many_bad_plan_array():
    """计划阶段非数组/非JSON → 优雅失败。"""
    llm = MockLLM(lambda msgs: "不是JSON")
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_many(_req(), n_hypotheses=2)
    assert all(not r.success for r in results)


def test_synthesize_many_dedup_names():
    """重名假设去重。"""
    dup = '[' + _PLAN_ARRAY[1:-1].split('},')[0] + '}, ' + _PLAN_ARRAY[1:-1].split('},')[0] + '}]'
    responses = iter([dup, _op_code("HypoA")])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_many(_req(), n_hypotheses=2)
    # 两个同名 HypoA, 去重后只合成一个
    assert len([r for r in results if r.success]) <= 1


# ---------------------------------------------------------------------------
# B4 温度透传: temperature=None → cfg.temperature (行为不变); 显式值覆盖
# ---------------------------------------------------------------------------


class _TempLLM(MockLLM):
    """记录每次 chat 的 temperature (温度透传断言用)。"""
    def __init__(self, responder):
        super().__init__(responder)
        self.temps = []

    def chat(self, messages, temperature=0.7, max_tokens=4096):
        self.temps.append(temperature)
        return self.responder(messages)


def test_synthesize_temperature_none_uses_cfg():
    responses = iter([_GOOD_PLAN, _GOOD_CODE])
    llm = _TempLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2, temperature=0.8))
    res = synth.synthesize(_req())
    assert res.success
    assert llm.temps == [0.8, 0.8]                    # plan + code 都用 cfg 默认


def test_synthesize_temperature_override():
    responses = iter([_GOOD_PLAN, _GOOD_CODE])
    llm = _TempLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2, temperature=0.8))
    res = synth.synthesize(_req(), temperature=0.3)
    assert res.success
    assert llm.temps == [0.3, 0.3]                    # 覆盖温度贯穿 plan + code


def test_synthesize_many_temperature_passthrough():
    responses = iter([_PLAN_ARRAY, _op_code("HypoA"), _op_code("HypoB")])
    llm = _TempLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1, temperature=0.8))
    results = synth.synthesize_many(_req(), n_hypotheses=2, temperature=0.55)
    assert sum(1 for r in results if r.success) == 2
    assert llm.temps == [0.55, 0.55, 0.55]            # 1 次计划 + 2 次代码


def test_synthesize_many_temperature_none_uses_cfg():
    responses = iter([_PLAN_ARRAY, _op_code("HypoA"), _op_code("HypoB")])
    llm = _TempLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1, temperature=0.8))
    synth.synthesize_many(_req(), n_hypotheses=2)
    assert llm.temps == [0.8, 0.8, 0.8]               # None → cfg.temperature (回归)


# ---------------------------------------------------------------------------
# B5 假设独立采样 (synthesize_independent) + 双模型路由
# ---------------------------------------------------------------------------


def _plan_json(name, composition, mechs=("a", "b")):
    return json.dumps({
        "operator_name": name, "rationale": "r", "shared_structure": "s",
        "composition": composition, "source_mechanisms": list(mechs),
        "expected_effect": "e",
    }, ensure_ascii=False)


def _bad_code_for(name):
    return (f'```python\nimport torch.nn as nn\nclass {name}(nn.Module):\n'
            f'    def __init__(self,channels,num_nodes,**kw):\n'
            f'        super().__init__(); self.l=nn.Linear(channels,channels)\n'
            f'    def forward(self,x,adj=None): return self.l(x).mean(dim=2)\n```')


class _ForcedCompliantLLM(MockLLM):
    """plan 阶段按 prompt 硬约束出合规 plan (算子名 Op_<文法>); code 阶段出该类名的好代码。"""

    def __init__(self):
        self.forced_seen: list[str] = []
        self.prompts: list[str] = []
        self.temps: list[float] = []
        self._last_name = None
        super().__init__(self._respond)

    def chat(self, messages, temperature=0.7, max_tokens=4096):
        self.temps.append(temperature)
        return super().chat(messages, temperature=temperature, max_tokens=max_tokens)

    def _respond(self, msgs):
        content = msgs[-1]["content"]
        self.prompts.append(content)
        m = re.search(r"composition=(\w+)", content)   # 硬约束行只在 plan prompt 里
        if m:                                          # plan 阶段
            comp = m.group(1)
            self.forced_seen.append(comp)
            self._last_name = f"Op_{comp}"
            return _plan_json(self._last_name, comp)
        return _op_code(self._last_name)               # code 阶段


def test_build_plan_prompt_unchanged_without_new_args():
    """回归: forced_composition/exemplars 为 None → prompt 与改动前逐字节一致 (无硬约束/履历段)。"""
    msgs = build_plan_prompt(_req())
    assert len(msgs) == 2 and msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert user.startswith("瓶颈与跨域机制集:")
    assert "硬约束" not in user and "成功先例" not in user


def test_build_plan_prompt_forced_composition_line():
    msgs = build_plan_prompt(_req(), forced_composition="gated_routed")
    assert "必须使用 composition=gated_routed" in msgs[1]["content"]


def test_synthesize_independent_rotates_all_four_compositions():
    """轮转组合文法命中 4 种 (sorted(COMPOSITION_OPS) 按假设序) + 硬约束进 prompt。"""
    llm = _ForcedCompliantLLM()
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_independent(_req(), n_hypotheses=4)
    assert llm.forced_seen == sorted(COMPOSITION_OPS)          # 4 种文法轮转命中
    assert all("必须使用 composition=" in p for p in llm.prompts[::2])   # plan prompt 均带硬约束
    assert sum(1 for r in results if r.success) == 4
    assert [r.plan.composition for r in results] == sorted(COMPOSITION_OPS)


def test_synthesize_independent_temperature_ladder_and_clamp():
    """温度阶梯 [t, t+0.1, t-0.1, t+0.2] clamp [0.3, 1.0]; None → cfg.temperature 为基准。"""
    llm = _ForcedCompliantLLM()
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1, temperature=0.8))
    synth.synthesize_independent(_req(), n_hypotheses=4)
    assert llm.temps[::2] == pytest.approx([0.8, 0.9, 0.7, 1.0])   # plan 阶段温度 (上沿 clamp 1.0)
    # 低基准: 下沿 clamp 0.3
    llm2 = _ForcedCompliantLLM()
    synth2 = OperatorSynthesizer(llm2, SynthesisConfig(max_retries=1, temperature=0.8))
    synth2.synthesize_independent(_req(), n_hypotheses=4, temperature=0.3)
    assert llm2.temps[::2] == pytest.approx([0.3, 0.4, 0.3, 0.5])   # 0.3-0.1=0.2 → clamp 0.3


def test_synthesize_independent_forced_mismatch_fails_before_code():
    """plan 实出文法与强制不符 → 记失败 (注明未按指定组合文法), 不进代码阶段。"""
    calls = []

    def responder(msgs):
        calls.append(msgs[-1]["content"])
        return _plan_json("DisobedientOp", "parallel")       # 永远出 parallel

    llm = MockLLM(responder)
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    results = synth.synthesize_independent(_req(), n_hypotheses=1)   # i=0 强制 additive_residual
    assert len(results) == 1
    r = results[0]
    assert not r.success
    assert "未按指定组合文法" in r.last_error
    assert r.plan is not None and r.plan.composition == "parallel"
    assert len(calls) == 1                                   # 只调了 plan, 未进代码阶段


def test_synthesize_independent_dedup_by_name():
    """去重形态一: operator_name 重名 → 后者跳过 (不计入返回)。"""
    responses = iter([_plan_json("SameNameOp", "additive_residual"), _op_code("SameNameOp"),
                      _plan_json("SameNameOp", "gated_routed"), _op_code("SameNameOp")])
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_independent(_req(), n_hypotheses=2)
    assert len(results) == 1
    assert results[0].success and results[0].plan.operator_name == "SameNameOp"


def test_synthesize_independent_dedup_by_mechanism_composition_sig():
    """去重形态二: (frozenset(source_mechanisms), composition) 与已接受结果重复 → 跳过。"""
    # h1 强制 gated_routed 但 LLM 违规出 additive_residual + 同源机制 → 与 h0 同签名, 跳过
    responses = iter([_plan_json("NameA", "additive_residual", ("a", "b")), _op_code("NameA"),
                      _plan_json("NameB", "additive_residual", ("b", "a"))])  # frozenset 无序
    llm = MockLLM(lambda msgs: next(responses))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    results = synth.synthesize_independent(_req(), n_hypotheses=2)
    assert len(results) == 1
    assert results[0].plan.operator_name == "NameA"


def test_synthesize_independent_dual_model_routing():
    """双模型路由: fast_llm 存在时第 0 个假设用强模型 (质量锚点), 其余用 fast_llm。"""
    strong = _ForcedCompliantLLM()
    fast = _ForcedCompliantLLM()
    synth = OperatorSynthesizer(strong, SynthesisConfig(max_retries=1), fast_llm=fast)
    results = synth.synthesize_independent(_req(), n_hypotheses=3)
    assert strong.forced_seen == sorted(COMPOSITION_OPS)[:1]        # h0 → 强模型
    assert fast.forced_seen == sorted(COMPOSITION_OPS)[1:3]         # h1/h2 → fast
    assert sum(1 for r in results if r.success) == 3
    assert strong.call_count == 2 and fast.call_count == 4          # 各自 plan+code, 无 escalation


def test_synthesize_independent_escalates_to_strong_after_fast_fails():
    """fast 假设重试全败 → 同槽位强模型补试一次 (escalation), 结果取强模型的。"""
    strong = _ForcedCompliantLLM()

    def fast_responder(msgs):
        content = msgs[-1]["content"]
        if "composition=" in content:                        # plan 阶段: 合规 (h1 强制 gated_routed)
            return _plan_json("FastOp", "gated_routed")
        return _bad_code_for("FastOp")                       # 代码永远坏 (丢节点维)

    fast = MockLLM(fast_responder)
    synth = OperatorSynthesizer(strong, SynthesisConfig(max_retries=2), fast_llm=fast)
    results = synth.synthesize_independent(_req(), n_hypotheses=2)
    assert len(results) == 2
    assert results[0].success                                # h0 强模型直接成功
    assert results[1].success                                # fast 全败 → 强模型补试成功
    assert results[1].plan.operator_name == "Op_gated_routed"  # 补试的 plan 来自强模型
    assert fast.call_count == 3                              # fast: 1 plan + 2 code (重试全败)
    assert strong.call_count == 4                            # strong: h0 (2) + escalation (2)


def test_synthesize_independent_no_fast_llm_uses_strong_for_all():
    """fast_llm=None → 全部假设用强模型 (现状兼容), 无 escalation。"""
    strong = _ForcedCompliantLLM()
    synth = OperatorSynthesizer(strong, SynthesisConfig(max_retries=1))     # 不传 fast_llm
    results = synth.synthesize_independent(_req(), n_hypotheses=3)
    assert len(strong.forced_seen) == 3
    assert sum(1 for r in results if r.success) == 3
    assert strong.call_count == 6                              # 3 plan + 3 code, 全走强模型


def test_synthesize_independent_passes_exemplars_to_plan_prompt():
    """exemplars 透传: 非空履历注入独立采样的 plan prompt (同 many 路径)。"""
    llm = _ForcedCompliantLLM()
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=1))
    exemplars = {"successes": [{"operator_name": "synth_x", "composition": "parallel",
                                "seed_val_mae": 18.5, "rationale": "r", "code": "class X: pass"}],
                 "failures": []}
    synth.synthesize_independent(_req(), n_hypotheses=1, exemplars=exemplars)
    assert "成功先例" in llm.prompts[0]
