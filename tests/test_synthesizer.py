"""synthesizer.py 的正确性回归测试 (MockLLM, 本地无网络/无真LLM)。

验证 plan-then-code 合成 + 验证守卫 + 重试的契约:
  - 好 LLM 响应 → 合成出过验证的算子
  - 坏代码 → 触发重试 → 最终失败(不崩)
  - 错误反馈进入下一轮 prompt
  - 代码抽取 / 沙箱 exec
"""

from __future__ import annotations

import json

import pytest

from darwin_st.creation import (
    FusionRequest,
    MockLLM,
    OperatorSynthesizer,
    SynthesisConfig,
    exec_operator_code,
    extract_code,
)
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
