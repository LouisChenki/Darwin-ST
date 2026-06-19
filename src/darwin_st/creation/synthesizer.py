"""算子合成器 (Operator Synthesizer) —— Tier-2 组合式创造的核心。

把跨域互补机制集融合成新 nn.Module 算子。plan-then-code(研究 +25% Pass@1):
  1. LLM 出【融合计划】(FusionPlan): 共享结构 / 组合算子 / 用哪些机制
  2. LLM 出【算子代码】(一个 nn.Module 类, forward(x,adj)->[B,T,N,C])
  3. 沙箱 exec 代码 → 验证 harness 全门(形状/可微/NaN/非平凡/参数量)
  4. 失败则把错误反馈给 LLM 重试(bounded, AI-Scientist MAX_ITERS=4)
  5. 通过 → SynthesizedOperator(待真实 MAE 评测)

防 reward hacking: LLM 只写算子体, 看不到测试/评测; 沙箱 exec 受限命名空间;
验证 harness 是执行接地的硬门; 真实 masked-MAE 才是最终裁判。

LLM 可插拔(MockLLM 本地测 / OpenAICompatLLM DeepSeek 服务器跑)。
"""

from __future__ import annotations

import re
import traceback
from dataclasses import dataclass, field

import torch.nn as nn

from darwin_st.creation.contracts import COMPOSITION_OPS, FusionPlan, FusionRequest, SynthesizedOperator
from darwin_st.creation.llm import LLMClient
from darwin_st.creation.validation import ValidationConfig, validate_operator

__all__ = ["SynthesisConfig", "SynthesisResult", "OperatorSynthesizer",
           "build_plan_prompt", "build_code_prompt", "extract_code", "exec_operator_code"]


@dataclass
class SynthesisConfig:
    max_retries: int = 4              # 验证失败重试上限 (AI-Scientist MAX_ITERS)
    temperature: float = 0.8
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass
class SynthesisResult:
    success: bool
    operator: SynthesizedOperator | None = None
    attempts: int = 0
    last_error: str = ""
    plan: FusionPlan | None = None


# ---------------------------------------------------------------------------
# Prompt 构造 (plan-then-code, 因果接地)
# ---------------------------------------------------------------------------

_SYS_PLAN = """你是时空预测模型的算子设计专家。给定一个性能瓶颈和一组来自【不同领域】的机制,
你的任务是设计一个【融合】它们的新算子来解决瓶颈。

关键原则:
- 按每个机制【为何有效(causal_behavior)】来融合, 不是表面堆叠。
- 融合方式必须从 4 种组合文法里选: sequential / parallel / additive_residual / gated_routed。
- additive_residual 最安全(零初始化残差, 坏融合不伤基线), 优先考虑。
- 先想清楚【融合计划】再写代码。

只输出一个 JSON(不要其它文字), 字段:
{"operator_name": "合法python标识符", "rationale": "为何这样融合(因果论证)",
 "shared_structure": "各机制共享的抽象结构", "composition": "4选1",
 "source_mechanisms": ["机制名..."], "expected_effect": "预期解决什么"}"""

_SYS_CODE = """你是 PyTorch 算子实现专家。根据给定的融合计划, 实现一个 nn.Module 算子。

【硬性契约 - 必须遵守】:
- 类名 = 计划里的 operator_name。
- __init__(self, channels, num_nodes, **kw): channels=特征维C, num_nodes=节点数N。
- forward(self, x, adj=None) -> 返回与 x 同形状 [B, T, N, C] 的张量。
- 张量约定 [Batch, Time, Nodes, Channels], 【绝不能丢节点维 N】(不要 flatten/mean 掉 N)。
- 必须可微(不要用 round/argmax/detach/.item()/原地操作破坏梯度)。
- 不要访问任何标签/目标(forward 只能用 x 和 adj)。
- 参数量适中(不要造百万级巨型层)。
- 优先用 additive_residual: 内部留一个 nn.Parameter(torch.zeros(1)) 作为残差缩放 α,
  输出 = 主路径 + α*新分支, 这样初始安全。

只输出一段 Python 代码(用 ```python ``` 包裹), 包含必要的 import(torch, torch.nn as nn)和这一个类。不要其它解释。"""


def build_plan_prompt(req: FusionRequest) -> list[dict]:
    ctx = req.to_prompt_context()
    import json
    user = "瓶颈与跨域机制集:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
    return [{"role": "system", "content": _SYS_PLAN}, {"role": "user", "content": user}]


def build_code_prompt(req: FusionRequest, plan: FusionPlan, prev_error: str = "") -> list[dict]:
    import json
    parts = [
        "融合计划:\n" + json.dumps({
            "operator_name": plan.operator_name, "composition": plan.composition,
            "shared_structure": plan.shared_structure, "rationale": plan.rationale,
            "source_mechanisms": plan.source_mechanisms,
        }, ensure_ascii=False, indent=2),
        "\n相关机制的数学结构(实现灵感):",
        json.dumps([{"name": m.name, "math_structure": m.math_structure} for m in req.mechanisms],
                   ensure_ascii=False, indent=2),
    ]
    if prev_error:
        parts.append(f"\n⚠️ 上一版代码验证失败, 错误如下, 请修正:\n{prev_error[:1200]}")
    return [{"role": "system", "content": _SYS_CODE}, {"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# 代码抽取 + 沙箱执行
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str:
    """从 LLM 响应里抽 ```python ... ``` 代码块(无围栏则返回原文)。"""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def extract_json(text: str) -> dict:
    """从 LLM 响应里抽第一个 JSON 对象。"""
    import json
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("响应中未找到 JSON")
    return json.loads(m.group(0))


def exec_operator_code(code: str, class_name: str):
    """在受限命名空间 exec 算子代码, 返回该类。沙箱: 仅给 torch/nn, 无 IO/网络。

    安全: 这是合成代码, 必须隔离。本函数只供 worktree 沙箱内调用。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ns: dict = {"torch": torch, "nn": nn, "F": F, "__builtins__": __builtins__}
    exec(compile(code, "<synthesized_op>", "exec"), ns)
    if class_name not in ns:
        raise ValueError(f"代码中未定义类 {class_name}")
    cls = ns[class_name]
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ValueError(f"{class_name} 不是 nn.Module 子类")
    return cls


# ---------------------------------------------------------------------------
# 合成器
# ---------------------------------------------------------------------------


class OperatorSynthesizer:
    """plan-then-code 算子合成 + 验证 harness 守卫 + bounded 重试。"""

    def __init__(self, llm: LLMClient, cfg: SynthesisConfig | None = None):
        self.llm = llm
        self.cfg = cfg or SynthesisConfig()

    def synthesize(self, req: FusionRequest, needs_adj: bool = False) -> SynthesisResult:
        # 1) 计划
        try:
            plan_text = self.llm.chat(build_plan_prompt(req), temperature=self.cfg.temperature)
            plan_d = extract_json(plan_text)
            plan = FusionPlan(
                operator_name=plan_d["operator_name"], rationale=plan_d.get("rationale", ""),
                shared_structure=plan_d.get("shared_structure", ""),
                composition=plan_d.get("composition", "additive_residual"),
                source_mechanisms=plan_d.get("source_mechanisms", []),
                expected_effect=plan_d.get("expected_effect", ""),
            )
            plan.validate()
        except Exception as e:
            return SynthesisResult(False, attempts=0, last_error=f"计划阶段失败: {type(e).__name__}: {e}")

        # 2-4) 代码 + 验证 + 重试
        prev_error = ""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                code_text = self.llm.chat(build_code_prompt(req, plan, prev_error),
                                          temperature=self.cfg.temperature)
                code = extract_code(code_text)
                cls = exec_operator_code(code, plan.operator_name)
            except Exception as e:
                prev_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
                continue

            # 验证 harness 全门
            def factory(channels, num_nodes):
                return cls(channels=channels, num_nodes=num_nodes)

            rep = validate_operator(factory, self.cfg.validation, needs_adj=needs_adj)
            if rep.passed:
                op = SynthesizedOperator(
                    name=plan.operator_name, code=code, plan=plan, needs_adj=needs_adj,
                    validated=True, validation_gate="all",
                )
                return SynthesisResult(True, operator=op, attempts=attempt, plan=plan)
            prev_error = f"验证未过, 门={rep.gate}: {rep.reason}"

        return SynthesisResult(False, attempts=self.cfg.max_retries,
                               last_error=f"重试 {self.cfg.max_retries} 次仍未过验证: {prev_error}",
                               plan=plan)
