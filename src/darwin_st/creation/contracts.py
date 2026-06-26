"""融合契约 (Fusion Contracts) —— harness ↔ LLM 的结构化数据接口。

研究 (docs/TIER2_RESEARCH_FINDINGS.md §7): 融合用 plan-then-code(先出显式融合计划,
再生成代码, 已验证 +25% Pass@1)。本模块定义这套交接的结构化数据:
  - FusionRequest:  harness → LLM, 含瓶颈 + 跨域互补机制集 + 约束
  - FusionPlan:     LLM → harness 第一步, 显式计划(共享结构/候选推断/组合算子)
  - SynthesizedOperator: LLM → harness 第二步, 算子代码 + 元信息

组合算子受 4 算子文法约束 (sequential/parallel/additive_residual/gated_routed),
防 Frankenstein 拼接。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin_st.knowledge.ontology import Mechanism

__all__ = ["COMPOSITION_OPS", "FusionRequest", "FusionPlan", "SynthesizedOperator"]

# 4 算子组合文法 (Modular DL survey) —— 约束 LLM 怎么组合
COMPOSITION_OPS = {
    "sequential",        # A 后接 B
    "parallel",          # A、B 并行后相加/拼接
    "additive_residual", # baseline + α·新分支 (零初始化, 最安全)
    "gated_routed",      # 门控选择 A/B
}


@dataclass
class FusionRequest:
    """harness → LLM: 请求把一组跨域机制融合成解决某瓶颈的新算子。

    由 orchestrator 从 P3-a 跨域检索结果 + 当前瓶颈组装。
    """

    bottleneck: str                        # 当前瓶颈描述
    target_preconditions: list[str]        # 瓶颈分解出的前提
    mechanisms: list[Mechanism]            # 跨域互补机制集 (P3-a 检索结果)
    baseline_operator: str = ""            # 要在其上做残差融合的基线算子名 (KEEP 的)
    constraints: dict = field(default_factory=lambda: {
        "tensor_shape": "[B,T,N,C] 进出, 保持节点维 N",
        "must_be_differentiable": True,
        "no_target_access": True,          # forward 不得访问标签 (防泄漏)
        "max_params": 2_000_000,
        "composition_grammar": sorted(COMPOSITION_OPS),
    })

    def to_prompt_context(self) -> dict:
        """组装成喂给 LLM 的结构化上下文 (每个机制陈述为'因果+解决哪个前提')。"""
        return {
            "bottleneck": self.bottleneck,
            "target_preconditions": self.target_preconditions,
            "mechanisms": [
                {
                    "name": m.name,
                    "abstract_function": m.abstract_function,
                    "causal_behavior": m.causal_behavior,    # 按"为何有效"融合
                    "math_structure": m.math_structure,      # 实现灵感种子
                    "addresses_preconditions": sorted(set(m.preconditions) & set(self.target_preconditions)),
                    "origin_domain": m.origin_domain,
                    "anti_patterns": m.anti_patterns,
                }
                for m in self.mechanisms
            ],
            "baseline_operator": self.baseline_operator,
            "constraints": self.constraints,
        }


@dataclass
class FusionPlan:
    """LLM → harness 第一步: 显式融合计划 (plan-then-code 的 plan)。

    强制 LLM 先想清楚再写代码: 共享结构是什么、要把哪个机制的什么性质投射过来、
    用哪个组合算子。harness 校验计划合法后才让其生成代码。
    """

    operator_name: str                     # 新算子名 (如 "masked_adaptive_graph")
    rationale: str                         # 为何这样融合 (因果论证)
    shared_structure: str                  # 各机制共享的抽象结构 (类比的对齐点)
    composition: str                       # ∈ COMPOSITION_OPS
    source_mechanisms: list[str]           # 用了哪些机制 (name)
    expected_effect: str                   # 预期解决什么瓶颈

    def validate(self) -> None:
        if self.composition not in COMPOSITION_OPS:
            raise ValueError(f"composition '{self.composition}' 不在 4 算子文法 {sorted(COMPOSITION_OPS)}")
        if not self.source_mechanisms:
            raise ValueError("融合至少需 1 个源机制")
        if not self.operator_name.isidentifier():
            raise ValueError(f"operator_name '{self.operator_name}' 须是合法 Python 标识符")


@dataclass
class SynthesizedOperator:
    """LLM → harness 第二步: 合成的算子 (代码 + 元信息)。

    code 是一段定义 nn.Module 的 Python 源 (由 Aider 写入 operators 副本)。
    经 validation harness 全门通过后, 才注册进算子库 + 真实评测。
    """

    name: str
    code: str                              # nn.Module 源码
    plan: FusionPlan
    needs_adj: bool = False                # forward 是否需要邻接
    validated: bool = False                # 是否已过验证 harness
    category: str = "spatiotemporal"       # 算子类别 spatial/temporal/spatiotemporal (融合算子默认时空一体)
    validation_gate: str = ""              # 验证停在哪个门 (或 "all")
    real_mae: float | None = None          # 真实评测 MAE (注入算子库后)
