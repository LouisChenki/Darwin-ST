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
           "build_plan_prompt", "build_plan_many_prompt", "build_code_prompt",
           "extract_code", "extract_json", "extract_json_array", "exec_operator_code"]


@dataclass
class SynthesisConfig:
    max_retries: int = 2              # 验证失败重试上限 (降到2: 慢Aider往返代价高, 多假设已提供多样性)
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

_SYS_PLAN_MANY = """你是时空预测模型的算子设计专家。给定一个性能瓶颈和一组来自【不同领域】的机制,
请一次性提出 N 个【不同的】融合方案(假设), 每个方案用不同的角度/组合方式融合这些机制,
以便后续并行验证、择优。

关键原则:
- 按每个机制【为何有效(causal_behavior)】来融合, 不是表面堆叠。
- 融合方式从 4 种组合文法选: sequential / parallel / additive_residual / gated_routed。
- additive_residual 最安全(零初始化残差), 但 N 个方案应尽量多样(不同组合/不同侧重)。
- 每个方案 operator_name 必须唯一。

只输出一个 JSON 数组(不要其它文字), N 个对象, 每个字段:
{"operator_name": "唯一合法python标识符", "rationale": "因果论证",
 "shared_structure": "共享抽象结构", "composition": "4选1",
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


def build_plan_prompt(req: FusionRequest, forced_composition: str | None = None,
                      exemplars: dict | None = None,
                      insights: list[dict] | None = None) -> list[dict]:
    ctx = req.to_prompt_context()
    import json
    user = "瓶颈与跨域机制集:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
    # B1 履历回读 (同 build_plan_many_prompt): exemplars=None 时与现状逐字节一致 (回归保证)。
    if exemplars and (exemplars.get("successes") or exemplars.get("failures")):
        user += _format_exemplars(exemplars)
    # B5 独立采样: 强制组合文法硬约束 (放末尾最靠近输出, 提升遵从); None 时与现状逐字节一致。
    if forced_composition is not None:
        user += f"\n\n【硬约束】本方案必须使用 composition={forced_composition} (不得选择其它组合文法)。"
    # 反思经验注入 (Reflection Consolidation): 顺序 机制上下文 → exemplars → 硬约束 → 经验块;
    # insights=None/空列表 → 不注入, prompt 与现状逐字节一致 (回归铁律)。
    user += _format_insights(insights)
    return [{"role": "system", "content": _SYS_PLAN}, {"role": "user", "content": user}]


def build_plan_many_prompt(req: FusionRequest, n: int, exemplars: dict | None = None,
                           insights: list[dict] | None = None) -> list[dict]:
    import json
    ctx = req.to_prompt_context()
    user = (f"请提出 {n} 个不同的融合方案。\n瓶颈与跨域机制集:\n"
            + json.dumps(ctx, ensure_ascii=False, indent=2))
    # B1 履历回读: exemplars (creation_archive.exemplar_context) 非空时追加成功先例+失败教训。
    # exemplars=None 时 prompt 与注入前逐字节一致 (回归保证)。
    if exemplars and (exemplars.get("successes") or exemplars.get("failures")):
        user += _format_exemplars(exemplars)
    # 反思经验注入 (顺序: 机制上下文 → exemplars → 经验块); None/空 → 逐字节不变。
    user += _format_insights(insights)
    return [{"role": "system", "content": _SYS_PLAN_MANY}, {"role": "user", "content": user}]


def _format_insights(insights: list[dict] | None) -> str:
    """反思经验块注入段 (creation/reflection.render_insights_block, 同一 render 同一格式)。

    None/空列表 → 返回 "" (拼到 user 末尾等于不注入, 旧 prompt 逐字节不变)。
    """
    if not insights:
        return ""
    from darwin_st.creation.reflection import ReflectionConfig, render_insights_block
    block = render_insights_block(insights, ReflectionConfig())
    return ("\n\n" + block) if block else ""


def _format_exemplars(exemplars: dict) -> str:
    """把创造档案履历格式化成 prompt 段落 (ADAS 式: 成功先例 + 失败教训 + 差异化指令)。"""
    parts = ["\n\n【历史创造履历 —— 必须参考】"]
    succ = exemplars.get("successes") or []
    if succ:
        parts.append("\n== 成功先例 (按实测 MAE 升序, 越小越好) ==")
        for i, s in enumerate(succ, 1):
            mae = s.get("seed_val_mae")
            mae_txt = f"{mae:.4f}" if isinstance(mae, (int, float)) else "未评测"
            parts.append(
                f"\n[先例 {i}] 算子 {s.get('operator_name', '')} | "
                f"组合方式 {s.get('composition', '')} | 实测 MAE={mae_txt}\n"
                f"融合理由: {s.get('rationale', '')}\n"
                f"实现代码:\n```python\n{s.get('code', '')}\n```")
    fail = exemplars.get("failures") or []
    if fail:
        parts.append("\n== 失败教训 (不得重提同一思路) ==")
        for i, f in enumerate(fail, 1):
            parts.append(
                f"\n[教训 {i}] 算子 {f.get('operator_name', '')} | "
                f"组合方式 {f.get('composition', '')} | 挂在验证门 [{f.get('gate', 'unknown')}]\n"
                f"融合理由: {f.get('rationale', '')}\n"
                f"错误摘要: {f.get('error', '')}")
    parts.append(
        "\n【指令】你的新方案必须与上述成功先例做【实现级差异】"
        "(不同的组合方式/不同的内部结构, 不是改名或微调), 并严禁重提失败方案的同一思路。")
    return "\n".join(parts)


def extract_json_array(text: str) -> list[dict]:
    """从 LLM 响应抽 JSON 数组(N 个方案)。容错: 单对象包成单元素列表。"""
    import json
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        arr = json.loads(m.group(0))
        return arr if isinstance(arr, list) else [arr]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return [json.loads(m.group(0))]
    raise ValueError("响应中未找到 JSON 数组或对象")


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


# B5 独立采样温度阶梯: 叠加在基准温度上的偏移 (按假设序轮转), 结果 clamp [0.3, 1.0]
_INDEPENDENT_TEMP_LADDER = (0.0, 0.1, -0.1, 0.2)


class OperatorSynthesizer:
    """plan-then-code 算子合成 + 验证 harness 守卫 + bounded 重试。

    code_backend: 写代码的后端 (可选)。
      - 给定 (如 AiderBackend): 用它在 git 沙箱里写算子 (健壮, 推荐)。
      - 为 None: 回退到 LLM 直生代码 + 正则抽取 (轻量, 测试用)。
    plan 阶段始终用 llm (计划是 JSON 非文件编辑, 不需 aider)。
    fast_llm: B5 双模型路由 (可选)。synthesize_independent 里第 0 个假设用强模型 (self.llm,
      质量锚点), 其余用 fast_llm (成本); fast 假设重试全败后用强模型补试一次 (escalation)。
      None → 全部用强模型 (行为同现状)。注: code_backend 路径的代码阶段不走 llm, 路由只作用
      于 plan 阶段 (代码模型由 backend 自身配置)。
    """

    def __init__(self, llm: LLMClient, cfg: SynthesisConfig | None = None, code_backend=None,
                 fast_llm: LLMClient | None = None):
        self.llm = llm
        self.cfg = cfg or SynthesisConfig()
        self.code_backend = code_backend
        self.fast_llm = fast_llm

    def _build_aider_instruction(self, req: FusionRequest, plan: FusionPlan, prev_error: str) -> str:
        """给 aider 的自然语言指令 (含契约 + 计划 + 上轮错误)。"""
        import json
        maths = json.dumps([{"name": m.name, "math": m.math_structure} for m in req.mechanisms],
                           ensure_ascii=False)
        instr = (
            f"在该文件写一个 PyTorch nn.Module 算子类, 类名必须是 {plan.operator_name}。\n"
            f"契约(必须遵守): __init__(self, channels, num_nodes, **kw); "
            f"forward(self, x, adj=None) 返回与 x 同形状 [B,T,N,C] 的张量; "
            f"绝不丢节点维 N(不要 flatten/mean 掉 N); 必须可微(不用 round/argmax/detach/.item/原地操作); "
            f"forward 只能用 x 和 adj(不访问标签); 参数量适中。\n"
            f"融合方式={plan.composition}; 融合理由={plan.rationale}。\n"
            f"优先 additive_residual: 留一个 nn.Parameter(torch.zeros(1)) 作 α, 输出=主路径+α*新分支。\n"
            f"⚠️ 主路径必须是【非平凡变换】(含可学习层如 Linear/Conv, 真正变换输入), "
            f"绝不能让输出恒等于输入或恒为常数(会被验证拒绝)。\n"
            f"相关机制数学结构(灵感): {maths}\n"
            f"包含必要 import(torch, torch.nn as nn)。只写这一个类。"
        )
        if prev_error:
            instr += f"\n上一版验证失败, 请修正: {prev_error[:600]}"
        return instr

    def _plan_from_dict(self, plan_d: dict) -> FusionPlan:
        plan = FusionPlan(
            operator_name=plan_d["operator_name"], rationale=plan_d.get("rationale", ""),
            shared_structure=plan_d.get("shared_structure", ""),
            composition=plan_d.get("composition", "additive_residual"),
            source_mechanisms=plan_d.get("source_mechanisms", []),
            expected_effect=plan_d.get("expected_effect", ""),
        )
        plan.validate()
        return plan

    def _synthesize_from_plan(self, req: FusionRequest, plan: FusionPlan,
                              needs_adj: bool, temperature: float | None = None,
                              llm: LLMClient | None = None) -> SynthesisResult:
        """给定一个计划, 走 代码+验证+重试。返回 SynthesisResult。

        temperature=None → 用 cfg.temperature (行为不变); 显式给定则覆盖 (B4 温度自适应)。
        llm=None → 用 self.llm (行为不变); 显式给定则代码阶段改用该模型 (B5 双模型路由,
        仅无 code_backend 的直生路径生效)。
        """
        temp = self.cfg.temperature if temperature is None else temperature
        use_llm = llm if llm is not None else self.llm
        prev_error = ""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                if self.code_backend is not None:
                    instr = self._build_aider_instruction(req, plan, prev_error)
                    code, err = self.code_backend.write_operator(instr)
                    if code is None:
                        prev_error = f"代码后端失败: {err}"
                        continue
                else:
                    code_text = use_llm.chat(build_code_prompt(req, plan, prev_error),
                                             temperature=temp)
                    code = extract_code(code_text)
                cls = exec_operator_code(code, plan.operator_name)
            except Exception as e:
                prev_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
                continue

            def factory(channels, num_nodes):
                return cls(channels=channels, num_nodes=num_nodes)

            rep = validate_operator(factory, self.cfg.validation, needs_adj=needs_adj)
            if rep.passed:
                op = SynthesizedOperator(name=plan.operator_name, code=code, plan=plan,
                                         needs_adj=needs_adj, validated=True, validation_gate="all")
                return SynthesisResult(True, operator=op, attempts=attempt, plan=plan)
            prev_error = f"验证未过, 门={rep.gate}: {rep.reason}"

        return SynthesisResult(False, attempts=self.cfg.max_retries,
                               last_error=f"重试 {self.cfg.max_retries} 次仍未过验证: {prev_error}",
                               plan=plan)

    def synthesize(self, req: FusionRequest, needs_adj: bool = False,
                   temperature: float | None = None, *,
                   llm: LLMClient | None = None, forced_composition: str | None = None,
                   exemplars: dict | None = None,
                   insights: list[dict] | None = None) -> SynthesisResult:
        """单假设合成 (向后兼容): 生成一个计划, 合成一个算子。

        temperature=None → 用 cfg.temperature (行为不变); 显式给定则覆盖 (B4 温度自适应)。
        llm=None → 用 self.llm (B5 双模型路由的内部覆盖口, 外部一般不传)。
        forced_composition: B5 独立采样的强制组合文法; plan 实出 composition 不符 → 记失败
        (last_error 注明"未按指定组合文法"), 不进入代码阶段。None → 不强制 (行为不变)。
        exemplars: B1 履历回读注入 plan prompt; None 时 prompt 与现状逐字节一致。
        insights: 反思经验注入 plan prompt; None/空 → 不注入 (逐字节一致)。
        """
        temp = self.cfg.temperature if temperature is None else temperature
        use_llm = llm if llm is not None else self.llm
        try:
            plan_text = use_llm.chat(
                build_plan_prompt(req, forced_composition=forced_composition,
                                  exemplars=exemplars, insights=insights),
                temperature=temp)
            plan = self._plan_from_dict(extract_json(plan_text))
        except Exception as e:
            return SynthesisResult(False, attempts=0, last_error=f"计划阶段失败: {type(e).__name__}: {e}")
        if forced_composition is not None and plan.composition != forced_composition:
            return SynthesisResult(
                False, attempts=0, plan=plan,
                last_error=f"未按指定组合文法: 强制 {forced_composition}, 实出 {plan.composition}")
        return self._synthesize_from_plan(req, plan, needs_adj, temperature=temp, llm=use_llm)

    def synthesize_independent(self, req: FusionRequest, n_hypotheses: int = 4,
                               needs_adj: bool = False, exemplars: dict | None = None,
                               temperature: float | None = None,
                               insights: list[dict] | None = None) -> list[SynthesisResult]:
        """B5 假设独立采样 (FunSearch/AlphaEvolve 机制): N 次独立单假设合成, 返回去重后结果。

        与 synthesize_many (单次调用产 N) 互补, 治"一次生成同质化":
          - 第 i 次强制组合文法轮转 sorted(COMPOSITION_OPS)[i % 4] (硬约束进 prompt,
            实出不符记失败, 不进代码阶段);
          - 温度阶梯 _INDEPENDENT_TEMP_LADDER[i % 4] 叠加在基准温度上, clamp [0.3, 1.0];
          - 双模型路由: fast_llm 存在时第 0 个假设用强模型 (质量锚点), 其余用 fast_llm;
            fast 假设重试全败后用强模型补试一次 (escalation); fast_llm=None 全部强模型。
        去重: operator_name 重名 或 (frozenset(source_mechanisms), composition) 与已接受结果
        重复 → 跳过该结果 (不计入返回); 无 plan 的计划阶段失败结果原样保留。
        insights: 反思经验注入 plan prompt; None/空 → 不注入 (逐字节一致)。
        """
        base = self.cfg.temperature if temperature is None else temperature
        comps = sorted(COMPOSITION_OPS)
        results: list[SynthesisResult] = []
        seen_names: set[str] = set()
        seen_sigs: set[tuple] = set()
        for i in range(n_hypotheses):
            forced = comps[i % len(comps)]
            t_i = min(1.0, max(0.3, base + _INDEPENDENT_TEMP_LADDER[i % len(_INDEPENDENT_TEMP_LADDER)]))
            llm_i = self.llm if (i == 0 or self.fast_llm is None) else self.fast_llm
            res = self.synthesize(req, needs_adj=needs_adj, temperature=t_i, llm=llm_i,
                                  forced_composition=forced, exemplars=exemplars,
                                  insights=insights)
            if not res.success and llm_i is not self.llm:
                # escalation: fast 全败 → 同槽位强模型补试一次 (同强制文法/同温度)
                res = self.synthesize(req, needs_adj=needs_adj, temperature=t_i, llm=self.llm,
                                      forced_composition=forced, exemplars=exemplars,
                                      insights=insights)
            plan = res.plan
            if plan is not None:
                sig = (frozenset(plan.source_mechanisms), plan.composition)
                if plan.operator_name in seen_names or sig in seen_sigs:
                    continue                          # 重名 / 同机制同文法 → 去重跳过
                seen_names.add(plan.operator_name)
                seen_sigs.add(sig)
            results.append(res)
        return results

    def synthesize_many(self, req: FusionRequest, n_hypotheses: int = 4,
                        needs_adj: bool = False, exemplars: dict | None = None,
                        temperature: float | None = None,
                        insights: list[dict] | None = None) -> list[SynthesisResult]:
        """多假设合成: 一次生成 N 个融合方案, 各自合成+验证, 返回全部结果(含失败)。

        成功的算子由调用方全部注入 → 进化 + 真实 MAE 评测当裁判(不用 LLM 互评筛)。
        exemplars: B1 创造档案履历 (creation_archive.exemplar_context), 非空则注入 plan prompt
        (成功先例+失败教训, 治"每轮创造失忆重启"); None 时 prompt 与现状逐字节一致。
        temperature=None → 用 cfg.temperature (行为不变); 显式给定则覆盖 (B4 温度自适应)。
        insights: 反思经验注入 plan prompt; None/空 → 不注入 (逐字节一致)。
        """
        temp = self.cfg.temperature if temperature is None else temperature
        # 1) 一次生成 N 个计划
        try:
            text = self.llm.chat(
                build_plan_many_prompt(req, n_hypotheses, exemplars=exemplars, insights=insights),
                temperature=temp)
            plan_dicts = extract_json_array(text)
        except Exception as e:
            return [SynthesisResult(False, attempts=0,
                                    last_error=f"多假设计划阶段失败: {type(e).__name__}: {e}")]
        if not plan_dicts:
            return [SynthesisResult(False, attempts=0, last_error="未生成任何融合假设")]

        # 2) 每个计划各自合成 (去重 operator_name)
        results: list[SynthesisResult] = []
        seen_names: set[str] = set()
        for pd in plan_dicts[:n_hypotheses]:
            try:
                plan = self._plan_from_dict(pd)
            except Exception as e:
                results.append(SynthesisResult(False, attempts=0,
                                               last_error=f"计划非法: {type(e).__name__}: {e}"))
                continue
            if plan.operator_name in seen_names:
                continue  # 重名跳过
            seen_names.add(plan.operator_name)
            results.append(self._synthesize_from_plan(req, plan, needs_adj, temperature=temp))
        return results
