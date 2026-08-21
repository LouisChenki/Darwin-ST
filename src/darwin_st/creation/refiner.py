"""算子精炼器 (Operator Refiner) —— B2 微进化: 好算子不是一锤子买卖。

FunSearch / ELM 的核心机制: 已验证的算子代码被 LLM【反复小幅改进】("生"之外加"养")。
与 synthesizer (从零融合) 互补:
  - synthesizer: 跨域机制 → 全新算子 (探索, 高温 0.8)。
  - refiner:     族内已有版本 → 同族新版本 ( exploit, 低温 0.6 求稳)。

两种精炼入口 (ELM commit-message 分档 + FunSearch best-shot):
  - refine(parent, mode): 单父版 + 一句话改动指令 (small/param/struct 三档), LLM 输出完整改后代码。
  - best_shot(versions): 族内 ≥2 个有实测 MAE 的版本时, 给"较差版+较好版"两份代码 (都标 MAE)
    + 空的新版类头 (docstring="Improved version of ..."), LLM 续写完整新版。

与 synthesizer 共用硬约束 ([B,T,N,C]/不丢 N/可微/不见标签/参数适中) 与执行接地
(exec_operator_code + validate_operator 6 门 + bounded 重试 + 错误反馈进下一轮 prompt)。
LLM 可插拔 (MockLLM 本地测 / OpenAICompatLLM 服务器跑), code_backend 可选 (Aider 沙箱写码)。
"""

from __future__ import annotations

import math
import traceback
from dataclasses import dataclass, field

from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator
from darwin_st.creation.llm import LLMClient
from darwin_st.creation.registry import SYNTH_PREFIX, family_of, version_of
from darwin_st.creation.synthesizer import (
    LLM_INFRA_TAG,
    SynthesisResult,
    exec_operator_code,
    extract_code,
)
from darwin_st.creation.validation import ValidationConfig, validate_operator

__all__ = ["RefineConfig", "OperatorRefiner", "REFINE_MODES", "REFINE_MODE_WEIGHTS",
           "build_refine_prompt", "build_best_shot_prompt"]

# ELM commit-message 三档改动指令 (权重在 CreationLoop 侧按 40/30/30 加权随机)
REFINE_MODES = ("small", "param", "struct")
REFINE_MODE_WEIGHTS = (0.4, 0.3, 0.3)

_MODE_INSTRUCTIONS = {
    "small": "微调内部系数或维度 (如卷积核大小/隐藏维度/缩放系数/dropout), 不改模块结构",
    "param": "只改数值超参 (系数初始化/维度数值等常量), 不改任何模块结构",
    "struct": "只改一处结构 (替换一个子模块, 或改变聚合/融合方式), 其余保持原样",
}

_SYS_REFINE = """你是 PyTorch 算子精炼专家 (进化式微改进)。给定一个已验证的时空算子父版代码与其实测 MAE,
按给定的【一句话改动指令】做最小改进, 生成同族新版本 —— 不是重写, 是 commit-message 式小改。

【硬性契约 - 必须遵守】:
- 类名 = 给定的版本名 (不是父版名)。
- __init__(self, channels, num_nodes, **kw): channels=特征维C, num_nodes=节点数N。
- forward(self, x, adj=None) -> 返回与 x 同形状 [B, T, N, C] 的张量。
- 张量约定 [Batch, Time, Nodes, Channels], 【绝不能丢节点维 N】(不要 flatten/mean 掉 N)。
- 必须可微(不要用 round/argmax/detach/.item()/原地操作破坏梯度)。
- 不要访问任何标签/目标(forward 只能用 x 和 adj)。
- 参数量适中(不要造百万级巨型层)。
- 保留父版整体结构, 只做指令要求的最小改动; 主路径必须仍是【非平凡变换】(含可学习层)。

只输出一段完整的改后 Python 代码(用 ```python ``` 包裹), 包含必要的 import(torch, torch.nn as nn)和这一个类。不要其它解释。"""


@dataclass
class RefineConfig:
    max_retries: int = 2              # 验证失败重试上限 (与 synthesizer 同: 慢 LLM 往返代价高)
    temperature: float = 0.6          # 比创造 (0.8) 低: 精炼要稳,  exploit 而非探索
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# Prompt 构造
# ---------------------------------------------------------------------------


def _mae_text(mae) -> str:
    return f"{mae:.4f}" if _finite(mae) else "未评测"


def build_refine_prompt(parent: dict, mode: str, new_name: str, prev_error: str = "") -> list[dict]:
    """单父版精炼 prompt: 父代码全文 + 父实测 MAE + mode 一句话指令 + 硬性契约。"""
    parts = [
        f"父算子 {parent.get('reg_name', '')} | 实测 MAE={_mae_text(parent.get('real_mae'))}",
        f"改动指令 (mode={mode}): {_MODE_INSTRUCTIONS[mode]}",
        f"新类名必须是: {new_name}",
        "\n父版完整代码:\n```python\n" + (parent.get("code") or "") + "\n```",
    ]
    if prev_error:
        parts.append(f"\n⚠️ 上一版代码验证失败, 错误如下, 请修正:\n{prev_error[:1200]}")
    return [{"role": "system", "content": _SYS_REFINE}, {"role": "user", "content": "\n".join(parts)}]


def build_best_shot_prompt(worse: dict, better: dict, new_name: str, prev_error: str = "") -> list[dict]:
    """FunSearch best-shot prompt: 较差版(标 MAE)+较好版(标 MAE)两份代码 + 空新版类头, LLM 续写。"""
    parts = [
        "同一算子族的两个版本如下 (较好版实测 MAE 更低)。请吸收较好版的优点、避开较差版的短板,",
        "续写出一个更好的新版本 (可做小幅结构调整, 保持非平凡变换)。",
        f"\n== 较差版 {worse.get('reg_name', '')} (实测 MAE={_mae_text(worse.get('real_mae'))}) ==",
        "```python\n" + (worse.get("code") or "") + "\n```",
        f"\n== 较好版 {better.get('reg_name', '')} (实测 MAE={_mae_text(better.get('real_mae'))}) ==",
        "```python\n" + (better.get("code") or "") + "\n```",
        "\n新版类头 (请补全为【完整实现】, 类名必须是 " + new_name + "):",
        "```python\nimport torch\nimport torch.nn as nn\n\n"
        f"class {new_name}(nn.Module):\n"
        f'    """Improved version of {better.get("reg_name", "")} and {worse.get("reg_name", "")}."""\n'
        "```",
    ]
    if prev_error:
        parts.append(f"\n⚠️ 上一版代码验证失败, 错误如下, 请修正:\n{prev_error[:1200]}")
    return [{"role": "system", "content": _SYS_REFINE}, {"role": "user", "content": "\n".join(parts)}]


# ---------------------------------------------------------------------------
# 精炼器
# ---------------------------------------------------------------------------


class OperatorRefiner:
    """族内微进化: refine (单父版 + mode 指令) / best_shot (双版本对比续写)。

    parent/versions 用 dict (registry.family_versions/best_in_family 的 enrich 条目,
    含 reg_name/code/real_mae/composition/source_mechanisms; next_version 可选) 或同名属性对象。
    """

    def __init__(self, llm: LLMClient, cfg: RefineConfig | None = None, code_backend=None):
        self.llm = llm
        self.cfg = cfg or RefineConfig()
        self.code_backend = code_backend

    # -- 入口 1: 单父版 + mode 指令 --

    def refine(self, parent, mode: str, round_idx: int) -> SynthesisResult:
        """按 mode (small/param/struct) 对父版做最小改动, 产出同族新版本。"""
        if mode not in _MODE_INSTRUCTIONS:
            raise ValueError(f"未知 refine mode '{mode}' (须 ∈ {list(_MODE_INSTRUCTIONS)})")
        p = _as_dict(parent)
        if not p.get("code"):
            return SynthesisResult(False, attempts=0, last_error="父版无代码可精炼")
        new_name = self._next_name(p)
        plan = self._make_plan(p, new_name, f"refine({mode}): {_MODE_INSTRUCTIONS[mode]}",
                               expected=f"在 {p.get('reg_name', '')} 基础上改进 (父 MAE={_mae_text(p.get('real_mae'))})")
        try:
            plan.validate()
        except Exception as e:
            return SynthesisResult(False, attempts=0, last_error=f"精炼计划非法: {e}", plan=plan)
        return self._generate_and_validate(
            lambda err: build_refine_prompt(p, mode, new_name, err),
            new_name, plan, needs_adj=bool(p.get("needs_adj", False)))

    # -- 入口 2: FunSearch best-shot (族内 ≥2 个有 MAE 版本) --

    def best_shot(self, versions: list, round_idx: int) -> SynthesisResult:
        """给较差版+较好版 (按实测 MAE 排序取两端) 两份代码 + 空新版类头, LLM 续写新版。"""
        vs = [_as_dict(v) for v in versions]
        scored = [v for v in vs if _finite(v.get("real_mae")) and v.get("code")]
        if len(scored) < 2:
            return SynthesisResult(False, attempts=0,
                                   last_error="best_shot 需族内 ≥2 个有实测 MAE 的版本")
        scored.sort(key=lambda v: v["real_mae"])
        better, worse = scored[0], scored[-1]     # 两端对比: 信号最强
        next_v = max(version_of(v.get("reg_name", "")) for v in vs) + 1
        new_name = self._version_name(better.get("reg_name", ""), next_v)
        plan = self._make_plan(
            better, new_name,
            rationale=(f"refine(best_shot): 吸收较好版 {better.get('reg_name', '')} "
                       f"(MAE={_mae_text(better.get('real_mae'))}) 的优点, 避开较差版 "
                       f"{worse.get('reg_name', '')} (MAE={_mae_text(worse.get('real_mae'))}) 的短板"),
            expected=f"超越族内最优 MAE {_mae_text(better.get('real_mae'))}")
        try:
            plan.validate()
        except Exception as e:
            return SynthesisResult(False, attempts=0, last_error=f"精炼计划非法: {e}", plan=plan)
        return self._generate_and_validate(
            lambda err: build_best_shot_prompt(worse, better, new_name, err),
            new_name, plan, needs_adj=bool(better.get("needs_adj", False)))

    # -- 内部: 命名 / 计划 / 生成+验证重试循环 --

    @staticmethod
    def _version_name(parent_reg_name: str, next_v: int) -> str:
        fam = family_of(parent_reg_name)
        base = fam[len(SYNTH_PREFIX):] if fam.startswith(SYNTH_PREFIX) else fam
        return f"{base}_v{next_v}"

    def _next_name(self, parent: dict) -> str:
        """新版本类名 <family_base>_v<N>: N 取 parent['next_version'] (CreationLoop 按族算好),
        缺省则父版版本号 + 1。"""
        reg = parent.get("reg_name", "")
        v = parent.get("next_version")
        if not isinstance(v, int) or v < 2:
            v = version_of(reg) + 1
        return self._version_name(reg, v)

    @staticmethod
    def _make_plan(parent: dict, new_name: str, rationale: str, expected: str) -> FusionPlan:
        """精炼版的 FusionPlan: composition/source_mechanisms 沿用父版, rationale 记改动指令。"""
        return FusionPlan(
            operator_name=new_name, rationale=rationale,
            shared_structure=parent.get("shared_structure", "") or "",
            composition=parent.get("composition") or "additive_residual",
            source_mechanisms=list(parent.get("source_mechanisms") or ["_refinement"]),
            expected_effect=expected)

    def _generate_and_validate(self, prompt_fn, new_name: str, plan: FusionPlan,
                               needs_adj: bool) -> SynthesisResult:
        """生成 → exec → 6 门验证, bounded 重试, 错误反馈进下一轮 prompt (仿 synthesizer._synthesize_from_plan)。"""
        prev_error = ""
        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                if self.code_backend is not None:
                    instr = "\n".join(m["content"] for m in prompt_fn(prev_error))
                    code, err = self.code_backend.write_operator(instr)
                    if code is None:
                        prev_error = f"{LLM_INFRA_TAG}代码后端失败: {err}"
                        continue
                else:
                    try:
                        text = self.llm.chat(prompt_fn(prev_error), temperature=self.cfg.temperature)
                    except Exception as e:     # LLM 服务异常 = 基础设施, 结构化标记
                        prev_error = f"{LLM_INFRA_TAG}{type(e).__name__}: {str(e)[:200]}"
                        continue
                    code = extract_code(text)
                cls = exec_operator_code(code, new_name)
            except Exception as e:
                prev_error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-600:]}"
                continue

            def factory(channels, num_nodes):
                return cls(channels=channels, num_nodes=num_nodes)

            rep = validate_operator(factory, self.cfg.validation, needs_adj=needs_adj)
            if rep.passed:
                op = SynthesizedOperator(name=new_name, code=code, plan=plan,
                                         needs_adj=needs_adj, validated=True, validation_gate="all")
                return SynthesisResult(True, operator=op, attempts=attempt, plan=plan)
            prev_error = f"验证未过, 门={rep.gate}: {rep.reason}"

        return SynthesisResult(False, attempts=self.cfg.max_retries,
                               last_error=f"重试 {self.cfg.max_retries} 次仍未过验证: {prev_error}",
                               plan=plan)


def _as_dict(parent) -> dict:
    """parent 归一成 dict: 支持 dict 或同名属性对象。"""
    if isinstance(parent, dict):
        return parent
    keys = ("reg_name", "code", "real_mae", "composition", "source_mechanisms",
            "shared_structure", "needs_adj", "version", "next_version")
    return {k: getattr(parent, k, None) for k in keys}


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
