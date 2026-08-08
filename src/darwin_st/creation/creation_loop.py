"""创造循环 (Creation Loop) —— Tier-2 接入优化, 完成闭环。

把跨域知识 + 算子合成 + 注入接成一个可被 orchestrator 在【停滞时】触发的步骤:
  停滞 → 诊断瓶颈 → 跨域检索互补机制集 → LLM 融合 + Aider 写算子 → 验证 →
  注入算子库 → 产出一个用新算子的 genotype 交给进化评估 → 创造履历记 B1 履历本。

B7 通道路由: 检索命中自监督/掩码族机制 (AUX_MECHANISM_FAMILIES, 见 _should_create_aux)
且 enable_aux_creation → 辅助任务通道 (synthesize_aux_task → register_aux → seed 挂
aux_op 槽, 架构不动); 未命中或开关关闭 → 原算子通道 (行为同 B7 前)。

设计 (契合两层自治, 用户拍板的职责分离):
  - 创造是 Tier-2 低频(仅停滞触发), Tier-1 进化高频。
  - 合成算子进【算子库+archive】(给进化), 不进机制库(机制库静态只读)。
  - 融合经验(work/不work) → B1 履历本(全量结构化记录, 下轮 prompt 回读), 不混进机制卡。
    memory insights 表不再由本模块写流水账 (信息是履历本子集且无人读取, 双写已停),
    今后只由反思环节写入 (_record_insight 保留给该环节, 见方法注释)。
  - 所有外部依赖(检索 store/embedder、synthesizer、registry、memory)注入 → 无 LLM/GPU 可 mock 全测。

诚实: 真实 MAE 评测合成算子由 orchestrator 的既有 eval_fn 完成(本模块只负责"造+注入+
产出待评 genotype")。评测器是最终裁判, 低质融合自然被进化淘汰。
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from dataclasses import dataclass, field

from darwin_st.creation.contracts import FusionRequest
from darwin_st.creation.synthesizer import OperatorSynthesizer
from darwin_st.creation.refiner import REFINE_MODES, REFINE_MODE_WEIGHTS, OperatorRefiner
from darwin_st.creation.registry import OperatorRegistry
from darwin_st.knowledge.retrieval import find_cross_domain_analogy
from darwin_st.search.genotype import Genotype, STBlock

__all__ = ["CreationConfig", "CreationOutcome", "CreationLoop", "diagnose_bottleneck",
           "parse_failure_gate", "AUX_MECHANISM_FAMILIES"]


@dataclass
class CreationConfig:
    target_domain: str = "ST"
    max_mechanisms: int = 3        # 跨域互补集大小
    n_hypotheses: int = 4          # 一次生成多少个融合假设 (单 Generator 生 N 个)
    seed: int = 0
    use_llm_diagnosis: bool = True  # 优先用 LLM 诊断瓶颈 (训练信号驱动多样化); 失败退回规则版
    seed_hpo_trials: int = 20      # 创造 seed 专项大 HPO 预算 (AlphaEvolve 式优胜者深评; 普通架构走 hpo_cfg)
    # B3 评估中间档 (proxy 粗筛): 合成算子过验证门后先短训打分, 前 top_k 才给深评大预算
    use_proxy: bool = True         # False → 全部 seed 直接大预算 (行为同 B3 之前)
    proxy_top_k: int = 2           # 深评名额: proxy 分升序前 k 个拿 seed_hpo_trials
    proxy_epochs: int = 4          # proxy 短训 epoch 数 (生产接线用, 见 run_creation_loop)
    proxy_max_batches: int = 60    # proxy 每 epoch 训练 batch 上限 (生产接线用)
    small_hpo_trials: int = 3      # 浅评小 HPO 预算 (仍被真实评测器裁决, 只是预算小)
    # B2 微进化 (算子谱系精炼): 候选族选择阈值
    refine_mae_window: float = 0.5       # 族 best 实测 MAE 距当前 best ≤ 此值视为"离前沿不远"
    refine_max_family_versions: int = 3  # 族版本数 < 此值视为"仍在成长期", 直接候选
    # B5 假设独立采样 (FunSearch/AlphaEvolve): True → synthesize_independent (N 次独立调用,
    # 组合文法轮转 + 温度阶梯 + 双模型路由); False → synthesize_many (单次调用产 N, 旧行为)
    independent_sampling: bool = True
    # B7 辅助任务创造通道: True (默认) 且检索命中自监督/掩码族机制 (AUX_MECHANISM_FAMILIES)
    # → 走辅助任务通道 (造自监督辅助损失模块, 挂 genotype aux_op 槽); False → 全部走算子通道 (行为同 B7 前)
    enable_aux_creation: bool = True


@dataclass
class CreationOutcome:
    """一次创造尝试的结果 (可含多个成功算子)。"""

    success: bool                          # 至少有一个算子成功合成+注入
    operator_names: list[str] = field(default_factory=list)   # 注入的算子注册名(多个)
    seed_genotypes: list[Genotype] = field(default_factory=list)  # 待评 genotype(多个)
    bottleneck: str = ""
    retrieved_mechanisms: list[str] = field(default_factory=list)
    n_hypotheses: int = 0                  # 生成的假设数
    n_success: int = 0                     # 成功合成的数
    reason: str = ""
    insight: str = ""

    # 向后兼容单算子访问
    @property
    def operator_name(self) -> str | None:
        return self.operator_names[0] if self.operator_names else None

    @property
    def seed_genotype(self) -> Genotype | None:
        return self.seed_genotypes[0] if self.seed_genotypes else None


def diagnose_bottleneck(best_genotype: Genotype | None, sota_gap: float | None) -> str:
    """从当前最优架构诊断瓶颈描述 (v1 启发式; 真实可读训练曲线增强)。

    依据当前最优用了哪些算子族, 推断"缺什么", 拼成瓶颈描述给跨域检索。
    """
    if best_genotype is None:
        return "模型整体精度不足,需要更强的时空表示能力"

    spatial = {b.spatial_op for b in best_genotype.blocks}
    temporal = {b.temporal_op for b in best_genotype.blocks}
    clues = []
    # 缺长程时序建模
    if not (temporal & {"attn", "gru"}):
        clues.append("长程时序依赖捕获不足")
    # 缺自适应/注意力空间
    if not (spatial & {"gat", "adaptive", "diffusion"}):
        clues.append("空间关系建模较弱,节点交互不充分")
    # 没用嵌入
    if not best_genotype.embedding.use_node:
        clues.append("节点身份不可区分")
    if sota_gap is not None and sota_gap > 2.0:
        clues.append("整体精度距 SOTA 差距较大,需要引入新机制")
    if not clues:
        clues.append("局部最优,需要跨域新机制打破瓶颈")
    return "; ".join(clues)


_GATE_RE = re.compile(r"门=([A-Za-z_]+)")


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def parse_failure_gate(last_error: str) -> str:
    """从 SynthesisResult.last_error 解析失败门名 (B1 履历用)。

    合成失败错误形如 "重试 N 次仍未过验证: 验证未过, 门=shape: ..." → 提取 "shape";
    计划阶段失败 ("计划阶段失败/计划非法/多假设计划阶段失败") → "plan"; 解析不出 → "unknown"。
    """
    if not last_error:
        return "unknown"
    m = _GATE_RE.search(last_error)
    if m:
        return m.group(1)
    if "计划" in last_error:
        return "plan"
    return "unknown"


# B7 自监督/掩码族机制卡名集合 (辅助任务通道路由判据)。
# 这类机制的因果行为天然是"设计一个自监督预训练/正则辅助任务" (STD-MAE 路线),
# 融合成架构算子反而语义错位 —— 命中即走辅助任务通道 (造 aux 损失模块, 非架构算子)。
# 可扩展: 新机制卡入库时把卡名 (knowledge/data/mechanism_cards.json 的 name 字段) 加入
# 本集合即可路由到 aux 通道, 无需改路由代码。
AUX_MECHANISM_FAMILIES = frozenset({
    "masked_autoencoding",                # 掩码自编码 (MAE 系, 掩码重建路线源头)
    "decoupled_st_masked_pretraining",    # 时空解耦掩码预训练 (STD-MAE 本体)
    "multi_strategy_patch_pretraining",   # 多策略 patch 掩码预训练
    "correlation_adaptive_augmentation",  # 相关自适应增强 (自监督增强系)
    "contrastive_learning",               # 对比学习 (自监督族)
})


def _should_create_aux(mechanisms) -> bool:
    """B7 通道路由判定: 检索到的机制卡名命中自监督/掩码族 (AUX_MECHANISM_FAMILIES) → True。

    互补集中任一机制命中即走辅助任务通道 (辅助任务是比架构融合更自然的创造对象);
    enable_aux_creation 开关的联合判定在 maybe_create 调用处做, 本函数纯看机制。
    """
    return any(m.name in AUX_MECHANISM_FAMILIES for m in mechanisms)


class CreationLoop:
    """Tier-2 创造步骤。orchestrator 在停滞时调 maybe_create()。"""

    def __init__(self, store, embedder, synthesizer: OperatorSynthesizer,
                 registry: OperatorRegistry, memory=None, config: CreationConfig | None = None,
                 llm=None, archive=None, refiner: OperatorRefiner | None = None,
                 proxy_fn=None, temperature: float = 0.8, state_path: str | None = None,
                 reflection=None):
        self.store = store
        self.embedder = embedder
        self.synthesizer = synthesizer
        self.registry = registry
        self.memory = memory
        self.cfg = config or CreationConfig()
        self.llm = llm                       # OpenAICompatLLM (注入); None 时 _diagnose 退规则版
        self.archive = archive               # CreationArchive (B1 履历本); None=不记录, 行为不变
        # 反思巩固环节 (ReflectionLoop, 可选注入): None → 不反思也不注入经验, 行为与接入前一致
        self.reflection = reflection
        # B3 proxy 粗筛: proxy_fn(genotype) -> float (短训 val MAE), 设备/数据绑定由外部注入;
        # None 或 cfg.use_proxy=False → 全部 seed 走大预算, 行为与 B3 之前完全一致
        self.proxy_fn = proxy_fn
        # 最近一次 _assign_seed_budgets 的逐候选 proxy 失败原因 (异常摘要/"inf"/None),
        # 与 _assign_seed_budgets 返回的 proxy MAE 列表对齐, 供履历 record 的 proxy_error 落盘。
        self._last_proxy_errors: list[str | None] = []
        # B2 微进化: 精炼器默认复用 synthesizer 的 LLM (低温配置独立); 显式传 None 之外的对象可 mock
        if refiner is not None:
            self.refiner = refiner
        elif synthesizer is not None and getattr(synthesizer, "llm", None) is not None:
            self.refiner = OperatorRefiner(synthesizer.llm)
        else:
            self.refiner = None              # 无 LLM → maybe_refine 恒 None (回落从零创造)
        self._refine_rng = random.Random(self.cfg.seed)  # mode 加权随机源 (seed 固定, 测试可复现)
        self._round_idx = 0                  # 第几次创造 (OPRO 轨迹用)
        # OPRO 轨迹: [{round, bottleneck, preconditions, result_mae, score}]
        # score = B4 方向信用分 (见 update_direction_outcome)
        self._history: list[dict] = []
        # B4 温度自适应 (LLMatic curiosity): 全灭升温探索 / 被 adopted 降温利用, clamp [0.5, 1.0]
        self._temperature = min(1.0, max(0.5, temperature))
        # B4 方向状态持久化: _history + _temperature 落盘, 重启回放不归零。路径显式注入优先;
        # 缺省与 B1 履历本同目录 (direction_state.json, 同生命周期, scope 隔离自然继承);
        # 无履历本且不显式给路径 → 不持久化 (行为同改动前)。
        if state_path is not None:
            self._state_path = state_path
        elif archive is not None:
            self._state_path = os.path.join(os.path.dirname(os.path.abspath(archive.path)),
                                            "direction_state.json")
        else:
            self._state_path = None
        self._restore_direction_state()      # 文件存在则回放; 坏文件/不存在 → 空启动

    def _prompt_insights(self, dataset: str) -> list[dict] | None:
        """反思经验注入口 (合成与诊断对称): memory 非 None 且 insights 非空 → 取 top-N。

        需接入 reflection (ReflectionLoop) 才生效 —— insights 表只由反思环节写入,
        未接反思时表恒空, 返回 None → 下游 prompt 与现状逐字节一致 (回归铁律)。
        读库异常 → None (经验注入是增益, 不中止创造闭环)。
        """
        if self.memory is None or self.reflection is None:
            return None
        try:
            insights = self.memory.list_insights_by_confidence(
                dataset, limit=self.reflection.cfg.prompt_top_n)
        except Exception:
            return None
        return insights or None

    def _diagnose(self, best_genotype: Genotype | None, sota_gap: float | None,
                  best_trace: dict | None, dataset: str = "PeMS04") -> tuple[str, list[str] | None]:
        """诊断瓶颈, 返回 (bottleneck 字符串, override_preconditions 或 None)。

        LLM 优先 (有 llm + 有 trace + 开关开): 训练信号 → 摘要 → LLM 诊断 → 受控前提词 (绕过关键词匹配)。
        任何失败 (无 llm / 无 trace / 解析失败) 退回规则版 diagnose_bottleneck, override=None (走原 decompose)。
        """
        if self.cfg.use_llm_diagnosis and self.llm is not None and best_trace is not None:
            try:
                from darwin_st.creation.diagnosis import (
                    diagnose_bottleneck_llm,
                    summarize_trace,
                )

                summary = summarize_trace(best_trace, best_genotype, sota_gap)
                diag = diagnose_bottleneck_llm(summary, self._history, self._round_idx, self.llm,
                                               insights=self._prompt_insights(dataset))
                if diag is not None and diag.preconditions:
                    return diag.bottleneck, diag.preconditions
            except Exception:
                pass  # 任何 LLM/解析异常 → 规则兜底 (创造闭环不中止)
        return diagnose_bottleneck(best_genotype, sota_gap), None

    def maybe_create(self, best_genotype: Genotype | None, sota_gap: float | None = None,
                     run_tag: str = "exp/auto", dataset: str = "PeMS04",
                     best_trace: dict | None = None, best_hps: dict | None = None) -> CreationOutcome:
        """尝试一次跨域创造。返回 CreationOutcome(含待评 genotype)。

        best_trace: 最优架构的训练动态轨迹 (TrainTrace asdict)。给定且 LLM 可用时走 LLM 诊断
        (训练信号驱动多样化瓶颈诊断); 否则退回规则版 diagnose_bottleneck。见 creation/diagnosis.py。
        best_hps: 最优架构的最优超参。作为创造 seed 的 warm-start (继承父超参省冷启动)。
        """
        # 1) 诊断瓶颈 (LLM 优先, 规则兜底; 反思经验注入诊断 prompt, 见 _prompt_insights)
        bottleneck, override_precs = self._diagnose(best_genotype, sota_gap, best_trace,
                                                    dataset=dataset)
        # OPRO 轨迹: 记本轮诊断方向 + 当前水位 (result_mae 用 best_trace 的 best_mae, 缺则用 gap 推)
        cur_mae = None
        if best_trace is not None and best_trace.get("best_mae") not in (None, float("inf")):
            cur_mae = best_trace.get("best_mae")
        round_idx = self._round_idx
        self._history.append({"round": round_idx, "bottleneck": bottleneck,
                              "preconditions": override_precs or [], "result_mae": cur_mae,
                              "score": 0.0})   # B4 方向信用分初始 0 (结局回授累加)
        self._round_idx += 1
        self._persist_direction_state()      # 轨迹 append 即落盘 (重启不归零)

        # 2) 跨域检索互补机制集 (override_precs 非空则直接用 LLM 给的前提词做 FAC 召回, 绕过关键词匹配)
        res = find_cross_domain_analogy(
            bottleneck, self.store, self.embedder,
            target_domain=self.cfg.target_domain, cross_domain_only=True,
            max_results=self.cfg.max_mechanisms,
            override_preconditions=override_precs)
        mech_names = [m.name for m in res.mechanisms]
        if not res.mechanisms:
            return CreationOutcome(False, bottleneck=bottleneck, reason="跨域检索无结果")

        # 3) 多假设合成: 一次生成 N 个融合方案, 各自合成+验证
        req = FusionRequest(
            bottleneck=bottleneck, target_preconditions=res.target_preconditions,
            mechanisms=res.mechanisms,
            baseline_operator=(best_genotype.blocks[0].spatial_op if best_genotype else ""))
        # B1 履历回读: 档案非空则把过往成功先例+失败教训喂进 plan prompt (治"每轮失忆重启")
        exemplars = None
        if self.archive is not None:
            ctx = self.archive.exemplar_context()
            if ctx["successes"] or ctx["failures"]:
                exemplars = ctx
        # 反思经验注入 (与诊断对称): memory 有反思产出的 insights → plan prompt 末尾经验块;
        # 无 → None, prompt 与现状逐字节一致
        insights = self._prompt_insights(dataset)
        # B7 通道路由: 检索命中自监督/掩码族机制且开关开 → 辅助任务通道 (造自监督辅助损失模块,
        # 挂 genotype aux_op 槽); 未命中或开关关闭 → 下方原算子通道 (行为同 B7 前)
        if self.cfg.enable_aux_creation and _should_create_aux(res.mechanisms):
            return self._maybe_create_aux(req, best_genotype, mech_names, round_idx,
                                          best_hps=best_hps, exemplars=exemplars,
                                          insights=insights)
        # B5: 默认假设独立采样 (组合文法轮转 + 温度阶梯 + 双模型路由);
        # independent_sampling=False → 单次调用产 N 的旧路径 (行为同 B5 前)
        if self.cfg.independent_sampling:
            results = self.synthesizer.synthesize_independent(
                req, n_hypotheses=self.cfg.n_hypotheses, needs_adj=False,
                exemplars=exemplars, temperature=self._temperature, insights=insights)
        else:
            results = self.synthesizer.synthesize_many(req, n_hypotheses=self.cfg.n_hypotheses,
                                                       needs_adj=False, exemplars=exemplars,
                                                       temperature=self._temperature,
                                                       insights=insights)
        successes = [r for r in results if r.success and r.operator is not None]

        # B1: 失败假设也进履历本 (失败教训是下一轮 prompt 的反例素材)
        if self.archive is not None:
            for r in results:
                if not (r.success and r.operator is not None):
                    self._record_to_archive(req, r, round_idx)

        if not successes:
            # B4: 全部假设挂门 (无 seed 产出, 结局立即可知) → 本轮方向记 −1 + 升温探索
            self.update_direction_outcome(round_idx, -1.0)
            self._temperature = min(1.0, self._temperature + 0.05)
            self._persist_direction_state()  # 升温在 update_direction_outcome 落盘之后, 补落一次
            err = results[0].last_error if results else "无结果"
            insight = f"融合 {mech_names} 解决'{bottleneck}': {len(results)} 个假设均未过验证"
            return CreationOutcome(False, bottleneck=bottleneck, retrieved_mechanisms=mech_names,
                                   n_hypotheses=len(results), n_success=0,
                                   reason=err[:200], insight=insight)

        # 4) 注入所有成功算子 + 各产出一个待评 genotype (+ B1 成功履历, 带注册名与 seed 签名)
        op_names, seeds = [], []
        for r in successes:
            reg_name = self.registry.register(r.operator)
            op_names.append(reg_name)
            seed = self._make_seed_genotype(best_genotype, reg_name, best_hps)
            seed._seed_meta["creation_round"] = round_idx  # B4: 结局回授归属的诊断轮次
            seeds.append(seed)

        # B3 proxy 粗筛: 短训排序分预算 (前 top_k 深评, 其余浅评), proxy 分+失败原因进履历
        proxy_maes = self._assign_seed_budgets(seeds)
        if self.archive is not None:
            for r, reg_name, seed, pmae, perr in zip(successes, op_names, seeds, proxy_maes,
                                                     self._last_proxy_errors):
                self._record_to_archive(req, r, round_idx, reg_name=reg_name,
                                        seed_signature=seed.signature(), proxy_mae=pmae,
                                        proxy_error=perr)

        compositions = [r.plan.composition for r in successes]
        insight = (f"融合 {mech_names} 解决'{bottleneck}': {len(results)} 假设中 "
                   f"{len(successes)} 个成功 → 注入 {op_names}(组合 {compositions}), 待评测")

        return CreationOutcome(
            True, operator_names=op_names, seed_genotypes=seeds,
            bottleneck=bottleneck, retrieved_mechanisms=mech_names,
            n_hypotheses=len(results), n_success=len(successes),
            reason="合成并注入成功", insight=insight)

    # ------------------------------------------------------------------
    # B7 辅助任务创造通道 (自监督辅助损失的 合成→注册→seed; 路由判据见 _should_create_aux)
    # ------------------------------------------------------------------

    def _maybe_create_aux(self, req: FusionRequest, best_genotype: Genotype | None,
                          mech_names: list[str], round_idx: int,
                          best_hps: dict | None = None, exemplars: dict | None = None,
                          insights: list[dict] | None = None) -> CreationOutcome:
        """辅助任务通道: synthesize_aux_task → register_aux → seed 挂 aux_op 槽 → 履历。

        单假设 (设计空间已被 AuxTaskPlan 文法约束, 多样性靠多轮触发而非单轮多假设)。
        失败纪律与算子通道一致: 不中止闭环 + 履历记失败 + 方向信用分 −1 + 升温探索。
        """
        result = self.synthesizer.synthesize_aux_task(
            req, exemplars=exemplars, insights=insights, temperature=self._temperature)
        if not (result.success and result.operator is not None):
            if self.archive is not None:
                self._record_aux_to_archive(req, result, round_idx)
            # B4 同纪律: 挂门 (无 seed 产出, 结局立即可知) → 本轮方向记 −1 + 升温
            self.update_direction_outcome(round_idx, -1.0)
            self._temperature = min(1.0, self._temperature + 0.05)
            self._persist_direction_state()  # 升温在 update_direction_outcome 落盘之后, 补落一次
            err = result.last_error or "无结果"
            insight = f"辅助任务 {mech_names} 解决'{req.bottleneck}': 未过防泄漏验证门"
            return CreationOutcome(False, bottleneck=req.bottleneck,
                                   retrieved_mechanisms=mech_names, n_hypotheses=1, n_success=0,
                                   reason=err[:200], insight=insight)

        reg_name = self.registry.register_aux(result.operator)
        seed = self._make_seed_aux_genotype(best_genotype, reg_name, best_hps)
        seed._seed_meta["creation_round"] = round_idx   # B4: 结局回授归属的诊断轮次
        # B3: 与算子通道共用同一套预算分配 (单候选 ≤ top_k 时不调 proxy, 直接大预算)
        proxy_maes = self._assign_seed_budgets([seed])
        if self.archive is not None:
            self._record_aux_to_archive(req, result, round_idx, reg_name=reg_name,
                                        seed_signature=seed.signature(),
                                        proxy_mae=proxy_maes[0],
                                        proxy_error=self._last_proxy_errors[0])
        pattern = result.plan.mask_pattern if result.plan else "?"
        insight = (f"辅助任务 {mech_names} 解决'{req.bottleneck}': 合成 {reg_name} "
                   f"(掩码模式 {pattern}) → 挂 aux_op 槽, 待评测")
        return CreationOutcome(True, operator_names=[reg_name], seed_genotypes=[seed],
                               bottleneck=req.bottleneck, retrieved_mechanisms=mech_names,
                               n_hypotheses=1, n_success=1,
                               reason="辅助任务合成并注入成功", insight=insight)

    def _make_seed_aux_genotype(self, best: Genotype | None, reg_name: str,
                                best_hps: dict | None = None) -> Genotype:
        """辅助通道 seed: 不改架构, 只挂 aux_op 槽 (best.copy(); 无 best 冷启动用默认骨架)。

        _seed_meta 机制与 _make_seed_genotype 一致 (warm_start_hps + seed_hpo_trials 大预算 +
        is_creation_seed + operator_name=aux 注册名), orchestrator 履历回填/方向回授零改动复用。
        """
        if best is not None:
            g = best.copy()
        else:
            # 冷启动默认骨架 (同 run_creation_loop 弱基线; 参照 _make_seed_genotype 的 best=None 分支)
            g = Genotype(blocks=[STBlock(spatial_op="gcn", temporal_op="tcn")], hidden=64)
        g.aux_op = reg_name
        g.validate()
        g._seed_meta = {"warm_start_hps": best_hps or None,
                        "hpo_trials": self.cfg.seed_hpo_trials,
                        "is_creation_seed": True,
                        "operator_name": reg_name}   # B1: orchestrator 据此回写履历本/registry
        return g

    def _record_aux_to_archive(self, req: FusionRequest, r, round_idx: int,
                               reg_name: str | None = None, seed_signature: str | None = None,
                               proxy_mae: float | None = None,
                               proxy_error: str | None = None) -> None:
        """辅助通道履历 (B1): composition 记 aux:<mask_pattern>, operator_name=注册名, mechanisms 照记。

        成功: gate="all" + 模块代码 + 设计表字段 + 注册名 + seed 签名; 失败: gate=失败门
        (parse_failure_gate 与算子通道复用同一解析) + error + plan(若有)。
        档案 IO 失败不中止创造闭环 (履历是增益, 非必需)。
        """
        if self.archive is None:
            return
        from datetime import datetime, timezone

        from darwin_st.creation.creation_archive import CreationRecord

        plan = r.plan
        if r.success and r.operator is not None:
            gate, error, code = "all", None, r.operator.code
            op_name = reg_name or r.operator.name
        else:
            gate = parse_failure_gate(r.last_error)
            error = r.last_error or None
            code = ""
            op_name = plan.task_name if plan else ""
        rec = CreationRecord(
            round_idx=round_idx, created_at=datetime.now(timezone.utc).isoformat(),
            bottleneck=req.bottleneck, preconditions=list(req.target_preconditions),
            mechanisms=[m.name for m in req.mechanisms],
            operator_name=op_name,
            composition=f"aux:{plan.mask_pattern}" if plan else "aux",
            rationale=plan.rationale if plan else "",
            code=code, gate=gate, error=error, seed_signature=seed_signature,
            proxy_mae=proxy_mae,          # B3: proxy 粗筛分 (未走 proxy 路径为 None)
            proxy_error=proxy_error)      # B3: proxy 失败原因 (异常摘要/"inf"; 成功为 None)
        try:
            self.archive.record(rec)
        except Exception:
            pass                     # 档案写失败不中止闭环

    # ------------------------------------------------------------------
    # B4 方向信用分 + 温度自适应 (LLMatic curiosity 机制)
    # ------------------------------------------------------------------

    def update_direction_outcome(self, round_idx: int, score_delta: float) -> None:
        """把一轮创造方向的实测结局回授成信用分, 累加进 OPRO 轨迹 (诊断 prompt 据此展示+禁令)。

        记分规则:
          - maybe_create 内全部假设挂门 (无 seed 产出, 立即可知) → 同步记 −1;
          - orchestrator _digest 回填: seed adopted (KEEP) → +1; DISCARD/CRASH → −0.5。
        温度自适应: +1 (方向被实测验证) → 合成温度 −0.05 (收敛利用), clamp [0.5, 1.0];
        降温只由 adopted 触发, 罚分不升温 (全灭升温在 maybe_create 轮末做)。
        找不到 round_idx (如外部手工喂的轮次) 静默跳过 —— 回授是增益, 不中止闭环。
        """
        for h in self._history:
            if h.get("round") == round_idx:
                h["score"] = h.get("score", 0.0) + score_delta
                if score_delta > 0:
                    self._temperature = max(0.5, self._temperature - 0.05)
                self._persist_direction_state()   # 改分/调温即落盘 (重启不归零)
                return

    # ------------------------------------------------------------------
    # B2 微进化: 算子谱系精炼 ("生"之外加"养")
    # ------------------------------------------------------------------

    def maybe_refine(self, round_idx: int = 0, best_genotype: Genotype | None = None,
                     best_hps: dict | None = None, current_best_mae: float | None = None,
                     dataset: str = "PeMS04") -> CreationOutcome | None:
        """从算子家谱挑有潜力的族, 让 LLM 小幅改进族内版本 (FunSearch/ELM 式微进化)。

        候选族规则 (保持简单; B4 再加信用分仲裁, 此处不做复杂策略):
          - 族 best_in_family 的 real_mae 有限且有代码 (无实测 MAE 无从判断潜力);
          - 且 (族 best 距当前 best ≤ cfg.refine_mae_window, 或族版本数 < cfg.refine_max_family_versions)。
            current_best_mae 缺省/无效时以各族 best 的最小值为参考 (最强族总会候选)。
          - 多候选时取族 best MAE 最小者 (前沿优先)。
        精炼方式: 族内 ≥2 个版本有实测 MAE → best_shot (FunSearch 双版本对比续写);
        否则按 40/30/30 加权随机 (small/param/struct, 随机源 seed 固定可复现) 精炼族 best 版。
        成功 → register_variant + 复用 _make_seed_genotype 产待评 genotype
        (_seed_meta 加 is_refinement/family) + 写 B1 履历 (composition=refine:<mode>)。
        返回 None = 无候选族/无精炼器 (调用方回落 maybe_create); 精炼失败返回 success=False。
        """
        if self.refiner is None or self.registry is None:
            return None
        pick = self._pick_refine_candidate(current_best_mae)
        if pick is None:
            return None
        family, best, versions = pick
        parent_reg = best["reg_name"]

        scored = [v for v in versions if _finite(v.get("real_mae")) and v.get("code")]
        if len(scored) >= 2:
            mode = "best_shot"
            result = self.refiner.best_shot(versions, round_idx)
        else:
            mode = self._refine_rng.choices(REFINE_MODES, weights=REFINE_MODE_WEIGHTS, k=1)[0]
            parent = dict(best)
            parent["next_version"] = len(versions) + 1   # refiner 据此命名 <family_base>_v<N>
            result = self.refiner.refine(parent, mode, round_idx)

        mechanisms = list(result.plan.source_mechanisms) if result.plan else []
        if not (result.success and result.operator is not None):
            self._record_refinement(family, mode, result, round_idx, mechanisms=mechanisms)
            err = (result.last_error or "精炼失败")[:200]
            insight = f"精炼 {family} ({mode}) 未过验证: {err}"
            return CreationOutcome(False, bottleneck=f"refine:{family}", n_hypotheses=1,
                                   n_success=0, reason=err, insight=insight)

        reg_name = self.registry.register_variant(result.operator, parent_reg, round_idx=round_idx)
        seed = self._make_seed_genotype(best_genotype, reg_name, best_hps)
        seed._seed_meta["is_refinement"] = True   # orchestrator 据此回填 registry real_mae
        seed._seed_meta["family"] = family
        seed._seed_meta["creation_round"] = round_idx  # B4: 结局回授归属轮次
        # B3: 与 maybe_create 共用同一套预算分配 (单候选 ≤ top_k 时不调 proxy, 直接大预算)
        proxy_maes = self._assign_seed_budgets([seed])
        self._record_refinement(family, mode, result, round_idx, reg_name=reg_name,
                                seed_signature=seed.signature(), mechanisms=mechanisms,
                                proxy_mae=proxy_maes[0], proxy_error=self._last_proxy_errors[0])
        insight = (f"精炼 {family} ({mode}): {parent_reg} → {reg_name}, 待评测 "
                   f"(族 best MAE={best.get('real_mae')})")
        return CreationOutcome(True, operator_names=[reg_name], seed_genotypes=[seed],
                               bottleneck=f"refine:{family}", n_hypotheses=1, n_success=1,
                               reason="精炼并注入成功", insight=insight)

    def _pick_refine_candidate(self, current_best_mae: float | None):
        """候选族选择 (规则见 maybe_refine docstring)。返回 (family, best条目, versions) 或 None。"""
        scored = []
        for fam in self.registry.families():
            versions = self.registry.family_versions(fam)
            best = self.registry.best_in_family(fam)
            if best is None or not best.get("code"):
                continue                       # 无实测 MAE / 无代码 → 无法精炼
            scored.append((fam, best, versions))
        if not scored:
            return None
        if _finite(current_best_mae):
            ref = current_best_mae
        else:
            ref = min(b["real_mae"] for _, b, _ in scored)   # 无外部参考 → 族际最强者
        cands = [(fam, best, vs) for fam, best, vs in scored
                 if len(vs) < self.cfg.refine_max_family_versions
                 or abs(best["real_mae"] - ref) <= self.cfg.refine_mae_window]
        if not cands:
            return None
        cands.sort(key=lambda t: t[1]["real_mae"])           # 前沿优先: 族 best 最小者
        return cands[0]

    def _record_refinement(self, family: str, mode: str, result, round_idx: int,
                           reg_name: str | None = None, seed_signature: str | None = None,
                           mechanisms: list[str] | None = None,
                           proxy_mae: float | None = None,
                           proxy_error: str | None = None) -> None:
        """精炼履历写 B1 履历本: composition 记 refine:<mode>, mechanisms 沿用父版 plan 源机制。"""
        if self.archive is None:
            return
        from datetime import datetime, timezone

        from darwin_st.creation.creation_archive import CreationRecord

        plan = result.plan
        if result.success and result.operator is not None:
            gate, error, code = "all", None, result.operator.code
            op_name = reg_name or result.operator.name
        else:
            gate = parse_failure_gate(result.last_error)
            error = result.last_error or None
            code = ""
            op_name = plan.operator_name if plan else ""
        rec = CreationRecord(
            round_idx=round_idx, created_at=datetime.now(timezone.utc).isoformat(),
            bottleneck=f"refine:{family}", mechanisms=list(mechanisms or []),
            operator_name=op_name,
            composition=f"refine:{mode}",
            rationale=plan.rationale if plan else "",
            code=code, gate=gate, error=error, seed_signature=seed_signature,
            proxy_mae=proxy_mae, proxy_error=proxy_error)
        try:
            self.archive.record(rec)
        except Exception:
            pass                     # 档案写失败不中止闭环 (履历是增益, 非必需)

    # ------------------------------------------------------------------
    # B3 评估中间档: proxy 短训粗筛 (AlphaEvolve 级联评估 / LLMatic 廉价首筛)
    # ------------------------------------------------------------------

    def _assign_seed_budgets(self, seeds: list[Genotype]) -> list[float | None]:
        """决定每个创造 seed 的 HPO 预算 (深评大预算 vs 浅评小预算), 返回对齐的 proxy MAE 列表。

        proxy_fn 已注入且 cfg.use_proxy 且候选数 > proxy_top_k 时:
          1. 每个 seed 调 proxy_fn 短训打分 (单个异常 → 记 inf, 不拖死整轮);
          2. 全部 seed 先挂浅评小预算 (small_hpo_trials);
          3. 按 proxy MAE 升序 (Python sorted 稳定, 并列/inf 保持原顺序), 前 proxy_top_k 个
             升深评大预算 (seed_hpo_trials), 深评名额不超 top_k。
        否则 (无 proxy_fn / 开关关 / 候选数 ≤ top_k): 全部保持 _make_seed_genotype 挂的大预算,
        proxy_fn 一次不调, 行为与 B3 之前完全一致; 返回全 None (履历 proxy_mae 记空)。

        可观测 (线上实证: 一轮 4 候选 proxy 全 inf —— 与 worker 争 cuda:0 致 OOM/NaN —— 粗筛
        形同虚设且无任何日志): proxy 异常/非有限返回都打 [B3] 警告 (候选简述+原因截断),
        失败原因同时留在 self._last_proxy_errors (与返回列表对齐) 供履历 proxy_error 落盘。
        """
        if (not self.cfg.use_proxy) or self.proxy_fn is None \
                or len(seeds) <= self.cfg.proxy_top_k:
            self._last_proxy_errors = [None] * len(seeds)
            return [None] * len(seeds)
        proxy_maes: list[float] = []
        proxy_errors: list[str | None] = []
        for seed in seeds:
            brief = self._proxy_brief(seed)
            try:
                mae = float(self.proxy_fn(seed))
            except Exception as e:
                mae = float("inf")            # 单个候选 proxy 崩溃不拖死整轮创造
                err: str | None = f"{type(e).__name__}: {e}"[:200]
                print(f"[B3] proxy 候选 {brief} 短训异常 → 记 inf 排尾 (不拖死整轮): {err}")
            else:
                err = None
                if not math.isfinite(mae):    # 训练 NaN/熔断 (或争卡 OOM 后) 返回 inf
                    err = "inf" if math.isinf(mae) else "nan"
                    print(f"[B3] proxy 候选 {brief} 短训返回 {err} → 记 inf 排尾 "
                          f"(疑似训练 NaN/时间熔断, 或与 worker 争卡 OOM), 粗筛继续")
            proxy_maes.append(mae)
            proxy_errors.append(err)
        self._last_proxy_errors = proxy_errors
        for seed in seeds:
            seed._seed_meta["hpo_trials"] = self.cfg.small_hpo_trials   # 暂挂浅评小预算
        order = sorted(range(len(seeds)), key=lambda i: proxy_maes[i])  # 稳定排序保并列原序
        for i in order[: self.cfg.proxy_top_k]:
            seeds[i]._seed_meta["hpo_trials"] = self.cfg.seed_hpo_trials  # 胜者升深评
        return proxy_maes

    @staticmethod
    def _proxy_brief(seed: Genotype) -> str:
        """候选简述 (proxy 日志用): 算子名 + genotype 签名截断。"""
        meta = getattr(seed, "_seed_meta", None) or {}
        return f"{meta.get('operator_name', '?')}(sig={seed.signature()[:12]})"

    def _make_seed_genotype(self, best: Genotype | None, op_name: str,
                            best_hps: dict | None = None) -> Genotype:
        """用新算子产出一个待评 genotype。以最优为基(若有), 否则新建。

        按算子类别正确放槽 (Stage 1): 时空一体算子 → joint block (一条边管时空);
        空间/时序类 → 对应槽。避免把时空算子强塞空间槽 (语义错位)。

        给 seed 挂非字段元数据 _seed_meta: 标记走专项大 HPO (seed_hpo_trials) + 父超参 warm-start。
        _seed_meta 不进 to_dict/signature (graveyard dedup 安全), 随 genotype pickle 到 worker (进程后端),
        copy() 不带过去 (走 to_dict/from_dict, 故挂在副本上无父代泄漏)。见 train.make_eval_fn 的检测。
        """
        from darwin_st.search.operators import op_category
        cat = op_category(op_name)
        if best is not None:
            g = best.copy()
            b0 = g.blocks[0]
            if cat == "temporal":
                b0.temporal_op = op_name          # 时序类 → 时序槽
            elif cat == "spatial":
                b0.spatial_op = op_name           # 空间类 → 空间槽
            else:                                  # spatiotemporal / any → joint block
                b0.joint_op = op_name             # 一条边管时空 (取代 S+T)
        else:
            if cat == "temporal":
                g = Genotype(blocks=[STBlock(spatial_op="gcn", temporal_op=op_name)], hidden=64)
            elif cat == "spatial":
                g = Genotype(blocks=[STBlock(spatial_op=op_name, temporal_op="tcn")], hidden=64)
            else:
                g = Genotype(blocks=[STBlock(spatial_op="identity", temporal_op="identity",
                                             joint_op=op_name)], hidden=64)
        g.validate()
        g._seed_meta = {"warm_start_hps": best_hps or None,
                        "hpo_trials": self.cfg.seed_hpo_trials,
                        "is_creation_seed": True,
                        "operator_name": op_name}   # B1: orchestrator 据此回写履历本
        return g

    def _record_to_archive(self, req: FusionRequest, r, round_idx: int,
                           reg_name: str | None = None, seed_signature: str | None = None,
                           proxy_mae: float | None = None,
                           proxy_error: str | None = None) -> None:
        """把一条 SynthesisResult 落成 CreationRecord 追加进履历本 (B1)。

        成功: gate="all" + 代码 + plan 字段 + 注册名 + seed 签名 (orchestrator 回填识别用)。
        失败: gate=失败门 (从 last_error 解析, 解析不出记 unknown) + error + plan(若有)。
        档案 IO 失败不中止创造闭环 (履历是增益, 非必需)。
        """
        from datetime import datetime, timezone

        from darwin_st.creation.creation_archive import CreationRecord

        plan = r.plan
        if r.success and r.operator is not None:
            gate, error, code = "all", None, r.operator.code
            op_name = reg_name or r.operator.name
        else:
            gate = parse_failure_gate(r.last_error)
            error = r.last_error or None
            code = ""
            op_name = plan.operator_name if plan else ""
        rec = CreationRecord(
            round_idx=round_idx, created_at=datetime.now(timezone.utc).isoformat(),
            bottleneck=req.bottleneck, preconditions=list(req.target_preconditions),
            mechanisms=[m.name for m in req.mechanisms],
            operator_name=op_name,
            composition=plan.composition if plan else "",
            rationale=plan.rationale if plan else "",
            code=code, gate=gate, error=error, seed_signature=seed_signature,
            proxy_mae=proxy_mae,          # B3: proxy 粗筛分 (未走 proxy 路径为 None)
            proxy_error=proxy_error)      # B3: proxy 失败原因 (异常摘要/"inf"; 成功为 None)
        try:
            self.archive.record(rec)
        except Exception:
            pass                     # 档案写失败不中止创造闭环

    def _record_insight(self, text: str, dataset: str, mechs: list[str], success: bool) -> None:
        """写一条 memory insight。【当前无人调用 —— 保留给后续反思环节 (reflection) 使用】

        流水账双写已停: 每轮创造的成败记录是 B1 履历本 (creation_archive, 含全量结构化
        字段且被 prompt 回读) 的子集, insights 表又无人读取, 故 maybe_create/maybe_refine
        不再调用本方法。insights 表今后只由反思环节写入 (经分析的洞察, 非轮次流水)。
        """
        if self.memory is None:
            return
        from datetime import datetime, timezone
        self.memory.add_insight(
            insight_text=text,
            created_at=datetime.now(timezone.utc).isoformat(),
            dataset=dataset,
            condition=f"fusion:{'+'.join(mechs)}",
            confidence=0.6 if success else 0.3,
        )

    # ------------------------------------------------------------------
    # B4 方向状态持久化 (OPRO 轨迹 + 温度, 重启回放不归零)
    # ------------------------------------------------------------------

    def _persist_direction_state(self) -> None:
        """把完整 _history + _temperature 原子落盘 (同目录临时文件 + os.replace, 防半文件)。

        时机: maybe_create 每轮 append 轨迹后 / update_direction_outcome 改分调温后
        (含轮末升温补落)。与 B1 履历本同目录同生命周期 (scope 隔离自然继承, 不跨 scope 泄漏)。
        落盘失败不中止创造闭环 (持久化是增益, 非必需)。
        """
        if self._state_path is None:
            return
        try:
            payload = {"temperature": self._temperature, "history": self._history}
            tmp = self._state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp, self._state_path)
        except Exception:
            pass                     # 状态落盘失败不中止闭环

    def _restore_direction_state(self) -> None:
        """启动回放 direction_state.json (容错: 坏文件/缺字段 → 空启动, 不崩)。

        回放 _history 与 _temperature (clamp [0.5, 1.0] 与构造一致); 轮次计数续到
        轨迹最大 round + 1, 防重启后 _round_idx 归零与旧轮次撞号 (回授找不到会静默丢分)。
        """
        if self._state_path is None or not os.path.exists(self._state_path):
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history")
            if isinstance(history, list):
                self._history = [h for h in history if isinstance(h, dict)]
                rounds = [h["round"] for h in self._history if isinstance(h.get("round"), int)]
                if rounds:
                    self._round_idx = max(rounds) + 1
            temp = data.get("temperature")
            if isinstance(temp, (int, float)) and not isinstance(temp, bool) \
                    and math.isfinite(temp):
                self._temperature = min(1.0, max(0.5, float(temp)))
        except Exception:
            self._history = []       # 坏文件 = 空启动 (温度/轮次保持构造值)
