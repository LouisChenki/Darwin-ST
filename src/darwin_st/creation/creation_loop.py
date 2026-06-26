"""创造循环 (Creation Loop) —— Tier-2 接入优化, 完成闭环。

把跨域知识 + 算子合成 + 注入接成一个可被 orchestrator 在【停滞时】触发的步骤:
  停滞 → 诊断瓶颈 → 跨域检索互补机制集 → LLM 融合 + Aider 写算子 → 验证 →
  注入算子库 → 产出一个用新算子的 genotype 交给进化评估 → 记录融合经验到 memory。

设计 (契合两层自治, 用户拍板的职责分离):
  - 创造是 Tier-2 低频(仅停滞触发), Tier-1 进化高频。
  - 合成算子进【算子库+archive】(给进化), 不进机制库(机制库静态只读)。
  - 融合经验(work/不work) → memory insights, 不混进机制卡。
  - 所有外部依赖(检索 store/embedder、synthesizer、registry、memory)注入 → 无 LLM/GPU 可 mock 全测。

诚实: 真实 MAE 评测合成算子由 orchestrator 的既有 eval_fn 完成(本模块只负责"造+注入+
产出待评 genotype")。评测器是最终裁判, 低质融合自然被进化淘汰。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin_st.creation.contracts import FusionRequest
from darwin_st.creation.synthesizer import OperatorSynthesizer
from darwin_st.creation.registry import OperatorRegistry
from darwin_st.knowledge.retrieval import find_cross_domain_analogy
from darwin_st.search.genotype import Genotype, STBlock

__all__ = ["CreationConfig", "CreationOutcome", "CreationLoop", "diagnose_bottleneck"]


@dataclass
class CreationConfig:
    target_domain: str = "ST"
    max_mechanisms: int = 3        # 跨域互补集大小
    n_hypotheses: int = 4          # 一次生成多少个融合假设 (单 Generator 生 N 个)
    seed: int = 0
    use_llm_diagnosis: bool = True  # 优先用 LLM 诊断瓶颈 (训练信号驱动多样化); 失败退回规则版
    seed_hpo_trials: int = 20      # 创造 seed 专项大 HPO 预算 (AlphaEvolve 式优胜者深评; 普通架构走 hpo_cfg)


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


class CreationLoop:
    """Tier-2 创造步骤。orchestrator 在停滞时调 maybe_create()。"""

    def __init__(self, store, embedder, synthesizer: OperatorSynthesizer,
                 registry: OperatorRegistry, memory=None, config: CreationConfig | None = None,
                 llm=None):
        self.store = store
        self.embedder = embedder
        self.synthesizer = synthesizer
        self.registry = registry
        self.memory = memory
        self.cfg = config or CreationConfig()
        self.llm = llm                       # OpenAICompatLLM (注入); None 时 _diagnose 退规则版
        self._round_idx = 0                  # 第几次创造 (OPRO 轨迹用)
        self._history: list[dict] = []       # 历轮诊断: [{preconditions, bottleneck, result_mae}]

    def _diagnose(self, best_genotype: Genotype | None, sota_gap: float | None,
                  best_trace: dict | None) -> tuple[str, list[str] | None]:
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
                diag = diagnose_bottleneck_llm(summary, self._history, self._round_idx, self.llm)
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
        # 1) 诊断瓶颈 (LLM 优先, 规则兜底)
        bottleneck, override_precs = self._diagnose(best_genotype, sota_gap, best_trace)
        # OPRO 轨迹: 记本轮诊断方向 + 当前水位 (result_mae 用 best_trace 的 best_mae, 缺则用 gap 推)
        cur_mae = None
        if best_trace is not None and best_trace.get("best_mae") not in (None, float("inf")):
            cur_mae = best_trace.get("best_mae")
        self._history.append({"round": self._round_idx, "bottleneck": bottleneck,
                              "preconditions": override_precs or [], "result_mae": cur_mae})
        self._round_idx += 1

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
        results = self.synthesizer.synthesize_many(req, n_hypotheses=self.cfg.n_hypotheses,
                                                   needs_adj=False)
        successes = [r for r in results if r.success and r.operator is not None]

        if not successes:
            err = results[0].last_error if results else "无结果"
            insight = f"融合 {mech_names} 解决'{bottleneck}': {len(results)} 个假设均未过验证"
            self._record_insight(insight, dataset, mech_names, success=False)
            return CreationOutcome(False, bottleneck=bottleneck, retrieved_mechanisms=mech_names,
                                   n_hypotheses=len(results), n_success=0,
                                   reason=err[:200], insight=insight)

        # 4) 注入所有成功算子 + 各产出一个待评 genotype
        op_names, seeds = [], []
        for r in successes:
            reg_name = self.registry.register(r.operator)
            op_names.append(reg_name)
            seeds.append(self._make_seed_genotype(best_genotype, reg_name, best_hps))

        compositions = [r.plan.composition for r in successes]
        insight = (f"融合 {mech_names} 解决'{bottleneck}': {len(results)} 假设中 "
                   f"{len(successes)} 个成功 → 注入 {op_names}(组合 {compositions}), 待评测")
        self._record_insight(insight, dataset, mech_names, success=True)

        return CreationOutcome(
            True, operator_names=op_names, seed_genotypes=seeds,
            bottleneck=bottleneck, retrieved_mechanisms=mech_names,
            n_hypotheses=len(results), n_success=len(successes),
            reason="合成并注入成功", insight=insight)

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
                        "is_creation_seed": True}
        return g

    def _record_insight(self, text: str, dataset: str, mechs: list[str], success: bool) -> None:
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
