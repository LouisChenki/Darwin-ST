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
    seed: int = 0


@dataclass
class CreationOutcome:
    """一次创造尝试的结果。"""

    success: bool
    operator_name: str | None = None      # 注入的算子注册名
    seed_genotype: Genotype | None = None # 用新算子产出的待评 genotype
    bottleneck: str = ""
    retrieved_mechanisms: list[str] = field(default_factory=list)
    reason: str = ""                      # 失败原因 / 成功说明
    insight: str = ""                     # 写入 memory 的融合经验


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
                 registry: OperatorRegistry, memory=None, config: CreationConfig | None = None):
        self.store = store
        self.embedder = embedder
        self.synthesizer = synthesizer
        self.registry = registry
        self.memory = memory
        self.cfg = config or CreationConfig()

    def maybe_create(self, best_genotype: Genotype | None, sota_gap: float | None = None,
                     run_tag: str = "exp/auto", dataset: str = "PeMS04") -> CreationOutcome:
        """尝试一次跨域创造。返回 CreationOutcome(含待评 genotype)。"""
        # 1) 诊断瓶颈
        bottleneck = diagnose_bottleneck(best_genotype, sota_gap)

        # 2) 跨域检索互补机制集
        res = find_cross_domain_analogy(
            bottleneck, self.store, self.embedder,
            target_domain=self.cfg.target_domain, cross_domain_only=True,
            max_results=self.cfg.max_mechanisms)
        mech_names = [m.name for m in res.mechanisms]
        if not res.mechanisms:
            return CreationOutcome(False, bottleneck=bottleneck, reason="跨域检索无结果")

        # 3) 合成 + 验证 (synthesizer 内部走 LLM/Aider + 验证 harness)
        req = FusionRequest(
            bottleneck=bottleneck, target_preconditions=res.target_preconditions,
            mechanisms=res.mechanisms,
            baseline_operator=(best_genotype.blocks[0].spatial_op if best_genotype else ""))
        synth_res = self.synthesizer.synthesize(req, needs_adj=False)

        if not synth_res.success:
            insight = f"融合 {mech_names} 解决'{bottleneck}'未通过合成/验证: {synth_res.last_error[:120]}"
            self._record_insight(insight, dataset, mech_names, success=False)
            return CreationOutcome(False, bottleneck=bottleneck,
                                   retrieved_mechanisms=mech_names,
                                   reason=synth_res.last_error[:200], insight=insight)

        # 4) 注入算子库
        reg_name = self.registry.register(synth_res.operator)

        # 5) 产出用新算子的 genotype 交给进化 (以最优为基, 把第一块空间算子换成新算子)
        seed_geno = self._make_seed_genotype(best_genotype, reg_name)

        insight = (f"成功融合 {mech_names} → 新算子 {reg_name}(组合方式 "
                   f"{synth_res.plan.composition})解决'{bottleneck}'; 已注入待评测")
        self._record_insight(insight, dataset, mech_names, success=True)

        return CreationOutcome(
            True, operator_name=reg_name, seed_genotype=seed_geno,
            bottleneck=bottleneck, retrieved_mechanisms=mech_names,
            reason="合成并注入成功", insight=insight)

    def _make_seed_genotype(self, best: Genotype | None, op_name: str) -> Genotype:
        """用新算子产出一个待评 genotype。以最优为基(若有), 否则新建。"""
        if best is not None:
            g = best.copy()
            g.blocks[0].spatial_op = op_name   # 把第一块空间算子换成新合成算子
        else:
            g = Genotype(blocks=[STBlock(spatial_op=op_name, temporal_op="tcn")], hidden=64)
        g.validate()
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
