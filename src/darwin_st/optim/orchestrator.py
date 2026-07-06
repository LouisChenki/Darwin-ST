"""自治编排 (Autonomous Orchestrator) —— P2 总装 + 两层自治循环。

把 P2 全部零件接成一个端到端、永不暂停的自主优化循环 (docs/P2_ALGORITHM_DESIGN.md §6):

  evolution 提议架构 (Tier-1 规则变异)
    → graveyard 硬否决 (已知失败/已见 跳过, 防无效优化)
    → scheduler 并行派发到多卡 (一卡一架构)
    → 每架构内层 HPO 调参 + 训练 + masked 评测 (由注入的 eval_fn 完成)
    → 落 memory (KEEP/DISCARD/CRASH) + 进 archive (MAP-Elites) + 回灌 evolution
    → **程序化判定是否超越 SOTA** (best_so_far vs baseline_registry)
    → 没超越就【无条件继续】下一轮 (永不停下来问人)

自主性铁律 (研究强调, 这是"防礼貌性暂停"的根本):
  - 循环驱动在【本编排器代码】里, 不依赖 LLM "记得别停"。
  - 停止判定是【代码比较】(best < SOTA), 绝非 LLM 口头判断。
  - 失败 (crash/nan) 是一个数据点 → 记 graveyard → 立即继续, 绝不中止循环。

eval_fn 注入: evaluate_architecture(genotype, device) → EvalResult。
  真实运行时它内部做 builder+hpo+训练+评测; 测试时可 mock。故本模块无需 GPU 可全测。

停止条件: 超越 SOTA, 或达 max_rounds/max_evals/预算 (任一)。无上限时为真 24/7。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from darwin_st.baseline_registry import get_sota
from darwin_st.optim.archive import MAPElitesArchive
from darwin_st.optim.scheduler import EvalResult, GPUScheduler
from darwin_st.search.evolution import AgingEvolution
from darwin_st.search.genotype import Genotype

__all__ = ["OrchestratorConfig", "RunState", "Orchestrator"]


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def _now_iso() -> str:
    """当前 UTC 时间 ISO8601 (memory 时间戳)。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@dataclass
class OrchestratorConfig:
    dataset: str = "PeMS04"
    run_tag: str = "exp/auto"
    space_version: str | None = None      # 搜索空间代际 (作用域隔离键); None=自动哈希派生
    population_size: int = 20
    tournament_size: int = 5
    seed: int = 0
    # 停止条件 (任一触发即停; None = 无此上限)
    max_rounds: int | None = None        # 最多多少轮 (每轮一批并行评估)
    max_evals: int | None = None         # 最多评估多少架构
    target_mae: float | None = None      # 自定目标 MAE (默认用 baseline_registry 的 SOTA)
    # KEEP/DISCARD 判定 (短训练下"劣于已发表基线"过严, 会清空 archive):
    #   - 冷启动前 warmup_keep 个有效评估一律 KEEP, 让 archive/进化先积累信号。
    #   - 之后: 相对当前 best 退化超过 discard_regression_factor 倍才 DISCARD (相对, 非绝对基线)。
    #   - discard_above 给定时用绝对阈值 (正式长训练跑可设为最差基线)。
    discard_above: float | None = None    # 绝对阈值; None = 用相对/warmup 策略
    warmup_keep: int = 16                 # 冷启动期一律 KEEP 的有效评估数
    discard_regression_factor: float = 1.5  # MAE > best*此倍数 → DISCARD
    # 创造节奏 (修两阶段衔接: 创造后给 NAS+HPO 充分发挥新算子的窗口, 别立刻再创造):
    #   创造成功后 creation_refine_rounds 轮内不再触发创造 (NAS 围绕新算子换骨架/精修), 也不 island_reset。
    creation_refine_rounds: int = 4       # 创造后精修窗口轮数 (冷却期)
    # 距离自适应创造调度 (Gap-Annealed Stagnation Patience): patience 随距 SOTA 远近自适应。
    #   动机=边际收益递减: 逼近 SOTA 时每点提升边际成本陡增, 应给精修更多耐心别过早切换探索。
    #   patience(gap) = min(cap, base + k/(gap+eps)), gap=best_mae-sota_mae。远→小(多探索), 近→大(重精修)。
    adaptive_patience: bool = False       # True 启用距离自适应; False 用固定 stagnation_patience
    patience_base: int = 3                # 远处基线耐心
    patience_k: float = 5.0               # 反比强度
    patience_cap: int = 12                # 上限 (防 gap→0 时发散)
    patience_eps: float = 0.5             # 分母平滑


@dataclass
class RunState:
    """循环运行状态 (供监控/停止判定/测试断言)。"""

    rounds: int = 0
    evals: int = 0
    n_keep: int = 0
    n_discard: int = 0
    n_crash: int = 0
    best_mae: float = float("inf")
    best_genotype: Genotype | None = None
    best_trace: dict | None = None       # 最优架构的训练动态轨迹 (供 Tier-2 LLM 瓶颈诊断)
    best_hps: dict = field(default_factory=dict)  # 最优架构的最优超参 (供创造 seed warm-start)
    beat_sota: bool = False
    sota_name: str | None = None
    sota_mae: float | None = None
    stop_reason: str | None = None
    history: list[dict] = field(default_factory=list)


# eval_fn 签名: (genotype, device) -> EvalResult (mae/rmse/hps/status...)
EvalFn = Callable[[Genotype, str], EvalResult]


class Orchestrator:
    """两层自治优化编排器。

    用法:
        orch = Orchestrator(cfg, eval_fn, devices=4, memory=mem)
        state = orch.run()       # 跑到满足停止条件 (或无上限则真 24/7)
    """

    def __init__(
        self,
        config: OrchestratorConfig,
        eval_fn: EvalFn | None = None,
        devices: list[str] | int | None = None,
        memory=None,                       # MemoryStore: 落库 + graveyard
        base_genotype: Genotype | None = None,
        protected: list[str] | None = None,
        on_round: Callable[[RunState], None] | None = None,
        creation_loop=None,                # Tier-2 CreationLoop: 停滞时触发跨域创造 (可选)
        eval_spec=None,                    # EvalSpec: 进程后端 worker 自建 eval_fn (避闭包pickle)
        backend: str = "auto",             # "auto"/"thread"/"process"; auto下无eval_spec→线程
    ):
        self.cfg = config
        self.memory = memory
        self.on_round = on_round
        self.creation_loop = creation_loop
        self._pending_seed_genotypes: list = []   # 创造产出的待评 genotype 队列
        self._creation_cooldown_until = 0         # 创造后精修窗口: rounds 到此前不再触发创造
        # 流式驱动状态: refresh 屏障标志 + 完成计数 (每 num_devices 次完成 = 1 轮)
        self._pending_refresh = False
        self._completions_since_round = 0

        # 实验作用域 (数据集 + 搜索空间代际): trial 落库带 space_version, evolution 按数据集查重。
        from darwin_st.scope import ExperimentScope
        self.scope = ExperimentScope.resolve(config.dataset, config.space_version)

        self.evo = AgingEvolution(
            population_size=config.population_size,
            tournament_size=config.tournament_size,
            seed=config.seed,
            protected=protected,
            memory=memory,
            base_genotype=base_genotype,
            dataset=config.dataset,
        )
        self.archive = MAPElitesArchive(seed=config.seed)
        self.scheduler = GPUScheduler(
            eval_fn, devices=devices, eval_spec=eval_spec, backend=backend)
        self.state = RunState()

        # SOTA 锚点 (程序化停止依据)
        name, mae = get_sota(config.dataset)
        self.state.sota_name, self.state.sota_mae = name, mae
        self._target = config.target_mae if config.target_mae is not None else mae

        # DISCARD 策略: 绝对阈值(若给定) 否则 warmup + 相对退化 (见 OrchestratorConfig)
        self._discard_above = config.discard_above
        self._n_valid = 0   # 已见的有限 MAE 评估数 (warmup 计数)

    def _classify(self, mae: float) -> tuple[str, str | None]:
        """定结局: CRASH(无效) / KEEP / DISCARD。返回 (status, fail_reason)。"""
        if not _finite(mae):
            return "CRASH", "nan_or_inf"
        # 绝对阈值优先 (正式长训练跑可设为最差基线)
        if self._discard_above is not None:
            if mae > self._discard_above:
                return "DISCARD", "worse_than_threshold"
            return "KEEP", None
        # 相对策略: 冷启动 warmup 期一律 KEEP, 让 archive/进化先积累
        if self._n_valid < self.cfg.warmup_keep:
            return "KEEP", None
        # warmup 后: 相对当前 best 退化超阈即 DISCARD
        if self.state.best_mae < float("inf") and mae > self.state.best_mae * self.cfg.discard_regression_factor:
            return "DISCARD", "regression_vs_best"
        return "KEEP", None

    # -- 停止判定 (纯代码, 非 LLM) --
    def _adaptive_patience(self) -> int:
        """距离自适应停滞耐心 (Gap-Annealed Stagnation Patience):
        patience(gap)=min(cap, base + k/(gap+eps)), gap=best_mae-sota_mae。

        远离 SOTA (gap 大) → 耐心小 → 快触发创造找新机制 (多探索);
        逼近 SOTA (gap 小) → 耐心大 → 给 NAS+HPO 充分精修别打断 (边际收益递减下每点提升都难得)。
        gap<=0 (已达/超 SOTA) 或 sota 未知 → 用 cap (最大耐心, 死磕精修)。
        """
        cfg = self.cfg
        if self.state.sota_mae is None or self.state.best_mae >= float("inf"):
            return cfg.patience_base
        gap = self.state.best_mae - self.state.sota_mae
        if gap <= 0:
            return cfg.patience_cap
        p = cfg.patience_base + cfg.patience_k / (gap + cfg.patience_eps)
        return int(min(cfg.patience_cap, round(p)))

    def should_stop(self) -> str | None:
        """返回停止原因字符串, 或 None (继续)。"""
        if self._target is not None and self.state.best_mae < self._target:
            return f"beat_sota(best={self.state.best_mae:.3f} < target={self._target})"
        if self.cfg.max_rounds is not None and self.state.rounds >= self.cfg.max_rounds:
            return f"max_rounds({self.cfg.max_rounds})"
        if self.cfg.max_evals is not None and self.state.evals >= self.cfg.max_evals:
            return f"max_evals({self.cfg.max_evals})"
        return None

    # -- 单轮: 提议一批 → 并行评估 → 消化结果 --
    def step(self) -> list[EvalResult]:
        """[legacy 批模式] 跑一轮: 取 (设备数) 个架构并行评估, 消化结果。返回本轮结果。

        run() 已改用流式驱动 (run_stream), 不再调 step()。保留供手动/兼容调用; 无测试直接依赖。
        """
        batch_size = self.scheduler.num_devices
        genotypes = []
        # 优先评估创造产出的待评 genotype (Tier-2 新算子), 余位由进化补
        while self._pending_seed_genotypes and len(genotypes) < batch_size:
            genotypes.append(self._pending_seed_genotypes.pop(0))
        while len(genotypes) < batch_size:
            genotypes.append(self.evo.ask())
        results = self.scheduler.run_batch(genotypes)
        for res in results:
            self._digest(res)
        self.state.rounds += 1
        if self.on_round is not None:
            self.on_round(self.state)
        return results

    # -- 流式驱动接线 (run_stream 回调): 4 卡持续满载, 破批同步 --

    def _next_genotype(self):
        """流式驱动取下一个待评 genotype。refresh 屏障待定时返 None (让在飞排空)。

        优先创造产出的待评 seed (Tier-2 新算子), 余由进化 ask 补 (同 step 的 211-217 语义)。
        """
        if self._pending_refresh:
            return None
        if self._pending_seed_genotypes:
            return self._pending_seed_genotypes.pop(0)
        return self.evo.ask()

    def _after_digest(self) -> None:
        """每完成一个评估后的轮次/停滞/创造 bookkeeping。

        每 num_devices 次完成 = 1 轮 (rounds == evals // num_devices, 与批模式恒等)。
        轮末做: 自适应耐心 + 停滞→创造/island_reset + on_round —— 逻辑与旧 run() 主体一致,
        保持停滞检测/创造/冷却全部按轮次粒度不变。
        """
        self._completions_since_round += 1
        if self._completions_since_round < self.scheduler.num_devices:
            return
        self._completions_since_round = 0
        self.state.rounds += 1
        # 停滞 → Tier-2 跨域创造(若接入)。创造后进精修窗口: N 轮内不再创造也不 island_reset。
        if self.cfg.adaptive_patience:
            self.archive.stagnation_patience = self._adaptive_patience()
        if self.archive.stagnated():
            if self.state.rounds >= self._creation_cooldown_until and self._try_creation():
                # 创造成功: 设冷却窗口, 不立即 island_reset (避免清掉刚注入算子的邻域)
                self._creation_cooldown_until = self.state.rounds + self.cfg.creation_refine_rounds
                self.archive._since_improve = 0   # 精修窗口干净计数
            elif self.state.rounds >= self._creation_cooldown_until:
                self.archive.island_reset()       # 没接创造 / 无产出 → 岛屿重置解停滞
        if self.on_round is not None:
            self.on_round(self.state)

    def _on_stream_result(self, res: EvalResult) -> None:
        """流式回调: 消化一个结果 + 轮次 bookkeeping。"""
        self._digest(res)
        self._after_digest()

    def _digest(self, res: EvalResult) -> None:
        """消化一个评估结果: 定状态 → 落 memory → 进 archive → 回灌 evolution → 更新最优。"""
        self.state.evals += 1
        geno = res.genotype

        # 1) 定结局: CRASH(评估崩) / DISCARD / KEEP
        if res.status == "CRASH":
            status, fail_reason = "CRASH", (res.fail_reason or "eval_crash")
        else:
            status, fail_reason = self._classify(res.mae)
            if _finite(res.mae):
                self._n_valid += 1   # warmup 计数 (仅有效 MAE)

        # 2) 落 memory (含 graveyard: CRASH/DISCARD 自动进, 防重复)
        trial_id = None
        if self.memory is not None:
            trial_id = self._record_memory(res, status, fail_reason)

        # 3) KEEP 才进 archive + 回灌 evolution 种群 (失败不污染种群/档案)
        if status == "KEEP":
            self.state.n_keep += 1
            num_params = int(res.extra.get("num_params", 0))
            self.archive.add(geno, fitness=res.mae, num_params=num_params, trial_id=trial_id)
            self.evo.tell(geno, fitness=res.mae, trial_id=trial_id)
            if res.mae < self.state.best_mae:
                self.state.best_mae = res.mae
                self.state.best_genotype = geno
                self.state.best_trace = res.extra.get("train_trace")  # 同步存最优架构训练轨迹
                self.state.best_hps = res.hps or {}                   # 最优超参 (供创造 seed warm-start)
                if self._target is not None and res.mae < self._target:
                    self.state.beat_sota = True
        elif status == "DISCARD":
            self.state.n_discard += 1
            # DISCARD 也回灌 evolution (aging 需要种群流动; 但 fitness 差自然被淘汰)
            self.evo.tell(geno, fitness=res.mae, trial_id=trial_id)
        else:  # CRASH
            self.state.n_crash += 1
            # crash 不回灌种群 (无有效 fitness), 但已进 graveyard 防重试

        self.state.history.append({
            "eval": self.state.evals, "status": status, "mae": res.mae,
            "best_mae": self.state.best_mae, "device": res.device,
        })

    def _record_memory(self, res: EvalResult, status: str, fail_reason: str | None) -> int | None:
        from darwin_st.memory.store import Trial

        trial = Trial(
            run_tag=self.cfg.run_tag, dataset=self.cfg.dataset,
            space_version=self.scope.space_version,
            genotype=res.genotype.to_dict(), hp=res.hps,
            status=status, created_at=_now_iso(),
            val_mae=res.mae if _finite(res.mae) else None,
            val_rmse=res.rmse if _finite(res.rmse) else None,
            fail_reason=fail_reason,
            num_params=int(res.extra.get("num_params", 0)) or None,
            wall_seconds=res.wall_seconds or None,
        )
        return self.memory.record_trial(trial)

    # -- 主循环 (永不暂停问人) --
    def run(self) -> RunState:
        """跑到满足停止条件 (流式驱动: 4 卡持续满载, 慢架构不阻塞快架构)。

        无 max_* 与可达 target → 真 24/7 (由外部中断)。停滞→创造/island_reset + 轮次 bookkeeping
        搬进 _after_digest (每 num_devices 次完成触发一次), 与旧批模式轮次粒度恒等。
        """
        try:
            self.scheduler.run_stream(
                next_genotype=self._next_genotype,
                on_result=self._on_stream_result,
                should_continue=lambda: self.should_stop() is None,
                pause_check=lambda: self._pending_refresh,
                on_resume=lambda: setattr(self, "_pending_refresh", False),
            )
            self.state.stop_reason = self.state.stop_reason or self.should_stop() or "stream_end"
        finally:
            self.scheduler.close()   # 进程后端清理持久池 (线程后端 no-op)
        return self.state

    def _try_creation(self) -> bool:
        """停滞时触发 Tier-2 跨域创造: 合成新算子 → 待评 genotype 入队。返回是否成功注入。

        失败/无创造层都不中止循环(自主性: 创造是低频增益, 非必需)。返回 True 表示有 seed 入队
        (调用方据此进精修窗口); False 表示无创造层/无产出/出错(调用方走 island_reset)。
        """
        if self.creation_loop is None:
            return False
        gap = None
        if self.state.sota_mae is not None and self.state.best_mae < float("inf"):
            gap = self.state.best_mae - self.state.sota_mae
        try:
            outcome = self.creation_loop.maybe_create(
                self.state.best_genotype, sota_gap=gap,
                run_tag=self.cfg.run_tag, dataset=self.cfg.dataset,
                best_trace=self.state.best_trace, best_hps=self.state.best_hps)
            if outcome.success and outcome.seed_genotypes:
                self._pending_seed_genotypes.extend(outcome.seed_genotypes)
                self.state.history.append({"event": "creation", "operators": outcome.operator_names,
                                           "bottleneck": outcome.bottleneck,
                                           "n_success": outcome.n_success})
                # 进程后端: 新 synth 算子已 persist → 需 refresh_workers 让新 worker load_persisted。
                # 流式下池常忙, 不能直接 shutdown (会 cancel 在飞丢结果) → 设标志, 由 run_stream 屏障
                # 排空在飞后再 refresh (线程后端 no-op; 主进程 SPATIAL_OPS 已含新算子)。
                self._pending_refresh = True
                return True
            return False
        except Exception as e:
            # 创造出错不中止优化
            self.state.history.append({"event": "creation_failed", "error": str(e)[:120]})
            return False
