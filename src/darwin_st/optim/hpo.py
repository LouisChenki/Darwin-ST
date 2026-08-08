"""内层超参优化 (Inner HPO) —— 每架构独立的 Optuna TPE + ASHA 调参。

实现 docs/P2_ALGORITHM_DESIGN.md §4 的两层分工内层: 外层进化拥有架构,
本模块对**单个固定架构**调其超参 (lr/dropout/...), 用 Optuna。

源码级要点 (已核实, 见 P2_ALGORITHM_DESIGN.md §4):
  - **架构不是 Optuna 参数**: 每个架构开独立 study (search space 干净, 剪枝同架构可比)。
  - Sampler: TPESampler(multivariate=True, group=True, constant_liar=True)
    · multivariate+group → 正确处理条件超参 (num_heads 仅头数可配的注意力架构存在)
    · constant_liar → 多 worker 并行防扎堆 (4 卡必须)
  - Pruner: SuccessiveHalvingPruner —— **它本身就是 ASHA** (异步, 无同步障碍);
    4 卡用它不用 HyperbandPruner (单 bracket, TPE 约 8 个完成 trial 即启动)。
  - fidelity = 训练 epoch; min_resource 设在 warmup 之后避开早期噪声; reduction_factor=3。
  - 接口 ask/tell; trial.report(mae, epoch) + should_prune(); 暖启动 enqueue_trial。
  - 僵尸防护: try/except + tell(state=FAIL)。
  - TPE 不学被剪枝 trial; 若饿了, 剪枝时 return 平滑 MAE 估计 (此处交训练函数决定)。

训练函数注入: optimize_architecture 接收 train_eval_fn(genotype, hps, trial) → final_mae,
故本模块**不依赖 GPU/真实训练**, 可用 mock 完整测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

from darwin_st.search.genotype import Genotype
from darwin_st.search.operators import accepts_num_heads

__all__ = ["HPOConfig", "suggest_hps", "make_study", "optimize_architecture", "HPOResult"]

# Optuna 日志默认很吵, 调到 WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOConfig:
    """HPO 配置 (每架构一次调参的预算与搜索空间边界)。"""

    n_trials: int = 12               # 每架构试多少组超参
    max_epochs: int = 27             # 单 trial 最大 epoch (= ASHA max_resource, 也是收敛早停硬上限)
    min_resource: int = 3            # ASHA 首 rung epoch (设在 warmup 之后)
    reduction_factor: int = 3        # ASHA eta (3 比默认 4 温和, 适合噪声 MAE)
    n_startup_trials: int = 8        # TPE 前 n 个随机 (小预算下调低)
    seed: int = 0
    # 收敛早停 (per-run/per-dataset 自适应训练预算): >0 才启用, 0=关(跑满 max_epochs)。
    early_stop_patience: int = 0     # val 连续几 epoch 无实质改进 → 停 (best 已存, 零损失)
    early_stop_min_delta: float = 0.001  # 相对改进阈值 (0.1%), scale-free 跨数据集
    # 搜索空间边界
    lr_range: tuple[float, float] = (1e-4, 1e-2)
    wd_range: tuple[float, float] = (1e-6, 1e-3)
    dropout_range: tuple[float, float] = (0.0, 0.5)
    batch_choices: tuple[int, ...] = (16, 32, 64)


def suggest_hps(trial: optuna.Trial, genotype: Genotype, cfg: HPOConfig) -> dict:
    """define-by-run 定义该架构的超参搜索空间, 返回采样到的 hps。

    无条件维度: lr/weight_decay/dropout/batch_size/lr_schedule/loss (损失类型也交给 HPO
    按架构择优: mae=归一化尺度, huber=真实尺度, 对齐 SOTA 训练实践)。
    条件超参: num_heads 仅当架构含【头数可配】的注意力算子时才存在 ——
    attn / st_graph_attn / series_decomp_attn (accepts_num_heads 按构造函数签名判定;
    gat/dynamic_gat 头数写死不可配, 不采 —— 采了也是没人消费的死参数)。
    aux_lambda 仅当 genotype.aux_op 非 None (挂了辅助任务) 时存在 (B7, 同模式)。
    配合 TPESampler(group=True) 正确建模。
    """
    hps = {
        "lr": trial.suggest_float("lr", *cfg.lr_range, log=True),          # log-uniform
        "weight_decay": trial.suggest_float("weight_decay", *cfg.wd_range, log=True),
        "dropout": trial.suggest_float("dropout", *cfg.dropout_range),
        "batch_size": trial.suggest_categorical("batch_size", list(cfg.batch_choices)),
        # lr schedule 种类作为可择优超参 (扩搜索空间, 让 HPO 自己选用不用、用哪种)
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["none", "cosine", "plateau"]),
        # 训练损失种类作为可择优超参: mae=归一化尺度 masked MAE (原默认), huber=真实尺度 masked
        # Huber (SOTA 实践, 见 train_one)。消费方 hps.get("loss", "mae") 兜底, 旧暖启动字典无此键不受影响
        "loss": trial.suggest_categorical("loss", ["mae", "huber"]),
    }
    # 条件超参: 任一块的空间/时序/一体槽坐了头数可配的注意力算子才调头数
    uses_attn = any(
        accepts_num_heads(op)
        for b in genotype.blocks
        for op in (b.spatial_op, b.temporal_op, b.joint_op)
        if op is not None
    )
    if uses_attn:
        hps["num_heads"] = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
    # 条件超参 (B7): 架构挂了辅助任务 (genotype.aux_op 非 None) 才采辅助损失权重
    # aux_lambda —— log-uniform [1e-3, 1e-1] (STD-MAE 量级); 无 aux 不采 (采了也是
    # 没人消费的死参数, 同 num_heads 的条件维度模式; 消费方 train_one hps.get("aux_lambda", 0.1))。
    if genotype.aux_op is not None:
        hps["aux_lambda"] = trial.suggest_float("aux_lambda", 1e-3, 1e-1, log=True)
    return hps


def make_study(study_name: str, cfg: HPOConfig, storage=None) -> optuna.Study:
    """为单个架构创建 study (TPE + ASHA)。storage=None → 内存 study (串行/测试)。"""
    sampler = TPESampler(
        multivariate=True,       # 建模超参交互
        group=True,              # 正确处理条件超参 (num_heads)
        constant_liar=True,      # 并行 worker 防扎堆
        n_startup_trials=cfg.n_startup_trials,
        seed=cfg.seed,
    )
    pruner = SuccessiveHalvingPruner(   # = ASHA (异步多保真)
        min_resource=cfg.min_resource,
        reduction_factor=cfg.reduction_factor,
        min_early_stopping_rate=0,
    )
    return optuna.create_study(
        study_name=study_name,
        direction="minimize",           # val MAE 越小越优
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )


@dataclass
class HPOResult:
    """一个架构调参的结果。"""

    best_mae: float
    best_hps: dict
    n_complete: int
    n_pruned: int
    n_failed: int
    all_trials: list[dict] = field(default_factory=list)
    best_trace: dict | None = None    # 最优 trial 的 TrainTrace (asdict), 供 Tier-2 LLM 诊断


# 训练函数签名: (genotype, hps, trial) -> final_mae
#   实现方在内部每 epoch 调 trial.report(mae, epoch) + 检查 trial.should_prune()
#   (抛 optuna.TrialPruned 提前停)。返回最终 val MAE。
TrainEvalFn = Callable[[Genotype, dict, optuna.Trial], float]


def optimize_architecture(
    genotype: Genotype,
    train_eval_fn: TrainEvalFn,
    cfg: HPOConfig | None = None,
    study_name: str | None = None,
    storage=None,
    warm_start_hps: dict | None = None,
    n_trials: int | None = None,
) -> HPOResult:
    """对单个架构跑 HPO, 返回最优超参与 MAE。

    train_eval_fn(genotype, hps, trial) → final_mae: 内部应每 epoch report + should_prune。
    warm_start_hps: 父架构的最优超参, 作为第一个 trial 暖启动 (省冷启动)。
    """
    cfg = cfg or HPOConfig()
    study_name = study_name or f"arch_{genotype.signature()[:10]}"
    study = make_study(study_name, cfg, storage=storage)

    if warm_start_hps:
        # 只 enqueue 与本架构搜索空间相容的键 (避免无效参数)
        study.enqueue_trial(warm_start_hps, skip_if_exists=True)

    budget = n_trials if n_trials is not None else cfg.n_trials
    for _ in range(budget):
        trial = study.ask()
        try:
            hps = suggest_hps(trial, genotype, cfg)
            mae = train_eval_fn(genotype, hps, trial)
            study.tell(trial, mae)
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        except Exception:
            study.tell(trial, state=TrialState.FAIL)   # 僵尸防护: 不留 RUNNING

    return _summarize(study)


def _summarize(study: optuna.Study) -> HPOResult:
    states = [t.state for t in study.trials]
    n_complete = states.count(TrialState.COMPLETE)
    n_pruned = states.count(TrialState.PRUNED)
    n_failed = states.count(TrialState.FAIL)

    completed = [t for t in study.trials if t.state == TrialState.COMPLETE and t.value is not None]
    if completed:
        best = min(completed, key=lambda t: t.value)
        best_mae, best_hps = float(best.value), dict(best.params)
        best_trace = best.user_attrs.get("trace")   # 最优 trial 的训练轨迹 (train_eval_fn 设的)
    else:
        best_mae, best_hps, best_trace = float("inf"), {}, None

    all_trials = [
        {"params": dict(t.params), "value": t.value, "state": t.state.name}
        for t in study.trials
    ]
    return HPOResult(
        best_mae=best_mae, best_hps=best_hps,
        n_complete=n_complete, n_pruned=n_pruned, n_failed=n_failed,
        all_trials=all_trials, best_trace=best_trace,
    )
