"""内层超参优化 (Inner HPO) —— 每架构独立的 Optuna TPE + ASHA 调参。

实现 docs/P2_ALGORITHM_DESIGN.md §4 的两层分工内层: 外层进化拥有架构,
本模块对**单个固定架构**调其超参 (lr/dropout/...), 用 Optuna。

源码级要点 (已核实, 见 P2_ALGORITHM_DESIGN.md §4):
  - **架构不是 Optuna 参数**: 每个架构开独立 study (search space 干净, 剪枝同架构可比)。
  - Sampler: TPESampler(multivariate=True, group=True, constant_liar=True)
    · multivariate+group → 正确处理条件超参 (num_heads 仅注意力架构存在)
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

__all__ = ["HPOConfig", "suggest_hps", "make_study", "optimize_architecture", "HPOResult"]

# Optuna 日志默认很吵, 调到 WARNING
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HPOConfig:
    """HPO 配置 (每架构一次调参的预算与搜索空间边界)。"""

    n_trials: int = 12               # 每架构试多少组超参
    max_epochs: int = 27             # 单 trial 最大 epoch (= ASHA max_resource)
    min_resource: int = 3            # ASHA 首 rung epoch (设在 warmup 之后)
    reduction_factor: int = 3        # ASHA eta (3 比默认 4 温和, 适合噪声 MAE)
    n_startup_trials: int = 8        # TPE 前 n 个随机 (小预算下调低)
    seed: int = 0
    # 搜索空间边界
    lr_range: tuple[float, float] = (1e-4, 1e-2)
    wd_range: tuple[float, float] = (1e-6, 1e-3)
    dropout_range: tuple[float, float] = (0.0, 0.5)
    batch_choices: tuple[int, ...] = (16, 32, 64)


def suggest_hps(trial: optuna.Trial, genotype: Genotype, cfg: HPOConfig) -> dict:
    """define-by-run 定义该架构的超参搜索空间, 返回采样到的 hps。

    条件超参: num_heads 仅当架构含注意力算子 (gat/attn) 时才存在 ——
    配合 TPESampler(group=True) 正确建模。
    """
    hps = {
        "lr": trial.suggest_float("lr", *cfg.lr_range, log=True),          # log-uniform
        "weight_decay": trial.suggest_float("weight_decay", *cfg.wd_range, log=True),
        "dropout": trial.suggest_float("dropout", *cfg.dropout_range),
        "batch_size": trial.suggest_categorical("batch_size", list(cfg.batch_choices)),
        # lr schedule 种类作为可择优超参 (扩搜索空间, 让 HPO 自己选用不用、用哪种)
        "lr_schedule": trial.suggest_categorical("lr_schedule", ["none", "cosine", "plateau"]),
    }
    # 条件超参: 架构用到注意力才调头数
    uses_attn = any(b.spatial_op == "gat" or b.temporal_op == "attn" for b in genotype.blocks)
    if uses_attn:
        hps["num_heads"] = trial.suggest_categorical("num_heads", [1, 2, 4, 8])
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
    else:
        best_mae, best_hps = float("inf"), {}

    all_trials = [
        {"params": dict(t.params), "value": t.value, "state": t.state.name}
        for t in study.trials
    ]
    return HPOResult(
        best_mae=best_mae, best_hps=best_hps,
        n_complete=n_complete, n_pruned=n_pruned, n_failed=n_failed,
        all_trials=all_trials,
    )
