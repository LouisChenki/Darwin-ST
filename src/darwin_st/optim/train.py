"""真实训练与评估 (Real Training & Evaluation) —— 自治循环的"点火"引擎。

把 builder(编译模型)+ hpo(Optuna 调参)+ 真实数据训练 + masked 评测 串成
orchestrator 需要的 eval_fn(genotype, device) → EvalResult。

单个架构的评估 = 一次内层 HPO study:
  对该架构采样多组超参 → 每组建模型、真实数据训练若干 epoch →
  每 epoch report val-MAE 给 ASHA 剪枝 → 取最优超参的 MAE 作为该架构得分。

训练纪律 (docs/P2_ALGORITHM_DESIGN.md):
  - masked MAE 训练 loss(归一化尺度)+ evaluate 真实尺度 masked 指标
  - **NaN 熔断**: loss 出现 nan/inf 立即停该 trial(记崩, 不空跑)
  - **时间熔断**: 单 trial 超时间预算即停(防重算子吃满算力)
  - 梯度裁剪(防时空图卷积梯度爆炸)
  - best-checkpoint: 取训练过程中 val-MAE 最优的那一步(防过拟合/震荡)

make_eval_fn(...) 工厂返回一个绑定了数据规格的 eval_fn, 注入给 Orchestrator。
"""

from __future__ import annotations

import os
import time

import numpy as np
import optuna
import torch

from darwin_st.data import prepare as P
from darwin_st.data.metrics import masked_mae
from darwin_st.data.protocol import DatasetProfile, get_profile
from darwin_st.optim.hpo import HPOConfig, optimize_architecture
from darwin_st.optim.scheduler import EvalResult
from darwin_st.search.builder import build_model, count_params
from darwin_st.search.genotype import Genotype

__all__ = ["train_one", "evaluate_architecture", "make_eval_fn", "limit_cpu_threads"]


def limit_cpu_threads(n_workers: int) -> None:
    """限制 PyTorch CPU intra-op 线程数, 防多 worker 并发训练时线程过订阅 (oversubscription)。

    根因 (标定跑卡死实测): 多卡机 208 vCPU 上 torch 默认开 ~100 intra-op 线程, 4 个
    ThreadPoolExecutor worker 并发 → 400 线程争 208 核 → 严重 thrashing, 训练几乎停滞。
    GPU 训练时 CPU 线程只管数据搬运/小算子, 不需要那么多。按 总核数/worker 分配, 留有余量。
    可被 DARWIN_ST_THREADS env 覆盖。串行/CPU 测试 (n_workers<=1) 不限制。
    """
    env = os.environ.get("DARWIN_ST_THREADS")
    if env:
        torch.set_num_threads(max(1, int(env)))
        return
    if n_workers and n_workers > 1:
        ncpu = os.cpu_count() or 8
        per = max(2, min(8, ncpu // (2 * n_workers)))  # 每 worker 限几线程, 上限 8 (GPU 训练够用)
        torch.set_num_threads(per)


def train_one(
    genotype: Genotype,
    hps: dict,
    data_dir: str,
    profile: DatasetProfile,
    adj: np.ndarray | None,
    device: str,
    trial: optuna.Trial | None = None,
    max_epochs: int = 27,
    time_budget_s: float = 2400.0,
) -> float:
    """用给定超参训练一个架构, 返回最优 val-MAE(真实尺度 masked)。

    每 epoch 向 trial.report 上报 val-MAE 供 ASHA 剪枝(trial 为 None 则跳过剪枝)。
    NaN/时间熔断触发时提前返回当前最优(或 inf)。
    """
    model = build_model(
        genotype, num_nodes=profile.num_nodes, in_channels=profile.num_channels,
        seq_len_in=profile.seq_len_in, seq_len_out=profile.seq_len_out, adj=adj,
    ).to(device)

    batch_size = int(hps.get("batch_size", 64))
    lr = float(hps.get("lr", 1e-3))
    wd = float(hps.get("weight_decay", 1e-4))
    train_loader = P.load_split(data_dir, "train", batch_size=batch_size, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # lr schedule 作为 HPO 可择优的超参 (none/cosine/plateau), 不硬设 —— 让 HPO 自己选用不用、用哪种。
    sched_kind = str(hps.get("lr_schedule", "none"))
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    elif sched_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None

    best_mae = float("inf")
    start = time.time()
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 4:                          # (x, tod, dow, y) STID 身份嵌入
                x, tod, dow, y = batch
                x, tod, dow, y = x.to(device), tod.to(device), dow.to(device), y.to(device)
                pred = model(x, tod_idx=tod, dow_idx=dow)
            else:                                        # (x, y) 回退
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
            opt.zero_grad()
            loss = masked_mae(pred, y)                    # 归一化尺度训练 loss
            if torch.isnan(loss) or torch.isinf(loss):   # NaN 熔断
                return best_mae if best_mae < float("inf") else float("inf")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪
            opt.step()

        # 真实尺度 masked 评测
        met = P.evaluate(model, data_dir, "val", batch_size=batch_size, device=device,
                         null_val=profile.null_val)
        mae = met["mae"]
        if mae < best_mae:                               # best-checkpoint(取最优步)
            best_mae = mae

        # lr schedule 步进: cosine 每 epoch 步, plateau 看 val-MAE
        if scheduler is not None:
            if sched_kind == "plateau":
                scheduler.step(mae)
            else:
                scheduler.step()

        if trial is not None:
            trial.report(mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if time.time() - start > time_budget_s:          # 时间熔断
            break

    return best_mae


def evaluate_architecture(
    genotype: Genotype,
    device: str,
    data_dir: str,
    profile: DatasetProfile,
    adj: np.ndarray | None,
    hpo_cfg: HPOConfig | None = None,
    warm_start_hps: dict | None = None,
) -> EvalResult:
    """对单个架构做内层 HPO + 训练, 返回最优结果(orchestrator 的 eval 单元)。"""
    hpo_cfg = hpo_cfg or HPOConfig()
    t0 = time.time()

    # 参数量(供 archive 行为描述子): 干实例化一次取, 无需训练
    probe = build_model(genotype, num_nodes=profile.num_nodes, in_channels=profile.num_channels,
                        seq_len_in=profile.seq_len_in, seq_len_out=profile.seq_len_out, adj=adj)
    n_params = count_params(probe)
    del probe

    def train_eval_fn(geno, hps, trial):
        return train_one(geno, hps, data_dir, profile, adj, device, trial=trial,
                         max_epochs=hpo_cfg.max_epochs)

    res = optimize_architecture(
        genotype, train_eval_fn, hpo_cfg,
        study_name=f"arch_{genotype.signature()[:10]}",
        warm_start_hps=warm_start_hps,
    )

    status = "OK" if res.best_mae < float("inf") else "CRASH"
    return EvalResult(
        genotype=genotype, status=status,
        mae=res.best_mae, rmse=float("inf"),   # rmse 此处不单独跑(评测函数已含, 可后续补)
        hps=res.best_hps, device=device,
        wall_seconds=time.time() - t0,
        fail_reason=None if status == "OK" else "all_hpo_trials_failed",
        extra={"num_params": n_params,
               "hpo_complete": res.n_complete, "hpo_pruned": res.n_pruned,
               "hpo_failed": res.n_failed},
    )


def make_eval_fn(dataset: str, hpo_cfg: HPOConfig | None = None):
    """工厂: 绑定数据集, 返回 orchestrator 用的 eval_fn(genotype, device) → EvalResult。

    首次调用前确保数据已 prepare(下载+切分+邻接)。
    """
    profile = get_profile(dataset)
    data_dir = P.prepare_dataset(dataset)
    adj = P.load_adj(data_dir)

    def eval_fn(genotype: Genotype, device: str) -> EvalResult:
        return evaluate_architecture(genotype, device, data_dir, profile, adj, hpo_cfg=hpo_cfg)

    return eval_fn
