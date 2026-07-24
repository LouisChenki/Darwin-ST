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
from dataclasses import asdict

import numpy as np
import optuna
import torch

from darwin_st.data import prepare as P
from darwin_st.data.metrics import masked_mae
from darwin_st.data.protocol import DatasetProfile, get_profile
from darwin_st.optim.checkpoint import maybe_save_best
from darwin_st.optim.hpo import HPOConfig, optimize_architecture
from darwin_st.optim.scheduler import EvalResult
from darwin_st.optim.trace import TrainTrace
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
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.001,
    checkpoint_path: str | None = None,
    checkpoint_meta: dict | None = None,
) -> tuple[float, TrainTrace]:
    """用给定超参训练一个架构, 返回 (最优 val-MAE 真实尺度 masked, TrainTrace 训练动态轨迹)。

    每 epoch 向 trial.report 上报 val-MAE 供 ASHA 剪枝(trial 为 None 则跳过剪枝)。
    NaN/时间/收敛熔断触发时提前返回当前最优(或 inf)。

    权重存档 (checkpoint_path 非 None 才启用, 默认行为完全不变): 每当 val-MAE 刷新
    best-checkpoint, 把 state_dict 搬到 CPU 调 maybe_save_best 原子落盘 (sidecar 比较
    保证同一路径权重只升不降; 写盘几 MB, 相对一次全量 val evaluate 可忽略)。
    checkpoint_meta 带调用方字段 (signature/dataset), best_epoch 在此按刷新点补。
    存档异常 (磁盘满等) 吞掉不影响训练主任务。

    收敛早停 (early_stop_patience>0 才启用): val-MAE 连续 patience 轮无 >min_delta(相对) 改进 → 停。
    因 best-checkpoint 已存最优, 早停零损失。max_epochs 是硬上限(安全), 早停是软自适应切(省算力)。
    相对 min_delta (默认 0.1%) 跨数据集 scale-free (PeMS04~18 与 METR-LA~3 同一相对阈值都有意义)。

    TrainTrace 在 epoch 边界聚合训练动态 (loss 曲线/梯度范数/收敛标志), 供 Tier-2 LLM 瓶颈诊断。
    采集点全在现有边界, 开销 <0.1% (每 epoch 几个 .item() + append, 相对一次全量 val evaluate 可忽略)。

    多卡并发铁律: 必须 set_device 把**当前线程的 CUDA 默认上下文**切到目标卡。
    只 .to(device) 不够 —— 张量在目标卡但临时分配/stream/cuDNN handle 仍挤在 cuda:0,
    多线程并发大模型时竞争同一上下文 → 死锁 (标定跑实测: 117线程全 futex_wait)。
    """
    # 把当前线程默认 CUDA 上下文切到目标卡 (多卡并发防上下文竞争死锁)
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(device)
    # HPO 旋钮接线: dropout (默认 0.0 恒等) 与 num_heads (仅头数可配的注意力算子消费, None=各算子默认)
    model = build_model(
        genotype, num_nodes=profile.num_nodes, in_channels=profile.num_channels,
        seq_len_in=profile.seq_len_in, seq_len_out=profile.seq_len_out, adj=adj,
        dropout=float(hps.get("dropout", 0.0)),
        num_heads=hps.get("num_heads"),
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

    trace = TrainTrace(max_epochs=max_epochs, lr_schedule=sched_kind, lr=lr)
    best_mae = float("inf")
    epochs_since_improve = 0                # 收敛早停计数: 连续多少 epoch 无 >min_delta(相对) 改进
    start = time.time()
    for epoch in range(max_epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0           # epoch 内训练 loss 累加 (epoch 末求均值)
        gn_sum, gn_max = 0.0, 0.0              # epoch 内梯度范数累加 + 最大 (clip 前)
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
                trace.nan_hit = True
                trace.n_epochs_run = epoch
                trace.best_mae = best_mae
                return (best_mae if best_mae < float("inf") else float("inf")), trace
            loss.backward()
            # clip_grad_norm_ 返回 **clip 前**的梯度总范数 (零额外计算) —— 之前丢弃, 现采集供梯度诊断
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            loss_sum += float(loss.detach())             # epoch 末一次性聚合, 不在 batch 内 sync
            gn_val = float(gn)
            gn_sum += gn_val
            gn_max = max(gn_max, gn_val)
            n_batches += 1

        # 真实尺度 masked 评测
        met = P.evaluate(model, data_dir, "val", batch_size=batch_size, device=device,
                         null_val=profile.null_val)
        mae = met["mae"]
        # 收敛早停计数: 用**相对**阈值判有无实质改进 (best 更新前算, scale-free 跨数据集)
        improved = mae < best_mae * (1.0 - early_stop_min_delta)
        epochs_since_improve = 0 if improved else epochs_since_improve + 1
        if mae < best_mae:                               # best-checkpoint(取最优步, 严格 <)
            best_mae = mae
            trace.best_epoch = epoch
            # 同 epoch 的 rmse/mape 一并接住 (评测已算出, 此前丢弃 → EvalResult.rmse 恒 inf)
            trace.best_rmse = float(met["rmse"])
            trace.best_mape = float(met["mape"])
            if checkpoint_path is not None:
                try:
                    sd_cpu = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                    maybe_save_best(sd_cpu, checkpoint_path,
                                    {**(checkpoint_meta or {}), "best_epoch": epoch}, mae)
                except Exception:
                    pass            # 存档失败 (磁盘满等) 不影响训练主任务

        # 采集本 epoch 动态信号 (epoch 边界, 低开销)
        trace.train_loss.append(loss_sum / max(n_batches, 1))
        trace.val_mae.append(float(mae))
        trace.grad_norm_mean.append(gn_sum / max(n_batches, 1))
        trace.grad_norm_max.append(gn_max)

        # lr schedule 步进: cosine 每 epoch 步, plateau 看 val-MAE
        if scheduler is not None:
            if sched_kind == "plateau":
                scheduler.step(mae)
            else:
                scheduler.step()

        if trial is not None:
            trial.report(mae, epoch)
            if trial.should_prune():
                trace.n_epochs_run = epoch + 1
                trace.best_mae = best_mae
                trace.final_train_loss = trace.train_loss[-1] if trace.train_loss else float("inf")
                raise optuna.TrialPruned()

        # 收敛早停: val 平台 patience 轮无实质改进 → 停 (best 已存, 零损失)。与 ASHA 正交:
        # ASHA 跨 trial 剪坏配置, 早停在单 trial 内切收敛好配置; 此 trial 仍是 COMPLETE(非 pruned)。
        if early_stop_patience and epochs_since_improve >= early_stop_patience:
            trace.converged = True
            break

        if time.time() - start > time_budget_s:          # 时间熔断
            trace.stopped_early = True
            break

    trace.n_epochs_run = len(trace.val_mae)
    trace.best_mae = best_mae
    trace.final_train_loss = trace.train_loss[-1] if trace.train_loss else float("inf")
    return best_mae, trace


def evaluate_architecture(
    genotype: Genotype,
    device: str,
    data_dir: str,
    profile: DatasetProfile,
    adj: np.ndarray | None,
    hpo_cfg: HPOConfig | None = None,
    warm_start_hps: dict | None = None,
    n_trials: int | None = None,
    checkpoint_dir: str | None = None,
) -> EvalResult:
    """对单个架构做内层 HPO + 训练, 返回最优结果(orchestrator 的 eval 单元)。

    n_trials: 覆盖 hpo_cfg.n_trials 的本次预算 (创造 seed 走大预算"优胜者深评"; None=用 cfg 默认)。
    warm_start_hps: 父架构最优超参, 作为第一个 trial 暖启动 (创造 seed 继承父超参省冷启动)。
    checkpoint_dir: 非 None 时启用权重存档 —— 本架构固定路径 <dir>/<dataset>_<sig[:12]>.pt,
      同架构的多个 HPO trial 共享 (同一 worker 内顺序执行, 靠 sidecar 比较只留最好 trial 的
      权重); 不同架构路径不同, 多 worker 并行无冲突。None (默认) = 完全不落盘, 行为不变。
    """
    hpo_cfg = hpo_cfg or HPOConfig()
    t0 = time.time()

    ckpt_path = None
    ckpt_meta = None
    if checkpoint_dir:
        sig = genotype.signature()
        ckpt_path = os.path.join(checkpoint_dir, f"{profile.name}_{sig[:12]}.pt")
        ckpt_meta = {"signature": sig, "dataset": profile.name}

    # 参数量(供 archive 行为描述子): 干实例化一次取, 无需训练
    probe = build_model(genotype, num_nodes=profile.num_nodes, in_channels=profile.num_channels,
                        seq_len_in=profile.seq_len_in, seq_len_out=profile.seq_len_out, adj=adj)
    n_params = count_params(probe)
    del probe

    def train_eval_fn(geno, hps, trial):
        mae, trace = train_one(geno, hps, data_dir, profile, adj, device, trial=trial,
                               max_epochs=hpo_cfg.max_epochs,
                               early_stop_patience=hpo_cfg.early_stop_patience,
                               early_stop_min_delta=hpo_cfg.early_stop_min_delta,
                               checkpoint_path=ckpt_path, checkpoint_meta=ckpt_meta)
        # 把训练轨迹挂到 trial user_attr (存 dict: Optuna 要求 JSON-able + 跨进程更稳)
        try:
            trial.set_user_attr("trace", asdict(trace))
        except Exception:
            pass  # user_attr 失败不影响调参 (诊断信号缺失退回规则版)
        return mae

    res = optimize_architecture(
        genotype, train_eval_fn, hpo_cfg,
        study_name=f"arch_{genotype.signature()[:10]}",
        warm_start_hps=warm_start_hps,
        n_trials=n_trials,
    )

    status = "OK" if res.best_mae < float("inf") else "CRASH"
    # 最优 trial best-checkpoint 的 rmse/mape 经 TrainTrace 带回 (评测每 epoch 已跑, 这里是
    # "把已算出的数接住", 零新增评测开销; CRASH/无 trace → inf, 由下游 _finite 判空)。
    best_trace = res.best_trace or {}
    best_rmse = float(best_trace.get("best_rmse", float("inf")))
    best_mape = float(best_trace.get("best_mape", float("inf")))
    return EvalResult(
        genotype=genotype, status=status,
        mae=res.best_mae, rmse=best_rmse,
        hps=res.best_hps, device=device,
        wall_seconds=time.time() - t0,
        fail_reason=None if status == "OK" else "all_hpo_trials_failed",
        extra={"num_params": n_params,
               "hpo_complete": res.n_complete, "hpo_pruned": res.n_pruned,
               "hpo_failed": res.n_failed,
               "val_mape": best_mape,
               "train_trace": res.best_trace},   # 最优 trial 的训练动态轨迹 (供 LLM 诊断)
    )


def make_eval_fn(dataset: str, hpo_cfg: HPOConfig | None = None,
                 checkpoint_dir: str | None = None):
    """工厂: 绑定数据集, 返回 orchestrator 用的 eval_fn(genotype, device) → EvalResult。

    首次调用前确保数据已 prepare(下载+切分+邻接)。
    checkpoint_dir 非 None → 训练中刷新纪录的模型权重落盘该目录 (默认 None=关闭)。
    """
    profile = get_profile(dataset)
    data_dir = P.prepare_dataset(dataset)
    adj = P.load_adj(data_dir)

    def eval_fn(genotype: Genotype, device: str) -> EvalResult:
        # 创造 seed 带非字段元数据 _seed_meta (随 genotype pickle 到 worker): 大 HPO 预算 + warm-start。
        # 普通进化架构无此属性, 走默认 (行为完全不变)。
        meta = getattr(genotype, "_seed_meta", None)
        warm_start_hps = meta.get("warm_start_hps") if meta else None
        n_trials = meta.get("hpo_trials") if meta else None     # None → 用 hpo_cfg.n_trials
        return evaluate_architecture(genotype, device, data_dir, profile, adj, hpo_cfg=hpo_cfg,
                                     warm_start_hps=warm_start_hps, n_trials=n_trials,
                                     checkpoint_dir=checkpoint_dir)

    return eval_fn
