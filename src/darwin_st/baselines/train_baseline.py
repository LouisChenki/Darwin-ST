"""基线统一训练器 (train_baseline) —— 与 optim/train.py 的 train_one 同炉语义。

逐项对齐 train_one 的训练纪律 (同炉 = 同一套优化器/损失/评测/早停):
  - Adam(lr=hps.get("lr",1e-3), weight_decay=hps.get("weight_decay",1e-4))
  - 梯度裁剪 clip_grad_norm_(…, 5.0)
  - 训练损失 masked_mae (归一化尺度, null_val=profile.null_val)
  - lr_schedule: hps.get("lr_schedule","plateau"); plateau=ReduceLROnPlateau
    (factor=0.5, patience=3), cosine=CosineAnnealingLR (T_max=max_epochs), none=无
    (train_one 默认 none, 此处默认 plateau —— 基线长跑惯例, 见同模块 docstring)
  - 每 epoch 末 P.evaluate(model, data_dir, "val", …) 评 val (真实尺度 masked)
  - best-checkpoint: 刷新时把 CPU state_dict 交 maybe_save_best 原子落盘
  - 早停: 连续 patience 轮无 >min_delta(相对) 改进则停 (scale-free, 同 train_one)
  - NaN 熔断: loss nan/inf 立即返回 (当前 best 或 inf)

多卡纪律: 进函数先 torch.cuda.set_device(device) —— train.py 的死锁教训:
只 .to(device) 时张量在目标卡但临时分配/stream/cuDNN handle 仍挤在 cuda:0,
多线程并发大模型竞争同一上下文 → 死锁 (见 optim/train.py train_one docstring)。
"""

from __future__ import annotations

import numpy as np
import torch

from darwin_st.data import prepare as P
from darwin_st.data.metrics import masked_mae
from darwin_st.data.protocol import DatasetProfile
from darwin_st.optim.checkpoint import maybe_save_best

__all__ = ["train_baseline", "DEFAULT_HPS"]

# 同炉统一超参: 与 train_one 的缺省完全一致 (lr/wd/batch), schedule 取 plateau
# (基线复现惯例; 评测口径与优化器配置四模型同一, 保证可比性)
DEFAULT_HPS: dict = {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 64,
                     "lr_schedule": "plateau"}


def train_baseline(
    model: torch.nn.Module,
    hps: dict,
    data_dir: str,
    profile: DatasetProfile,
    adj: np.ndarray | None,
    device: str,
    max_epochs: int = 100,
    early_stop_patience: int = 10,
    early_stop_min_delta: float = 0.001,
    checkpoint_path: str | None = None,
) -> tuple[float, dict]:
    """训练一个基线模型, 返回 (best_val_mae, {"best_rmse", "best_mape", "n_epochs"})。

    adj 为与 train_one 对齐的契约参数: 各模型构造时已把所需图 (转移矩阵 buffer 或
    自适应嵌入) 内化, 此处不再使用。

    早停相对 min_delta (默认 0.1%): val-MAE 需比历史 best 低超过 0.1% 才算改进。
    NaN 熔断: 训练 loss 出现 nan/inf 立即返回 (best 有限则返回 best, 否则 inf)。
    """
    # 多卡并发铁律: 把当前线程默认 CUDA 上下文切到目标卡 (train.py 死锁教训, 见模块 docstring)
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.set_device(device)
    model = model.to(device)
    del adj

    batch_size = int(hps.get("batch_size", 64))
    lr = float(hps.get("lr", 1e-3))
    wd = float(hps.get("weight_decay", 1e-4))
    train_loader = P.load_split(data_dir, "train", batch_size=batch_size, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    sched_kind = str(hps.get("lr_schedule", "plateau"))
    if sched_kind == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    elif sched_kind == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=3)
    else:
        scheduler = None

    best_mae = float("inf")
    best_rmse = float("inf")
    best_mape = float("inf")
    epochs_since_improve = 0
    n_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            if len(batch) == 4:                          # (x, tod, dow, y) STID 身份嵌入
                x, tod, dow, y = batch
                x, tod, dow, y = (x.to(device), tod.to(device),
                                  dow.to(device), y.to(device))
                pred = model(x, tod_idx=tod, dow_idx=dow)
            else:                                        # (x, y) 回退
                x, y = batch
                x, y = x.to(device), y.to(device)
                pred = model(x)
            opt.zero_grad()
            loss = masked_mae(pred, y, null_val=profile.null_val)   # 归一化尺度 (同 train_one)
            if torch.isnan(loss) or torch.isinf(loss):   # NaN 熔断: 不空跑
                return (best_mae if best_mae < float("inf") else float("inf")), {
                    "best_rmse": best_rmse, "best_mape": best_mape, "n_epochs": n_epochs}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        # 真实尺度 masked 评测 (与 train_one 同一 evaluate)
        met = P.evaluate(model, data_dir, "val", batch_size=batch_size, device=device,
                         null_val=profile.null_val)
        n_epochs = epoch + 1
        mae = met["mae"]
        # 相对阈值判实质改进 (best 更新前算, scale-free 跨数据集)
        improved = mae < best_mae * (1.0 - early_stop_min_delta)
        epochs_since_improve = 0 if improved else epochs_since_improve + 1
        if mae < best_mae:                               # best-checkpoint (严格 <)
            best_mae = float(mae)
            best_rmse = float(met["rmse"])
            best_mape = float(met["mape"])
            if checkpoint_path is not None:
                try:
                    sd_cpu = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                    maybe_save_best(sd_cpu, checkpoint_path,
                                    {"baseline": model.__class__.__name__,
                                     "dataset": profile.name, "best_epoch": epoch}, best_mae)
                except Exception:
                    pass            # 存档失败 (磁盘满等) 不影响训练主任务 (同 train_one)

        # lr schedule 步进: cosine 每 epoch 步, plateau 看 val-MAE
        if scheduler is not None:
            if sched_kind == "plateau":
                scheduler.step(mae)
            else:
                scheduler.step()

        # 收敛早停: val 平台 patience 轮无实质改进 → 停 (best 已存, 零损失)
        if early_stop_patience and epochs_since_improve >= early_stop_patience:
            break

    return best_mae, {"best_rmse": best_rmse, "best_mape": best_mape,
                      "n_epochs": n_epochs}
