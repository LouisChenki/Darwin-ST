"""训练动态轨迹 (TrainTrace) —— 逐 epoch 训练过程信号, 供 LLM 瓶颈诊断。

现状根因 (见 plan): diagnose_bottleneck 只看**静态架构**(算子集合+gap), 6 次创造诊断同一瓶颈、
检索同一机制, best 卡 18.69 无法突破。要让诊断多样化, LLM 需要**训练动态**: loss 曲线怎么降、
梯度稳不稳、是欠拟合还是过拟合还是欠训。这些信号现在全丢了 (train_one 只返回 best_mae 标量,
clip_grad_norm_ 的返回值——梯度范数——直接丢弃)。

TrainTrace 在 epoch 边界聚合这些信号 (不存 per-batch, 省开销/体积):
  - 逐 epoch 短列表: train_loss / val_mae / grad_norm_mean / grad_norm_max
    (80 epoch × 4 列 ≈ 320 floats, pickle <3KB, 跨 spawn 进程零压力)。
  - 标量: best_epoch / n_epochs_run / stopped_early / nan_hit / lr_schedule / lr。

铁律: **全字段纯标量 / list[float] / str / bool / int → 可 pickle** (阶段1多进程后, 训练在 worker
子进程, trace 要经 Optuna user_attr + EvalResult.extra 跨进程回到主进程)。故此模块**不导入 torch**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["TrainTrace"]


@dataclass
class TrainTrace:
    """单次训练 (一个 trial) 的逐 epoch 动态轨迹。全字段可 pickle / JSON-able。"""

    # 逐 epoch 曲线 (len = 实际跑的 epoch 数, ≤ max_epochs)
    train_loss: list[float] = field(default_factory=list)      # 每 epoch 训练 loss 均值 (尺度依 loss 字段: mae=归一化, huber=真实)
    val_mae: list[float] = field(default_factory=list)         # 每 epoch val MAE (真实尺度 masked)
    grad_norm_mean: list[float] = field(default_factory=list)  # 每 epoch clip 前梯度总范数均值
    grad_norm_max: list[float] = field(default_factory=list)   # 每 epoch 梯度总范数最大值

    # 标量
    n_epochs_run: int = 0           # 实际跑了几 epoch (时间/NaN/收敛熔断可能 < max)
    max_epochs: int = 0
    best_epoch: int = -1            # val_mae 最优的 epoch 下标
    best_mae: float = float("inf")
    best_rmse: float = float("inf")  # best-checkpoint 同 epoch 的 val RMSE (接住已算出的评测, 零额外开销)
    best_mape: float = float("inf")  # 同上, val MAPE
    final_train_loss: float = float("inf")
    stopped_early: bool = False     # 被时间熔断截断 (非自然收敛)
    converged: bool = False         # 被收敛早停截断 (val 曲线平台 patience 轮无实质改进; best 已存零损失)
    nan_hit: bool = False           # 训练中遇到 NaN/Inf loss
    lr_schedule: str = "none"       # 该 trial 用的调度 (none/cosine/plateau)
    lr: float = 0.0                 # 该 trial lr (辅助梯度诊断)
    loss: str = "mae"               # 该 trial 训练损失类型 (mae=归一化尺度 masked MAE / huber=真实尺度 masked Huber)
