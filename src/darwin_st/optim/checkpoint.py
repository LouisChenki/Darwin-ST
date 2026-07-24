"""权重存档 (Checkpoint) —— 训练中新纪录模型的"权重+元数据"落盘, 只留 top-K 防撑盘。

现状痛点: 系统只把 (genotype, hps) 以 JSON 存进 SQLite memory, 从不保存权重 ——
复查/复现历史最优只能从头重训。本模块提供最小闭环:

  - maybe_save_best: 训练中 val-MAE 刷新纪录时, 把 CPU state_dict 原子落盘 +
    旁挂 sidecar 元数据 (<path>.meta.json)。sidecar 里已记的 val_mae 是"只升不降"
    的比较基准: 只有更优的新 MAE 才覆盖旧权重。
  - prune_to_top_k:  跑完按 sidecar val_mae 升序保留前 K 个, 其余连权重带 sidecar 删。
  - load_meta:       容错读 sidecar (缺/坏 → None)。

并发与原子性论证 (多进程后端, worker 直接写共享磁盘):
  - 同一路径只会被"评估同一架构的那个 worker"写: 同架构的多个 HPO trial 在同一 worker
    内顺序执行, 共享同一 ckpt 路径, 靠 sidecar 比较自然实现"同架构只留最好 trial 的权重"。
  - 不同架构路径不同 (文件名含 dataset + signature[:12]), 不同 worker 并行评不同架构
    互不冲突。极端情形 (同架构重复提议到两张卡) 也只是 sidecar 比较竞态 —— 结果可能
    非最优保留, 但文件绝不撕裂 (见下), 可接受。
  - 原子写: 先写同目录临时文件再 os.replace 改名 —— 读者要么看到旧文件要么新文件,
    绝不看到写一半的 .pt / .json。崩溃残留 `<path>.<pid>.tmp` 不匹配 `*.pt` 扫描,
    不参与 prune, 可手工清理。
  - 写序: 先权重后 sidecar (sidecar 作提交点)。中途崩 → 权重可能新于 sidecar,
    下次出现更优 MAE 时一并重写自愈; sidecar 缺失整体视为首次写。

张量不跨进程 pickle: 权重由 worker 进程内直接写共享目录, EvalResult 只带标量。
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime

import torch

__all__ = ["maybe_save_best", "prune_to_top_k", "load_meta", "meta_path_for"]


def meta_path_for(path: str) -> str:
    """权重文件 path 旁的 sidecar 元数据路径。"""
    return path + ".meta.json"


def load_meta(path: str) -> dict | None:
    """容错读 sidecar: 不存在 / JSON 损坏 / 非 dict → None (调用方视为"无历史")。"""
    try:
        with open(meta_path_for(path), encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def _atomic_write(path: str, write_fn) -> None:
    """先写同目录临时文件再 os.replace 原子改名 (读写双方都不会看到半写文件)。

    临时文件名带 pid: 极端情况下两进程写同一路径也不共用临时文件; 崩溃残留可识别。
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        write_fn(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def maybe_save_best(state_dict_cpu: dict, path: str, meta: dict, new_mae: float) -> bool:
    """仅当 new_mae 优于 sidecar 已记 val_mae (或无 sidecar=首次) 才原子落盘。

    state_dict_cpu: 调用方已搬到 CPU 的 state_dict (GPU 张量先 .detach().cpu(),
                    本函数不做设备搬运, 保持纯逻辑可测)。
    meta: 调用方给的业务字段 (signature / dataset / best_epoch 等); 本函数补
          val_mae=new_mae 与 created_at 后整体写入 sidecar。
    返回是否发生了写入 (False = 历史更优或持平, 未动盘)。
    """
    old = load_meta(path)
    if old is not None:
        try:
            old_mae = float(old.get("val_mae"))
        except (TypeError, ValueError):
            old_mae = float("inf")      # sidecar 在但 val_mae 坏 → 视为可被覆盖
        if not (new_mae < old_mae):     # 严格更优才覆盖 (持平保留先到的)
            return False

    payload = {**meta, "val_mae": float(new_mae),
               "created_at": datetime.now().isoformat(timespec="seconds")}
    _atomic_write(path, lambda tmp: torch.save(state_dict_cpu, tmp))

    def _dump(tmp: str) -> None:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    _atomic_write(meta_path_for(path), _dump)
    return True


def _mae_key(path: str) -> tuple[int, float]:
    """prune 排序键: 有有效 sidecar 的按 val_mae 升序在前; 缺/坏 sidecar 排尾 (先删)。"""
    m = load_meta(path)
    if m is None:
        return (1, float("inf"))
    try:
        return (0, float(m["val_mae"]))
    except (KeyError, TypeError, ValueError):
        return (1, float("inf"))


def prune_to_top_k(ckpt_dir: str, keep: int) -> list[str]:
    """按 sidecar val_mae 升序保留前 keep 个 .pt, 删除其余 (连同 sidecar)。

    无 sidecar / sidecar 损坏的 .pt 视为最差 (排在保留区之外)。keep<=0 清空。
    返回被删除的 .pt 路径列表。目录不存在 → 空列表 (不报错)。
    """
    if not os.path.isdir(ckpt_dir):
        return []
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")), key=_mae_key)
    deleted: list[str] = []
    for p in pts[max(keep, 0):]:
        for q in (p, meta_path_for(p)):
            try:
                os.unlink(q)
            except OSError:
                pass            # sidecar 本就缺 / 删除竞态, 不致命
        deleted.append(p)
    return deleted
