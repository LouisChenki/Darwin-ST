"""并行调度 (GPU Scheduler) —— 跨多卡并行评估架构。

把外层进化提议的架构, 并行派发到多张 GPU 评估 (每个架构的评估 = 一次内层 HPO study)。
模型: **一个架构占一张卡** (架构内的 HPO trial 在该卡上串行/多保真早停)。

并行机制: ThreadPoolExecutor, worker 数 = GPU 数。PyTorch 训练在 C/CUDA 层执行并释放
GIL, 故线程池能拿到真并行 (无需多进程, 省去 CUDA 上下文复制)。每个 worker 从设备队列
领一张卡、评估完归还。

设计 (契合两层自治, 见 docs/P2_ALGORITHM_DESIGN.md §4/§6):
  - eval_fn(genotype, device) → EvalResult 注入, 故本模块**不依赖真实训练/GPU**, 可 mock 全测。
  - 失败 (OOM/NaN/超时) 被捕获为 EvalResult(status=CRASH), 不让整个调度崩 —— 自主性要求。
  - 结果按完成顺序回调 (on_result), orchestrator 据此 tell 进化 + 落 memory。
"""

from __future__ import annotations

import queue
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from darwin_st.search.genotype import Genotype

__all__ = ["EvalResult", "GPUScheduler", "resolve_devices"]


@dataclass
class EvalResult:
    """单个架构的评估结果。"""

    genotype: Genotype
    status: str                       # KEEP/DISCARD/CRASH (由 orchestrator 据 SOTA 判定; 这里给 OK/CRASH)
    mae: float = float("inf")
    rmse: float = float("inf")
    hps: dict = field(default_factory=dict)
    device: str = ""
    wall_seconds: float = 0.0
    fail_reason: str | None = None
    extra: dict = field(default_factory=dict)


def resolve_devices(devices: list[str] | int | None = None) -> list[str]:
    """规范化设备列表。

    - None: 自动探测 CUDA 卡数 → ['cuda:0', ...]; 无 CUDA → ['cpu']
    - int N: ['cuda:0'..'cuda:N-1']
    - list: 原样返回
    """
    if isinstance(devices, list):
        return devices
    if isinstance(devices, int):
        return [f"cuda:{i}" for i in range(devices)]
    # 自动探测
    try:
        import torch

        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 0 else ["cpu"]
    except Exception:
        return ["cpu"]


# eval_fn 签名: (genotype, device) -> EvalResult
#   实现方在 device 上做该架构的 HPO+训练+评测, 返回 EvalResult。
EvalFn = Callable[[Genotype, str], EvalResult]


class GPUScheduler:
    """跨多卡并行评估架构的调度器。

    用法:
        sched = GPUScheduler(eval_fn, devices=4)        # 或 ['cuda:0',...]
        results = sched.run_batch([g1, g2, g3, ...])    # 并行评估一批, 返回结果列表
      或流式:
        sched.submit(g); ... ; for r in sched.drain(): ...
    """

    def __init__(
        self,
        eval_fn: EvalFn,
        devices: list[str] | int | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
    ):
        self.eval_fn = eval_fn
        self.devices = resolve_devices(devices)
        self.on_result = on_result
        # 设备资源池: 线程从中领卡, 用完归还 → 保证一卡同时只跑一个架构
        self._device_pool: queue.Queue[str] = queue.Queue()
        for d in self.devices:
            self._device_pool.put(d)
        self._lock = threading.Lock()

    @property
    def num_devices(self) -> int:
        return len(self.devices)

    def _eval_one(self, genotype: Genotype) -> EvalResult:
        """领一张卡 → 评估 → 归还。失败兜底为 CRASH, 绝不抛出 (调度不崩)。"""
        device = self._device_pool.get()  # 阻塞直到有空闲卡
        try:
            return self.eval_fn(genotype, device)
        except Exception as e:
            return EvalResult(
                genotype=genotype, status="CRASH", device=device,
                fail_reason=f"{type(e).__name__}: {e}",
                extra={"traceback": traceback.format_exc()},
            )
        finally:
            self._device_pool.put(device)  # 无论成败都归还卡

    def run_batch(self, genotypes: list[Genotype]) -> list[EvalResult]:
        """并行评估一批架构, 全部完成后返回结果列表 (顺序与输入对应)。

        并发上限 = GPU 数 (设备池天然限流)。on_result 在每个完成时回调。
        """
        if not genotypes:
            return []
        results: list[EvalResult | None] = [None] * len(genotypes)
        with ThreadPoolExecutor(max_workers=self.num_devices) as pool:
            future_to_idx = {pool.submit(self._eval_one, g): i for i, g in enumerate(genotypes)}
            for fut in as_completed(future_to_idx):
                i = future_to_idx[fut]
                res = fut.result()  # _eval_one 不抛, 总有 EvalResult
                results[i] = res
                if self.on_result is not None:
                    with self._lock:        # 回调串行化, 便于安全更新共享状态
                        self.on_result(res)
        return [r for r in results if r is not None]
