"""并行调度 (GPU Scheduler) —— 跨多卡并行评估架构。

把外层进化提议的架构, 并行派发到多张 GPU 评估 (每个架构的评估 = 一次内层 HPO study)。
模型: **一个架构占一张卡** (架构内的 HPO trial 在该卡上串行/多保真早停)。

并行机制 (双后端):
  - **进程后端** (默认真实路径, backend="auto"+eval_spec 或 "process"): ProcessPoolExecutor +
    spawn, 每进程独立 GIL + 独立 CUDA 上下文 → 真并行 (~卡数倍)。worker 靠 EvalSpec 自建
    eval_fn (避闭包 pickle), 进程内惰性初始化一次 (set_device + limit_cpu_threads + 重载 synth)。
    实测教训: 线程池假设"PyTorch 训练释放 GIL 故线程真并行"是错的 (本地仅 1.17x, 4 卡各跑不满),
    Python 侧训练循环 GIL-bound, 必须多进程才能真并行。
  - **线程后端** (backend="thread", 或 auto 下无 eval_spec / 非 cuda): ThreadPoolExecutor,
    eval_fn 直接注入。mock/CPU 测试走这条 (注入的闭包 eval_fn 不可 pickle, 天然回退)。

设计 (契合两层自治, 见 docs/P2_ALGORITHM_DESIGN.md §4/§6):
  - eval_fn(genotype, device) → EvalResult 注入 (线程) 或 EvalSpec 重建 (进程), 故本模块**不依赖
    真实训练/GPU**, 可 mock 全测 (线程后端) + CPU spawn 实测 (进程后端, 见 test_scheduler_process)。
  - 失败 (OOM/NaN/超时/worker 崩) 被捕获为 EvalResult(status=CRASH), 不让整个调度崩 —— 自主性要求。
  - 结果按完成顺序回调 (on_result, 主进程串行), orchestrator 据此 tell 进化 + 落 memory。
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import traceback
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from typing import Callable

from darwin_st.search.genotype import Genotype

__all__ = ["EvalResult", "EvalSpec", "GPUScheduler", "resolve_devices"]


@dataclass
class EvalSpec:
    """进程后端用: worker 子进程靠它自建 eval_fn (避开闭包 pickle)。

    make_eval_fn 的闭包只闭包 dataset(str)+hpo_cfg(可pickle dataclass), profile/data_dir/adj
    全从 dataset 名重建。故 worker 只需这三项即可重建 eval_fn。全字段可 pickle (跨 spawn 进程)。
    """

    dataset: str
    hpo_cfg: object = None                 # HPOConfig (纯标量 dataclass, 可 pickle)
    synth_persist_dir: str | None = None   # registry persist_dir; worker 启动 load_persisted 补 synth 算子
    eval_factory: str | None = None        # dotted "module:attr" 工厂, 签名 (dataset, hpo_cfg)->eval_fn;
                                           # None=默认 make_eval_fn。用于注入替代评测后端或 CPU 测试 spawn 通路。
    checkpoint_dir: str | None = None      # 权重存档目录 (worker 直接写共享盘); None=关闭。
                                           # 仅默认 make_eval_fn 消费 (自定义工厂签名不变, 向后兼容)。


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


# ---------------------------------------------------------------------------
# 进程后端 worker (模块级, spawn 可定位) + 进程内缓存
# ---------------------------------------------------------------------------

# 每个 worker 子进程的进程级缓存: 首次调用重建 eval_fn + 重载 synth 算子, 之后复用。
_WORKER_EVAL_FN = None
_WORKER_INITED = False


def _worker_init(eval_spec: "EvalSpec", n_workers: int) -> None:
    """worker 进程首次惰性初始化 (进程级一次): 限线程 + 重载 synth 算子 + 重建 eval_fn。

    多进程铁律:
      - limit_cpu_threads 必须在**子进程内**调 (父进程那次对 spawn 子进程无效)。
      - synth 算子存活于进程全局 SPATIAL_OPS; spawn 子进程没有 → load_persisted 从盘补齐,
        否则 build_model 解析 genotype 里的 synth_xxx 算子会 KeyError。
    """
    global _WORKER_EVAL_FN, _WORKER_INITED
    if _WORKER_INITED:
        return
    from darwin_st.optim.train import limit_cpu_threads

    limit_cpu_threads(n_workers)  # 按 worker(=进程)数限 torch intra-op 线程
    if eval_spec.synth_persist_dir and os.path.isdir(eval_spec.synth_persist_dir):
        try:
            from darwin_st.creation.registry import OperatorRegistry

            OperatorRegistry(persist_dir=eval_spec.synth_persist_dir).load_persisted()
        except Exception:
            pass  # 无 synth 算子或加载失败不致命 (纯 Tier-1 跑无 synth)
    if eval_spec.eval_factory:                     # 注入式工厂 (替代后端 / CPU 测试)
        import importlib

        mod_name, attr = eval_spec.eval_factory.split(":")
        factory = getattr(importlib.import_module(mod_name), attr)
        _WORKER_EVAL_FN = factory(eval_spec.dataset, eval_spec.hpo_cfg)
    else:
        from darwin_st.optim.train import make_eval_fn
        _WORKER_EVAL_FN = make_eval_fn(eval_spec.dataset, eval_spec.hpo_cfg,
                                       checkpoint_dir=eval_spec.checkpoint_dir)
    _WORKER_INITED = True


def _process_worker(eval_spec: "EvalSpec", n_workers: int, device: str,
                    genotype: Genotype) -> EvalResult:
    """spawn 子进程评估一个架构。全程兜底 CRASH (绝不让异常逃出, 否则 future.result 抛坏池)。"""
    try:
        if isinstance(device, str) and device.startswith("cuda"):
            import torch

            torch.cuda.set_device(device)
        _worker_init(eval_spec, n_workers)
        return _WORKER_EVAL_FN(genotype, device)
    except Exception as e:
        return EvalResult(
            genotype=genotype, status="CRASH", device=device,
            fail_reason=f"{type(e).__name__}: {e}",
            extra={"traceback": traceback.format_exc()},
        )


class GPUScheduler:
    """跨多卡并行评估架构的调度器。

    用法:
        sched = GPUScheduler(eval_fn, devices=4)        # 或 ['cuda:0',...]
        results = sched.run_batch([g1, g2, g3, ...])    # 批模式: 并行一批, 全返回才出结果
      或流式 (推荐, 4 卡持续满载, 慢架构不阻塞快架构):
        sched.run_stream(next_genotype, on_result, should_continue)
    """

    def __init__(
        self,
        eval_fn: EvalFn | None = None,
        devices: list[str] | int | None = None,
        on_result: Callable[[EvalResult], None] | None = None,
        eval_spec: "EvalSpec | None" = None,
        backend: str = "auto",
    ):
        self.eval_fn = eval_fn
        self.devices = resolve_devices(devices)
        self.on_result = on_result
        self.eval_spec = eval_spec
        # 设备资源池: 线程从中领卡, 用完归还 → 保证一卡同时只跑一个架构 (线程后端用)
        self._device_pool: queue.Queue[str] = queue.Queue()
        for d in self.devices:
            self._device_pool.put(d)
        self._lock = threading.Lock()
        # 进程后端: 持久池惰性建 (整个 orchestrator 生命周期复用, 摊销 CUDA 初始化)
        self._pool = None
        self._use_process = self._resolve_backend(backend)

    def _resolve_backend(self, backend: str) -> bool:
        """决定走进程后端还是线程后端。auto: 仅当显式给了 eval_spec 才进程 (真实路径),
        否则 (mock eval_fn / 无 spec) 回退线程 —— 385 测试天然走这条 (它们 eval_fn 不可 pickle)。"""
        if backend == "thread":
            return False
        if backend == "process":
            if self.eval_spec is None:
                raise ValueError("backend='process' 需提供 eval_spec (worker 自建 eval_fn)")
            return True
        # auto: 有 eval_spec 且全 cuda 设备 → 进程; 否则线程
        return self.eval_spec is not None and all(
            isinstance(d, str) and d.startswith("cuda") for d in self.devices)

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

        并发上限 = GPU 数。on_result 在每个完成时回调 (主进程串行)。
        进程后端 (真实多卡训练) 避开 GIL 串行化; 线程后端 (mock/测试) 保持现状。
        """
        if not genotypes:
            return []
        if self._use_process:
            return self._run_batch_process(genotypes)
        return self._run_batch_thread(genotypes)

    def _run_batch_thread(self, genotypes: list[Genotype]) -> list[EvalResult]:
        """线程后端 (现状逻辑, 零行为变化): mock eval_fn / CPU 测试用。"""
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

    def _ensure_pool(self) -> None:
        """惰性建 spawn 进程池 (整个生命周期复用)。"""
        if self._pool is None:
            ctx = mp.get_context("spawn")  # CUDA 多进程铁律: 必须 spawn (fork 后用 CUDA 崩)
            self._pool = ProcessPoolExecutor(max_workers=self.num_devices, mp_context=ctx)

    def refresh_workers(self) -> None:
        """回收并下次重建进程池 —— 创造注入新 synth 算子后调, 让新 worker load_persisted 拿到最新。"""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def close(self) -> None:
        """清理进程池 (orchestrator run 结束/异常时调)。"""
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _run_batch_process(self, genotypes: list[Genotype]) -> list[EvalResult]:
        """进程后端 (真实多卡): spawn 持久池, 每架构取模绑一张卡 (batch=卡数时一卡一个)。

        BrokenProcessPool (worker segfault/OOM-kill 连坐整池) → 本批剩余标 CRASH + 重建池,
        下轮恢复, 不传播到主循环 (自主性铁律: 崩溃是数据点不是中止)。
        """
        from concurrent.futures.process import BrokenProcessPool

        n = self.num_devices
        results: list[EvalResult | None] = [None] * len(genotypes)
        try:
            self._ensure_pool()
            fut_to_idx = {}
            for i, g in enumerate(genotypes):
                device = self.devices[i % n]
                fut = self._pool.submit(_process_worker, self.eval_spec, n, device, g)
                fut_to_idx[fut] = i
            for fut in as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                try:
                    res = fut.result()
                except Exception as e:           # worker 已兜底 CRASH; 这里防 future 层异常
                    res = EvalResult(genotype=genotypes[i], status="CRASH",
                                     fail_reason=f"future: {type(e).__name__}: {e}")
                results[i] = res
                if self.on_result is not None:
                    self.on_result(res)          # 主进程串行回调
        except BrokenProcessPool as e:
            # 整池坏: 未完成的标 CRASH, 重建池下轮恢复
            for i, r in enumerate(results):
                if r is None:
                    results[i] = EvalResult(genotype=genotypes[i], status="CRASH",
                                            fail_reason=f"BrokenProcessPool: {e}")
                    if self.on_result is not None:
                        self.on_result(results[i])
            self.refresh_workers()
        return [r for r in results if r is not None]

    # -- 流式驱动 (run_stream): 保持 num_devices 个在飞, 4 卡持续满载 --

    def run_stream(
        self,
        next_genotype: Callable[[], "Genotype | None"],
        on_result: Callable[[EvalResult], None],
        should_continue: Callable[[], bool],
        pause_check: Callable[[], bool] | None = None,
        on_resume: Callable[[], None] | None = None,
    ) -> None:
        """连续流式评估: 始终保持 ≤num_devices 个评估在飞, 一个完成立即取下一个提交。

        破批同步瓶颈: 批模式下一批必须全返回才落库, 慢架构拖住整批 (4 卡 3 闲)。流式下
        一个 worker 一空就补新任务 → 卡永远满载。ask/tell 已乱序安全, 每 genotype 仍恰好评估+
        digest 一次, 计数与批模式恒等, 只变完成顺序与提交时机。

        回调契约:
          - next_genotype(): 下一个待评 genotype; None = 暂无 (含 refresh 待定期, 让在飞排空)。
          - on_result(res):  每完成回调 (主进程/线程串行), 调用方在此 digest。
          - should_continue(): False → 停止提交、排空剩余在飞 (仍 digest, 不丢结果)、返回。
          - pause_check():   True → 进 refresh 屏障 (排空在飞 → refresh_workers → on_resume → 续)。
          - on_resume():     屏障后调 (调用方清 pause 标志)。
        """
        pause_check = pause_check or (lambda: False)
        if self._use_process:
            self._run_stream_process(next_genotype, on_result, should_continue,
                                     pause_check, on_resume)
        else:
            self._run_stream_thread(next_genotype, on_result, should_continue,
                                    pause_check, on_resume)

    def _run_stream_thread(self, next_genotype, on_result, should_continue,
                           pause_check, on_resume) -> None:
        """线程后端流式 (mock/CPU 测试): 一卡一架构由 _device_pool 保证 (同 _eval_one)。"""
        with ThreadPoolExecutor(max_workers=self.num_devices) as pool:
            in_flight: dict = {}          # future -> genotype

            def _fill() -> None:
                while len(in_flight) < self.num_devices:
                    if pause_check():
                        break             # refresh 待定: 停止提交, 让在飞排空
                    g = next_genotype()
                    if g is None:
                        break
                    in_flight[pool.submit(self._eval_one, g)] = g

            _fill()
            while True:
                if not in_flight:
                    _fill()               # 屏障后/冷启动: 重新 prime
                    if not in_flight:
                        break             # 无在飞且无新任务 (next_genotype 枯竭) → 结束
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in done:
                    if fut not in in_flight:
                        continue          # 已被 _drain 消费 (同批 done 多 future 时防重复)
                    g = in_flight.pop(fut)
                    res = self._safe_future_result(fut, g)
                    with self._lock:      # 回调串行化 (同批模式 238)
                        on_result(res)
                    if not should_continue():
                        self._drain(in_flight, on_result, lock=self._lock)
                        return
                    if pause_check():
                        # 屏障: 排空在飞 → refresh → 恢复 (线程后端 refresh 是 no-op, 仍走流程保一致)
                        self._drain(in_flight, on_result, lock=self._lock)
                        self.refresh_workers()
                        if on_resume is not None:
                            on_resume()
                        break             # 回外层 while, 顶部重新 prime
                    _fill()

    def _run_stream_process(self, next_genotype, on_result, should_continue,
                            pause_check, on_resume) -> None:
        """进程后端流式 (真实多卡): 持久 spawn 池, round-robin 绑卡 (同 _run_batch_process:274)。"""
        from concurrent.futures.process import BrokenProcessPool

        n = self.num_devices
        in_flight: dict = {}              # future -> genotype
        rr = 0

        def _fill() -> None:
            nonlocal rr
            while len(in_flight) < n:
                if pause_check():
                    break
                g = next_genotype()
                if g is None:
                    break
                device = self.devices[rr % n]
                fut = self._pool.submit(_process_worker, self.eval_spec, n, device, g)
                in_flight[fut] = g
                rr += 1

        self._ensure_pool()
        try:
            _fill()
            while True:
                if not in_flight:
                    _fill()               # 屏障后/冷启动: 重新 prime
                    if not in_flight:
                        break             # 无在飞且无新任务 → 结束
                done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for fut in done:
                    if fut not in in_flight:
                        continue          # 已被 _drain 消费
                    g = in_flight.pop(fut)
                    res = self._safe_future_result(fut, g)
                    on_result(res)        # 主进程本就串行
                    if not should_continue():
                        self._drain(in_flight, on_result)
                        return
                    if pause_check():
                        self._drain(in_flight, on_result)
                        self.refresh_workers()
                        if on_resume is not None:
                            on_resume()
                        self._ensure_pool()   # refresh 后重建池, 供下轮 submit
                        break             # 回外层 while, 顶部重新 prime
        except BrokenProcessPool as e:
            # 整池坏: 在飞全标 CRASH + 重建池, 由调用方停止判定决定是否续 (自主性: 崩不中止)
            for fut, g in list(in_flight.items()):
                res = EvalResult(genotype=g, status="CRASH",
                                 fail_reason=f"BrokenProcessPool: {e}")
                on_result(res)
            in_flight.clear()
            self.refresh_workers()

    def _safe_future_result(self, fut, genotype: Genotype) -> EvalResult:
        """取 future 结果, future 层异常兜底 CRASH (worker 内已兜底, 这里防池层异常)。"""
        try:
            return fut.result()
        except Exception as e:
            return EvalResult(genotype=genotype, status="CRASH",
                              fail_reason=f"future: {type(e).__name__}: {e}")

    def _drain(self, in_flight: dict, on_result, lock=None) -> None:
        """排空剩余在飞 future: 全部等到完成并 digest (不丢结果), 清空 in_flight。"""
        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                g = in_flight.pop(fut)
                res = self._safe_future_result(fut, g)
                if lock is not None:
                    with lock:
                        on_result(res)
                else:
                    on_result(res)

