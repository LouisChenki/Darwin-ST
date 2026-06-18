"""scheduler.py 的正确性回归测试 (无 GPU, 用 fake eval_fn)。

核心契约:
  - run_batch 评估所有架构, 结果与输入对应
  - 并发上限 = 设备数 (同时在跑的不超过 GPU 数)
  - 一卡同时只跑一个架构 (设备池互斥)
  - 失败被兜底为 CRASH, 不让调度崩
  - on_result 回调每个完成都触发
  - resolve_devices 规范化设备
"""

from __future__ import annotations

import threading
import time

from darwin_st.optim.scheduler import EvalResult, GPUScheduler, resolve_devices
from darwin_st.search.genotype import random_genotype


def _genos(n):
    # 用不同 spatial 保证签名不同
    pool = ["gcn", "gat", "cheb", "diffusion", "adaptive"]
    return [random_genotype(depth=1, spatial=pool[i % len(pool)]) for i in range(n)]


# ---------------------------------------------------------------------------
# resolve_devices
# ---------------------------------------------------------------------------


def test_resolve_devices_int():
    assert resolve_devices(4) == ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]


def test_resolve_devices_list_passthrough():
    assert resolve_devices(["cuda:0", "cuda:2"]) == ["cuda:0", "cuda:2"]


# ---------------------------------------------------------------------------
# run_batch 基本
# ---------------------------------------------------------------------------


def test_run_batch_evaluates_all():
    def fake_eval(g, device):
        return EvalResult(genotype=g, status="OK", mae=18.0, device=device)

    sched = GPUScheduler(fake_eval, devices=["cpu0", "cpu1"])
    genos = _genos(5)
    results = sched.run_batch(genos)
    assert len(results) == 5
    assert all(r.status == "OK" for r in results)
    # 结果与输入一一对应 (顺序保持)
    for g, r in zip(genos, results):
        assert r.genotype.signature() == g.signature()


def test_run_batch_empty():
    sched = GPUScheduler(lambda g, d: EvalResult(g, "OK"), devices=2)
    assert sched.run_batch([]) == []


# ---------------------------------------------------------------------------
# 并行度 + 设备互斥
# ---------------------------------------------------------------------------


def test_concurrency_capped_at_device_count():
    """同时在跑的架构数不超过设备数。"""
    n_devices = 2
    active = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake_eval(g, device):
        with lock:
            active["now"] += 1
            active["peak"] = max(active["peak"], active["now"])
        time.sleep(0.05)  # 模拟训练耗时, 制造重叠
        with lock:
            active["now"] -= 1
        return EvalResult(genotype=g, status="OK", device=device)

    sched = GPUScheduler(fake_eval, devices=n_devices)
    sched.run_batch(_genos(8))
    assert active["peak"] <= n_devices, f"并发 {active['peak']} 超过设备数 {n_devices}"
    assert active["peak"] == n_devices  # 确实用满了并行


def test_one_arch_per_device():
    """同一设备同一时刻只跑一个架构 (设备池互斥)。"""
    seen_concurrent = {"bad": False}
    busy = set()
    lock = threading.Lock()

    def fake_eval(g, device):
        with lock:
            if device in busy:
                seen_concurrent["bad"] = True  # 同卡并发 = 错误
            busy.add(device)
        time.sleep(0.03)
        with lock:
            busy.discard(device)
        return EvalResult(genotype=g, status="OK", device=device)

    sched = GPUScheduler(fake_eval, devices=["cuda:0", "cuda:1"])
    sched.run_batch(_genos(6))
    assert not seen_concurrent["bad"], "同一卡上并发跑了多个架构!"


def test_actually_parallel_faster():
    """并行应比串行快 (4 任务各 0.1s, 2 卡应 ~0.2s 而非 0.4s)。"""
    def slow_eval(g, device):
        time.sleep(0.1)
        return EvalResult(genotype=g, status="OK", device=device)

    sched = GPUScheduler(slow_eval, devices=2)
    t0 = time.time()
    sched.run_batch(_genos(4))
    elapsed = time.time() - t0
    assert elapsed < 0.35  # 串行需 0.4s; 并行 2 卡约 0.2s


# ---------------------------------------------------------------------------
# 失败兜底
# ---------------------------------------------------------------------------


def test_crash_does_not_break_scheduling():
    """某架构评估抛异常 → 兜底为 CRASH, 其余正常完成。"""
    def flaky_eval(g, device):
        if g.blocks[0].spatial_op == "gat":
            raise RuntimeError("模拟 OOM")
        return EvalResult(genotype=g, status="OK", mae=18.0, device=device)

    sched = GPUScheduler(flaky_eval, devices=2)
    results = sched.run_batch(_genos(5))
    assert len(results) == 5  # 全部有结果, 没崩
    crashed = [r for r in results if r.status == "CRASH"]
    ok = [r for r in results if r.status == "OK"]
    assert len(crashed) >= 1
    assert len(ok) >= 1
    assert all(r.fail_reason for r in crashed)  # 崩的有死因


def test_device_returned_after_crash():
    """崩溃后设备应归还池 (否则会逐渐耗尽卡)。"""
    calls = {"n": 0}

    def always_crash_first(g, device):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("boom")
        return EvalResult(genotype=g, status="OK", device=device)

    sched = GPUScheduler(always_crash_first, devices=2)
    # 若崩溃不归还卡, 2 卡崩完就死锁; 能完成说明归还了
    results = sched.run_batch(_genos(6))
    assert len(results) == 6


# ---------------------------------------------------------------------------
# on_result 回调
# ---------------------------------------------------------------------------


def test_on_result_callback():
    received = []

    def fake_eval(g, device):
        return EvalResult(genotype=g, status="OK", mae=18.0, device=device)

    sched = GPUScheduler(fake_eval, devices=2, on_result=lambda r: received.append(r))
    sched.run_batch(_genos(4))
    assert len(received) == 4
