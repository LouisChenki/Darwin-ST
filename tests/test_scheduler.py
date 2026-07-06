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


# ---------------------------------------------------------------------------
# run_stream 流式驱动 (破批同步: 4 卡持续满载, 慢架构不阻塞快架构)
# ---------------------------------------------------------------------------


def test_run_stream_keeps_pool_full():
    """核心修复: 峰值并发==设备数; 慢架构不阻塞快架构 (快的先完成)。"""
    peak = [0]; cur = [0]; lk = threading.Lock(); order = []

    def eval_fn(g, dev):
        with lk:
            cur[0] += 1; peak[0] = max(peak[0], cur[0])
        time.sleep(0.4 if g.hidden == 999 else 0.03)   # hidden=999 是慢架构
        with lk:
            cur[0] -= 1
        return EvalResult(genotype=g, status="OK", mae=20.0, device=dev, extra={"num_params": 5000})

    sched = GPUScheduler(eval_fn, devices=4, backend="thread")
    from darwin_st.search.genotype import Genotype, STBlock
    genos = ([Genotype(blocks=[STBlock("gcn", "tcn")], hidden=999)]
             + [Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64) for _ in range(11)])
    it = iter(genos); results = []
    sched.run_stream(lambda: next(it, None),
                     lambda r: (order.append(r.genotype.hidden), results.append(r)),
                     should_continue=lambda: True)
    assert len(results) == 12
    assert peak[0] == 4, f"峰值并发 {peak[0]} != 4 (没满载)"
    assert order[0] != 999, "慢架构最先完成 (它阻塞了快的)"
    assert 999 in order, "慢架构最终也要完成"


def test_run_stream_stop_drains_inflight():
    """停止时排空在飞: should_continue 翻 False 后, 在飞的仍 digest (不丢结果)。"""
    def eval_fn(g, dev):
        time.sleep(0.03)
        return EvalResult(genotype=g, status="OK", mae=20.0, device=dev, extra={"num_params": 5000})
    sched = GPUScheduler(eval_fn, devices=4, backend="thread")
    from darwin_st.search.genotype import Genotype, STBlock
    it = iter([Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64) for _ in range(20)])
    n = [0]
    sched.run_stream(lambda: next(it, None), lambda r: n.__setitem__(0, n[0] + 1),
                     should_continue=lambda: n[0] < 3)
    # 停在第3个完成, 但 ≤num_devices-1 个在飞排空 → 3..6, 不丢
    assert 3 <= n[0] <= 3 + sched.num_devices - 1


def test_run_stream_refresh_barrier():
    """refresh 屏障: pause_check True → 排空在飞到 0 才调 refresh_workers → on_resume 后续跑。"""
    calls = {"refresh": 0}
    def eval_fn(g, dev):
        time.sleep(0.02)
        return EvalResult(genotype=g, status="OK", mae=20.0, device=dev, extra={"num_params": 5000})
    sched = GPUScheduler(eval_fn, devices=2, backend="thread")
    sched.refresh_workers = lambda: calls.__setitem__("refresh", calls["refresh"] + 1)
    from darwin_st.search.genotype import Genotype, STBlock
    it = iter([Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64) for _ in range(8)])
    done = [0]; paused = [False]
    def on_res(r):
        done[0] += 1
        if done[0] == 2:
            paused[0] = True   # 第2个完成后触发屏障
    sched.run_stream(lambda: next(it, None), on_res, should_continue=lambda: done[0] < 8,
                     pause_check=lambda: paused[0], on_resume=lambda: paused.__setitem__(0, False))
    assert calls["refresh"] >= 1, "屏障未调 refresh_workers"
    assert done[0] == 8, "屏障后未继续跑完 (seed/后续没恢复)"


def test_run_stream_exhausts_source():
    """next_genotype 枯竭 (返 None) → 排空在飞后干净结束, 每个都 digest 恰好一次。"""
    def eval_fn(g, dev):
        return EvalResult(genotype=g, status="OK", mae=20.0, device=dev, extra={"num_params": 5000})
    sched = GPUScheduler(eval_fn, devices=4, backend="thread")
    it = iter(_genos(7)); seen = []
    sched.run_stream(lambda: next(it, None), lambda r: seen.append(r), should_continue=lambda: True)
    assert len(seen) == 7   # 恰好 7 个, 无丢无重
