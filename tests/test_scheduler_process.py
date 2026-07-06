"""进程后端 (ProcessPoolExecutor + spawn) 测试 —— 无 GPU, CPU 即可跑通整条回路。

线程后端的覆盖在 test_scheduler.py。这里专测进程后端: EvalSpec.eval_factory 注入
可导入的 fake eval (见 _fake_eval_backend.py), 让 spawn 子进程自建 eval_fn, 从而验证:
  - 真跨进程执行 (worker pid ≠ 主进程)
  - 取模绑卡 (devices[i % n]) 正确
  - 持久池复用 + refresh_workers (创造后重建) + close 生命周期
  - worker 兜底 CRASH (子进程抛异常绝不传播坏池)
  - backend 解析 (auto/thread/process) + EvalSpec 可 pickle (跨进程铁律)
"""

from __future__ import annotations

import os
import pickle

import pytest

from darwin_st.optim.scheduler import EvalResult, EvalSpec, GPUScheduler
from darwin_st.search.genotype import random_genotype

from _fake_eval_backend import CRASH_HIDDEN

FACTORY = "_fake_eval_backend:make_fake_eval"


def _spec():
    return EvalSpec(dataset="PeMS04", hpo_cfg=None, eval_factory=FACTORY)


def _make_sched(devices, on_result=None):
    return GPUScheduler(eval_fn=None, devices=devices, on_result=on_result,
                        eval_spec=_spec(), backend="process")


def test_eval_spec_is_picklable():
    """跨 spawn 进程铁律: EvalSpec 必须可 pickle。"""
    spec = _spec()
    spec.synth_persist_dir = "/tmp/whatever"
    back = pickle.loads(pickle.dumps(spec))
    assert back.dataset == "PeMS04"
    assert back.eval_factory == FACTORY


def test_resolve_backend_logic():
    """auto: 有 eval_spec 且全 cuda → process; 无 spec / 含 cpu → thread。显式 process 无 spec → 报错。"""
    # auto + cpu 设备 → 线程 (本测试机无 GPU, auto 不会误入进程)
    s = GPUScheduler(eval_fn=lambda g, d: None, devices=["cpu"], eval_spec=_spec())
    assert s._use_process is False
    # auto + cuda 设备 + spec → 进程
    s = GPUScheduler(eval_fn=None, devices=["cuda:0", "cuda:1"], eval_spec=_spec())
    assert s._use_process is True
    # auto + cuda 但无 spec → 线程 (回退; 385 mock 测试走这条)
    s = GPUScheduler(eval_fn=lambda g, d: None, devices=["cuda:0"])
    assert s._use_process is False
    # 显式 process 但无 spec → 报错
    with pytest.raises(ValueError):
        GPUScheduler(eval_fn=None, devices=["cpu"], backend="process")
    # 显式 thread 即便给了 spec 也走线程
    s = GPUScheduler(eval_fn=None, devices=["cuda:0"], eval_spec=_spec(), backend="thread")
    assert s._use_process is False


def test_process_backend_runs_cross_process():
    """进程后端真跑: 结果回填、worker 在别的进程、绑到指定 device。"""
    sched = _make_sched(["cpu", "cpu", "cpu"])
    try:
        genos = [random_genotype(depth=1, spatial=s, hidden=64)
                 for s in ("gcn", "adaptive", "cheb", "gcn", "adaptive")]
        results = sched.run_batch(genos)
        assert len(results) == len(genos)
        assert all(isinstance(r, EvalResult) for r in results)
        assert all(r.status == "OK" for r in results)
        assert all(r.mae < float("inf") for r in results)
        # 真跨进程: worker pid 必须 ≠ 主进程 pid
        pids = {r.extra["worker_pid"] for r in results}
        assert all(p != os.getpid() for p in pids)
        # 绑卡: 第 i 个架构绑 devices[i % n] (n=3)
        devs = ["cpu", "cpu", "cpu"]
        for i, r in enumerate(results):
            assert r.extra["seen_device"] == devs[i % 3]
    finally:
        sched.close()


def test_process_backend_crash_is_contained():
    """worker 抛异常 → 兜底成 CRASH EvalResult, 绝不传播; 同批其余正常完成。"""
    sched = _make_sched(["cpu", "cpu"])
    try:
        genos = [
            random_genotype(depth=1, spatial="gcn", hidden=64),
            random_genotype(depth=1, spatial="gcn", hidden=CRASH_HIDDEN),  # 哨兵 → 抛
            random_genotype(depth=1, spatial="adaptive", hidden=64),
        ]
        results = sched.run_batch(genos)
        assert len(results) == 3
        status = {r.genotype.hidden: r.status for r in results}
        assert status[CRASH_HIDDEN] == "CRASH"
        assert status[64] == "OK"
        crashed = [r for r in results if r.status == "CRASH"][0]
        assert "sentinel crash" in (crashed.fail_reason or "")
    finally:
        sched.close()


def test_on_result_callback_fires_in_main_process():
    """on_result 在主进程串行回调 (回填共享状态安全)。"""
    seen = []
    sched = _make_sched(["cpu", "cpu"], on_result=lambda r: seen.append(r))
    try:
        genos = [random_genotype(depth=1, spatial="gcn", hidden=64) for _ in range(4)]
        sched.run_batch(genos)
        assert len(seen) == 4
        assert all(isinstance(r, EvalResult) for r in seen)
    finally:
        sched.close()


def test_refresh_and_reuse_pool():
    """池复用 + refresh_workers 重建 (创造注入新算子后调) + close 幂等。"""
    sched = _make_sched(["cpu", "cpu"])
    try:
        g = [random_genotype(depth=1, spatial="gcn", hidden=64) for _ in range(2)]
        r1 = sched.run_batch(g)
        assert len(r1) == 2 and sched._pool is not None
        # refresh: 回收旧池, 下批惰性重建
        sched.refresh_workers()
        assert sched._pool is None
        r2 = sched.run_batch(g)
        assert len(r2) == 2 and sched._pool is not None
    finally:
        sched.close()
    assert sched._pool is None
    sched.close()  # 幂等: 再调不报错


def test_empty_batch_returns_empty():
    sched = _make_sched(["cpu"])
    try:
        assert sched.run_batch([]) == []
    finally:
        sched.close()


# ---------------------------------------------------------------------------
# run_stream 进程后端 (流式真跨进程 + 崩溃兜底)
# ---------------------------------------------------------------------------


def test_run_stream_process_cross_process():
    """流式进程后端: 真跨进程评估, 每个恰好 digest 一次, 结果回填。"""
    sched = _make_sched(["cpu", "cpu", "cpu"])
    try:
        genos = [random_genotype(depth=1, spatial=s, hidden=64)
                 for s in ("gcn", "adaptive", "cheb", "gcn", "adaptive", "cheb", "gcn")]
        it = iter(genos); seen = []
        sched.run_stream(lambda: next(it, None), lambda r: seen.append(r),
                         should_continue=lambda: True)
        assert len(seen) == len(genos)
        assert all(r.status == "OK" for r in seen)
        pids = {r.extra["worker_pid"] for r in seen}
        assert all(p != os.getpid() for p in pids)   # 真跨进程
    finally:
        sched.close()


def test_run_stream_process_crash_contained():
    """流式下 worker 抛异常 → 兜底 CRASH, 不传播, 其余正常; 每个仍 digest 一次。"""
    sched = _make_sched(["cpu", "cpu"])
    try:
        genos = [
            random_genotype(depth=1, spatial="gcn", hidden=64),
            random_genotype(depth=1, spatial="gcn", hidden=CRASH_HIDDEN),  # 哨兵 → 抛
            random_genotype(depth=1, spatial="adaptive", hidden=64),
            random_genotype(depth=1, spatial="cheb", hidden=64),
        ]
        it = iter(genos); seen = []
        sched.run_stream(lambda: next(it, None), lambda r: seen.append(r),
                         should_continue=lambda: True)
        assert len(seen) == 4
        status = {r.genotype.hidden: r.status for r in seen}
        assert status[CRASH_HIDDEN] == "CRASH"
        assert sum(1 for r in seen if r.status == "OK") == 3
    finally:
        sched.close()
