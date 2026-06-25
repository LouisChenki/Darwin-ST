"""Tier-1 ↔ Tier-2 衔接集成测试 (进程后端) —— 项目核心是两层一体, 衔接必须经得起多进程。

最险的衔接点③: Tier-2 在主进程合成并注册 synth 算子 (注入主进程 SPATIAL_OPS + 持久化到盘),
但 Tier-1 的真实评测在 **spawn 子进程**, 其 SPATIAL_OPS 是全新的、不含 synth 算子。
设计靠: register 持久化 → orchestrator refresh_workers 杀旧池 → 下轮新 worker load_persisted
从盘补 synth → build 能解析。本测试在 CPU 上真 spawn 把这条链钉死 (无 GPU/数据/LLM)。
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

from darwin_st.optim.scheduler import EvalSpec, GPUScheduler
from darwin_st.search.genotype import Genotype, STBlock

PROBE = "_synth_probe_backend:make_synth_probe_eval"

# 一个最小合成算子 (nn.Module, 签名 channels/num_nodes), 写进 persist_dir 模拟 Tier-2 register 产物
_SYNTH_CODE = textwrap.dedent('''
    import torch.nn as nn
    class MySynthOp(nn.Module):
        def __init__(self, channels, num_nodes, **kw):
            super().__init__()
            self.proj = nn.Linear(channels, channels)
        def forward(self, x, adj=None):
            return self.proj(x)
''').strip()


def _write_persisted_synth(persist_dir: str, reg_name: str = "synth_MySynthOp") -> str:
    """模拟 OperatorRegistry._persist: 写 <reg_name>.py + .json 到盘。"""
    os.makedirs(persist_dir, exist_ok=True)
    with open(os.path.join(persist_dir, reg_name + ".py"), "w") as f:
        f.write(_SYNTH_CODE)
    with open(os.path.join(persist_dir, reg_name + ".json"), "w") as f:
        json.dump({"reg_name": reg_name, "class_name": "MySynthOp",
                   "needs_adj": False, "plan_name": "my_synth",
                   "composition": "test", "source_mechanisms": ["x"], "real_mae": None}, f)
    return reg_name


def _synth_genotype(reg_name: str) -> Genotype:
    """一个用 synth 算子作 spatial_op 的 genotype (Tier-2 产出的 seed 形态)。"""
    return Genotype(blocks=[STBlock(spatial_op=reg_name, temporal_op="tcn")], hidden=16)


def test_spawn_worker_recovers_synth_op_from_disk(tmp_path):
    """衔接点③正路: persist_dir 给定 → spawn worker load_persisted 补回 synth → 解析成功 OK。

    这正是创造后下一轮评测走的路径 (refresh_workers 后新 worker 自建)。
    """
    persist_dir = str(tmp_path / "dynamic_ops")
    reg_name = _write_persisted_synth(persist_dir)

    spec = EvalSpec(dataset="PeMS04", hpo_cfg=None,
                    synth_persist_dir=persist_dir, eval_factory=PROBE)
    sched = GPUScheduler(eval_fn=None, devices=["cpu", "cpu"], eval_spec=spec, backend="process")
    try:
        results = sched.run_batch([_synth_genotype(reg_name), _synth_genotype(reg_name)])
        assert len(results) == 2
        for r in results:
            assert r.status == "OK", f"衔接断裂: {r.fail_reason}"
            assert r.extra["resolved_synth"] == reg_name
            assert r.extra["synth_in_worker"] is True      # synth 算子确实在 worker 进程里
            assert r.extra["num_params"] > 0               # 真的实例化了
    finally:
        sched.close()


def test_spawn_worker_without_persist_dir_crashes(tmp_path):
    """负对照: 不给 synth_persist_dir → worker SPATIAL_OPS 没 synth 算子 → 评测 CRASH (不静默假成功)。

    证明"正路成功"是 load_persisted 的功劳, 不是 worker 碰巧有算子。
    """
    persist_dir = str(tmp_path / "dynamic_ops")
    reg_name = _write_persisted_synth(persist_dir)

    # 故意不传 synth_persist_dir
    spec = EvalSpec(dataset="PeMS04", hpo_cfg=None,
                    synth_persist_dir=None, eval_factory=PROBE)
    sched = GPUScheduler(eval_fn=None, devices=["cpu"], eval_spec=spec, backend="process")
    try:
        results = sched.run_batch([_synth_genotype(reg_name)])
        assert len(results) == 1
        assert results[0].status == "CRASH"               # worker 解析不到 synth → KeyError 兜底
        assert reg_name in (results[0].fail_reason or "")
    finally:
        sched.close()


def test_refresh_workers_picks_up_newly_persisted_synth(tmp_path):
    """衔接点③时序: 第一批无 synth (空池) → 持久化新 synth + refresh_workers → 第二批解析成功。

    精确复刻 orchestrator: 停滞→创造(持久化新算子)→refresh_workers→下轮评测含 synth 的 seed。
    """
    persist_dir = str(tmp_path / "dynamic_ops")
    os.makedirs(persist_dir, exist_ok=True)   # 先空目录

    spec = EvalSpec(dataset="PeMS04", hpo_cfg=None,
                    synth_persist_dir=persist_dir, eval_factory=PROBE)
    sched = GPUScheduler(eval_fn=None, devices=["cpu"], eval_spec=spec, backend="process")
    try:
        # 模拟创造: 主进程注册新 synth 算子 → 持久化到盘
        reg_name = _write_persisted_synth(persist_dir)
        # 模拟 orchestrator._try_creation 成功后: 回收旧池, 下批 worker 重载最新
        sched.refresh_workers()
        # 下一轮评测含 synth 的 seed genotype
        results = sched.run_batch([_synth_genotype(reg_name)])
        assert results[0].status == "OK", f"refresh 后仍解析不到 synth: {results[0].fail_reason}"
        assert results[0].extra["resolved_synth"] == reg_name
    finally:
        sched.close()
