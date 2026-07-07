"""train.py 的正确性回归测试 (用合成小数据, 本地 CPU, 无下载/无 GPU)。

验证真实训练-评估引擎的契约:
  - train_one: 能在合成数据上训练并返回有限 MAE; NaN/时间熔断生效
  - evaluate_architecture: 返回 EvalResult 含 num_params; HPO 跑通
  - make_eval_fn → orchestrator 可用的 eval_fn
不依赖真实数据集下载 (monkeypatch CACHE_DIR + 预置 tiny npy)。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from darwin_st.data import prepare as P
from darwin_st.data.protocol import get_profile
from darwin_st.optim.hpo import HPOConfig
from darwin_st.optim.train import evaluate_architecture, train_one
from darwin_st.search.genotype import random_genotype


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """预置一个 tiny 的已处理数据集 (10 节点, 少量样本), 跳过下载。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
    # 自定义一个迷你 profile (10 节点) 注入 PROFILES
    from darwin_st.data.protocol import DatasetProfile, PROFILES
    prof = DatasetProfile(
        name="TINY", num_nodes=10, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    )
    monkeypatch.setitem(PROFILES, "TINY", prof)

    data_dir = P.data_dir_for("TINY")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    N, C = 10, 3
    for split, n in (("train", 64), ("val", 24), ("test", 24)):
        x = rng.random((n, 12, N, C)).astype(np.float32)
        y = rng.random((n, 12, N)).astype(np.float32)
        np.save(os.path.join(data_dir, f"{split}_x.npy"), x)
        np.save(os.path.join(data_dir, f"{split}_y.npy"), y)
    np.save(os.path.join(data_dir, "scaler.npy"), np.array([50.0, 20.0]))
    # 简单邻接
    adj = np.eye(N, dtype=np.float32)
    np.save(os.path.join(data_dir, "adj.npy"), adj)
    return prof, data_dir, adj


def test_train_one_returns_finite_mae(tiny_data):
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    mae, trace = train_one(geno, {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16},
                           data_dir, prof, adj, device="cpu", max_epochs=2)
    assert np.isfinite(mae)
    assert mae > 0  # 真实尺度 MAE
    # TrainTrace 采集: 逐 epoch 曲线长度 = 实际 epoch 数, 梯度范数被捡回 (非空)
    assert trace.n_epochs_run == 2
    assert len(trace.train_loss) == 2 and len(trace.val_mae) == 2
    assert len(trace.grad_norm_mean) == 2 and len(trace.grad_norm_max) == 2
    assert all(g >= 0 for g in trace.grad_norm_mean)   # 梯度范数非负 (clip 前)
    assert 0 <= trace.best_epoch < 2
    assert trace.lr == 1e-3 and trace.lr_schedule == "none"
    assert not trace.nan_hit


def test_train_one_with_tod_dow_and_lr_schedule(tiny_data):
    """tod/dow 时间索引接通 + lr_schedule=cosine: 4 元组路径走通, 返回有限 MAE。

    锁住冲 SOTA 的两项关键能力: STID 身份嵌入数据链 + HPO 可择优的 lr schedule。
    """
    prof, data_dir, adj = tiny_data
    # 给 tiny_data 补 tod/dow 索引 (触发 load_split 的 4 元组路径 + STEmbedding 真实时间嵌入)
    for split, n in (("train", 64), ("val", 24), ("test", 24)):
        tod = (np.arange(n)[:, None] + np.arange(12)[None, :]) % 288
        dow = ((np.arange(n)[:, None] + np.arange(12)[None, :]) // 288) % 7
        np.save(os.path.join(data_dir, f"{split}_tod.npy"), tod.astype(np.int64))
        np.save(os.path.join(data_dir, f"{split}_dow.npy"), dow.astype(np.int64))
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    # geno 默认 use_tod/use_dow=True → 模型会消费 tod/dow
    mae, trace = train_one(geno, {"lr": 1e-2, "batch_size": 16, "lr_schedule": "cosine"},
                           data_dir, prof, adj, device="cpu", max_epochs=3)
    assert np.isfinite(mae) and mae > 0
    assert trace.lr_schedule == "cosine" and trace.n_epochs_run == 3


def test_train_one_time_budget(tiny_data):
    """时间熔断: 预算 0 秒 → 至多跑 1 epoch 即停, 仍返回有限值。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, hidden=16)
    mae, trace = train_one(geno, {"lr": 1e-3, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=50, time_budget_s=0.0)
    assert np.isfinite(mae)  # 第一个 epoch 后熔断
    assert trace.stopped_early and trace.n_epochs_run <= 2


def test_evaluate_architecture_returns_result(tiny_data):
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    cfg = HPOConfig(n_trials=2, max_epochs=2, min_resource=1, n_startup_trials=1)
    res = evaluate_architecture(geno, "cpu", data_dir, prof, adj, hpo_cfg=cfg)
    assert res.status == "OK"
    assert np.isfinite(res.mae)
    assert res.extra["num_params"] > 0      # 参数量供 archive
    assert res.hps  # 最优超参非空


def test_make_eval_fn_routes_seed_meta(tiny_data, monkeypatch):
    """A: 创造 seed 带 _seed_meta → eval_fn 把 warm_start_hps + n_trials 透传给 evaluate_architecture;
    普通架构(无 meta)走默认(warm_start_hps=None, n_trials=None)。"""
    import darwin_st.optim.train as train_mod
    from darwin_st.optim.scheduler import EvalResult
    from darwin_st.optim.train import make_eval_fn

    captured = []

    def fake_eval(genotype, device, data_dir, profile, adj, hpo_cfg=None,
                  warm_start_hps=None, n_trials=None):
        captured.append({"warm_start_hps": warm_start_hps, "n_trials": n_trials})
        return EvalResult(genotype=genotype, status="OK", mae=10.0, device=device,
                          extra={"num_params": 1})

    monkeypatch.setattr(train_mod, "evaluate_architecture", fake_eval)
    cfg = HPOConfig(n_trials=3, max_epochs=2)
    eval_fn = make_eval_fn("TINY", hpo_cfg=cfg)   # tiny_data 已预置 TINY 处理好的 npy + 注入 PROFILES

    # 普通架构: 无 _seed_meta → 默认
    g_plain = random_genotype(depth=1, spatial="gcn", hidden=16)
    eval_fn(g_plain, "cpu")
    assert captured[-1] == {"warm_start_hps": None, "n_trials": None}

    # 创造 seed: 带 _seed_meta → 透传
    g_seed = random_genotype(depth=1, spatial="gcn", hidden=16)
    g_seed._seed_meta = {"warm_start_hps": {"lr": 1e-3}, "hpo_trials": 20, "is_creation_seed": True}
    eval_fn(g_seed, "cpu")
    assert captured[-1] == {"warm_start_hps": {"lr": 1e-3}, "n_trials": 20}


def test_evaluate_architecture_as_orchestrator_eval_fn(tiny_data):
    """evaluate_architecture 闭包可直接当 orchestrator 的 eval_fn 用。"""
    prof, data_dir, adj = tiny_data
    cfg = HPOConfig(n_trials=2, max_epochs=2, min_resource=1, n_startup_trials=1)

    def eval_fn(geno, device):
        return evaluate_architecture(geno, device, data_dir, prof, adj, hpo_cfg=cfg)

    from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
    ocfg = OrchestratorConfig(dataset="PeMS04", population_size=3, tournament_size=2,
                              max_rounds=2, target_mae=0.0)  # 不可达 target → max_rounds 停
    orch = Orchestrator(ocfg, eval_fn, devices=1)
    state = orch.run()
    assert state.rounds == 2
    assert state.evals >= 2


# ---------------------------------------------------------------------------
# 收敛早停 (Part A: 依训练动态自适应训练预算, 多 benchmark 地基)
# ---------------------------------------------------------------------------


def test_train_one_early_stops_on_plateau(tiny_data):
    """lr=0 → 权重不更新 → val 立即平台 → 收敛早停切在 patience 后, best 已存零损失。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, hidden=16)
    mae, trace = train_one(geno, {"lr": 0.0, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=30, early_stop_patience=3)
    assert trace.converged, "平台未触发收敛早停"
    assert trace.n_epochs_run < 30, f"没提前停 (跑满 {trace.n_epochs_run})"
    assert np.isfinite(mae) and abs(mae - trace.best_mae) < 1e-9, "返回值非 best"


def test_train_one_no_early_stop_when_improving(tiny_data):
    """正常 lr 训练几 epoch: 还在改进 (或没连续平台够 patience) → 不误停跑满。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    # patience 设得比 max_epochs 大 → 即使有噪声也不会触发 (验证不误停)
    mae, trace = train_one(geno, {"lr": 1e-2, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=5, early_stop_patience=10)
    assert not trace.converged
    assert trace.n_epochs_run == 5   # 跑满


def test_train_one_early_stop_off_by_default(tiny_data):
    """early_stop_patience=0 (默认) → 检测关, 跑满 max_epochs (向后兼容)。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, hidden=16)
    mae, trace = train_one(geno, {"lr": 0.0, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=6)   # 不传 early_stop_patience → 默认 0
    assert not trace.converged
    assert trace.n_epochs_run == 6   # 平台但默认关 → 跑满


def test_train_one_early_stop_reports_asha_every_epoch(tiny_data):
    """ASHA 共存: 即使收敛早停开, trial.report 仍每 epoch 调 (两机制正交不冲突)。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, hidden=16)

    class _StubTrial:
        def __init__(self): self.reports = []
        def report(self, mae, epoch): self.reports.append(epoch)
        def should_prune(self): return False   # 不剪, 让早停自己触发

    stub = _StubTrial()
    mae, trace = train_one(geno, {"lr": 0.0, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=20, trial=stub, early_stop_patience=3)
    assert trace.converged
    # report 调用数 == 实际跑的 epoch 数 (每 epoch 都 report, 早停不跳过 ASHA)
    assert len(stub.reports) == trace.n_epochs_run
