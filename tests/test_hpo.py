"""hpo.py 的正确性回归测试 (无 GPU/真实训练, 用 mock train_eval_fn)。

核心契约:
  - suggest_hps 定义正确搜索空间; 条件超参 num_heads 仅头数可配的注意力架构存在
  - optimize_architecture 找到接近最优的超参 (在合成 objective 上)
  - 剪枝/失败被正确计数, 不污染最优选择
  - 暖启动 enqueue 生效
  - study 配置 = TPE + ASHA
"""

from __future__ import annotations

import optuna
import pytest

from darwin_st.optim.hpo import (
    HPOConfig,
    make_study,
    optimize_architecture,
    suggest_hps,
)
from darwin_st.search.genotype import Genotype, STBlock, random_genotype


def _geno(spatial="gcn", temporal="tcn"):
    return Genotype(blocks=[STBlock(spatial, temporal)])


# ---------------------------------------------------------------------------
# 搜索空间
# ---------------------------------------------------------------------------


def test_suggest_hps_basic_keys():
    cfg = HPOConfig()
    study = optuna.create_study()
    trial = study.ask()
    hps = suggest_hps(trial, _geno("gcn", "tcn"), cfg)
    assert set(hps) >= {"lr", "weight_decay", "dropout", "batch_size"}
    assert "num_heads" not in hps  # 无注意力 → 无 num_heads


def test_suggest_hps_conditional_num_heads():
    """头数可配的注意力算子才有 num_heads (条件超参)。

    代码事实: attn/series_decomp_attn (时序槽) 与 st_graph_attn (一体槽) 的构造函数
    接受 num_heads; gat/dynamic_gat 头数写死不可配 → 不采 (采了也是死参数)。
    """
    cfg = HPOConfig()
    study = optuna.create_study()
    yes = [
        _geno("gcn", "attn"),
        _geno("gcn", "series_decomp_attn"),
        Genotype(blocks=[STBlock("identity", "identity", joint_op="st_graph_attn")]),
    ]
    for geno in yes:
        trial = study.ask()
        hps = suggest_hps(trial, geno, cfg)
        assert "num_heads" in hps, f"{geno.blocks[0]} 应采 num_heads"


def test_suggest_hps_no_num_heads_for_fixed_head_ops():
    """头数写死的注意力算子 (gat/dynamic_gat) 不采 num_heads —— 死参数清出搜索空间。"""
    cfg = HPOConfig()
    study = optuna.create_study()
    no = [
        _geno("gat", "tcn"),
        Genotype(blocks=[STBlock("identity", "identity", joint_op="dynamic_gat")]),
    ]
    for geno in no:
        trial = study.ask()
        hps = suggest_hps(trial, geno, cfg)
        assert "num_heads" not in hps, f"{geno.blocks[0]} 不应采 num_heads (头数不可配)"


def test_suggest_hps_lr_in_range():
    cfg = HPOConfig(lr_range=(1e-4, 1e-2))
    study = optuna.create_study()
    for _ in range(10):
        trial = study.ask()
        hps = suggest_hps(trial, _geno(), cfg)
        assert 1e-4 <= hps["lr"] <= 1e-2
        study.tell(trial, 1.0)


def test_suggest_hps_loss_dimension():
    """loss 作为 HPO 维度: 每 trial 采 mae/huber 之一; seeded 采样下两类都出现 (维度真实生效)。"""
    cfg = HPOConfig()
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen = set()
    for _ in range(10):
        trial = study.ask()
        hps = suggest_hps(trial, _geno(), cfg)
        assert hps["loss"] in {"mae", "huber"}
        seen.add(hps["loss"])
        study.tell(trial, 1.0)
    assert seen == {"mae", "huber"}


# ---------------------------------------------------------------------------
# study 配置
# ---------------------------------------------------------------------------


def test_make_study_is_tpe_asha():
    from optuna.pruners import SuccessiveHalvingPruner
    from optuna.samplers import TPESampler

    cfg = HPOConfig()
    study = make_study("test", cfg)
    assert isinstance(study.sampler, TPESampler)
    assert isinstance(study.pruner, SuccessiveHalvingPruner)  # = ASHA
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


# ---------------------------------------------------------------------------
# 端到端 HPO (mock 训练)
# ---------------------------------------------------------------------------


def test_optimize_finds_good_lr():
    """合成 objective: MAE 在 lr≈1e-3 处最小。HPO 应找到接近的 lr。"""
    import math

    def fake_train(geno, hps, trial):
        # 谷底在 lr=1e-3: |log10(lr) - (-3)| 越小越好
        return abs(math.log10(hps["lr"]) + 3.0) + 0.1 * hps["dropout"]

    cfg = HPOConfig(n_trials=40, seed=0, n_startup_trials=8)
    res = optimize_architecture(_geno(), fake_train, cfg)
    assert res.best_mae < 0.5             # 接近谷底
    assert 3e-4 < res.best_hps["lr"] < 3e-3  # lr 落在 1e-3 附近


def test_pruning_counted_and_not_best():
    """被剪枝的 trial 不应成为最优。"""
    def fake_train(geno, hps, trial):
        # 一半 trial 在中途被剪 (report 高 MAE 触发)
        for epoch in range(10):
            mae = 5.0 if hps["dropout"] > 0.25 else 1.0
            trial.report(mae, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return mae

    cfg = HPOConfig(n_trials=20, min_resource=1, reduction_factor=2, seed=1)
    res = optimize_architecture(_geno(), fake_train, cfg)
    # 最优应来自低 dropout 分支 (MAE≈1.0), 不是被剪的高 MAE
    assert res.best_mae <= 1.5
    assert res.n_complete >= 1


def test_failed_trial_handled():
    """训练抛异常 → 计为 FAIL, 不崩溃, 不影响最优。"""
    calls = {"n": 0}

    def flaky_train(geno, hps, trial):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise RuntimeError("模拟 OOM")
        return hps["dropout"]  # dropout 越小越好

    cfg = HPOConfig(n_trials=12, seed=0)
    res = optimize_architecture(_geno(), flaky_train, cfg)
    assert res.n_failed >= 1            # 有失败
    assert res.n_complete >= 1          # 也有成功
    assert res.best_mae < float("inf")  # 仍能给出最优


def test_warm_start_enqueue():
    """暖启动: 父超参作为第一个 trial 被评估。"""
    seen_lrs = []

    def fake_train(geno, hps, trial):
        seen_lrs.append(hps["lr"])
        return hps["lr"]

    cfg = HPOConfig(n_trials=5, seed=0)
    warm = {"lr": 0.0033, "weight_decay": 1e-5, "dropout": 0.1, "batch_size": 32}
    optimize_architecture(_geno(), fake_train, cfg, warm_start_hps=warm)
    # 第一个被评估的 lr 应是暖启动值
    assert abs(seen_lrs[0] - 0.0033) < 1e-9


def test_all_failed_returns_inf():
    """全部失败 → best_mae=inf, 不崩。"""
    def always_fail(geno, hps, trial):
        raise RuntimeError("boom")

    cfg = HPOConfig(n_trials=5, seed=0)
    res = optimize_architecture(_geno(), always_fail, cfg)
    assert res.best_mae == float("inf")
    assert res.best_hps == {}
    assert res.n_failed == 5
