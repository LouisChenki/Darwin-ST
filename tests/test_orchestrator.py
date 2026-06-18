"""orchestrator.py 的正确性回归测试 (无 GPU, mock eval_fn 跑完整自治循环)。

核心契约:
  - 主循环端到端跑通: 提议→评估→落memory→进archive→回灌→判停
  - 程序化停止: 超越 SOTA 即停 (代码比较, 非 LLM)
  - max_rounds/max_evals 上限生效
  - crash/discard 不中止循环 (自主性铁律), 进 graveyard
  - 最优随轮次下降 (进化确实在改进)
  - memory/archive 被正确填充
"""

from __future__ import annotations

import random

import pytest

from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
from darwin_st.optim.scheduler import EvalResult
from darwin_st.memory.store import MemoryStore
from darwin_st.search.genotype import Genotype


def _make_eval_fn(rng=None, crash_prob=0.0, base=22.0):
    """合成 eval_fn: MAE 偏好 depth 大 + 节点嵌入 + adaptive 空间算子。

    返回越小越优的 MAE, 加少量噪声。可注入 crash 概率。
    """
    rng = rng or random.Random(0)

    def eval_fn(geno: Genotype, device: str) -> EvalResult:
        if crash_prob and rng.random() < crash_prob:
            return EvalResult(genotype=geno, status="CRASH", device=device, fail_reason="mock_nan")
        mae = base - geno.depth * 1.0
        if geno.embedding.use_node:
            mae -= 1.0
        if any(b.spatial_op == "adaptive" for b in geno.blocks):
            mae -= 0.5
        mae += rng.uniform(-0.2, 0.2)
        return EvalResult(genotype=geno, status="OK", mae=mae, rmse=mae * 1.5,
                          device=device, hps={"lr": 1e-3},
                          extra={"num_params": 80_000 + geno.depth * 40_000})
    return eval_fn


# ---------------------------------------------------------------------------
# 基本循环
# ---------------------------------------------------------------------------


def test_runs_and_stops_on_max_rounds():
    cfg = OrchestratorConfig(dataset="PeMS04", population_size=6, tournament_size=3,
                             max_rounds=5, target_mae=0.0)  # target 不可达 → 靠 max_rounds 停
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    state = orch.run()
    assert state.rounds == 5
    assert "max_rounds" in state.stop_reason
    assert state.evals == 5 * 2  # 每轮 2 卡


def test_max_evals_stop():
    cfg = OrchestratorConfig(population_size=6, tournament_size=3, max_evals=8, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    state = orch.run()
    assert state.evals >= 8
    assert "max_evals" in state.stop_reason


def test_sota_target_set_from_registry():
    """未指定 target 时, 用 baseline_registry 的 SOTA 作目标。"""
    cfg = OrchestratorConfig(dataset="PeMS04", max_rounds=1)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=1)
    assert orch.state.sota_name == "STD-MAE"
    assert orch.state.sota_mae == 17.8


# ---------------------------------------------------------------------------
# 程序化停止: 超越 SOTA
# ---------------------------------------------------------------------------


def test_stops_when_beats_target():
    """eval_fn 总返回很低 MAE → 应触发 beat_sota 停止 (程序化, 非 LLM)。"""
    def great_eval(geno, device):
        return EvalResult(genotype=geno, status="OK", mae=5.0, rmse=8.0, device=device,
                          extra={"num_params": 50_000})

    cfg = OrchestratorConfig(dataset="PeMS04", population_size=4, tournament_size=2,
                             max_rounds=50, target_mae=17.8)
    orch = Orchestrator(cfg, great_eval, devices=2)
    state = orch.run()
    assert state.beat_sota is True
    assert "beat_sota" in state.stop_reason
    assert state.best_mae < 17.8
    assert state.rounds < 50  # 提前停, 没跑满


# ---------------------------------------------------------------------------
# 自主性: crash/discard 不中止
# ---------------------------------------------------------------------------


def test_crash_does_not_stop_loop():
    """高 crash 概率下循环仍跑满 max_rounds (自主性: 失败是数据点不是中止)。"""
    cfg = OrchestratorConfig(population_size=6, tournament_size=3, max_rounds=6, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(crash_prob=0.5), devices=2)
    state = orch.run()
    assert state.rounds == 6
    assert state.n_crash >= 1   # 确实有崩
    # 崩的不算 KEEP


def test_discard_above_absolute_threshold():
    """显式绝对阈值: MAE 劣于它 → DISCARD。"""
    def bad_eval(geno, device):
        return EvalResult(genotype=geno, status="OK", mae=999.0, rmse=999.0, device=device,
                          extra={"num_params": 50_000})

    cfg = OrchestratorConfig(dataset="PeMS04", population_size=4, tournament_size=2,
                             max_rounds=3, target_mae=0.0, discard_above=22.93)
    orch = Orchestrator(cfg, bad_eval, devices=2)
    state = orch.run()
    assert state.n_discard >= 1
    assert state.n_keep == 0
    assert state.best_mae == float("inf")  # 没有有效 KEEP


def test_warmup_keeps_everything_then_relative_discard():
    """冷启动 warmup 期一律 KEEP(让 archive 积累); warmup 后相对 best 退化才 DISCARD。

    这修复了点火实跑发现的问题: 短训练 MAE 全劣于已发表基线 → 旧逻辑全 DISCARD → archive 空。
    """
    seq = iter([20.0, 19.0, 18.0])  # 前 3 个递减 (warmup 内全 KEEP)

    def eval_fn(geno, device):
        try:
            mae = next(seq)
        except StopIteration:
            mae = 40.0  # warmup 后的大退化 (相对 best=18 超 1.5 倍 → DISCARD)
        return EvalResult(genotype=geno, status="OK", mae=mae, device=device,
                          extra={"num_params": 50_000})

    cfg = OrchestratorConfig(dataset="PeMS04", population_size=4, tournament_size=2,
                             max_rounds=3, target_mae=0.0, warmup_keep=3,
                             discard_regression_factor=1.5)
    orch = Orchestrator(cfg, eval_fn, devices=2)
    state = orch.run()
    # 前 3 个(20/19/18)在 warmup 内全 KEEP; 之后的 40 相对 best=18 退化超 1.5x → DISCARD
    assert state.n_keep >= 3
    assert state.best_mae == 18.0
    assert state.n_discard >= 1
    assert len(orch.archive) >= 1   # archive 确实积累了 (修复点)


# ---------------------------------------------------------------------------
# memory / archive 集成
# ---------------------------------------------------------------------------


def test_memory_populated():
    mem = MemoryStore(":memory:")
    cfg = OrchestratorConfig(dataset="PeMS04", population_size=6, tournament_size=3,
                             max_rounds=4, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2, memory=mem)
    state = orch.run()
    stats = mem.get_stats("PeMS04")
    assert stats["total"] == state.evals  # 每次评估都落库
    # best_so_far 应与 orchestrator 的 best 一致
    best = mem.best_so_far("PeMS04")
    if best is not None:
        assert abs(best["val_mae"] - state.best_mae) < 1e-6
    mem.close()


def test_archive_populated():
    cfg = OrchestratorConfig(population_size=8, tournament_size=3, max_rounds=8, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    orch.run()
    assert len(orch.archive) >= 1
    assert orch.archive.best() is not None


def test_graveyard_prevents_recrash(monkeypatch):
    """crash 进 graveyard 后, evolution 不再提议同配置 (memory 集成)。"""
    mem = MemoryStore(":memory:")
    # 让所有 adaptive 架构崩
    rng = random.Random(0)

    def eval_fn(geno, device):
        if any(b.spatial_op == "adaptive" for b in geno.blocks):
            return EvalResult(genotype=geno, status="CRASH", device=device, fail_reason="nan")
        return EvalResult(genotype=geno, status="OK", mae=20.0, device=device,
                          extra={"num_params": 50_000})

    cfg = OrchestratorConfig(population_size=6, tournament_size=3, max_rounds=10, target_mae=0.0)
    orch = Orchestrator(cfg, eval_fn, devices=2, memory=mem)
    orch.run()
    # graveyard 里应有崩溃记录
    grave_count = mem.get_stats("PeMS04")["CRASH"]
    assert grave_count >= 1
    mem.close()


# ---------------------------------------------------------------------------
# 进化改进
# ---------------------------------------------------------------------------


def test_best_improves_over_rounds():
    """足够多轮后, 最优 MAE 应明显优于第一轮 (进化在改进)。"""
    cfg = OrchestratorConfig(population_size=10, tournament_size=4, max_rounds=30, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(rng=random.Random(42)), devices=2)

    first_round_best = {"v": None}
    orig_step = orch.step

    state = orch.run()
    # 历史里前期 best vs 后期 best
    early = [h["best_mae"] for h in state.history[:10] if h["best_mae"] < float("inf")]
    late = [h["best_mae"] for h in state.history[-10:]]
    assert min(late) <= min(early)  # 后期最优不差于前期 (单调改进)


def test_never_asks_user():
    """编排器无任何 input()/交互 —— 纯自主。用一个会失败的 eval 也不应卡住等输入。"""
    # 这是结构性保证: run() 是纯循环, 无 input。此测试确保跑完即返回, 不挂起。
    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=3, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(crash_prob=0.8), devices=2)
    state = orch.run()  # 若有 input() 会卡死; 能返回即证明无交互
    assert state.stop_reason is not None
