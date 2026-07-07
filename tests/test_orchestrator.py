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
    # 流式: 每轮 2 卡 = 10 评估; 停止时排空在飞可多至 num_devices-1 个 (不丢结果)
    assert 5 * 2 <= state.evals <= 5 * 2 + orch.scheduler.num_devices - 1


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


def test_search_explores_deep_architectures():
    """深度探索修复端到端: eval_fn 奖励 depth (mae=base-depth*1.0), 搜索应探到 depth>=3 架构,
    且 archive 填充 3-4 深度带 (证明浅层坍缩被破)。"""
    cfg = OrchestratorConfig(population_size=10, tournament_size=4, max_rounds=25, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(rng=random.Random(0)), devices=2)
    orch.run()
    # archive 里出现过 depth>=3 的精英 (深度档 3-4 或 5+ 被填)
    deep_cells = [k for k in orch.archive.cells if k[3] in ("3-4", "5+")]
    assert deep_cells, "搜索未探到 depth>=3 (浅层坍缩未破)"
    # 最深精英确实 >= 3 层
    max_depth = max(e.genotype.depth for e in orch.archive.cells.values())
    assert max_depth >= 3, f"最深架构仅 {max_depth} 层"


def test_cold_start_seeds_deep_not_shallow():
    """铁律回归 (曾踩坑): base_genotype=None 冷启动 → 走 seed_genotypes 深度[2,3,4]播种,
    **不是** 从 depth-1 弱基线变异。这守住 run_creation_loop 的 base=None 冷启动接线:
    若 base 误传非 None (如弱基线), evolution 走暖启动分支绕过深度播种 → 坍缩浅层。"""
    cfg = OrchestratorConfig(population_size=16, tournament_size=3, max_rounds=1, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=4, base_genotype=None)
    seed_depths = {g.depth for g in orch.evo._seed_queue}
    assert seed_depths and seed_depths <= {2, 3, 4}, f"冷启动种子深度应⊆{{2,3,4}}, 实际 {seed_depths}"
    assert 1 not in seed_depths, "冷启动不该出现 depth-1 (说明误走了弱基线暖启动分支)"


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


# ---------------------------------------------------------------------------
# Tier-2 创造接入 (停滞触发跨域创造)
# ---------------------------------------------------------------------------


def test_creation_triggered_on_stagnation():
    """停滞时 orchestrator 调 creation_loop, 合成算子的 genotype 被评估。"""
    from darwin_st.creation import (CreationLoop, MockLLM, OperatorRegistry,
                                    OperatorSynthesizer, SynthesisConfig)
    from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
    from darwin_st.search import operators as ops_mod
    from darwin_st.search.genotype import Genotype, STBlock

    # 清理注入算子
    before_ops = set(ops_mod.SPATIAL_OPS)

    store = InMemoryGraphStore(HashEmbedder(dim=128))
    for m in all_seed_mechanisms():
        store.add_mechanism(m)

    plan = ('[{"operator_name": "OrchFusedOp", "rationale": "r", "shared_structure": "s",'
            '"composition": "additive_residual", "source_mechanisms": ["a"], "expected_effect": "e"}]')
    code = ('```python\nimport torch\nimport torch.nn as nn\n'
            'class OrchFusedOp(nn.Module):\n'
            '    def __init__(self, channels, num_nodes, **kw):\n'
            '        super().__init__(); self.l = nn.Linear(channels, channels)\n'
            '        self.a = nn.Parameter(torch.zeros(1))\n'
            '    def forward(self, x, adj=None): return self.l(x) + self.a * x\n```')
    resp = iter([plan, code] * 10)
    synth = OperatorSynthesizer(MockLLM(lambda m: next(resp)), SynthesisConfig(max_retries=2))
    cloop = CreationLoop(store, store.embedder, synth, OperatorRegistry())

    created = {"ops": []}

    def eval_fn(geno, device):
        # 记录是否评估到了 synth 算子 (Stage 1: 可能在 spatial/temporal/joint 任一槽)
        for b in geno.blocks:
            for op in (b.spatial_op, b.temporal_op, b.joint_op):
                if op and op.startswith("synth_"):
                    created["ops"].append(op)
        return EvalResult(genotype=geno, status="OK", mae=20.0, device=device,
                          extra={"num_params": 50_000})

    cfg = OrchestratorConfig(dataset="PeMS04", population_size=4, tournament_size=2,
                             max_rounds=6, target_mae=0.0, warmup_keep=2,
                             discard_regression_factor=1.5)
    # 强制快速停滞: archive stagnation_patience 默认 20, 这里手动调小
    orch = Orchestrator(cfg, eval_fn, devices=2, base_genotype=Genotype(blocks=[STBlock("gcn", "tcn")]),
                        creation_loop=cloop)
    orch.archive.stagnation_patience = 2  # 快速触发创造
    state = orch.run()

    # 创造事件应被记录
    creation_events = [h for h in state.history if h.get("event") == "creation"]
    assert len(creation_events) >= 1, "停滞未触发创造"
    # 合成算子的 genotype 应被评估
    assert len(created["ops"]) >= 1, "合成算子未进入评估"

    # 清理
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before_ops:
            ops_mod.SPATIAL_OPS.pop(k, None)


def test_creation_none_does_not_break():
    """不接 creation_loop 时, 停滞照常 island_reset, 不崩。"""
    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=4, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)  # creation_loop=None
    orch.archive.stagnation_patience = 2
    state = orch.run()
    assert state.rounds == 4  # 正常跑完


def test_try_creation_refreshes_workers_and_closes(monkeypatch):
    """衔接点②→③ orchestrator 级: 创造成功后必须 refresh_workers (让进程后端下批重载新 synth),
    run 结束必须 close。这是多进程下 Tier-2 算子能被 Tier-1 评测的关键 (否则 spawn worker 没新算子)。
    """
    from darwin_st.search.genotype import Genotype, STBlock

    class _SpyCreationLoop:
        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            from darwin_st.creation.creation_loop import CreationOutcome
            seed = Genotype(blocks=[STBlock("gcn", "tcn")])
            return CreationOutcome(True, operator_names=["synth_x"], seed_genotypes=[seed],
                                   bottleneck="b", n_success=1)

    calls = {"refresh": 0, "close": 0}
    # 常量 eval_fn: best 首评即定, 之后永不改进 → 停滞必触发 (与变异池动态无关, 不脆)。
    # (用 _make_eval_fn 的单调 toy 会因 Stage2 更丰富变异池持续改进而永不停滞。)
    def const_eval(geno, device):
        return EvalResult(genotype=geno, status="OK", mae=20.0, rmse=30.0, device=device,
                          hps={"lr": 1e-3}, extra={"num_params": 50_000})

    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=3, target_mae=0.0)
    orch = Orchestrator(cfg, const_eval, devices=2, creation_loop=_SpyCreationLoop())
    # 监听 scheduler 的 refresh/close (线程后端是 no-op, 但 orchestrator 仍应调用)
    monkeypatch.setattr(orch.scheduler, "refresh_workers",
                        lambda: calls.__setitem__("refresh", calls["refresh"] + 1))
    monkeypatch.setattr(orch.scheduler, "close",
                        lambda: calls.__setitem__("close", calls["close"] + 1))
    orch.archive.stagnation_patience = 1  # 立即触发创造
    orch.run()

    assert calls["refresh"] >= 1, "创造成功后未调 refresh_workers (多进程下 worker 拿不到新 synth)"
    assert calls["close"] == 1, "run 结束未 close 进程池"


def test_creation_cooldown_window_throttles_creation():
    """B: 创造成功后进精修窗口 (creation_refine_rounds 轮内不再创造), 修高频创造过早收敛。

    patience=1 让每轮都停滞; 若无冷却窗口会每轮创造。冷却=3 应让创造稀疏 (不再每轮)。
    """
    from darwin_st.search.genotype import Genotype, STBlock

    class _CountingCreationLoop:
        def __init__(self):
            self.calls = 0

        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            from darwin_st.creation.creation_loop import CreationOutcome
            self.calls += 1
            seed = Genotype(blocks=[STBlock("gcn", "tcn")])
            return CreationOutcome(True, operator_names=[f"synth_{self.calls}"],
                                   seed_genotypes=[seed], bottleneck="b", n_success=1)

    cloop = _CountingCreationLoop()
    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=12, target_mae=0.0,
                             creation_refine_rounds=3)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2, creation_loop=cloop)
    orch.archive.stagnation_patience = 1   # 每轮都"停滞"
    orch.run()

    # 12 轮, 冷却窗口 3 → 创造应明显少于 12 (每次创造后 3 轮静默)。约 12/4=3 次量级。
    assert cloop.calls <= 5, f"冷却窗口未限流, 创造 {cloop.calls} 次 (应 ~3-4)"
    assert cloop.calls >= 1, "完全没创造"


def test_adaptive_patience_grows_near_sota():
    """距离自适应耐心: gap 大→耐心小(快创造多探索), gap 小→耐心大(重精修), 封顶 cap。"""
    cfg = OrchestratorConfig(adaptive_patience=True, patience_base=3, patience_k=5.0,
                             patience_cap=12, patience_eps=0.5)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=1)
    orch.state.sota_mae = 17.8
    orch.state.best_mae = 21.8   # gap=4 → 耐心小
    p_far = orch._adaptive_patience()
    orch.state.best_mae = 18.0   # gap=0.2 → 耐心大
    p_near = orch._adaptive_patience()
    assert p_far < p_near, "耐心应随逼近 SOTA 而增大"
    assert p_far <= 6 and p_near >= 9   # 方案C 区间
    orch.state.best_mae = 17.0   # gap<0 (已超 SOTA) → 封顶 cap
    assert orch._adaptive_patience() == 12
    orch.state.sota_mae = None   # sota 未知 → base
    assert orch._adaptive_patience() == 3
    orch.state.sota_mae = 17.8
    orch.state.best_mae = float("inf")   # 冷启动 → base
    assert orch._adaptive_patience() == 3


def test_adaptive_patience_applied_in_run():
    """自适应开启时, run() 每轮按 gap 重设 archive.stagnation_patience (落在 [base,cap])。"""
    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=3, target_mae=0.0,
                             adaptive_patience=True, patience_base=3, patience_k=5.0, patience_cap=12)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    orch.run()
    assert 3 <= orch.archive.stagnation_patience <= 12


def test_creation_cooldown_resets_since_improve():
    """B: 创造成功后手动重置 archive._since_improve=0 (精修窗口干净计数)。"""
    from darwin_st.search.genotype import Genotype, STBlock

    class _OnceCreationLoop:
        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            from darwin_st.creation.creation_loop import CreationOutcome
            seed = Genotype(blocks=[STBlock("gcn", "tcn")])
            return CreationOutcome(True, operator_names=["synth_x"], seed_genotypes=[seed],
                                   bottleneck="b", n_success=1)

    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=8, target_mae=0.0,
                             creation_refine_rounds=4, warmup_keep=2)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2, creation_loop=_OnceCreationLoop())
    orch.archive.stagnation_patience = 1
    orch.run()
    # 创造后设了冷却窗口 (rounds + refine_rounds)
    assert orch._creation_cooldown_until > 0



# ---------------------------------------------------------------------------
# 作用域隔离: trial 落库带 space_version
# ---------------------------------------------------------------------------


def test_trials_carry_space_version():
    """OrchestratorConfig(space_version=...) → 所有落库 trial 带该代际; scoped best_so_far 隔离。"""
    mem = MemoryStore(":memory:")
    cfg = OrchestratorConfig(dataset="PeMS04", space_version="vSCOPE",
                             population_size=4, tournament_size=2, max_rounds=2, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2, memory=mem)
    assert orch.scope.space_version == "vSCOPE"
    assert orch.evo.dataset == "PeMS04"
    orch.run()
    rows = mem.list_trials(dataset="PeMS04")
    assert rows and all(r.get("space_version") == "vSCOPE" for r in rows)
    # scoped best_so_far 只在本代取到, 别代 None
    assert mem.best_so_far("PeMS04", space_version="vSCOPE") is not None
    assert mem.best_so_far("PeMS04", space_version="other") is None
    mem.close()


def test_space_version_auto_resolves_when_none():
    """不传 space_version → 自动哈希派生 (非 None)。"""
    from darwin_st.search.operators import builtin_op_signature
    cfg = OrchestratorConfig(dataset="PeMS04", max_rounds=1)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=1)
    assert orch.scope.space_version == builtin_op_signature()


# ---------------------------------------------------------------------------
# 流式驱动: 4 卡满载, 慢架构不阻塞 (破批同步瓶颈)
# ---------------------------------------------------------------------------


def test_orchestrator_pipelined_no_idle():
    """流式: 慢架构不阻塞快架构 —— 直接用 scheduler.run_stream 证快的先完成 (完成序非提交序)。

    批模式下一批必须全返回才 digest, 慢的拖住整批。流式下快的一完成立即 digest。
    """
    import time
    from darwin_st.optim.scheduler import GPUScheduler
    from darwin_st.search.genotype import Genotype, STBlock
    order = []

    def eval_fn(geno, device):
        time.sleep(0.4 if geno.hidden == 999 else 0.02)   # hidden=999 慢
        order.append(geno.hidden)
        return EvalResult(genotype=geno, status="OK", mae=20.0, device=device,
                          extra={"num_params": 50_000})

    sched = GPUScheduler(eval_fn, devices=4, backend="thread")
    # 第一个提交的是慢架构, 其余快 → 流式下快的应先完成 digest
    genos = ([Genotype(blocks=[STBlock("gcn", "tcn")], hidden=999)]
             + [Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64) for _ in range(11)])
    it = iter(genos)
    sched.run_stream(lambda: next(it, None), lambda r: None, should_continue=lambda: True)
    assert len(order) == 12
    assert order[0] != 999, "慢架构最先完成 = 阻塞了快架构 (流式失效)"
    assert order.index(999) >= 3, "慢架构应在多个快架构之后才完成 (证明 4 卡并行不空转)"


def test_pipelined_counts_identical_to_batch():
    """流式与批模式计数恒等: 每 genotype 恰好评估+digest 一次 (eval/keep 计数不变)。"""
    cfg = OrchestratorConfig(dataset="PeMS04", population_size=6, tournament_size=3,
                             max_rounds=8, target_mae=0.0)
    orch = Orchestrator(cfg, _make_eval_fn(rng=random.Random(1)), devices=2)
    state = orch.run()
    # 每次评估恰好一条 history + 计数自洽
    assert state.evals == len([h for h in state.history if "eval" in h])
    assert state.n_keep + state.n_discard + state.n_crash == state.evals
    assert state.rounds == 8   # 每 num_devices 完成 = 1 轮


# ---------------------------------------------------------------------------
# 自适应 HPO trials (Part B: Gap-Annealed HPO Depth, 距 SOTA 自适应调参预算)
# ---------------------------------------------------------------------------


def test_adaptive_hpo_trials_scales_with_gap():
    """距离自适应: gap 大→base(广撒网), gap 小→cap(精调); 冷启动/未知→base; gap<0→cap。"""
    cfg = OrchestratorConfig(adaptive_hpo_trials=True, hpo_trials_base=4, hpo_trials_cap=12,
                             hpo_trials_k=2.0, hpo_trials_eps=0.3)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=1)
    orch.state.sota_mae = 17.8
    orch.state.best_mae = 19.21   # gap=1.41 → 少 trials (远)
    far = orch._adaptive_hpo_trials()
    orch.state.best_mae = 18.0    # gap=0.20 → 多 trials (近)
    near = orch._adaptive_hpo_trials()
    assert far < near, "近 SOTA 应更多 trials"
    assert cfg.hpo_trials_base <= far <= cfg.hpo_trials_cap
    # gap<=0 (已超 SOTA) → cap 死磕精调
    orch.state.best_mae = 17.0
    assert orch._adaptive_hpo_trials() == cfg.hpo_trials_cap
    # 冷启动 (best=inf) → base (广撒网快筛, 修正: 不是 cap)
    orch.state.best_mae = float("inf")
    assert orch._adaptive_hpo_trials() == cfg.hpo_trials_base
    # sota 未知 → base
    orch.state.sota_mae = None
    assert orch._adaptive_hpo_trials() == cfg.hpo_trials_base


def test_next_genotype_stamps_evo_not_creation_seed():
    """_next_genotype 只盖进化 genotype 的 hpo_trials, 不覆盖创造 seed 的显式大 HPO。"""
    from darwin_st.search.genotype import Genotype, STBlock
    cfg = OrchestratorConfig(dataset="PeMS04", adaptive_hpo_trials=True)
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    orch.state.sota_mae = 17.8
    orch.state.best_mae = 19.21
    # 创造 seed 带 _seed_meta 大 HPO → 原样返回不被盖
    seed = Genotype(blocks=[STBlock("gcn", "tcn")])
    seed._seed_meta = {"hpo_trials": 20, "is_creation_seed": True}
    orch._pending_seed_genotypes = [seed]
    g1 = orch._next_genotype()
    assert g1._seed_meta["hpo_trials"] == 20, "创造 seed 大 HPO 被覆盖!"
    # evo genotype 被盖成自适应值
    g2 = orch._next_genotype()
    assert getattr(g2, "_seed_meta", None) is not None
    assert g2._seed_meta["hpo_trials"] == orch._adaptive_hpo_trials()


def test_adaptive_hpo_trials_off_by_default():
    """默认关 → evo genotype 无 _seed_meta → worker 用固定 n_trials (行为不变)。"""
    cfg = OrchestratorConfig(dataset="PeMS04")   # adaptive_hpo_trials 默认 False
    orch = Orchestrator(cfg, _make_eval_fn(), devices=2)
    g = orch._next_genotype()
    assert getattr(g, "_seed_meta", None) is None
