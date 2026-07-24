"""evolution.py 的正确性回归测试 (设备无关, 纯 CPU 逻辑)。

核心契约:
  - random_mutation 总产合法子代, 且尊重保护区
  - ask/tell: bootstrap 填满种群后进入锦标赛进化
  - **aging = 移除最老 (FIFO), 而非最差** —— 算法核心, 最易写错
  - tournament 选最优为父代
  - graveyard 集成: 跳过已知失败/已见配置
  - 端到端: 在一个合成 fitness 上, 进化能朝更优方向推进
"""

from __future__ import annotations

import random

import pytest

from darwin_st.search.evolution import (
    AgingEvolution,
    Member,
    random_mutation,
    seed_genotypes,
)
from darwin_st.search.genotype import Genotype, random_genotype


# ---------------------------------------------------------------------------
# random_mutation
# ---------------------------------------------------------------------------


def test_random_mutation_always_valid():
    rng = random.Random(0)
    g = random_genotype(depth=2)
    for _ in range(100):
        child = random_mutation(g, rng)
        child.validate()  # 不抛即合法
        g = child  # 链式变异也始终合法


def test_random_mutation_respects_protected():
    """保护区算子在任意随机变异下都不被删除。"""
    rng = random.Random(1)
    g = random_genotype(depth=2, spatial="adaptive", protected=["adaptive"])
    for _ in range(200):
        child = random_mutation(g, rng)
        # adaptive 仍在某 block 中 (保护区铁律)
        assert child._uses_op("adaptive"), "保护区算子被变异删除了!"
        g = child


def test_random_mutation_returns_new_object():
    rng = random.Random(0)
    g = random_genotype(depth=1)
    child = random_mutation(g, rng)
    assert child is not g


def test_seed_genotypes_diverse_and_valid():
    rng = random.Random(0)
    seeds = seed_genotypes(10, rng)
    assert len(seeds) == 10
    for s in seeds:
        s.validate()
    # 至少有一定多样性 (不是全相同签名)
    sigs = {s.signature() for s in seeds}
    assert len(sigs) >= 5


# ---------------------------------------------------------------------------
# ask/tell + bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_then_evolve():
    evo = AgingEvolution(population_size=5, tournament_size=3, seed=0)
    assert evo.is_bootstrapping
    # 前 5 次 ask 是 bootstrap 种子
    for i in range(5):
        g = evo.ask()
        evo.tell(g, fitness=20.0 - i)  # 任意 fitness
    assert not evo.is_bootstrapping
    assert len(evo) == 5
    # 之后 ask 应来自锦标赛+变异, 仍合法
    g = evo.ask()
    g.validate()


def test_population_capacity_enforced():
    evo = AgingEvolution(population_size=4, tournament_size=2, seed=0)
    for i in range(20):
        g = evo.ask()
        evo.tell(g, fitness=float(i))
    assert len(evo) == 4  # 不超容


# ---------------------------------------------------------------------------
# aging = FIFO eviction (算法核心)
# ---------------------------------------------------------------------------


def test_aging_removes_oldest_not_worst():
    """关键: 超容时移除【最老】而非【最差】。

    构造: 先填满种群, 其中最老的成员 fitness 最优(最小)。再加一个新成员,
    若是 aging(移除最老), 则那个最优的老成员应被逐出; 若错误地移除最差, 它会留下。
    """
    evo = AgingEvolution(population_size=3, tournament_size=1, seed=0)
    # 手动控制种群: 跳过 bootstrap, 直接塞
    evo._seed_queue.clear()
    evo.population.clear()
    g1 = random_genotype(depth=1, spatial="gcn")
    g2 = random_genotype(depth=2, spatial="cheb")
    g3 = random_genotype(depth=3, spatial="diffusion")
    g4 = random_genotype(depth=2, spatial="gat")
    evo.tell(g1, fitness=1.0)   # 最老, 且 fitness 最优
    evo.tell(g2, fitness=50.0)
    evo.tell(g3, fitness=60.0)
    assert len(evo) == 3
    # 再加一个 → 超容 → 应逐出最老的 g1 (尽管它 fitness 最优)
    evo.tell(g4, fitness=70.0)
    assert len(evo) == 3
    remaining = [m.fitness for m in evo.population]
    assert 1.0 not in remaining, "最优但最老的成员未被逐出 → 退化成移除最差, 不是 aging!"
    assert 70.0 in remaining     # 最新的在


def test_tell_appends_right_evicts_left():
    """新成员入右端, 逐出从左端 (FIFO 顺序)。"""
    evo = AgingEvolution(population_size=2, tournament_size=1, seed=0)
    evo._seed_queue.clear()
    evo.population.clear()
    a = random_genotype(depth=1, spatial="gcn")
    b = random_genotype(depth=1, spatial="gat")
    c = random_genotype(depth=1, spatial="cheb")
    evo.tell(a, 10.0); evo.tell(b, 20.0)
    evo.tell(c, 30.0)  # 超容, 逐出最老的 a
    fits = [m.fitness for m in evo.population]
    assert fits == [20.0, 30.0]  # a(10) 被逐, 顺序保持


# ---------------------------------------------------------------------------
# tournament
# ---------------------------------------------------------------------------


def test_tournament_picks_best():
    evo = AgingEvolution(population_size=5, tournament_size=5, seed=0)
    evo._seed_queue.clear()
    evo.population.clear()
    for i, f in enumerate([30.0, 10.0, 50.0, 20.0, 40.0]):
        evo.tell(random_genotype(depth=1, spatial=list(["gcn","gat","cheb","diffusion","adaptive"])[i]), f)
    # tournament_size=5=全部 → 必选 fitness 最小 (10.0)
    parent = evo._tournament()
    assert parent.fitness == 10.0


def test_best_returns_min_fitness():
    evo = AgingEvolution(population_size=5, tournament_size=2, seed=0)
    for i in range(5):
        g = evo.ask(); evo.tell(g, fitness=float(10 + i))
    assert evo.best().fitness == 10.0


# ---------------------------------------------------------------------------
# graveyard 集成
# ---------------------------------------------------------------------------


def test_graveyard_skip(monkeypatch):
    """ask 不应返回 graveyard 中已知失败的配置。"""
    from darwin_st.memory.store import MemoryStore, Trial

    mem = MemoryStore(":memory:")
    evo = AgingEvolution(population_size=3, tournament_size=2, seed=0, memory=mem)
    # 完成 bootstrap
    seen = []
    for _ in range(3):
        g = evo.ask(); evo.tell(g, fitness=20.0); seen.append(g)
    # 把"下一个会被提议的"配置预先标记为 graveyard, 验证被跳过
    # 先 ask 一次拿到一个候选, 记其签名, 然后塞进 graveyard 并确认后续不再出现
    cand = evo.ask()
    evo.tell(cand, fitness=22.0)
    mem.record_trial(Trial(run_tag="t", dataset="PeMS04", genotype=cand.to_dict(),
                           status="CRASH", created_at="2026-06-18T00:00:00", fail_reason="nan"))
    # 之后多次 ask 都不应再返回 cand 的签名
    for _ in range(20):
        g = evo.ask()
        assert g.signature() != cand.signature(), "graveyard 配置被重新提议了!"
        evo.tell(g, fitness=21.0)
    mem.close()


# ---------------------------------------------------------------------------
# 端到端: 进化朝更优推进
# ---------------------------------------------------------------------------


def test_evolution_improves_on_synthetic_fitness():
    """合成 fitness: 偏好 depth 大 + 有节点嵌入。进化应逐步找到更优 (更小 fitness)。

    fitness = 10 - depth - (2 if use_node else 0) + 噪声, 越小越优。
    """
    rng = random.Random(0)
    evo = AgingEvolution(population_size=10, tournament_size=4, seed=0)

    def synth_fitness(g: Genotype) -> float:
        base = 10.0 - g.depth - (2.0 if g.embedding.use_node else 0.0)
        return base + rng.uniform(-0.3, 0.3)  # 评估噪声

    # bootstrap + 进化共 80 轮
    early, late = [], []
    for t in range(80):
        g = evo.ask()
        f = synth_fitness(g)
        evo.tell(g, f)
        if 10 <= t < 25:
            early.append(f)
        elif t >= 65:
            late.append(f)
    # 后期平均 fitness 应优于早期 (进化在改进)
    assert sum(late) / len(late) < sum(early) / len(early)


# ---------------------------------------------------------------------------
# 作用域: ask() 按数据集查重 (修跨数据集泄漏)
# ---------------------------------------------------------------------------


def test_ask_passes_dataset_to_graveyard_and_seen():
    """AgingEvolution(dataset=...) → ask() 用 dataset 调 query_graveyard/seen_signature。

    修跨数据集泄漏: METR-LA 的进化不该被 PeMS04 的崩溃/已见签名阻断。
    """
    calls = {"graveyard": [], "seen": []}

    class _SpyMemory:
        def query_graveyard(self, genotype, hp=None, dataset=None, space_version=None):
            calls["graveyard"].append(dataset)
            return None
        def seen_signature(self, genotype, hp=None, dataset=None, space_version=None):
            calls["seen"].append(dataset)
            return False

    evo = AgingEvolution(population_size=6, tournament_size=3, seed=0,
                         memory=_SpyMemory(), dataset="METR-LA")
    # 跑过 bootstrap 进入进化路径 (ask/tell 若干轮)
    for _ in range(10):
        g = evo.ask()
        evo.tell(g, 5.0)
    assert calls["graveyard"], "ask() 未调 query_graveyard"
    assert all(d == "METR-LA" for d in calls["graveyard"]), "graveyard 未按数据集查重"
    assert all(d == "METR-LA" for d in calls["seen"]), "seen 未按数据集查重"


def test_ask_dataset_none_backward_compat():
    """不传 dataset → 查重 dataset=None (全局, 旧行为)。"""
    seen_ds = []

    class _SpyMemory:
        def query_graveyard(self, genotype, hp=None, dataset=None, space_version=None):
            seen_ds.append(dataset); return None
        def seen_signature(self, genotype, hp=None, dataset=None, space_version=None):
            return False

    evo = AgingEvolution(population_size=6, tournament_size=3, seed=0, memory=_SpyMemory())
    for _ in range(8):
        evo.tell(evo.ask(), 5.0)
    assert seen_ds and all(d is None for d in seen_ds)


# ---------------------------------------------------------------------------
# 根因 A 修复: 变异池来源加权 (_weighted_pick) —— 保内置可达
# ---------------------------------------------------------------------------


def _register_fake_synth(n: int) -> list[str]:
    """注册 n 个假 synth 空间算子 (类别 spatiotemporal), 返回注册名。用后由调用方清理。"""
    from darwin_st.search import operators as ops_mod
    names = []
    for i in range(n):
        name = f"synth_fake_{i}"
        ops_mod.SPATIAL_OPS[name] = ops_mod.SPATIAL_OPS["gcn"]  # 复用工厂, 仅测采样比例
        ops_mod.OP_CATEGORY[name] = "spatiotemporal"
        names.append(name)
    return names


def _cleanup_synth(names: list[str]) -> None:
    from darwin_st.search import operators as ops_mod
    for n in names:
        ops_mod.SPATIAL_OPS.pop(n, None)
        ops_mod.OP_CATEGORY.pop(n, None)


def test_weighted_pick_lifts_builtin_share():
    """117 synth 淹没 6 内置时, 权重 0.5 下内置整体命中率应 ~50% (远高于均匀的 ~5%)。"""
    from darwin_st.search.evolution import _weighted_pick, _spatial_slot_pool
    from darwin_st.search.operators import SYNTH_PREFIX

    names = _register_fake_synth(117)
    try:
        rng = random.Random(0)
        pool = _spatial_slot_pool()
        # 前提: 池里 synth 远多于内置 (淹没场景)
        n_builtin = sum(1 for o in pool if not o.startswith(SYNTH_PREFIX))
        n_synth = sum(1 for o in pool if o.startswith(SYNTH_PREFIX))
        assert n_synth > 5 * n_builtin
        picks = [_weighted_pick(pool, rng, 0.5) for _ in range(4000)]
        builtin_share = sum(1 for p in picks if not p.startswith(SYNTH_PREFIX)) / len(picks)
        assert 0.4 < builtin_share < 0.6   # ~50%, 而非均匀的 ~5%
    finally:
        _cleanup_synth(names)


def test_weighted_pick_zero_weight_is_uniform():
    """权重=0 退回对全池均匀 (向后兼容): 内置命中率回到 ~n_builtin/n_pool。"""
    from darwin_st.search.evolution import _weighted_pick, _spatial_slot_pool
    from darwin_st.search.operators import SYNTH_PREFIX

    names = _register_fake_synth(117)
    try:
        rng = random.Random(0)
        pool = _spatial_slot_pool()
        n_builtin = sum(1 for o in pool if not o.startswith(SYNTH_PREFIX))
        expected = n_builtin / len(pool)
        picks = [_weighted_pick(pool, rng, 0.0) for _ in range(4000)]
        share = sum(1 for p in picks if not p.startswith(SYNTH_PREFIX)) / len(picks)
        assert abs(share - expected) < 0.03   # 贴近均匀期望
    finally:
        _cleanup_synth(names)


def test_weighted_pick_single_source_falls_back():
    """池里只有内置 (无 synth, 如冷启动) → 加权退化为均匀, 不崩且全返内置。"""
    from darwin_st.search.evolution import _weighted_pick
    from darwin_st.search.operators import SYNTH_PREFIX

    rng = random.Random(0)
    pool = ["gcn", "gat", "diffusion"]   # 纯内置
    picks = {_weighted_pick(pool, rng) for _ in range(200)}
    assert picks <= set(pool)
    assert all(not p.startswith(SYNTH_PREFIX) for p in picks)


def test_builtin_weight_env_read_once_at_construction(monkeypatch):
    """BUILTIN_MUTATION_WEIGHT 在构造期读一次: 构造后改 env 不影响已有实例 (治运行期静默生效)。"""
    monkeypatch.setenv("BUILTIN_MUTATION_WEIGHT", "0.9")
    evo = AgingEvolution(population_size=4, tournament_size=2, seed=0)
    assert evo._builtin_weight == 0.9
    # 构造后改 env → 已有实例不变 (冻结); 新实例才读到新值
    monkeypatch.setenv("BUILTIN_MUTATION_WEIGHT", "0.0")
    assert evo._builtin_weight == 0.9
    evo2 = AgingEvolution(population_size=4, tournament_size=2, seed=0)
    assert evo2._builtin_weight == 0.0


def test_builtin_weight_env_invalid_and_clamped(monkeypatch):
    """env 非法值容错回默认 0.5; 越界值裁剪到 [0,1] (构造期解析语义不变)。"""
    monkeypatch.setenv("BUILTIN_MUTATION_WEIGHT", "not_a_float")
    assert AgingEvolution(population_size=4, tournament_size=2, seed=0)._builtin_weight == 0.5
    monkeypatch.setenv("BUILTIN_MUTATION_WEIGHT", "2.5")
    assert AgingEvolution(population_size=4, tournament_size=2, seed=0)._builtin_weight == 1.0
    monkeypatch.delenv("BUILTIN_MUTATION_WEIGHT")
    assert AgingEvolution(population_size=4, tournament_size=2, seed=0)._builtin_weight == 0.5


def test_reseed_enqueues_survivors():
    """reseed 把 genotype 塞回 bootstrap 种子队列, 下次 ask 优先派发; 去重已在队/在评估的。"""
    rng_geno = [random_genotype(depth=2 + i) for i in range(3)]
    evo = AgingEvolution(population_size=4, tournament_size=2, seed=0)
    # 先耗尽 bootstrap 种子, 让 _seed_queue 空
    while evo.is_bootstrapping:
        evo.tell(evo.ask(), 5.0)
    assert not evo.is_bootstrapping
    added = evo.reseed(rng_geno)
    assert added == 3
    assert evo.is_bootstrapping        # 重新有种子待派
    # 再 reseed 同样的 → 去重, 不重复入队
    added2 = evo.reseed(rng_geno)
    assert added2 == 0
