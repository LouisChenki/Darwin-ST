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
