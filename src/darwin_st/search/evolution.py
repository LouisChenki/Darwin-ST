"""进化式架构搜索 (Aging/Regularized Evolution) —— 替换"随机游走"的核心引擎。

实现 Real et al. 2019 (arXiv:1802.01548) 的 aging evolution, 这是 docs/P2_ALGORITHM_DESIGN.md §3
选定的算法。核心循环:
  种群 = 固定容量 P 的队列;
  每轮: 随机采 S 个 → 取最优为父代(锦标赛) → 变异生子代 → 评估 → 入队右端 →
        **移除最老 (FIFO), 而非最差**。

为何移除最老 (关键): 20 分钟评估有噪声。移除最差会让"侥幸高分"的架构永久霸榜;
aging 给每个个体有限寿命, 血脉只能靠后代**重新训练到高分**才能存活 → 抗噪声正则化。
我们付不起重复评估, 故 aging 是廉价且有效的噪声防御。

设计 (ask/tell, 与 Optuna 一致, 契合两层自治 orchestrator):
  - evolution 只负责"提议下一个待评架构"与"接收评估结果", **绝不自己训练**
    (训练由 scheduler 派发到 GPU)。
  - ask() → 下一个 genotype (bootstrap 期给多样化种子; 之后 锦标赛+变异)
  - tell(genotype, fitness) → 入队 + 超容则逐出最老
  - 与 memory.graveyard 集成: 跳过已知失败/已试配置 (防无效优化)
  - 变异 mutation-only (无 crossover, 它破坏图基因合法性), 非法子代拒绝重采。

fitness 约定: **越小越优** (masked val-MAE)。
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

from darwin_st.search.genotype import (
    SPATIAL_OPS,
    TEMPORAL_OPS,
    VALID_ADJ_MODE,
    VALID_FUSION,
    EmbeddingConfig,
    Genotype,
    STBlock,
    mutate,
    random_genotype,
)

__all__ = ["Member", "AgingEvolution", "random_mutation", "seed_genotypes"]

_HIDDEN_CHOICES = (16, 32, 64, 128)
_EMB_DIM_CHOICES = EmbeddingConfig.VALID_DIMS  # (16, 32, 64)


# ---------------------------------------------------------------------------
# 随机变异算子 (Tier-1 规则变异): 产出一个合法子代
# ---------------------------------------------------------------------------


def random_mutation(geno: Genotype, rng: random.Random, max_attempts: int = 50) -> Genotype:
    """对 genotype 施加一个**随机且合法**的变异, 返回新 genotype。

    随机挑变异类型 + 随机参数, 触及保护区或产生非法结构则换一种重试 (拒绝-重采)。
    变异类型按研究价值加权: 算子替换 / 嵌入开关 是高价值变异。
    所有尝试失败 (极少见) 则回退到必定合法的 change_hidden。
    """
    # (类型, 权重) —— op-swap 与 embedding-toggle 高权重(研究指明高价值)
    choices = [
        ("swap_spatial", 3),
        ("swap_temporal", 3),
        ("toggle_embedding", 3),
        ("change_fusion", 2),
        ("change_hidden", 2),
        ("change_adj_mode", 1),
        ("add_block", 1),
        ("remove_block", 1),
    ]
    ops = [c for c, _ in choices]
    weights = [w for _, w in choices]

    for _ in range(max_attempts):
        op = rng.choices(ops, weights=weights, k=1)[0]
        try:
            return _apply_random(geno, op, rng)
        except (ValueError, KeyError):
            continue  # 触及保护区/非法 → 换一种重试
    # 兜底: change_hidden 几乎总合法
    new_h = rng.choice([h for h in _HIDDEN_CHOICES if h != geno.hidden] or list(_HIDDEN_CHOICES))
    return mutate(geno, "change_hidden", new_hidden=new_h)


def _apply_random(geno: Genotype, op: str, rng: random.Random) -> Genotype:
    """为给定变异类型随机生成参数并应用 (可能抛异常, 由调用方捕获重试)。"""
    n = len(geno.blocks)

    if op == "swap_spatial":
        i = rng.randrange(n)
        cur = geno.blocks[i].spatial_op
        cands = [o for o in SPATIAL_OPS if o != cur]
        return mutate(geno, "swap_spatial", index=i, new_op=rng.choice(cands))

    if op == "swap_temporal":
        i = rng.randrange(n)
        cur = geno.blocks[i].temporal_op
        cands = [o for o in TEMPORAL_OPS if o != cur]
        return mutate(geno, "swap_temporal", index=i, new_op=rng.choice(cands))

    if op == "change_fusion":
        i = rng.randrange(n)
        cur = geno.blocks[i].fusion
        cands = [f for f in VALID_FUSION if f != cur]
        return mutate(geno, "change_fusion", index=i, new_fusion=rng.choice(cands))

    if op == "change_hidden":
        cands = [h for h in _HIDDEN_CHOICES if h != geno.hidden]
        return mutate(geno, "change_hidden", new_hidden=rng.choice(cands))

    if op == "change_adj_mode":
        cands = [m for m in VALID_ADJ_MODE if m != geno.adj_mode]
        return mutate(geno, "change_adj_mode", new_adj_mode=rng.choice(cands))

    if op == "add_block":
        nb = STBlock(
            spatial_op=rng.choice(list(SPATIAL_OPS)),
            temporal_op=rng.choice(list(TEMPORAL_OPS)),
            fusion=rng.choice(list(VALID_FUSION)),
        )
        return mutate(geno, "add_block", new_block=nb)

    if op == "remove_block":
        if n <= 1:
            raise ValueError("只剩 1 块不可删")
        return mutate(geno, "remove_block", index=rng.randrange(n))

    if op == "toggle_embedding":
        which = rng.choice(["node", "tod", "dow"])
        if rng.random() < 0.5:
            cur = getattr(geno.embedding, f"use_{which}")
            return mutate(geno, "toggle_embedding", which=which, enable=not cur)
        return mutate(geno, "toggle_embedding", which=which, dim=rng.choice(_EMB_DIM_CHOICES))

    raise ValueError(f"未知变异类型: {op}")


def seed_genotypes(n: int, rng: random.Random, protected: list[str] | None = None) -> list[Genotype]:
    """生成 n 个多样化的初始 genotype (覆盖不同 空间×时序 算子组合)。

    用于 bootstrap 种群; 比全用同一个起点更利于早期探索。
    """
    # 偏好"有意义"的算子 (排除 identity 作为主算子, 但允许出现在变异中)
    s_pool = [o for o in SPATIAL_OPS if o != "identity"]
    t_pool = [o for o in TEMPORAL_OPS if o != "identity"]
    out = []
    for _ in range(n):
        g = random_genotype(
            depth=rng.choice([1, 2, 3]),
            spatial=rng.choice(s_pool),
            temporal=rng.choice(t_pool),
            fusion=rng.choice(list(VALID_FUSION)),
            hidden=rng.choice(_HIDDEN_CHOICES),
            adj_mode=rng.choice(list(VALID_ADJ_MODE)),
            protected=protected,
        )
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# 种群成员 + Aging Evolution
# ---------------------------------------------------------------------------


@dataclass
class Member:
    """种群成员: 一个已评估的架构。fitness 越小越优 (masked val-MAE)。"""

    genotype: Genotype
    fitness: float
    trial_id: int | None = None  # 关联 memory.experiments.id


class AgingEvolution:
    """Aging/Regularized Evolution 控制器 (ask/tell 接口)。

    用法 (orchestrator 驱动):
        evo = AgingEvolution(population_size=20, tournament_size=5, seed=0)
        geno = evo.ask()                  # 取下一个待评架构
        fitness = train_and_eval(geno)    # 由 scheduler 在 GPU 上做
        evo.tell(geno, fitness)           # 回灌结果
        # 循环往复, 永不停 (停止判定在 orchestrator)
    """

    def __init__(
        self,
        population_size: int = 20,
        tournament_size: int = 5,
        seed: int = 0,
        protected: list[str] | None = None,
        memory=None,                  # 可选 MemoryStore: 用于 graveyard/已见 查重
        base_genotype: Genotype | None = None,  # 若给定, bootstrap 以它为起点变异
    ):
        if tournament_size > population_size:
            raise ValueError("tournament_size 不应超过 population_size")
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)
        self.protected = protected or []
        self.memory = memory
        self.base_genotype = base_genotype

        self.population: deque[Member] = deque()  # FIFO: 左老右新
        self._seed_queue: list[Genotype] = []     # 待提议的 bootstrap 种子
        self._init_seeds()
        self._pending: dict[str, Genotype] = {}    # 已 ask 未 tell 的 (签名→genotype)

    def _init_seeds(self) -> None:
        """准备 bootstrap 种子: 种群需先填满 P 个才进入锦标赛进化。"""
        if self.base_genotype is not None:
            # 以基准为中心: 一半原样, 一半变异 (围绕用户给定创新点探索)
            self._seed_queue.append(self.base_genotype.copy())
            for _ in range(self.population_size - 1):
                self._seed_queue.append(random_mutation(self.base_genotype, self.rng))
        else:
            self._seed_queue = seed_genotypes(self.population_size, self.rng, self.protected)

    # -- ask/tell --
    def ask(self) -> Genotype:
        """返回下一个待评估的 genotype。

        bootstrap 期: 派发种子;之后: 锦标赛选父 → 变异 → graveyard 查重(命中重采)。
        """
        # 1) bootstrap: 还有种子未派发
        if self._seed_queue:
            geno = self._seed_queue.pop(0)
            self._pending[geno.signature()] = geno
            return geno

        # 2) 进化: 锦标赛 + 变异 + 查重
        for _ in range(100):  # 重采上限, 防止极端情况死循环
            parent = self._tournament()
            child = random_mutation(parent.genotype, self.rng)
            sig = child.signature()
            if sig in self._pending:
                continue  # 正在评估中, 换一个
            if self.memory is not None:
                if self.memory.query_graveyard(child.to_dict()) is not None:
                    continue  # 已知失败, 跳过 (硬阻断无效优化)
                if self.memory.seen_signature(child.to_dict()):
                    continue  # 已评估过, 跳过
            self._pending[sig] = child
            return child
        # 兜底: 实在采不出新的, 返回一个父代的强制变异 (允许重复)
        parent = self._tournament()
        child = random_mutation(parent.genotype, self.rng)
        self._pending[child.signature()] = child
        return child

    def tell(self, genotype: Genotype, fitness: float, trial_id: int | None = None) -> None:
        """回灌评估结果: 入队右端, 超容则逐出最老 (aging)。

        fitness 越小越优。crash/inf 一般不入种群 (由 orchestrator 决定是否调用)。
        """
        self._pending.pop(genotype.signature(), None)
        self.population.append(Member(genotype=genotype, fitness=fitness, trial_id=trial_id))
        while len(self.population) > self.population_size:
            self.population.popleft()  # **移除最老 (FIFO), 而非最差** —— aging 核心

    # -- 查询 --
    def _tournament(self) -> Member:
        """锦标赛选择: 随机采 min(S, |pop|) 个, 取 fitness 最小 (最优) 者为父代。"""
        if not self.population:
            raise RuntimeError("种群为空, 无法锦标赛 (bootstrap 未完成)")
        k = min(self.tournament_size, len(self.population))
        contenders = self.rng.sample(list(self.population), k)
        return min(contenders, key=lambda m: m.fitness)

    def best(self) -> Member | None:
        """当前种群中的最优成员 (注意: aging 下最优可能已被逐出, 全局最优由 memory 持有)。"""
        if not self.population:
            return None
        return min(self.population, key=lambda m: m.fitness)

    @property
    def is_bootstrapping(self) -> bool:
        return len(self._seed_queue) > 0

    def __len__(self) -> int:
        return len(self.population)
