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
from darwin_st.search.operators import SPATIOTEMPORAL_OPS, SYNTH_PREFIX, op_category

__all__ = ["Member", "AgingEvolution", "random_mutation", "seed_genotypes"]

_HIDDEN_CHOICES = (64, 128, 192)   # 砍掉 256: 冷启动撞上 hidden-256 → 121万参数吃31GB显存,
                                   # 是深度探索的评估提速陷阱 (18.348 best 也才 hidden-128)。深度是要探的
                                   # 容量维度, 宽度不是; 保留到 192 够用, 去掉最笨重的 256。
_EMB_DIM_CHOICES = EmbeddingConfig.VALID_DIMS  # (32, 64, 96, 128)


# ---------------------------------------------------------------------------
# 类别兼容候选池 (Stage 2): 让新算子 + joint 时空算子进得了变异池
# ---------------------------------------------------------------------------


def _spatial_slot_pool() -> list[str]:
    """可坐"空间槽"的算子: 空间类 + 时空一体类 (类别兼容, 放宽后签名相同)。

    含运行时注入的 synth 算子 (它们登记进 SPATIAL_OPS dict + OP_CATEGORY)。
    """
    return [o for o in _all_search_ops() if op_category(o) in ("spatial", "spatiotemporal", "any")]


def _temporal_slot_pool() -> list[str]:
    """可坐"时序槽"的算子: 时序类 + 时空一体类 (类别兼容)。"""
    return [o for o in _all_search_ops() if op_category(o) in ("temporal", "spatiotemporal", "any")]


def _joint_pool() -> list[str]:
    """可坐"一体槽"的时空算子: spatiotemporal 类 (含 synth_ 默认归此类)。"""
    return [o for o in _all_search_ops() if op_category(o) == "spatiotemporal"]


def _builtin_weight() -> float:
    """解析 env BUILTIN_MUTATION_WEIGHT (仅 AgingEvolution 构造期调一次)。默认 0.5 (内置与 synth 各半)。

    env BUILTIN_MUTATION_WEIGHT 覆盖; 非法值容错回 0.5; 裁剪到 [0,1]; =0 退回旧的对全池均匀
    rng.choice (向后兼容)。集中在构造期读取: 运行期再改 env 不再静默生效 (防行为突变不可复现)。
    根因 A 修复: 117 synth 淹没 6 内置 → 均匀采样下内置命中率仅 ~5%, 达到旧最优的
    adaptive+gru+attn 组合统计上不可达。按来源加权把内置整体命中率抬回 ~50%。
    """
    import os
    raw = os.environ.get("BUILTIN_MUTATION_WEIGHT")
    if raw is None:
        return 0.5
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.5


def _weighted_pick(pool: list[str], rng: random.Random, builtin_weight: float = 0.5) -> str:
    """从候选池按**来源加权**采一个算子: 内置整体占 w, synth 整体占 1-w, 组内均匀。

    w=builtin_weight (由调用方从 AgingEvolution 构造期读定的实例属性透传, 不在此读 env)。
    w<=0 或某一侧为空 → 退回对全池均匀 rng.choice (向后兼容/退化保护)。
    组内均匀 = 每个内置算子概率 w/n_builtin, 每个 synth 算子 (1-w)/n_synth。
    """
    w = builtin_weight
    if w <= 0.0:
        return rng.choice(pool)
    builtin = [o for o in pool if not o.startswith(SYNTH_PREFIX)]
    synth = [o for o in pool if o.startswith(SYNTH_PREFIX)]
    if not builtin or not synth:
        return rng.choice(pool)   # 只有一侧 → 均匀即可
    weights = [w / len(builtin) if not o.startswith(SYNTH_PREFIX) else (1.0 - w) / len(synth)
               for o in pool]
    return rng.choices(pool, weights=weights, k=1)[0]


def _all_search_ops() -> list[str]:
    """三表全部算子名 (含注入 synth)。去重保稳定顺序 (dict 有序)。"""
    seen: dict[str, None] = {}
    for name in (*SPATIAL_OPS, *TEMPORAL_OPS, *SPATIOTEMPORAL_OPS):
        seen.setdefault(name, None)
    return list(seen)


# ---------------------------------------------------------------------------
# 随机变异算子 (Tier-1 规则变异): 产出一个合法子代
# ---------------------------------------------------------------------------


def random_mutation(geno: Genotype, rng: random.Random, max_attempts: int = 50,
                    builtin_weight: float = 0.5) -> Genotype:
    """对 genotype 施加一个**随机且合法**的变异, 返回新 genotype。

    随机挑变异类型 + 随机参数, 触及保护区或产生非法结构则换一种重试 (拒绝-重采)。
    变异类型按研究价值加权: 算子替换 / 嵌入开关 是高价值变异。
    所有尝试失败 (极少见) 则回退到必定合法的 change_hidden。
    builtin_weight: 内置算子整体采样权重 (AgingEvolution 构造期从 env 读定后透传)。
    """
    # (类型, 权重) —— op-swap 与 embedding-toggle 高权重(研究指明高价值)
    # Stage2: swap 池纳入类别兼容全算子; 加 swap_joint/toggle_joint 让 joint 时空算子进搜索。
    # 深度探索修复: add_block 权重 3 > remove_block 1 —— 温和向深偏置打破 depth=2 坍缩 (实测 depth-4
    # 一次没探到)。非单向: remove_block 仍在, 变异是父代 ±1 层双向游走; 差的深架构被 archive 严格改进
    # + aging 逐出自动淘汰, 浅层好架构在自己深度格保留 (配 archive depth_bucket 维度)。
    choices = [
        ("swap_spatial", 3),
        ("swap_temporal", 3),
        ("toggle_embedding", 3),
        ("change_fusion", 2),
        ("swap_joint", 2),
        ("toggle_joint", 2),
        ("change_hidden", 2),
        ("change_adj_mode", 1),
        ("add_block", 3),
        ("remove_block", 1),
    ]
    ops = [c for c, _ in choices]
    weights = [w for _, w in choices]

    for _ in range(max_attempts):
        op = rng.choices(ops, weights=weights, k=1)[0]
        try:
            return _apply_random(geno, op, rng, builtin_weight)
        except (ValueError, KeyError):
            continue  # 触及保护区/非法 → 换一种重试
    # 兜底: change_hidden 几乎总合法
    new_h = rng.choice([h for h in _HIDDEN_CHOICES if h != geno.hidden] or list(_HIDDEN_CHOICES))
    return mutate(geno, "change_hidden", new_hidden=new_h)


def _apply_random(geno: Genotype, op: str, rng: random.Random,
                  builtin_weight: float = 0.5) -> Genotype:
    """为给定变异类型随机生成参数并应用 (可能抛异常, 由调用方捕获重试)。"""
    n = len(geno.blocks)

    if op == "swap_spatial":
        i = rng.randrange(n)
        cur = geno.blocks[i].spatial_op
        cands = [o for o in _spatial_slot_pool() if o != cur]
        return mutate(geno, "swap_spatial", index=i, new_op=_weighted_pick(cands, rng, builtin_weight))

    if op == "swap_temporal":
        i = rng.randrange(n)
        cur = geno.blocks[i].temporal_op
        cands = [o for o in _temporal_slot_pool() if o != cur]
        return mutate(geno, "swap_temporal", index=i, new_op=_weighted_pick(cands, rng, builtin_weight))

    if op == "swap_joint":
        # 仅对已处于一体模式的块生效 (否则 mutate 抛错→重采)
        joint_idx = [j for j, b in enumerate(geno.blocks) if b.joint_op is not None]
        if not joint_idx:
            raise ValueError("无一体模式块可换 joint 算子")
        i = rng.choice(joint_idx)
        cur = geno.blocks[i].joint_op
        cands = [o for o in _joint_pool() if o != cur]
        if not cands:
            raise ValueError("无候选时空一体算子")
        return mutate(geno, "swap_joint", index=i, new_op=_weighted_pick(cands, rng, builtin_weight))

    if op == "toggle_joint":
        i = rng.randrange(n)
        b = geno.blocks[i]
        if b.joint_op is not None:
            return mutate(geno, "toggle_joint", index=i)      # 退回分离模式
        pool = _joint_pool()
        if not pool:
            raise ValueError("无时空一体算子, 不能进一体模式")
        return mutate(geno, "toggle_joint", index=i, new_op=rng.choice(pool))  # 进一体模式

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
        # 低概率产 joint block (Stage2: 让一体时空块进入深度扩展); 多数仍分离块
        joint_pool = _joint_pool()
        if joint_pool and rng.random() < 0.25:
            nb = STBlock(
                spatial_op="identity", temporal_op="identity",
                fusion=rng.choice(list(VALID_FUSION)),
                joint_op=rng.choice(joint_pool),
            )
        else:
            nb = STBlock(
                spatial_op=_weighted_pick(_spatial_slot_pool(), rng, builtin_weight),
                temporal_op=_weighted_pick(_temporal_slot_pool(), rng, builtin_weight),
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
    Stage2: 候选池含新原子算子 (mixhop/multiscale_tcn/ssm 等) + 类别兼容时空算子。
    """
    # 偏好"有意义"的算子 (排除 identity 作为主算子, 但允许出现在变异中)
    s_pool = [o for o in _spatial_slot_pool() if o != "identity"]
    t_pool = [o for o in _temporal_slot_pool() if o != "identity"]
    out = []
    for _ in range(n):
        g = random_genotype(
            depth=rng.choice([2, 3, 4]),   # 深度探索: 播种深一档 (原[1,2,3]) 保证 3-4 深度带 bootstrap 就被采样;
            spatial=rng.choice(s_pool),     # 丢最弱的 depth-1 (实测 d1 best=20.60 远差于 d2)。
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
        dataset: str | None = None,   # 作用域: graveyard/seen 按数据集查重 (防跨数据集误阻断)
    ):
        if tournament_size > population_size:
            raise ValueError("tournament_size 不应超过 population_size")
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)
        self.protected = protected or []
        self.memory = memory
        self.base_genotype = base_genotype
        self.dataset = dataset
        # env 开关构造期读一次存为实例属性 (构造后运行期改 env 不再生效, 行为可复现);
        # 经 random_mutation 透传给 _weighted_pick (模块级函数不再自读 env)。
        self._builtin_weight = _builtin_weight()

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
                self._seed_queue.append(
                    random_mutation(self.base_genotype, self.rng, builtin_weight=self._builtin_weight))
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
            child = random_mutation(parent.genotype, self.rng, builtin_weight=self._builtin_weight)
            sig = child.signature()
            if sig in self._pending:
                continue  # 正在评估中, 换一个
            if self.memory is not None:
                if self.memory.query_graveyard(child.to_dict(), dataset=self.dataset) is not None:
                    continue  # 已知失败, 跳过 (硬阻断无效优化; 按数据集查重防跨集误阻断)
                if self.memory.seen_signature(child.to_dict(), dataset=self.dataset):
                    continue  # 已评估过, 跳过
            self._pending[sig] = child
            return child
        # 兜底: 实在采不出新的, 返回一个父代的强制变异 (允许重复)
        parent = self._tournament()
        child = random_mutation(parent.genotype, self.rng, builtin_weight=self._builtin_weight)
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

    def reseed(self, genotypes: list[Genotype]) -> int:
        """把一批 genotype 塞回 bootstrap 种子队列 (下次 ask 优先派发)。返回入队数。

        根因 B 修复: island_reset 清掉 archive 差格后, 把存活精英回灌进化种子队列, 让
        逃逸机制真正作用到被 ask() 消费的种群 (否则 island_reset 只重置停滞计数, 是 no-op)。
        去重 (按签名) 避免与在评估中的重复; 不改 population 本身 (aging FIFO 语义不变)。
        """
        pending_sigs = set(self._pending)
        queued_sigs = {g.signature() for g in self._seed_queue}
        added = 0
        for g in genotypes:
            sig = g.signature()
            if sig in pending_sigs or sig in queued_sigs:
                continue
            self._seed_queue.append(g.copy())
            queued_sigs.add(sig)
            added += 1
        return added

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
