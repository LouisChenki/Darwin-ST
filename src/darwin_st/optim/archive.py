"""MAP-Elites 质量-多样性档案 (Quality-Diversity Archive)。

实现 docs/P2_ALGORITHM_DESIGN.md §5 选定的小网格 MAP-Elites + aging 混合, 专为
小评估预算 (~100-300 trial) 特化 (纯 MAP-Elites 需 10^5 评估, 不可行)。

核心:
  - 网格按行为描述子 (BD) 离散, 每格只留最优精英 (严格改进才替换)。
  - BD 四轴 (108 格): 主导空间族 × 主导时序族 × 参数量档 × 深度档 —— 从 genotype 直接算, 无需训练。
  - 选择 = 混合: 40% 从填充格均匀采 (QD 多样性), 60% 最近精英锦标赛 (aging 抗噪)。
  - 停滞 → 岛屿重置 (FunSearch 式): 清差的一半格, 从最优精英重播种。
  - empty_cells() 暴露未探索区, 供 LLM 定向填充 (ICQD 式)。

BD 不能与 fitness 相关 (否则 QD 退化成纯优化) —— 空间/时序算子族与精度正交, 是好 BD。
**深度轴 (depth_bucket) 是关键**: 无它则 depth-2 与 depth-4 若家族+参数档相同会落同一格竞争,
便宜的浅架构挤掉深架构 → 搜索坍缩在浅层 (实测 depth-4 一次没探到)。加深度轴让每个深度带各留 elite,
浅的挤不掉深的 (QD/MAP-Elites for NAS 破容量坍缩的标准解, 见 Schneider 2022)。深度与 masked-MAE 正交, 是合法 BD。
fitness 约定: 越小越优 (masked val-MAE)。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from darwin_st.search.genotype import Genotype

__all__ = ["Elite", "behavior_descriptor", "cell_index", "MAPElitesArchive",
           "SPATIAL_FAMILIES", "TEMPORAL_FAMILIES", "PARAM_BUCKETS", "DEPTH_BUCKETS"]

# 主导空间族分桶 (4): 把 6 个空间算子归并为 4 个有意义的族
SPATIAL_FAMILIES = {
    "identity": "none",
    "gcn": "local_conv", "cheb": "local_conv",
    "gat": "attention",
    "diffusion": "diffusion_adaptive", "adaptive": "diffusion_adaptive",
}
_SPATIAL_FAMILY_ORDER = ["none", "local_conv", "attention", "diffusion_adaptive"]

# 主导时序族分桶 (3)
TEMPORAL_FAMILIES = {
    "identity": "conv",  # identity 极少作主导; 归入 conv 桶避免空族
    "tcn": "conv",
    "gru": "recurrent",
    "attn": "attention",
}
_TEMPORAL_FAMILY_ORDER = ["conv", "recurrent", "attention"]

# 参数量档 (3): <50k / 50k-200k / >200k
PARAM_BUCKETS = [(0, 50_000), (50_000, 200_000), (200_000, float("inf"))]
_N_PARAM_BUCKETS = len(PARAM_BUCKETS)

# 深度档 (3): 1-2 / 3-4 / 5+。用字符串标签自描述 (且不和 int param_bucket 视觉混)。
# 用 3 个带而非逐深度: 小评估预算下逐深度过碎; 3 带映射实测缺口 (depth-2 坍缩 vs depth-4 目标)。
DEPTH_BUCKETS = [(1, 3), (3, 5), (5, float("inf"))]
_DEPTH_BUCKET_LABELS = ["1-2", "3-4", "5+"]

# synth 前缀 (与 operators.SYNTH_PREFIX 一致; 此处本地常量避免 archive→operators 反向依赖膨胀)
_SYNTH_PREFIX = "synth_"


def _synth_family(op: str, families_order: list[str]) -> str:
    """把 synth 算子按注册名稳定哈希散到已有族 (而非全塌默认族)。

    根因 C 修复: synth 算子在 SPATIAL/TEMPORAL_FAMILIES 字典里查不到 → .get 全落 none/conv →
    368/368 KEEP 架构落同一族, 108 格 QD 网格塌成 ~9 格, 深度保护失效。按注册名哈希散开,
    保证不同 synth 算子分布到不同族, 网格重新展开。哈希用名字 (稳定, 跨进程一致), 排除 'none'
    (none 语义是"无空间算子", 不该被 synth 占用)。
    """
    import hashlib
    pool = [f for f in families_order if f != "none"] or families_order
    h = int(hashlib.sha1(op.encode()).hexdigest(), 16)
    return pool[h % len(pool)]


def _family_of(op: str, table: dict, families_order: list[str], default: str) -> str:
    """查算子所属族: 内置查 table; synth_ 前缀走稳定哈希散族; 其余落 default。"""
    if op in table:
        return table[op]
    if op.startswith(_SYNTH_PREFIX):
        return _synth_family(op, families_order)
    return default


def _dominant_spatial_family(geno: Genotype) -> str:
    """该 genotype 的主导空间族 = 出现最多的空间算子所属族 (平局取更深块的)。"""
    counts: dict[str, int] = {}
    for b in geno.blocks:
        fam = _family_of(b.spatial_op, SPATIAL_FAMILIES, _SPATIAL_FAMILY_ORDER, "none")
        counts[fam] = counts.get(fam, 0) + 1
    # 平局时按 _SPATIAL_FAMILY_ORDER 靠后 (更"重") 优先
    return max(counts, key=lambda f: (counts[f], _SPATIAL_FAMILY_ORDER.index(f)))


def _dominant_temporal_family(geno: Genotype) -> str:
    counts: dict[str, int] = {}
    for b in geno.blocks:
        fam = _family_of(b.temporal_op, TEMPORAL_FAMILIES, _TEMPORAL_FAMILY_ORDER, "conv")
        counts[fam] = counts.get(fam, 0) + 1
    return max(counts, key=lambda f: (counts[f], _TEMPORAL_FAMILY_ORDER.index(f)))


def _param_bucket(num_params: int) -> int:
    for i, (lo, hi) in enumerate(PARAM_BUCKETS):
        if lo <= num_params < hi:
            return i
    return _N_PARAM_BUCKETS - 1


def _depth_bucket(depth: int) -> str:
    """深度 → 深度档标签 (1-2 / 3-4 / 5+)。depth = len(genotype.blocks)。"""
    for (lo, hi), label in zip(DEPTH_BUCKETS, _DEPTH_BUCKET_LABELS):
        if lo <= depth < hi:
            return label
    return _DEPTH_BUCKET_LABELS[-1]


def behavior_descriptor(geno: Genotype, num_params: int) -> dict:
    """计算行为描述子 (BD): {spatial_family, temporal_family, param_bucket, depth_bucket}。

    num_params 由调用方提供 (orchestrator 从 builder 实例化的模型取), 无需训练。
    depth_bucket 让 MAP-Elites 在深度上保持多样性 (深架构不被浅架构挤掉)。
    """
    return {
        "spatial_family": _dominant_spatial_family(geno),
        "temporal_family": _dominant_temporal_family(geno),
        "param_bucket": _param_bucket(num_params),
        "depth_bucket": _depth_bucket(geno.depth),
    }


def cell_index(bd: dict) -> tuple[str, str, int, str]:
    """BD → 网格 cell 键 (可哈希元组)。"""
    return (bd["spatial_family"], bd["temporal_family"], bd["param_bucket"], bd["depth_bucket"])


def all_cells() -> list[tuple[str, str, int, str]]:
    """枚举全部 108 个可能 cell (用于 coverage / empty_cells)。"""
    return [
        (s, t, p, d)
        for s in _SPATIAL_FAMILY_ORDER
        for t in _TEMPORAL_FAMILY_ORDER
        for p in range(_N_PARAM_BUCKETS)
        for d in _DEPTH_BUCKET_LABELS
    ]


@dataclass
class Elite:
    """一个 cell 的精英: genotype + fitness + 元信息。"""

    genotype: Genotype
    fitness: float                 # masked val-MAE, 越小越优
    num_params: int
    bd: dict
    trial_id: int | None = None
    insert_order: int = 0          # 插入序号 (aging 用)


class MAPElitesArchive:
    """MAP-Elites 网格档案 + aging 混合选择。

    用法 (orchestrator 驱动):
        arc = MAPElitesArchive(seed=0)
        inserted = arc.add(geno, fitness=18.5, num_params=120_000)   # 严格改进才入
        parent = arc.select()        # 混合选择下一个父代
        if arc.stagnated(): arc.island_reset()
    """

    TOTAL_CELLS = len(all_cells())  # 108 (4 空间族 × 3 时序族 × 3 参数档 × 3 深度档)

    def __init__(self, seed: int = 0, p_uniform: float = 0.4, tournament_size: int = 4,
                 stagnation_patience: int = 20):
        import random
        self.rng = random.Random(seed)
        self.p_uniform = p_uniform
        self.tournament_size = tournament_size
        self.stagnation_patience = stagnation_patience

        self.cells: dict[tuple, Elite] = {}        # cell → 精英
        self._order = 0                            # 全局插入计数
        self._recent: deque = deque(maxlen=64)     # 最近插入的 cell 键 (aging 锦标赛池)
        self._best_fitness = float("inf")
        self._since_improve = 0                    # 距上次全局最优改进的轮数

    # -- 插入 --
    def add(self, geno: Genotype, fitness: float, num_params: int,
            trial_id: int | None = None) -> bool:
        """尝试把 genotype 放进对应 cell。每格精英, **严格改进才替换**。返回是否入选。"""
        bd = behavior_descriptor(geno, num_params)
        key = cell_index(bd)
        incumbent = self.cells.get(key)
        if incumbent is not None and fitness >= incumbent.fitness:
            self._since_improve += 1
            return False  # 没打败现任 (严格 <), 不入

        self._order += 1
        self.cells[key] = Elite(genotype=geno, fitness=fitness, num_params=num_params,
                                bd=bd, trial_id=trial_id, insert_order=self._order)
        self._recent.append(key)
        if fitness < self._best_fitness:
            self._best_fitness = fitness
            self._since_improve = 0
        else:
            self._since_improve += 1
        return True

    # -- 选择 (混合 QD + aging) --
    def select(self) -> Elite | None:
        """选一个父代精英: p_uniform 概率均匀采填充格 (多样性); 否则最近精英锦标赛 (aging)。"""
        if not self.cells:
            return None
        if self.rng.random() < self.p_uniform:
            return self.rng.choice(list(self.cells.values()))     # QD: 均匀采
        return self._tournament_recent()                          # aging: 最近精英锦标赛

    def _tournament_recent(self) -> Elite:
        # 从最近插入的 cell (去重且仍存活) 里锦标赛, 偏向新近精英
        recent_keys = [k for k in dict.fromkeys(reversed(self._recent)) if k in self.cells]
        pool = recent_keys[: max(self.tournament_size * 2, 8)] or list(self.cells.keys())
        k = min(self.tournament_size, len(pool))
        contenders = [self.cells[key] for key in self.rng.sample(pool, k)]
        return min(contenders, key=lambda e: e.fitness)

    # -- 岛屿重置 (停滞解法) --
    def stagnated(self) -> bool:
        return self._since_improve >= self.stagnation_patience

    def island_reset(self) -> int:
        """清掉较差的一半 cell, 保留较优一半 (FunSearch 式重启多样性)。返回清掉的数量。

        被清的区域成为 empty_cells, 可交给 LLM 定向填充。重置改进计数器。
        """
        if len(self.cells) <= 1:
            self._since_improve = 0
            return 0
        ranked = sorted(self.cells.items(), key=lambda kv: kv[1].fitness)
        keep_n = max(1, len(ranked) // 2)
        survivors = dict(ranked[:keep_n])
        cleared = len(self.cells) - len(survivors)
        self.cells = survivors
        self._recent = deque((k for k in self._recent if k in self.cells), maxlen=64)
        self._since_improve = 0
        return cleared

    # -- 查询 --
    def best(self) -> Elite | None:
        if not self.cells:
            return None
        return min(self.cells.values(), key=lambda e: e.fitness)

    def filled_cells(self) -> list[tuple]:
        return list(self.cells.keys())

    def empty_cells(self) -> list[tuple]:
        """未填充的 cell (供 LLM 定向探索)。"""
        return [c for c in all_cells() if c not in self.cells]

    def coverage(self) -> float:
        """填充率 = 已填 / 总 36 格。"""
        return len(self.cells) / self.TOTAL_CELLS

    def qd_score(self) -> float:
        """QD-score = Σ 各 cell (1/(1+fitness)), 同时奖励覆盖广度与单格质量 (越大越好)。"""
        return sum(1.0 / (1.0 + e.fitness) for e in self.cells.values())

    def elites(self) -> list[Elite]:
        return list(self.cells.values())

    def __len__(self) -> int:
        return len(self.cells)
