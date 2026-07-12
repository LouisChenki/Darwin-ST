"""archive.py 的正确性回归测试 (设备无关)。

核心契约:
  - BD 计算: 主导空间/时序族 + 参数量档正确
  - 插入: 每格精英, 严格改进才替换
  - coverage/empty_cells/qd_score
  - 混合选择返回精英
  - 岛屿重置清差半留优半
  - 不同 BD 进不同 cell (多样性), 同 BD 竞争同 cell
"""

from __future__ import annotations

from darwin_st.optim.archive import (
    MAPElitesArchive,
    behavior_descriptor,
    cell_index,
    all_cells,
)
from darwin_st.search.genotype import Genotype, STBlock, random_genotype


def _geno(spatial="gcn", temporal="tcn", depth=1):
    return Genotype(blocks=[STBlock(spatial, temporal) for _ in range(depth)])


# ---------------------------------------------------------------------------
# 行为描述子
# ---------------------------------------------------------------------------


def test_bd_spatial_families():
    assert behavior_descriptor(_geno("gcn"), 1000)["spatial_family"] == "local_conv"
    assert behavior_descriptor(_geno("cheb"), 1000)["spatial_family"] == "local_conv"
    assert behavior_descriptor(_geno("gat"), 1000)["spatial_family"] == "attention"
    assert behavior_descriptor(_geno("diffusion"), 1000)["spatial_family"] == "diffusion_adaptive"
    assert behavior_descriptor(_geno("adaptive"), 1000)["spatial_family"] == "diffusion_adaptive"
    assert behavior_descriptor(_geno("identity"), 1000)["spatial_family"] == "none"


def test_bd_temporal_families():
    assert behavior_descriptor(_geno(temporal="tcn"), 1000)["temporal_family"] == "conv"
    assert behavior_descriptor(_geno(temporal="gru"), 1000)["temporal_family"] == "recurrent"
    assert behavior_descriptor(_geno(temporal="attn"), 1000)["temporal_family"] == "attention"


def test_bd_param_buckets():
    assert behavior_descriptor(_geno(), 10_000)["param_bucket"] == 0    # <50k
    assert behavior_descriptor(_geno(), 100_000)["param_bucket"] == 1   # 50k-200k
    assert behavior_descriptor(_geno(), 500_000)["param_bucket"] == 2   # >200k


def test_bd_dominant_family_by_count():
    """多块时主导族 = 出现最多的。"""
    g = Genotype(blocks=[STBlock("gcn", "tcn"), STBlock("gcn", "tcn"), STBlock("gat", "tcn")])
    assert behavior_descriptor(g, 1000)["spatial_family"] == "local_conv"  # gcn x2 > gat x1


def test_all_cells_is_108():
    """4 轴: 空间族(4) × 时序族(3) × 参数档(3) × 深度档(3) = 108。"""
    assert len(all_cells()) == 108
    assert len(set(all_cells())) == 108   # 无重复


def test_bd_depth_buckets():
    """深度档: 1-2 / 3-4 / 5+。"""
    assert behavior_descriptor(_geno(depth=1), 1000)["depth_bucket"] == "1-2"
    assert behavior_descriptor(_geno(depth=2), 1000)["depth_bucket"] == "1-2"
    assert behavior_descriptor(_geno(depth=3), 1000)["depth_bucket"] == "3-4"
    assert behavior_descriptor(_geno(depth=4), 1000)["depth_bucket"] == "3-4"
    assert behavior_descriptor(_geno(depth=5), 1000)["depth_bucket"] == "5+"
    assert behavior_descriptor(_geno(depth=8), 1000)["depth_bucket"] == "5+"


def test_cell_index_is_4_tuple():
    bd = behavior_descriptor(_geno("gcn", "tcn", depth=2), 10_000)
    ci = cell_index(bd)
    assert len(ci) == 4
    assert ci == ("local_conv", "conv", 0, "1-2")


def test_depth_separates_cells():
    """核心回归 (守住浅层坍缩 bug): 家族+参数档相同但深度不同 → 落不同 cell,
    且更差的深架构**不被**更好的浅架构挤掉 (深探索是零成本加保险)。"""
    arc = MAPElitesArchive(seed=0)
    shallow = _geno("gcn", "tcn", depth=2)   # local_conv/conv/param/1-2
    deep = _geno("gcn", "tcn", depth=4)      # local_conv/conv/param/3-4 (仅深度档不同)
    # 同 num_params 强制家族+参数档一致, 只有深度档区分
    assert cell_index(behavior_descriptor(shallow, 60_000)) \
        != cell_index(behavior_descriptor(deep, 60_000))
    arc.add(shallow, fitness=18.0, num_params=60_000)   # 好的浅
    arc.add(deep, fitness=20.0, num_params=60_000)      # 差的深 —— 不该被挤掉
    assert len(arc) == 2, "深架构被浅架构挤掉了 (坍缩 bug 复现)"


# ---------------------------------------------------------------------------
# 插入: 每格精英 + 严格改进
# ---------------------------------------------------------------------------


def test_first_insert_succeeds():
    arc = MAPElitesArchive(seed=0)
    assert arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000) is True
    assert len(arc) == 1


def test_strict_improvement_replaces():
    arc = MAPElitesArchive(seed=0)
    # 同 BD (gcn/tcn/<50k) 竞争同一 cell
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)
    assert arc.add(_geno("gcn", "tcn"), fitness=18.0, num_params=12_000) is True  # 更优 → 替换
    assert arc.add(_geno("gcn", "tcn"), fitness=19.0, num_params=11_000) is False  # 更差 → 拒
    assert arc.add(_geno("gcn", "tcn"), fitness=18.0, num_params=11_000) is False  # 相等 → 拒(严格)
    assert len(arc) == 1
    assert arc.best().fitness == 18.0


def test_different_bd_different_cells():
    """不同 BD 进不同 cell (多样性保持)。"""
    arc = MAPElitesArchive(seed=0)
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)      # local_conv/conv/0
    arc.add(_geno("gat", "gru"), fitness=22.0, num_params=10_000)      # attention/recurrent/0
    arc.add(_geno("gcn", "tcn"), fitness=21.0, num_params=300_000)     # local_conv/conv/2
    assert len(arc) == 3  # 三个不同 cell


# ---------------------------------------------------------------------------
# coverage / empty / qd
# ---------------------------------------------------------------------------


def test_coverage_and_empty():
    arc = MAPElitesArchive(seed=0)
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)
    assert arc.coverage() == 1 / 108
    assert len(arc.empty_cells()) == 107
    assert cell_index(behavior_descriptor(_geno("gcn", "tcn"), 10_000)) not in arc.empty_cells()


def test_qd_score_increases_with_better_elites():
    arc = MAPElitesArchive(seed=0)
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)
    q1 = arc.qd_score()
    arc.add(_geno("gcn", "tcn"), fitness=10.0, num_params=10_000)  # 更优精英
    assert arc.qd_score() > q1  # 质量提升 → QD 升


# ---------------------------------------------------------------------------
# 选择
# ---------------------------------------------------------------------------


def test_select_returns_elite():
    arc = MAPElitesArchive(seed=0)
    assert arc.select() is None  # 空档案
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)
    e = arc.select()
    assert e is not None
    assert e.fitness == 20.0


def test_select_mix_uniform_and_tournament():
    """混合选择: 多次选择应能取到不同精英 (有多样性)。"""
    arc = MAPElitesArchive(seed=0, p_uniform=0.5)
    pool = ["gcn", "gat", "cheb", "diffusion", "adaptive"]
    for i, sp in enumerate(pool):
        arc.add(_geno(sp, "tcn"), fitness=20.0 - i, num_params=10_000 + i * 100_000)
    picks = {id(arc.select()) for _ in range(50)}
    assert len(picks) >= 2  # 不是每次都选同一个


# ---------------------------------------------------------------------------
# 岛屿重置
# ---------------------------------------------------------------------------


def test_stagnation_detection():
    arc = MAPElitesArchive(seed=0, stagnation_patience=3)
    arc.add(_geno("gcn", "tcn"), fitness=20.0, num_params=10_000)
    assert not arc.stagnated()
    # 连续没改进 (更差插入)
    for _ in range(4):
        arc.add(_geno("gcn", "tcn"), fitness=25.0, num_params=10_000)  # 被拒, since_improve++
    assert arc.stagnated()


def test_island_reset_keeps_better_half():
    arc = MAPElitesArchive(seed=0)
    pool = ["gcn", "gat", "cheb", "diffusion"]
    # 4 个不同 cell, fitness 20/19/18/17
    for i, sp in enumerate(pool):
        arc.add(_geno(sp, "tcn"), fitness=20.0 - i, num_params=10_000 + i * 100_000)
    assert len(arc) == 4
    cleared = arc.island_reset()
    assert cleared == 2          # 清掉差的一半
    assert len(arc) == 2
    # 保留的应是较优的 (fitness 17, 18)
    fits = sorted(e.fitness for e in arc.elites())
    assert fits == [17.0, 18.0]
    assert not arc.stagnated()   # 计数器重置


def test_island_reset_clears_to_empty_cells():
    """重置后被清的格成为 empty_cells (供 LLM 定向)。"""
    arc = MAPElitesArchive(seed=0)
    for i, sp in enumerate(["gcn", "gat", "cheb", "diffusion"]):
        arc.add(_geno(sp, "tcn"), fitness=20.0 - i, num_params=10_000 + i * 100_000)
    before_empty = len(arc.empty_cells())
    arc.island_reset()
    assert len(arc.empty_cells()) > before_empty  # 清出更多空格


# ---------------------------------------------------------------------------
# 端到端: 档案在搜索流中累积多样精英
# ---------------------------------------------------------------------------


def test_archive_accumulates_diverse_elites():
    import random
    rng = random.Random(0)
    arc = MAPElitesArchive(seed=0)
    pool_s = ["gcn", "gat", "cheb", "diffusion", "adaptive", "identity"]
    pool_t = ["tcn", "gru", "attn"]
    for _ in range(100):
        g = _geno(rng.choice(pool_s), rng.choice(pool_t))
        arc.add(g, fitness=rng.uniform(15, 25), num_params=rng.choice([10_000, 100_000, 300_000]))
    # 应填充多个 cell (多样性), 但不超过 36
    assert 5 <= len(arc) <= 36
    assert arc.best() is not None


# ---------------------------------------------------------------------------
# 根因 C 修复: BD 对 synth 算子分族 (不再全塌 none/conv)
# ---------------------------------------------------------------------------


def test_synth_ops_disperse_across_families():
    """多个不同 synth 空间算子应散到多个空间族 (而非全塌 'none'), 否则 108 格 QD 塌缩。"""
    from darwin_st.optim.archive import _dominant_spatial_family
    fams = set()
    for i in range(12):
        g = Genotype(blocks=[STBlock(f"synth_op_{i}", "tcn")])
        fams.add(_dominant_spatial_family(g))
    assert "none" not in fams        # synth 不该被判成"无空间算子"
    assert len(fams) >= 2            # 散开到多族 (哈希分散)


def test_synth_temporal_disperse_not_all_conv():
    """多个 synth 时序算子不应全落 'conv' 族。"""
    from darwin_st.optim.archive import _dominant_temporal_family
    fams = set()
    for i in range(12):
        g = Genotype(blocks=[STBlock("gcn", f"synth_t_{i}")])
        fams.add(_dominant_temporal_family(g))
    assert len(fams) >= 2


def test_synth_family_is_stable():
    """同一 synth 名多次映射到同一族 (稳定哈希, 跨进程一致)。"""
    from darwin_st.optim.archive import _dominant_spatial_family
    g = Genotype(blocks=[STBlock("synth_stable_xyz", "tcn")])
    f1 = _dominant_spatial_family(g)
    f2 = _dominant_spatial_family(g)
    assert f1 == f2


def test_builtin_family_unchanged():
    """内置算子分族行为不变 (回归): gcn→local_conv, gat→attention。"""
    from darwin_st.optim.archive import _dominant_spatial_family
    assert _dominant_spatial_family(Genotype(blocks=[STBlock("gcn", "tcn")])) == "local_conv"
    assert _dominant_spatial_family(Genotype(blocks=[STBlock("gat", "tcn")])) == "attention"
    assert _dominant_spatial_family(Genotype(blocks=[STBlock("identity", "tcn")])) == "none"


def test_all_cells_still_108():
    """网格总格数保持 108 (4×3×3×3)。"""
    assert len(all_cells()) == 108
    assert MAPElitesArchive.TOTAL_CELLS == 108
