"""adjacency.py 的正确性回归测试 (设备无关, 本地 CPU 即可全绿)。

验证:
  1. 阈值高斯核公式与 DCRNN 口径一致 (exp(-d²/σ²)、κ 阈值置零)
  2. 两条路径产出一致形态 (distance.csv 计算 vs adj_mx.pkl 加载)
  3. 归一化数学性质 (随机游走行和=1、对称归一化对称、孤立节点不崩)
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from darwin_st.data.adjacency import (
    add_self_loops,
    build_adj_from_distance_csv,
    build_adjacency,
    gaussian_kernel,
    load_adj_csv,
    load_adj_pkl,
    random_walk_normalize,
    symmetric_normalize,
)
from darwin_st.data.protocol import get_profile


# ---------------------------------------------------------------------------
# 高斯核公式
# ---------------------------------------------------------------------------


def test_gaussian_kernel_formula():
    """W_ij = exp(-d²/σ²), σ=有效距离标准差。"""
    # 三条有效距离 [1,2,3] (对称位置), 对角与无边为 inf
    dist = np.array([
        [np.inf, 1.0, 2.0],
        [1.0, np.inf, 3.0],
        [2.0, 3.0, np.inf],
    ])
    sigma = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]).std()  # 用全部有限元素
    out = gaussian_kernel(dist, normalized_k=0.0)  # κ=0 不裁剪, 纯看公式
    assert np.isclose(out[0, 1], np.exp(-(1.0 ** 2) / sigma ** 2), atol=1e-6)
    assert np.isclose(out[1, 2], np.exp(-(3.0 ** 2) / sigma ** 2), atol=1e-6)


def test_gaussian_kernel_inf_becomes_zero():
    """inf 距离 (无边/自距离) → exp(-inf) = 0。"""
    dist = np.array([[np.inf, 1.0], [1.0, np.inf]])
    out = gaussian_kernel(dist, normalized_k=0.0)
    assert out[0, 0] == 0.0
    assert out[1, 1] == 0.0


def test_gaussian_kernel_threshold_zeroes_weak_edges():
    """权重 < κ 的边被置 0 (稀疏化)。"""
    dist = np.array([
        [np.inf, 1.0, 10.0],
        [1.0, np.inf, 10.0],
        [10.0, 10.0, np.inf],
    ])
    out = gaussian_kernel(dist, normalized_k=0.5)
    # 远距离 (10) 的权重应很小 → 被 κ=0.5 裁掉
    assert out[0, 2] == 0.0
    # 近距离 (1) 的权重应较大 → 保留
    assert out[0, 1] > 0.0


def test_gaussian_kernel_all_inf_no_crash():
    """全 inf (无任何有效边) 应退化不崩。"""
    dist = np.full((3, 3), np.inf)
    out = gaussian_kernel(dist)
    assert np.all(out == 0.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# 路径一: distance.csv 构建
# ---------------------------------------------------------------------------


def test_build_from_distance_csv(tmp_path):
    csv = tmp_path / "distance.csv"
    csv.write_text("from,to,cost\n0,1,1.0\n1,2,2.0\n")
    adj = build_adj_from_distance_csv(str(csv), num_nodes=3, normalized_k=0.0)
    assert adj.shape == (3, 3)
    # 对称化: 0-1 边应双向
    assert adj[0, 1] > 0 and adj[1, 0] > 0
    assert np.isclose(adj[0, 1], adj[1, 0])  # 对称
    # 无边 0-2 应为 0
    assert adj[0, 2] == 0.0


def test_build_from_distance_csv_skips_out_of_range(tmp_path):
    """越界节点边被跳过, 不崩。"""
    csv = tmp_path / "distance.csv"
    csv.write_text("from,to,cost\n0,1,1.0\n0,99,5.0\n")  # 99 越界
    adj = build_adj_from_distance_csv(str(csv), num_nodes=2, normalized_k=0.0)
    assert adj.shape == (2, 2)
    assert adj[0, 1] > 0


def test_build_from_distance_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        build_adj_from_distance_csv("/nonexistent/distance.csv", num_nodes=3)


# ---------------------------------------------------------------------------
# 路径二: adj_mx.pkl 加载
# ---------------------------------------------------------------------------


def test_load_adj_pkl_three_tuple(tmp_path):
    """标准 DCRNN 3 元组格式。"""
    pkl = tmp_path / "adj_mx.pkl"
    sensor_ids = ["a", "b", "c"]
    sensor_id_to_ind = {"a": 0, "b": 1, "c": 2}
    adj_mx = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.0]], dtype=np.float32)
    with open(pkl, "wb") as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f)

    loaded = load_adj_pkl(str(pkl))
    assert loaded.shape == (3, 3)
    assert np.allclose(loaded, adj_mx)


def test_load_adj_pkl_bare_matrix(tmp_path):
    """兼容直接 pickle 裸矩阵的非标准情形。"""
    pkl = tmp_path / "adj_mx.pkl"
    mat = np.eye(4, dtype=np.float32)
    with open(pkl, "wb") as f:
        pickle.dump(mat, f)
    loaded = load_adj_pkl(str(pkl))
    assert loaded.shape == (4, 4)


def test_load_adj_pkl_missing_file():
    with pytest.raises(FileNotFoundError):
        load_adj_pkl("/nonexistent/adj_mx.pkl")


# ---------------------------------------------------------------------------
# 路径三: wam_*.csv 稠密邻接矩阵加载 (dcrnn_csv 谱系)
# ---------------------------------------------------------------------------


def test_load_adj_csv_dense_matrix(tmp_path):
    wam = tmp_path / "wam_metr_la.csv"
    mat = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.0]])
    wam.write_text("\n".join(",".join(str(v) for v in row) for row in mat))
    loaded = load_adj_csv(str(wam))
    assert loaded.shape == (3, 3)
    assert np.allclose(loaded, mat)


def test_load_adj_csv_missing_file():
    with pytest.raises(FileNotFoundError):
        load_adj_csv("/nonexistent/wam.csv")


def test_build_adjacency_dcrnn_csv_path(tmp_path):
    """METR-LA/PEMS-BAY 谱系走 wam_*.csv 加载路径。"""
    from darwin_st.data.protocol import DatasetProfile

    wam = tmp_path / "wam_metr_la.csv"
    mat = np.eye(3, dtype=np.float32)
    wam.write_text("\n".join(",".join(str(v) for v in row) for row in mat))
    mini = DatasetProfile(
        name="METR-LA", num_nodes=3, num_channels=1, target_channel=0,
        split_ratios=(0.7, 0.1, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="per_horizon", report_horizons=(3, 6, 12), null_val=0.0,
        graph_source="dcrnn_csv",
    )
    adj = build_adjacency(mini, str(tmp_path), normalize=None)
    assert adj.shape == (3, 3)
    assert np.allclose(adj, mat)


# ---------------------------------------------------------------------------
# 归一化数学性质
# ---------------------------------------------------------------------------


def test_self_loops():
    adj = np.zeros((3, 3))
    out = add_self_loops(adj)
    assert np.allclose(np.diag(out), 1.0)


def test_random_walk_row_sums_to_one():
    """随机游走归一化后每行 (有度的) 和为 1 = 转移概率。"""
    adj = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = random_walk_normalize(adj, add_self_loop=True)
    row_sums = out.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_symmetric_normalize_is_symmetric():
    """对称归一化结果应保持对称 (输入对称时)。"""
    adj = np.array([[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    out = symmetric_normalize(adj, add_self_loop=True)
    assert np.allclose(out, out.T, atol=1e-6)


def test_normalize_isolated_node_no_nan():
    """孤立节点 (度 0) 不应产生 nan/inf。"""
    # 节点 2 完全孤立 (不加自环时)
    adj = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sym = symmetric_normalize(adj, add_self_loop=False)
    rw = random_walk_normalize(adj, add_self_loop=False)
    assert np.all(np.isfinite(sym))
    assert np.all(np.isfinite(rw))


# ---------------------------------------------------------------------------
# 统一入口: 按 profile 分流
# ---------------------------------------------------------------------------


def test_build_adjacency_distance_csv_path(tmp_path):
    """PeMS 谱系走 distance.csv 计算路径。需 num_nodes 对齐, 用迷你 profile 替身。"""
    from darwin_st.data.protocol import DatasetProfile

    csv = tmp_path / "distance.csv"
    csv.write_text("from,to,cost\n0,1,1.0\n1,2,1.0\n")
    mini = DatasetProfile(
        name="mini", num_nodes=3, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=(12,), null_val=0.0,
        graph_source="distance_csv",
    )
    adj = build_adjacency(mini, str(tmp_path), normalize=None)
    assert adj.shape == (3, 3)


def test_build_adjacency_pkl_path_with_node_check(tmp_path):
    """speed 谱系走 adj_mx.pkl 加载路径, 并校验节点数。"""
    from darwin_st.data.protocol import DatasetProfile

    pkl = tmp_path / "adj_mx.pkl"
    adj_mx = np.eye(5, dtype=np.float32)
    with open(pkl, "wb") as f:
        pickle.dump((None, None, adj_mx), f)
    mini = DatasetProfile(
        name="mini5", num_nodes=5, num_channels=1, target_channel=0,
        split_ratios=(0.7, 0.1, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="per_horizon", report_horizons=(3, 6, 12), null_val=0.0,
        graph_source="adj_pkl",
    )
    adj = build_adjacency(mini, str(tmp_path), normalize="rw")
    assert adj.shape == (5, 5)
    # 单位阵 + 自环 → 随机游走行和为 1
    assert np.allclose(adj.sum(axis=1), 1.0, atol=1e-6)


def test_build_adjacency_node_mismatch_raises(tmp_path):
    """adj_mx 节点数与 profile 不符应报错 (防张冠李戴)。"""
    from darwin_st.data.protocol import DatasetProfile

    pkl = tmp_path / "adj_mx.pkl"
    with open(pkl, "wb") as f:
        pickle.dump((None, None, np.eye(3, dtype=np.float32)), f)
    mini = DatasetProfile(
        name="mismatch", num_nodes=207, num_channels=1, target_channel=0,
        split_ratios=(0.7, 0.1, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="per_horizon", report_horizons=(3, 6, 12), null_val=0.0,
        graph_source="adj_pkl",
    )
    with pytest.raises(ValueError):
        build_adjacency(mini, str(tmp_path))


def test_build_adjacency_bad_normalize_mode(tmp_path):
    csv = tmp_path / "distance.csv"
    csv.write_text("from,to,cost\n0,1,1.0\n")
    prof = get_profile("PeMS04")
    # 节点数对齐: 用真实 profile 但只有 2 节点的 csv → 仍能构建 307×307 (其余为 0)
    with pytest.raises(ValueError):
        build_adjacency(prof, str(tmp_path), normalize="bogus")
