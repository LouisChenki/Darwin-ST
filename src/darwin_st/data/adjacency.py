"""邻接矩阵构建 (Adjacency Matrix Construction) —— 图卷积类模型的静态先验图。

时空预测的两条数据集谱系提供的「图原料」形态不同, 但**数学口径必须一致**:

  - METR-LA / PEMS-BAY: 仓库自带 `adj_mx.pkl` (已算好的 N×N 加权矩阵), 直接加载。
  - PeMS04 / PeMS08:   仓库只给 `distance.csv` 边列表, 需用**阈值高斯核**算成矩阵。

两者用的是**同一个公式** (DCRNN gen_adj_mx.py):

    W_ij = exp(-dist(i,j)² / σ²)   若 ≥ 阈值 κ, 否则置 0
    σ = 所有【有效边】距离的标准差;  κ 默认 0.1

这样「加载」与「计算」两条路产出的是同一种东西, 跨谱系可比。

⚠️ 本模块产出的是**可选的静态先验图**, 仅供需要预定义图的模型 (DCRNN/STGCN/ASTGCN)。
   自学习图模型 (Graph WaveNet/MTGNN/AGCRN) 与无图模型 (STID/STAEformer) 不使用它。

参考: DCRNN gen_adj_mx.py  https://github.com/liyaguang/DCRNN
本模块为 numpy 实现 (数据准备时, CPU), 设备无关, 可本地 pytest 验证。
"""

from __future__ import annotations

import os
import pickle

import numpy as np

__all__ = [
    "gaussian_kernel",
    "build_adj_from_distance_csv",
    "load_adj_pkl",
    "load_adj_csv",
    "add_self_loops",
    "symmetric_normalize",
    "random_walk_normalize",
    "build_adjacency",
]


# ---------------------------------------------------------------------------
# 核心: 阈值高斯核 (两条谱系共用的统一公式)
# ---------------------------------------------------------------------------


def gaussian_kernel(dist_mx: np.ndarray, normalized_k: float = 0.1) -> np.ndarray:
    """对距离矩阵施加阈值高斯核, 返回加权邻接矩阵。

    dist_mx: N×N 距离矩阵, **无边处填 np.inf** (exp(-inf)=0 自动断开)。
             对角线一般也填 inf (无自距离), 自环由归一化阶段显式添加。
    normalized_k: 稀疏化阈值 κ, 小于此权重的边置 0 (默认 0.1, 同 DCRNN)。

    σ 取所有【有效(有限)】距离的标准差——若全为 inf 则退化为不缩放(σ=1)。
    """
    finite = dist_mx[~np.isinf(dist_mx)]
    std = float(finite.std()) if finite.size > 0 else 0.0
    if std == 0.0:
        std = 1.0  # 退化保护: 无有效边或零方差时不缩放

    adj = np.exp(-np.square(dist_mx / std))
    adj[adj < normalized_k] = 0.0
    return adj.astype(np.float32)


# ---------------------------------------------------------------------------
# 路径一: 从 distance.csv 构建 (PeMS04 / PeMS08)
# ---------------------------------------------------------------------------


def build_adj_from_distance_csv(
    csv_path: str,
    num_nodes: int,
    normalized_k: float = 0.1,
    symmetric: bool = True,
) -> np.ndarray:
    """从边列表 distance.csv 构建加权邻接矩阵 (PeMS04/08 谱系)。

    csv 列约定: from,to,cost  (节点为 0-indexed 整数, cost 为距离)。首行表头会被跳过。
    symmetric=True 时把有向距离对称化 (取两方向较小距离), 符合 PeMS 路网无向惯例。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"distance.csv 不存在: {csv_path}")

    dist_mx = np.full((num_nodes, num_nodes), np.inf, dtype=np.float64)

    # 用 numpy 读三列, 跳过可能的表头
    raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=np.float64)
    if raw.ndim == 1:  # 仅一行边
        raw = raw[None, :]

    for row in raw:
        i, j, cost = int(row[0]), int(row[1]), float(row[2])
        if i >= num_nodes or j >= num_nodes:
            continue  # 越界节点跳过 (防脏数据)
        dist_mx[i, j] = cost
        if symmetric:
            # 对称化: 取两方向已知距离的较小者
            dist_mx[j, i] = min(dist_mx[j, i], cost)

    return gaussian_kernel(dist_mx, normalized_k=normalized_k)


# ---------------------------------------------------------------------------
# 路径二: 加载现成 adj_mx.pkl (METR-LA / PEMS-BAY)
# ---------------------------------------------------------------------------


def load_adj_pkl(pkl_path: str) -> np.ndarray:
    """加载 DCRNN 格式的 adj_mx.pkl, 返回 N×N 加权邻接矩阵。

    文件为 3 元组 (sensor_ids, sensor_id_to_ind, adj_mx)。该 pkl 多由 Python2 生成,
    故用 latin1 编码兜底反序列化。也兼容直接 pickle 了裸矩阵的非标准情形。
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"adj_mx.pkl 不存在: {pkl_path}")

    with open(pkl_path, "rb") as f:
        try:
            obj = pickle.load(f)
        except UnicodeDecodeError:
            f.seek(0)
            obj = pickle.load(f, encoding="latin1")

    # 标准: 3 元组的第三项是矩阵; 兼容: 直接是矩阵
    if isinstance(obj, (tuple, list)) and len(obj) == 3:
        adj = np.asarray(obj[2], dtype=np.float32)
    else:
        adj = np.asarray(obj, dtype=np.float32)

    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adj_mx 形状非法: {adj.shape}, 期望方阵")
    return adj


def load_adj_csv(csv_path: str) -> np.ndarray:
    """加载稠密 N×N 邻接矩阵 CSV (hazdzz/dcrnn_data 的 wam_*.csv)。

    该文件已是算好的加权邻接矩阵 (无表头、无索引列), 直接读取即可——
    它与 adj_mx.pkl 同源 (同一阈值高斯核生成), 口径一致。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"邻接矩阵 CSV 不存在: {csv_path}")
    adj = np.genfromtxt(csv_path, delimiter=",", dtype=np.float32)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"wam 邻接矩阵形状非法: {adj.shape}, 期望方阵")
    return adj


# ---------------------------------------------------------------------------
# 归一化 (供图卷积使用) —— 对称 / 随机游走 两种
# ---------------------------------------------------------------------------


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """Ã = A + I (Kipf GCN 惯例: 聚合时纳入节点自身)。"""
    return adj + np.eye(adj.shape[0], dtype=adj.dtype)


def symmetric_normalize(adj: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """对称归一化 D^{-1/2} Ã D^{-1/2} (STGCN/ASTGCN 的 Chebyshev 卷积用)。

    孤立节点 (度为 0) 的 D^{-1/2} 用 0 兜底, 避免除零产生 inf/nan。
    """
    a = add_self_loops(adj) if add_self_loop else adj.copy()
    deg = a.sum(axis=1)
    # out= 预置 0, 仅在 deg>0 处求幂; 孤立节点 (deg<=0) 保持 0, 避免除零与未初始化内存
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0, out=np.zeros_like(deg, dtype=np.float64))
    d_mat = np.diag(d_inv_sqrt)
    return (d_mat @ a @ d_mat).astype(np.float32)


def random_walk_normalize(adj: np.ndarray, add_self_loop: bool = True) -> np.ndarray:
    """随机游走归一化 D^{-1} Ã (DCRNN/Graph WaveNet 的扩散卷积用)。

    每行归一化为转移概率 (行和为 1, 孤立节点行全 0)。
    """
    a = add_self_loops(adj) if add_self_loop else adj.copy()
    deg = a.sum(axis=1)
    # out= 预置 0, 仅在 deg>0 处求倒数; 孤立节点行保持全 0
    d_inv = np.power(deg, -1.0, where=deg > 0, out=np.zeros_like(deg, dtype=np.float64))
    d_mat = np.diag(d_inv)
    return (d_mat @ a).astype(np.float32)


# ---------------------------------------------------------------------------
# 统一入口: 按 profile 分流
# ---------------------------------------------------------------------------


def build_adjacency(
    profile,
    data_dir: str,
    normalize: str | None = None,
    normalized_k: float = 0.1,
):
    """按数据集 profile 自动选择「加载」或「计算」路径, 返回邻接矩阵。

    profile: protocol.DatasetProfile (用其 graph_source 与 num_nodes 分流)。
    data_dir: 存放 distance.csv / adj_mx.pkl 的目录。
    normalize: None(原始加权图) | "sym"(对称归一化) | "rw"(随机游走归一化)。
               作为搜索维度交由上层选择。
    """
    if profile.graph_source == "distance_csv":
        csv_path = os.path.join(data_dir, "distance.csv")
        adj = build_adj_from_distance_csv(csv_path, profile.num_nodes, normalized_k=normalized_k)
    elif profile.graph_source == "dcrnn_csv":
        # hazdzz 镜像: wam_<name>.csv 是现成的稠密邻接矩阵
        wam_path = os.path.join(data_dir, f"wam_{profile.name.lower().replace('-', '_')}.csv")
        adj = load_adj_csv(wam_path)
        if adj.shape[0] != profile.num_nodes:
            raise ValueError(
                f"{profile.name}: wam 邻接矩阵节点数 {adj.shape[0]} 与 profile {profile.num_nodes} 不符"
            )
    elif profile.graph_source == "adj_pkl":
        pkl_path = os.path.join(data_dir, "adj_mx.pkl")
        adj = load_adj_pkl(pkl_path)
        if adj.shape[0] != profile.num_nodes:
            raise ValueError(
                f"{profile.name}: adj_mx 节点数 {adj.shape[0]} 与 profile {profile.num_nodes} 不符"
            )
    else:
        raise ValueError(f"未知 graph_source: {profile.graph_source}")

    if normalize is None:
        return adj
    if normalize == "sym":
        return symmetric_normalize(adj)
    if normalize == "rw":
        return random_walk_normalize(adj)
    raise ValueError(f"未知 normalize 模式: {normalize} (可选 None/'sym'/'rw')")
