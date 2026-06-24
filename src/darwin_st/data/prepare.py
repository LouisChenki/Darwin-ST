"""数据管道整合层 (Data Pipeline) —— 把 protocol/adjacency/metrics 串成端到端流程。

职责:
  下载 (按 profile, 修复占位符 URL, 支持 .npz 与 .h5)
    → 滑动窗口 [T,N,C] → (X:[S,T_in,N,C], Y:[S,T_out,N])
    → 按 profile 时序三切分 (train/val/**test**, 防泄漏)
    → train-only ZScoreScaler (仅目标通道)
    → 落盘 train/val/test + scaler + 邻接矩阵
    → 运行时: load_split() 出 DataLoader, evaluate() 出 masked 指标

与旧 prepare.py 的关键差异 (修复 docs/ANALYSIS.md 所列硬伤):
  - 真 test 集 (旧版只有 train/val, 0.6+0.2 丢了 0.2)
  - 按数据集 profile 分流 split (旧版全局硬编码 0.6/0.2)
  - masked 指标 (旧版把 0 当真值算 MAE)
  - 邻接矩阵 (旧版完全不产出, GNN 无法运行)
  - 修复 METR-LA/PEMS-BAY 占位符假 URL

设备说明: 数据准备 (下载/窗口/切分/归一化) 为 numpy/CPU, 设备无关可本地验证;
仅 DataLoader/evaluate 涉及 torch device (CPU/MPS/CUDA 通吃)。

详见 docs/BLUEPRINT.md §2。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from darwin_st.data.adjacency import build_adjacency
from darwin_st.data.metrics import compute_all_metrics
from darwin_st.data.protocol import DatasetProfile, ZScoreScaler, chronological_split, get_profile

__all__ = [
    "CACHE_DIR",
    "TIME_BUDGET",
    "DATASET_URLS",
    "data_dir_for",
    "generate_windows",
    "make_time_indices",
    "prepare_dataset",
    "load_split",
    "load_scaler",
    "load_adj",
    "evaluate",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TIME_BUDGET = 1200  # 单次训练时间预算 (秒) = 20 分钟, 供 train.py 时间熔断使用
# 缓存目录: 默认 ~/.cache/darwin-st; 可用 DARWIN_ST_CACHE 覆盖。
# ⚠️ 服务器上系统盘常很小(如 30GB), 应指向数据盘, 例:
#    export DARWIN_ST_CACHE=/root/autodl-tmp/darwin-st-cache
CACHE_DIR = os.environ.get(
    "DARWIN_ST_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "darwin-st")
)

# 数据集下载源 (已修复占位符)。值为 {本地文件名: 远程URL} 映射 (一个数据集可能多文件)。
# - PeMS04/08: ASTGCN 仓库直挂 .npz, raw 可直接下载 (图来自同目录 distance.csv, 需另置)。
# - METR-LA/PEMS-BAY: hazdzz/dcrnn_data 镜像把【速度数据 vel_*.csv】与【邻接矩阵 wam_*.csv】
#   都用 CSV 直挂在 GitHub raw, 可直链、无需 Google Drive、邻接矩阵现成。
_ASTGCN_BASE = "https://github.com/Davidham3/ASTGCN/raw/master/data"
_DCRNN_BASE = "https://raw.githubusercontent.com/hazdzz/dcrnn_data/main"
DATASET_URLS: dict[str, dict[str, str]] = {
    "PeMS04": {
        "pems04.npz": f"{_ASTGCN_BASE}/PEMS04/pems04.npz",
        "distance.csv": f"{_ASTGCN_BASE}/PEMS04/distance.csv",  # 图原料 → 邻接矩阵
    },
    "PeMS08": {
        "pems08.npz": f"{_ASTGCN_BASE}/PEMS08/pems08.npz",
        "distance.csv": f"{_ASTGCN_BASE}/PEMS08/distance.csv",
    },
    "METR-LA": {
        "vel_metr_la.csv": f"{_DCRNN_BASE}/metr_la/vel_metr_la.csv",
        "wam_metr_la.csv": f"{_DCRNN_BASE}/metr_la/wam_metr_la.csv",
    },
    "PEMS-BAY": {
        "vel_pems_bay.csv": f"{_DCRNN_BASE}/pems_bay/vel_pems_bay.csv",
        "wam_pems_bay.csv": f"{_DCRNN_BASE}/pems_bay/wam_pems_bay.csv",
    },
}


def data_dir_for(dataset_name: str) -> str:
    """该数据集的缓存目录 (~/.cache/darwin-st/<dataset>/)。"""
    return os.path.join(CACHE_DIR, dataset_name.lower())


# ---------------------------------------------------------------------------
# 下载 + 原始数据读取 (按格式分流: .npz vs .h5)
# ---------------------------------------------------------------------------


def _apply_mirror(url: str) -> str:
    """若设置了环境变量 GITHUB_MIRROR, 给 GitHub 系 URL 加镜像前缀以加速 (国内服务器友好)。

    例: GITHUB_MIRROR=https://gh-proxy.com/ 会把
        https://github.com/... → https://gh-proxy.com/https://github.com/...
    仅对 github.com / raw.githubusercontent.com 生效; 其它源原样返回。
    未设置则不改动 (默认行为不变)。
    """
    mirror = os.environ.get("GITHUB_MIRROR", "").strip()
    if not mirror:
        return url
    if ("github.com" in url) or ("githubusercontent.com" in url):
        return mirror.rstrip("/") + "/" + url
    return url


def _download(url: str, dest: str, retries: int = 3) -> None:
    """流式下载到 dest, 带重试 (GitHub 偶发 connection reset)。全部失败抛异常由调用方降级。

    支持 GITHUB_MIRROR 环境变量加速 (见 _apply_mirror)。
    """
    import time

    import requests

    fetch_url = _apply_mirror(url)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(fetch_url, stream=True, timeout=60)
            resp.raise_for_status()
            tmp = dest + ".part"
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            os.replace(tmp, dest)  # 原子落盘, 避免半截文件冒充成品
            return
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(2 * attempt)  # 退避重试
    raise last_err  # type: ignore[misc]


def _raw_data_filename(profile: DatasetProfile) -> str:
    """该数据集承载【时序数据】的文件名 (PeMS=.npz; dcrnn=vel_*.csv)。"""
    if profile.graph_source == "distance_csv":
        return f"{profile.name.lower()}.npz"
    if profile.graph_source == "dcrnn_csv":
        return f"vel_{profile.name.lower().replace('-', '_')}.csv"
    return f"{profile.name.lower()}.h5"  # adj_pkl 谱系: 数据为 .h5


def _ensure_raw_data(profile: DatasetProfile, data_dir: str) -> str:
    """确保原始数据文件就绪; 缺失则尝试下载该数据集的【全部】登记文件。

    返回承载时序数据的主文件路径。下载失败则给手动放置指引并退出。
    """
    data_filename = _raw_data_filename(profile)
    data_path = os.path.join(data_dir, data_filename)

    urls = DATASET_URLS.get(profile.name, {})
    for fname, url in urls.items():
        dest = os.path.join(data_dir, fname)
        if os.path.exists(dest):
            continue
        print(f"正在下载 {profile.name}/{fname} 从 {url} ...")
        try:
            _download(url, dest)
            print(f"  ✓ {fname}")
        except Exception as e:  # 网络/镜像失效 → 降级为手动放置, 不崩
            print(f"  ⚠️ 下载失败 ({e})")

    if os.path.exists(data_path):
        return data_path

    print(
        f"请手动下载 {profile.name} 数据并放到 {data_dir}/ (主数据文件: {data_filename})\n"
        f"  - PeMS04/08(.npz): ASTGCN 仓库 data/PEMS0X/, 图另需 distance.csv\n"
        f"  - METR-LA/PEMS-BAY: hazdzz/dcrnn_data 镜像的 vel_*.csv(数据) 与 wam_*.csv(邻接)"
    )
    sys.exit(1)


def _load_raw_array(profile: DatasetProfile, raw_path: str) -> np.ndarray:
    """读原始时序文件为统一的 [T, N, C] float32 数组 (按 graph_source 分流格式)。"""
    if profile.graph_source == "distance_csv":
        # ASTGCN .npz: 含 'data' 键, shape [T, N, C]
        f = np.load(raw_path)
        data = f["data"] if "data" in f else f[f.files[0]]
    elif profile.graph_source == "dcrnn_csv":
        # hazdzz vel_*.csv: 首列时间戳索引 + 首行传感器表头 → 取数值体, [T, N] → [T, N, 1]
        import pandas as pd

        df = pd.read_csv(raw_path, index_col=0)
        data = np.asarray(df.values, dtype=np.float32)[:, :, None]
    elif profile.graph_source == "adj_pkl":
        # DCRNN .h5: pandas DataFrame [T, N] (单通道 speed) → [T, N, 1]
        import pandas as pd

        df = pd.read_hdf(raw_path)
        data = np.asarray(df.values)[:, :, None]
    else:
        raise ValueError(f"未知 graph_source: {profile.graph_source}")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"{profile.name}: 原始数据应为 [T,N,C], 实为 {data.shape}")
    if data.shape[1] != profile.num_nodes:
        raise ValueError(
            f"{profile.name}: 节点数 {data.shape[1]} 与 profile {profile.num_nodes} 不符"
        )
    return data


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------


def make_time_indices(num_samples: int, seq_len_in: int,
                      steps_per_day: int = 288) -> tuple[np.ndarray, np.ndarray]:
    """为每个输入窗口生成 time-of-day / day-of-week 索引 (STID 式身份嵌入用)。

    PeMS 等固定 5min 间隔采样, **无需原始时间戳**: 样本序号即时间槽。
    窗口 i 的输入是 data[i : i+seq_len_in], 故其第 t 步绝对时刻 = i+t。
      tod = (i+t) % steps_per_day         (一天 288 槽, 0..287)
      dow = ((i+t) // steps_per_day) % 7  (一周 7 天, 0..6)
    返回 (tod[S, seq_len_in], dow[S, seq_len_in]) int64。
    """
    base = np.arange(num_samples)[:, None] + np.arange(seq_len_in)[None, :]  # [S, T_in] 绝对时刻
    tod = (base % steps_per_day).astype(np.int64)
    dow = ((base // steps_per_day) % 7).astype(np.int64)
    return tod, dow


def generate_windows(
    data: np.ndarray, seq_len_in: int, seq_len_out: int, target_channel: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """滑动窗口切片。

    输入 data: [T, N, C]
    返回:
        X: [S, seq_len_in, N, C]    (全通道历史)
        Y: [S, seq_len_out, N]      (仅目标通道未来值)
    S = T - seq_len_in - seq_len_out + 1
    """
    total_len = data.shape[0]
    num_samples = total_len - seq_len_in - seq_len_out + 1
    if num_samples <= 0:
        raise ValueError(
            f"序列太短: T={total_len} 不足以切出 in={seq_len_in}+out={seq_len_out} 的窗口"
        )

    N, C = data.shape[1], data.shape[2]
    X = np.zeros((num_samples, seq_len_in, N, C), dtype=np.float32)
    Y = np.zeros((num_samples, seq_len_out, N), dtype=np.float32)
    for i in range(num_samples):
        X[i] = data[i : i + seq_len_in]
        Y[i] = data[i + seq_len_in : i + seq_len_in + seq_len_out, :, target_channel]
    return X, Y


# ---------------------------------------------------------------------------
# 端到端准备并落盘
# ---------------------------------------------------------------------------


def prepare_dataset(dataset_name: str, force: bool = False) -> str:
    """完整准备一个数据集: 下载→窗口→三切分→train-only 归一化→落盘(含邻接矩阵)。

    返回该数据集的 data_dir。幂等: 已处理过则跳过 (除非 force=True)。
    """
    profile = get_profile(dataset_name)
    data_dir = data_dir_for(profile.name)
    os.makedirs(data_dir, exist_ok=True)

    marker = os.path.join(data_dir, "test_x.npy")
    if os.path.exists(marker) and not force:
        print(f"数据已预处理 (cached): {data_dir}")
        return data_dir

    raw_path = _ensure_raw_data(profile, data_dir)
    data = _load_raw_array(profile, raw_path)  # [T, N, C]

    X, Y = generate_windows(data, profile.seq_len_in, profile.seq_len_out, profile.target_channel)
    tod, dow = make_time_indices(X.shape[0], profile.seq_len_in)  # [S, T_in] 时间槽 (STID 身份嵌入)
    tr, va, te = chronological_split(X.shape[0], profile)

    splits = {"train": (X[tr], Y[tr]), "val": (X[va], Y[va]), "test": (X[te], Y[te])}
    time_splits = {"train": (tod[tr], dow[tr]), "val": (tod[va], dow[va]), "test": (tod[te], dow[te])}

    # train-only scaler, 仅目标通道
    train_x = splits["train"][0]
    scaler = ZScoreScaler.fit(train_x[..., profile.target_channel])

    for name, (sx, sy) in splits.items():
        sx = sx.copy()
        sx[..., profile.target_channel] = scaler.transform(sx[..., profile.target_channel])
        sy = scaler.transform(sy.copy())  # 标签同尺度归一化 (评测时再 inverse)
        np.save(os.path.join(data_dir, f"{name}_x.npy"), sx)
        np.save(os.path.join(data_dir, f"{name}_y.npy"), sy)
        # 时间索引 (tod/dow) 与 x/y 同切分对齐落盘, 供 STID 身份嵌入
        stod, sdow = time_splits[name]
        np.save(os.path.join(data_dir, f"{name}_tod.npy"), stod)
        np.save(os.path.join(data_dir, f"{name}_dow.npy"), sdow)

    np.save(os.path.join(data_dir, "scaler.npy"), np.array([scaler.mean, scaler.std], dtype=np.float64))

    # 邻接矩阵 (若图原料存在则产出原始加权图; 归一化交由模型侧选择)
    try:
        adj = build_adjacency(profile, data_dir, normalize=None)
        np.save(os.path.join(data_dir, "adj.npy"), adj)
        print(f"邻接矩阵已生成 (adj): {adj.shape}")
    except FileNotFoundError:
        print(
            f"ℹ️ 未找到图原料 ({'distance.csv' if profile.graph_source=='distance_csv' else 'adj_mx.pkl'}), "
            f"跳过邻接矩阵。需图卷积类模型时请放置后重跑。"
        )

    print(
        f"预处理完成 (Ready)! {profile.name}: "
        f"train={splits['train'][0].shape[0]} val={splits['val'][0].shape[0]} test={splits['test'][0].shape[0]}"
    )
    return data_dir


# ---------------------------------------------------------------------------
# 运行时组件 (供 train.py / 评测使用)
# ---------------------------------------------------------------------------


def load_scaler(data_dir: str) -> ZScoreScaler:
    arr = np.load(os.path.join(data_dir, "scaler.npy"))
    return ZScoreScaler(mean=float(arr[0]), std=float(arr[1]), fitted=True)


def load_adj(data_dir: str) -> np.ndarray | None:
    path = os.path.join(data_dir, "adj.npy")
    return np.load(path) if os.path.exists(path) else None


def load_split(
    data_dir: str, split: str, batch_size: int, device: str = "cpu", drop_last: bool | None = None
) -> DataLoader:
    """加载某切分的 DataLoader。train 默认 shuffle+drop_last, val/test 不丢尾。

    若该切分有时间索引 (tod/dow) 落盘, 每个 batch 产出 (x, tod, dow, y) 四元组
    (供 STID 身份嵌入); 否则回退 (x, y) 二元组 (旧缓存/无时间特征)。
    消费方 (train_one/evaluate) 用 len(batch) 区分两种形态。
    """
    x_path = os.path.join(data_dir, f"{split}_x.npy")
    y_path = os.path.join(data_dir, f"{split}_y.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"切分 {split} 不存在: {x_path} (先运行 prepare_dataset)")

    X = torch.from_numpy(np.load(x_path)).float()
    Y = torch.from_numpy(np.load(y_path)).float()
    is_accel = ("cuda" in str(device)) or ("mps" in str(device))
    if drop_last is None:
        drop_last = split == "train"

    tod_path = os.path.join(data_dir, f"{split}_tod.npy")
    dow_path = os.path.join(data_dir, f"{split}_dow.npy")
    if os.path.exists(tod_path) and os.path.exists(dow_path):
        tod = torch.from_numpy(np.load(tod_path)).long()
        dow = torch.from_numpy(np.load(dow_path)).long()
        dataset: TensorDataset = TensorDataset(X, tod, dow, Y)   # (x, tod, dow, y)
    else:
        dataset = TensorDataset(X, Y)                            # (x, y) 回退

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        pin_memory=is_accel,
        drop_last=drop_last,
    )


@torch.no_grad()
def evaluate(
    model, data_dir: str, split: str, batch_size: int, device: str = "cpu", null_val: float = 0.0
) -> dict[str, float]:
    """在指定切分上算 masked MAE/RMSE/MAPE (已 inverse-transform 回真实尺度)。

    NaN 熔断: 预测含 NaN 直接返回 inf 指标, 供外围进程快速识别并跳过该次演化。
    """
    model.eval()
    loader = load_split(data_dir, split, batch_size, device=device, drop_last=False)
    scaler = load_scaler(data_dir)

    preds_all, labels_all = [], []
    for batch in loader:
        if len(batch) == 4:                       # (x, tod, dow, y)
            x, tod, dow, y = batch
            x, tod, dow, y = x.to(device), tod.to(device), dow.to(device), y.to(device)
            preds = model(x, tod_idx=tod, dow_idx=dow)
        else:                                     # (x, y) 回退
            x, y = batch
            x, y = x.to(device), y.to(device)
            preds = model(x)
        if torch.isnan(preds).any():
            return {"mae": float("inf"), "rmse": float("inf"), "mape": float("inf")}
        # inverse-transform 回真实交通流尺度后再算指标
        preds_all.append((preds * scaler.std + scaler.mean).cpu())
        labels_all.append((y * scaler.std + scaler.mean).cpu())

    preds_real = torch.cat(preds_all, dim=0)
    labels_real = torch.cat(labels_all, dim=0)
    model.train()
    return compute_all_metrics(preds_real, labels_real, null_val=null_val)


if __name__ == "__main__":
    # 用法: DATASET=PeMS04 python -m darwin_st.data.prepare
    ds = os.environ.get("DATASET", "PeMS04")
    print(f"准备时空预测数据 (Prepare Spatio-Temporal Data): {ds}")
    prepare_dataset(ds)
