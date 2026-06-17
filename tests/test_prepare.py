"""prepare.py 的正确性回归测试 (设备无关, 本地 CPU, 无网络)。

用合成原始数据 (假 .npz + 迷你 distance.csv) 走完整管道, 验证:
  1. 滑动窗口形状与目标通道切片正确
  2. 端到端三切分齐全 (含真 test)、scaler 仅由训练集拟合 (防泄漏)
  3. 邻接矩阵随管道产出
  4. evaluate() 的 inverse-transform + masked 指标闭环正确
所有下载被合成文件短路, 不触网。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from darwin_st.data import prepare as P
from darwin_st.data.protocol import get_profile


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------


def test_generate_windows_shapes():
    T, N, C = 100, 5, 3
    data = np.random.rand(T, N, C).astype(np.float32)
    X, Y = P.generate_windows(data, seq_len_in=12, seq_len_out=12, target_channel=0)
    S = T - 12 - 12 + 1
    assert X.shape == (S, 12, N, C)
    assert Y.shape == (S, 12, N)  # 仅目标通道, 无 C 维


def test_generate_windows_content_alignment():
    """X 是历史窗口, Y 是紧随其后的目标通道未来值。"""
    T, N, C = 30, 2, 3
    data = np.arange(T * N * C, dtype=np.float32).reshape(T, N, C)
    X, Y = P.generate_windows(data, seq_len_in=4, seq_len_out=3, target_channel=0)
    # 第 0 个样本: X 取 data[0:4], Y 取 data[4:7] 的通道 0
    assert np.allclose(X[0], data[0:4])
    assert np.allclose(Y[0], data[4:7, :, 0])


def test_generate_windows_too_short_raises():
    data = np.random.rand(10, 3, 1).astype(np.float32)
    with pytest.raises(ValueError):
        P.generate_windows(data, seq_len_in=12, seq_len_out=12)


def test_generate_windows_target_channel():
    """target_channel 决定 Y 取哪个通道。"""
    T, N, C = 30, 2, 3
    data = np.zeros((T, N, C), dtype=np.float32)
    data[..., 1] = 7.0  # 只有通道 1 非零
    X, Y = P.generate_windows(data, 4, 3, target_channel=1)
    assert np.all(Y == 7.0)


# ---------------------------------------------------------------------------
# 端到端管道 (合成原始数据, 不触网)
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_pems(tmp_path, monkeypatch):
    """把缓存目录指到 tmp, 并预置一个假的 PeMS04 .npz + 迷你 distance.csv。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
    profile = get_profile("PeMS04")
    data_dir = P.data_dir_for("PeMS04")
    import os

    os.makedirs(data_dir, exist_ok=True)

    # 合成 [T, N=307, C=3] 原始数据 (T 取够切窗口即可)
    T, N, C = 120, profile.num_nodes, 3
    rng = np.random.default_rng(0)
    raw = rng.random((T, N, C)).astype(np.float32) * 100.0
    np.savez(os.path.join(data_dir, "pems04.npz"), data=raw)

    # 迷你 distance.csv (几条边即可触发邻接矩阵产出)
    with open(os.path.join(data_dir, "distance.csv"), "w") as f:
        f.write("from,to,cost\n0,1,100.0\n1,2,200.0\n2,3,150.0\n")

    return profile, data_dir, raw


def test_prepare_dataset_end_to_end(fake_pems):
    profile, data_dir, raw = fake_pems
    import os

    out_dir = P.prepare_dataset("PeMS04")
    assert out_dir == data_dir

    # 三切分文件齐全 (含 test!)
    for split in ("train", "val", "test"):
        assert os.path.exists(os.path.join(data_dir, f"{split}_x.npy"))
        assert os.path.exists(os.path.join(data_dir, f"{split}_y.npy"))
    assert os.path.exists(os.path.join(data_dir, "scaler.npy"))
    assert os.path.exists(os.path.join(data_dir, "adj.npy"))  # 邻接矩阵已产出


def test_prepare_split_proportions(fake_pems):
    """切分样本数符合 6/2/2 (余数归 test)。"""
    profile, data_dir, raw = fake_pems
    P.prepare_dataset("PeMS04")

    n_train = np.load(f"{data_dir}/train_x.npy").shape[0]
    n_val = np.load(f"{data_dir}/val_x.npy").shape[0]
    n_test = np.load(f"{data_dir}/test_x.npy").shape[0]
    S = 120 - 12 - 12 + 1  # 97 个窗口
    assert n_train == int(S * 0.6)
    assert n_val == int(S * 0.2)
    assert n_train + n_val + n_test == S  # 无丢失


def test_prepare_scaler_train_only(fake_pems):
    """落盘的 scaler 必须等于【仅训练集目标通道】的统计量, 不含 val/test。"""
    profile, data_dir, raw = fake_pems
    P.prepare_dataset("PeMS04")

    # 复算: 手动切窗口 + 时序切分, 取 train 段目标通道统计量
    X, Y = P.generate_windows(raw, 12, 12, 0)
    from darwin_st.data.protocol import chronological_split

    tr, _, _ = chronological_split(X.shape[0], profile)
    expected_mean = float(X[tr][..., 0].mean())
    expected_std = float(X[tr][..., 0].std())

    scaler = P.load_scaler(data_dir)
    assert np.isclose(scaler.mean, expected_mean, atol=1e-4)
    assert np.isclose(scaler.std, expected_std, atol=1e-4)


def test_prepare_idempotent(fake_pems):
    """重复调用应走缓存, 不报错。"""
    P.prepare_dataset("PeMS04")
    P.prepare_dataset("PeMS04")  # 第二次走 cached 分支


def test_load_split_returns_loader(fake_pems):
    profile, data_dir, raw = fake_pems
    P.prepare_dataset("PeMS04")
    loader = P.load_split(data_dir, "test", batch_size=8, drop_last=False)
    x, y = next(iter(loader))
    assert x.shape[1:] == (12, profile.num_nodes, 3)  # [B, T_in, N, C]
    assert y.shape[1:] == (12, profile.num_nodes)     # [B, T_out, N]


def test_evaluate_identity_model_low_error(fake_pems):
    """用「完美模型」(直接返回真实标签的归一化值) 验证 evaluate 闭环: 误差应≈0。"""
    profile, data_dir, raw = fake_pems
    P.prepare_dataset("PeMS04")
    scaler = P.load_scaler(data_dir)

    # 完美模型: 已知 test 的 y(归一化) 就是答案; 这里构造一个直接吐 y 的模型
    class PerfectModel(torch.nn.Module):
        """前向忽略 x, 返回与该 batch 对应的归一化标签 —— 仅用于验证评测闭环。"""
        def __init__(self, data_dir):
            super().__init__()
            self._y = torch.from_numpy(np.load(f"{data_dir}/test_y.npy")).float()
            self._cursor = 0

        def forward(self, x):
            b = x.shape[0]
            out = self._y[self._cursor : self._cursor + b]
            self._cursor += b
            return out

    model = PerfectModel(data_dir)
    # 评测需顺序读取 (不能 shuffle), test split 默认不 shuffle
    metrics = P.evaluate(model, data_dir, "test", batch_size=8, null_val=0.0)
    assert metrics["mae"] < 1e-3   # 完美预测 → masked MAE ≈ 0
    assert metrics["rmse"] < 1e-3


def test_evaluate_nan_fuse(fake_pems):
    """模型吐 NaN → evaluate 立即返回 inf 指标 (NaN 熔断)。"""
    profile, data_dir, raw = fake_pems
    P.prepare_dataset("PeMS04")

    class NaNModel(torch.nn.Module):
        def forward(self, x):
            out = torch.zeros(x.shape[0], 12, x.shape[2])
            out[0, 0, 0] = float("nan")
            return out

    metrics = P.evaluate(NaNModel(), data_dir, "test", batch_size=8)
    assert metrics["mae"] == float("inf")


# ---------------------------------------------------------------------------
# URL 修复
# ---------------------------------------------------------------------------


def test_dataset_urls_no_placeholder():
    """确认占位符假 URL 已被修复 (不再含 placeholder 字样)。"""
    for name, files in P.DATASET_URLS.items():
        for fname, url in files.items():
            assert "placeholder" not in url.lower(), f"{name}/{fname} 仍是占位符 URL"
            assert url.startswith("https://")


def test_all_profiles_have_url():
    """四大数据集都要有下载源。"""
    for name in ("PeMS04", "PeMS08", "METR-LA", "PEMS-BAY"):
        assert name in P.DATASET_URLS
        assert len(P.DATASET_URLS[name]) >= 1


def test_apply_mirror_off_by_default(monkeypatch):
    """未设 GITHUB_MIRROR 时 URL 原样返回。"""
    monkeypatch.delenv("GITHUB_MIRROR", raising=False)
    url = "https://github.com/x/y/raw/main/a.npz"
    assert P._apply_mirror(url) == url


def test_apply_mirror_prefixes_github(monkeypatch):
    """设了镜像则给 GitHub 系 URL 加前缀。"""
    monkeypatch.setenv("GITHUB_MIRROR", "https://gh-proxy.com/")
    url = "https://github.com/x/y/raw/main/a.npz"
    assert P._apply_mirror(url) == "https://gh-proxy.com/https://github.com/x/y/raw/main/a.npz"
    raw = "https://raw.githubusercontent.com/x/y/main/a.csv"
    assert P._apply_mirror(raw).startswith("https://gh-proxy.com/")


def test_apply_mirror_ignores_non_github(monkeypatch):
    """非 GitHub 源不加镜像。"""
    monkeypatch.setenv("GITHUB_MIRROR", "https://gh-proxy.com/")
    url = "https://example.com/data.npz"
    assert P._apply_mirror(url) == url


def test_dcrnn_datasets_have_vel_and_wam():
    """METR-LA/PEMS-BAY 应各含速度数据(vel)与邻接矩阵(wam)两个文件。"""
    for name in ("METR-LA", "PEMS-BAY"):
        fnames = list(P.DATASET_URLS[name])
        assert any(f.startswith("vel_") for f in fnames)
        assert any(f.startswith("wam_") for f in fnames)
