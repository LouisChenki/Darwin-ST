"""四个文献基线 (DCRNN/GWNet/AGCRN/STID) 的同炉契约测试 (CPU, 合成小数据)。

验证:
  - 形状冒烟: [B,T_in,N,C] → [B,T_out,N], 无 NaN; tod/dow=None 容忍
  - train_baseline: 合成小数据集 (tmp_path 预置 npy 切分 + scaler + adj + tod/dow)
    上 2 epoch 不炸、返回有限 MAE、best-checkpoint 落盘可回载; lr=0 平台触发早停
  - run_baseline.py 纯函数: parse_seeds 环境变量解析 + summarize_results 汇总
不依赖真实数据集下载 (沿用 test_train.py 的 tiny_data 思路)。
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import pytest
import torch

from darwin_st.baselines import BASELINES, build_baseline
from darwin_st.baselines.agcrn import AGCRN
from darwin_st.baselines.dcrnn import DCRNN
from darwin_st.baselines.gwnet import GWNet
from darwin_st.baselines.stid import STID
from darwin_st.baselines.train_baseline import train_baseline
from darwin_st.data import prepare as P

# 各模型的极小配置 (CPU 冒烟用; 论文默认配置在各模块 docstring)
_TINY_KW = {
    "dcrnn": {"hidden_dim": 16},
    "gwnet": {"residual_channels": 8, "dilation_channels": 8,
              "skip_channels": 16, "end_channels": 32},
    "agcrn": {"embed_dim": 4, "hidden_dim": 16},
    "stid": {"embed_dim": 8, "node_dim": 8, "tod_dim": 8, "dow_dim": 8, "num_layer": 2},
}


def _rand_adj(n: int, seed: int = 0) -> np.ndarray:
    """合成稀疏非负加权邻接 (约半边, 无自环)。"""
    rng = np.random.default_rng(seed)
    a = rng.random((n, n)).astype(np.float32)
    a[a < 0.5] = 0.0
    np.fill_diagonal(a, 0.0)
    return a


# ---------------------------------------------------------------------------
# 1) 形状冒烟: 四个模型 [2,12,8,3] → [2,12,8], 无 NaN, tod/dow 容忍 None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["dcrnn", "gwnet", "agcrn", "stid"])
def test_shape_smoke(name):
    torch.manual_seed(0)
    model = build_baseline(name, num_nodes=8, in_channels=3, seq_len_in=12,
                           seq_len_out=12, adj=_rand_adj(8), **_TINY_KW[name])
    x = torch.randn(2, 12, 8, 3)
    tod = torch.randint(0, 288, (2, 12))
    dow = torch.randint(0, 7, (2, 12))
    out = model(x, tod_idx=tod, dow_idx=dow)          # 带时间索引
    assert out.shape == (2, 12, 8)
    assert torch.isfinite(out).all()
    out_none = model(x)                               # tod/dow=None 容忍 (非 STID 忽略)
    assert out_none.shape == (2, 12, 8)
    assert torch.isfinite(out_none).all()


def test_adj_none_fallback():
    """无图兜底: dcrnn/gwnet 在 adj=None 时 (自环/纯自适应) 也能前向。"""
    for name in ("dcrnn", "gwnet"):
        torch.manual_seed(0)
        model = build_baseline(name, num_nodes=8, in_channels=3, seq_len_in=12,
                               seq_len_out=12, adj=None, **_TINY_KW[name])
        out = model(torch.randn(2, 12, 8, 3))
        assert out.shape == (2, 12, 8)
        assert torch.isfinite(out).all()


def test_registry_and_unknown():
    assert set(BASELINES) == {"dcrnn", "gwnet", "agcrn", "stid"}
    assert isinstance(build_baseline("STID", num_nodes=8, in_channels=3,
                                     seq_len_in=12, seq_len_out=12, adj=None), STID)
    assert isinstance(BASELINES["dcrnn"](8, 3, 12, 12), DCRNN)
    assert isinstance(BASELINES["gwnet"](8, 3, 12, 12), GWNet)
    assert isinstance(BASELINES["agcrn"](8, 3, 12, 12), AGCRN)
    with pytest.raises(ValueError, match="未知基线"):
        build_baseline("stgcn", num_nodes=8, in_channels=3, seq_len_in=12, seq_len_out=12)


# ---------------------------------------------------------------------------
# 2) train_baseline 冒烟: 合成小数据集 2 epoch 不炸, 有限 MAE, checkpoint 落盘
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """预置 tiny 已处理数据集 (8 节点, 含 tod/dow 时间索引), 跳过下载。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
    from darwin_st.data.protocol import DatasetProfile, PROFILES
    prof = DatasetProfile(
        name="TINYB", num_nodes=8, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    )
    monkeypatch.setitem(PROFILES, "TINYB", prof)

    data_dir = P.data_dir_for("TINYB")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    N, C = 8, 3
    for split, n in (("train", 48), ("val", 16), ("test", 16)):
        x = rng.random((n, 12, N, C)).astype(np.float32)
        y = rng.random((n, 12, N)).astype(np.float32)
        np.save(os.path.join(data_dir, f"{split}_x.npy"), x)
        np.save(os.path.join(data_dir, f"{split}_y.npy"), y)
        # tod/dow 时间索引 → load_split 走 (x, tod, dow, y) 四元组路径
        tod = (np.arange(n)[:, None] + np.arange(12)[None, :]) % 288
        dow = ((np.arange(n)[:, None] + np.arange(12)[None, :]) // 288) % 7
        np.save(os.path.join(data_dir, f"{split}_tod.npy"), tod.astype(np.int64))
        np.save(os.path.join(data_dir, f"{split}_dow.npy"), dow.astype(np.int64))
    np.save(os.path.join(data_dir, "scaler.npy"), np.array([50.0, 20.0]))
    adj = _rand_adj(N)
    np.save(os.path.join(data_dir, "adj.npy"), adj)
    return prof, data_dir, adj


@pytest.mark.parametrize("name", ["dcrnn", "gwnet", "agcrn", "stid"])
def test_train_baseline_smoke(tiny_data, name):
    """四模型各训 2 epoch: 返回有限 MAE, n_epochs=2, best 权重落盘且可回载。"""
    prof, data_dir, adj = tiny_data
    torch.manual_seed(0)
    model = build_baseline(name, num_nodes=prof.num_nodes,
                           in_channels=prof.num_channels,
                           seq_len_in=prof.seq_len_in,
                           seq_len_out=prof.seq_len_out, adj=adj, **_TINY_KW[name])
    ckpt = os.path.join(data_dir, f"ck_{name}.pt")
    best, info = train_baseline(
        model, {"lr": 1e-3, "weight_decay": 1e-4, "batch_size": 16},
        data_dir, prof, adj, device="cpu", max_epochs=2,
        early_stop_patience=0, checkpoint_path=ckpt)
    assert np.isfinite(best) and best > 0             # 真实尺度 masked MAE
    assert info["n_epochs"] == 2
    assert np.isfinite(info["best_rmse"]) and info["best_rmse"] > 0
    assert np.isfinite(info["best_mape"]) and info["best_mape"] > 0
    assert os.path.exists(ckpt)                       # best-checkpoint 已落盘
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd)                         # 回载形状兼容


def test_train_baseline_early_stop(tiny_data):
    """lr=0 → 权重不更新 → val 立即平台 → 早停在 patience 后 (跑不满 max_epochs)。"""
    prof, data_dir, adj = tiny_data
    torch.manual_seed(0)
    model = build_baseline("stid", num_nodes=prof.num_nodes,
                           in_channels=prof.num_channels,
                           seq_len_in=prof.seq_len_in,
                           seq_len_out=prof.seq_len_out, adj=adj, **_TINY_KW["stid"])
    best, info = train_baseline(
        model, {"lr": 0.0, "batch_size": 16}, data_dir, prof, adj,
        device="cpu", max_epochs=30, early_stop_patience=2)
    assert info["n_epochs"] < 30, "平台未触发早停"
    assert np.isfinite(best) and best > 0


def test_train_baseline_cosine_schedule(tiny_data):
    """lr_schedule=cosine 走通 (scheduler 非 plateau 分支)。"""
    prof, data_dir, adj = tiny_data
    torch.manual_seed(0)
    model = build_baseline("agcrn", num_nodes=prof.num_nodes,
                           in_channels=prof.num_channels,
                           seq_len_in=prof.seq_len_in,
                           seq_len_out=prof.seq_len_out, adj=adj, **_TINY_KW["agcrn"])
    best, info = train_baseline(
        model, {"lr": 1e-3, "batch_size": 16, "lr_schedule": "cosine"},
        data_dir, prof, adj, device="cpu", max_epochs=2, early_stop_patience=0)
    assert np.isfinite(best)
    assert info["n_epochs"] == 2


# ---------------------------------------------------------------------------
# 3) run_baseline.py 纯函数: parse_seeds / summarize_results
# ---------------------------------------------------------------------------


def _load_run_baseline():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "run_baseline.py")
    spec = importlib.util.spec_from_file_location("run_baseline", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_seeds():
    mod = _load_run_baseline()
    assert mod.parse_seeds("", 42) == [42]            # 空 SEEDS 回退单 SEED
    assert mod.parse_seeds("0,1,2") == [0, 1, 2]
    assert mod.parse_seeds(" 7 , 8 ,") == [7, 8]      # 空格/尾逗号容错


def _rec(seed, test_mae, val_mae):
    return {"seed": seed, "retrained_val_mae": val_mae,
            "val": {"mae": val_mae, "rmse": 0.0, "mape": 0.0},
            "test": {"mae": test_mae, "rmse": 0.0, "mape": 0.0},
            "offset_test_minus_val": test_mae - val_mae}


def test_summarize_results_single_seed():
    mod = _load_run_baseline()
    s = mod.summarize_results([_rec(42, 18.40, 18.35)], "stid", "PeMS04")
    assert s["baseline"] == "stid" and s["dataset"] == "PeMS04"
    assert s["n_seeds"] == 1
    assert abs(s["test_mae_mean"] - 18.40) < 1e-9
    assert s["test_mae_std"] == 0.0                   # 单 seed 不报假方差
    assert abs(s["offset_mean"] - 0.05) < 1e-9


def test_summarize_results_multi_seed():
    mod = _load_run_baseline()
    recs = [_rec(0, 18.40, 18.35), _rec(1, 18.50, 18.35), _rec(2, 18.45, 18.40)]
    s = mod.summarize_results(recs, "dcrnn", "PeMS04")
    assert s["n_seeds"] == 3
    assert abs(s["test_mae_mean"] - 18.45) < 1e-9
    assert s["test_mae_std"] > 0                      # 多 seed 出 std
    assert abs(s["val_mae_mean"] - (18.35 + 18.35 + 18.40) / 3) < 1e-9
    assert abs(s["offset_mean"] - (0.25 / 3)) < 1e-6  # 偏移 = test−val
    assert len(s["results"]) == 3                     # 原始记录全保留
