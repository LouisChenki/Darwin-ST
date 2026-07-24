"""checkpoint.py 权重存档 + 训练接线的正确性测试 (纯 CPU, 无下载/无 GPU)。

覆盖契约:
  - maybe_save_best: 首次写 / 更优覆盖 / 更差或持平不覆盖 / sidecar 损坏容错 / 原子写无临时文件残留
  - prune_to_top_k: 按 sidecar val_mae 留优删劣, sidecar 同删, 缺 sidecar 视为最差
  - train_one 接线: 玩具训练 2 epoch 后 ckpt 落盘且 meta 字段正确
  - evaluate_architecture / make_eval_fn: checkpoint_dir 透传
  - EvalSpec.checkpoint_dir 默认 None 向后兼容
  - render_model_card 权重存档行 (有/无 weights_file)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from darwin_st.data import prepare as P
from darwin_st.data.protocol import get_profile
from darwin_st.optim.checkpoint import load_meta, maybe_save_best, meta_path_for, prune_to_top_k
from darwin_st.optim.hpo import HPOConfig
from darwin_st.optim.scheduler import EvalSpec
from darwin_st.optim.train import evaluate_architecture, train_one
from darwin_st.search.genotype import random_genotype


def _sd(v: float) -> dict:
    """玩具 state_dict (一个标量张量, 值可区分第几次写)。"""
    return {"w": torch.tensor([v])}


# ---------------------------------------------------------------------------
# maybe_save_best
# ---------------------------------------------------------------------------

def test_maybe_save_best_first_write(tmp_path):
    path = str(tmp_path / "m.pt")
    meta = {"signature": "abc123", "dataset": "TINY", "best_epoch": 0}
    assert maybe_save_best(_sd(1.0), path, meta, new_mae=10.0) is True
    assert os.path.exists(path)
    side = load_meta(path)
    # sidecar 字段: 调用方业务字段 + 本函数补的 val_mae/created_at
    assert side["signature"] == "abc123" and side["dataset"] == "TINY"
    assert side["best_epoch"] == 0
    assert side["val_mae"] == 10.0
    assert side["created_at"]
    sd = torch.load(path, weights_only=True)
    assert float(sd["w"]) == 1.0


def test_maybe_save_best_better_overwrites(tmp_path):
    path = str(tmp_path / "m.pt")
    maybe_save_best(_sd(1.0), path, {"signature": "s", "dataset": "TINY", "best_epoch": 0}, 10.0)
    assert maybe_save_best(_sd(2.0), path, {"signature": "s", "dataset": "TINY", "best_epoch": 3}, 8.5) is True
    side = load_meta(path)
    assert side["val_mae"] == 8.5 and side["best_epoch"] == 3
    assert float(torch.load(path, weights_only=True)["w"]) == 2.0


def test_maybe_save_best_worse_or_equal_keeps_old(tmp_path):
    path = str(tmp_path / "m.pt")
    maybe_save_best(_sd(1.0), path, {"signature": "s", "dataset": "TINY", "best_epoch": 0}, 10.0)
    # 更差: 不写
    assert maybe_save_best(_sd(9.0), path, {"signature": "s", "dataset": "TINY", "best_epoch": 1}, 12.0) is False
    # 持平: 严格 < 不覆盖, 保留先到的
    assert maybe_save_best(_sd(9.0), path, {"signature": "s", "dataset": "TINY", "best_epoch": 2}, 10.0) is False
    side = load_meta(path)
    assert side["val_mae"] == 10.0 and side["best_epoch"] == 0
    assert float(torch.load(path, weights_only=True)["w"]) == 1.0


def test_maybe_save_best_corrupt_sidecar_tolerated(tmp_path):
    path = str(tmp_path / "m.pt")
    torch.save(_sd(0.0), path)
    with open(meta_path_for(path), "w") as f:
        f.write("{not valid json")                     # sidecar 损坏
    assert load_meta(path) is None                       # 容错读 → None
    # 视为首次: 即使 mae 不优也重写, 且 sidecar 被修复为合法 JSON
    assert maybe_save_best(_sd(3.0), path, {"signature": "s", "dataset": "TINY"}, 99.0) is True
    side = load_meta(path)
    assert side["val_mae"] == 99.0
    # sidecar 是 list 而非 dict 也容错
    with open(meta_path_for(path), "w") as f:
        json.dump([1, 2], f)
    assert load_meta(path) is None


def test_maybe_save_best_atomic_no_temp_residue(tmp_path, monkeypatch):
    path = str(tmp_path / "sub" / "m.pt")               # 父目录不存在 → 自动建
    assert maybe_save_best(_sd(1.0), path, {"signature": "s"}, 5.0) is True
    names = sorted(os.listdir(tmp_path / "sub"))
    assert names == ["m.pt", "m.pt.meta.json"]           # 无 .tmp 残留

    # 写权重中途失败 (模拟磁盘错): 临时文件被清理, 已有文件不被破坏
    maybe_save_best(_sd(2.0), path, {"signature": "s"}, 4.0)
    real_save = torch.save

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(OSError):
        maybe_save_best(_sd(3.0), path, {"signature": "s"}, 1.0)
    monkeypatch.setattr(torch, "save", real_save)
    assert sorted(os.listdir(tmp_path / "sub")) == ["m.pt", "m.pt.meta.json"]
    assert float(torch.load(path, weights_only=True)["w"]) == 2.0   # 旧权重完好


# ---------------------------------------------------------------------------
# prune_to_top_k
# ---------------------------------------------------------------------------

def _mk_ckpt(d, name, mae):
    p = os.path.join(d, name)
    if mae is None:
        torch.save(_sd(0.0), p)          # 只有 .pt, 无 sidecar
    else:
        maybe_save_best(_sd(mae), p, {"signature": name, "dataset": "TINY"}, mae)
    return p


def test_prune_to_top_k_keeps_best_deletes_rest(tmp_path):
    d = str(tmp_path)
    names_maes = [("a.pt", 5.0), ("b.pt", 1.0), ("c.pt", 3.0),
                  ("d.pt", 2.0), ("e.pt", 6.0), ("f.pt", 4.0)]
    for n, m in names_maes:
        _mk_ckpt(d, n, m)
    _mk_ckpt(d, "g.pt", None)                            # 无 sidecar → 视为最差, 最先删

    deleted = prune_to_top_k(d, keep=3)
    assert sorted(os.path.basename(p) for p in deleted) == ["a.pt", "e.pt", "f.pt", "g.pt"]
    # 留下的恰是 mae 最小的 3 个 (b=1, d=2, c=3), sidecar 同留
    assert sorted(os.listdir(d)) == ["b.pt", "b.pt.meta.json",
                                     "c.pt", "c.pt.meta.json",
                                     "d.pt", "d.pt.meta.json"]
    # 幂等: 再 prune 不再删
    assert prune_to_top_k(d, keep=3) == []
    # keep=0 清空; 目录不存在不报错
    assert len(prune_to_top_k(d, keep=0)) == 3
    assert os.listdir(d) == []
    assert prune_to_top_k(str(tmp_path / "nonexistent"), keep=3) == []


# ---------------------------------------------------------------------------
# train_one / evaluate_architecture / make_eval_fn 接线 (玩具数据, 同 test_train.py 风格)
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """预置一个 tiny 的已处理数据集 (10 节点, 少量样本), 跳过下载。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
    from darwin_st.data.protocol import DatasetProfile, PROFILES
    prof = DatasetProfile(
        name="TINY", num_nodes=10, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    )
    monkeypatch.setitem(PROFILES, "TINY", prof)

    data_dir = P.data_dir_for("TINY")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    N, C = 10, 3
    for split, n in (("train", 64), ("val", 24), ("test", 24)):
        x = rng.random((n, 12, N, C)).astype(np.float32)
        y = rng.random((n, 12, N)).astype(np.float32)
        np.save(os.path.join(data_dir, f"{split}_x.npy"), x)
        np.save(os.path.join(data_dir, f"{split}_y.npy"), y)
    np.save(os.path.join(data_dir, "scaler.npy"), np.array([50.0, 20.0]))
    adj = np.eye(N, dtype=np.float32)
    np.save(os.path.join(data_dir, "adj.npy"), adj)
    return prof, data_dir, adj


def test_train_one_saves_checkpoint_on_improve(tiny_data, tmp_path):
    """checkpoint_path 非 None: 训练刷新 best 时权重落盘, meta 与 trace 一致。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    ckpt = str(tmp_path / "ck" / "TINY_x.pt")
    mae, trace = train_one(geno, {"lr": 1e-3, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=2,
                           checkpoint_path=ckpt,
                           checkpoint_meta={"signature": geno.signature(), "dataset": "TINY"})
    assert os.path.exists(ckpt)
    side = load_meta(ckpt)
    assert side["signature"] == geno.signature() and side["dataset"] == "TINY"
    assert side["val_mae"] == pytest.approx(mae)         # 存的就是 best-checkpoint 的 MAE
    assert side["best_epoch"] == trace.best_epoch
    assert side["created_at"]
    # 权重可加载且 shape 与模型一致 (CPU 张量)
    sd = torch.load(ckpt, weights_only=True)
    assert all(isinstance(v, torch.Tensor) and v.device.type == "cpu" for v in sd.values())
    # 无临时文件残留
    assert sorted(os.listdir(tmp_path / "ck")) == ["TINY_x.pt", "TINY_x.pt.meta.json"]


def test_train_one_default_no_checkpoint(tiny_data, tmp_path):
    """checkpoint_path=None (默认): 行为完全不变, 不产生任何文件。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, hidden=16)
    mae, _ = train_one(geno, {"lr": 1e-3, "batch_size": 16}, data_dir, prof, adj,
                       device="cpu", max_epochs=1)
    assert np.isfinite(mae)
    assert os.listdir(tmp_path) == [] or all(
        not f.endswith(".pt") for f in os.listdir(tmp_path))


def test_evaluate_architecture_checkpoint_dir(tiny_data, tmp_path):
    """checkpoint_dir 非 None: 本架构固定路径 <dir>/<dataset>_<sig[:12]>.pt, 多 trial 共享只留最优。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    ckpt_dir = str(tmp_path / "ckpts")
    cfg = HPOConfig(n_trials=2, max_epochs=2, min_resource=1, n_startup_trials=1)
    res = evaluate_architecture(geno, "cpu", data_dir, prof, adj, hpo_cfg=cfg,
                                checkpoint_dir=ckpt_dir)
    expected = os.path.join(ckpt_dir, f"TINY_{geno.signature()[:12]}.pt")
    assert os.path.exists(expected)
    side = load_meta(expected)
    assert side["signature"] == geno.signature() and side["dataset"] == "TINY"
    # sidecar 的 val_mae ≤ 最优 trial 的 MAE (同一路径只升不降; HPO 最优即该 trial 内 best)
    assert side["val_mae"] == pytest.approx(res.mae)


def test_make_eval_fn_passes_checkpoint_dir(tiny_data, monkeypatch):
    """make_eval_fn(checkpoint_dir=...) 透传到 evaluate_architecture; 默认 None。"""
    import darwin_st.optim.train as train_mod
    from darwin_st.optim.scheduler import EvalResult
    from darwin_st.optim.train import make_eval_fn

    captured = []

    def fake_eval(genotype, device, data_dir, profile, adj, hpo_cfg=None,
                  warm_start_hps=None, n_trials=None, checkpoint_dir=None):
        captured.append(checkpoint_dir)
        return EvalResult(genotype=genotype, status="OK", mae=10.0, device=device,
                          extra={"num_params": 1})

    monkeypatch.setattr(train_mod, "evaluate_architecture", fake_eval)
    g = random_genotype(depth=1, spatial="gcn", hidden=16)

    make_eval_fn("TINY", hpo_cfg=HPOConfig(n_trials=1, max_epochs=1))(g, "cpu")
    assert captured[-1] is None                          # 默认关

    make_eval_fn("TINY", hpo_cfg=HPOConfig(n_trials=1, max_epochs=1),
                 checkpoint_dir="/tmp/ck")(g, "cpu")
    assert captured[-1] == "/tmp/ck"


# ---------------------------------------------------------------------------
# EvalSpec / 模型卡
# ---------------------------------------------------------------------------

def test_eval_spec_checkpoint_dir_default_none():
    """EvalSpec.checkpoint_dir 默认 None, 旧构造方式 (位置/关键字) 完全兼容。"""
    assert EvalSpec(dataset="PeMS04").checkpoint_dir is None
    assert EvalSpec(dataset="PeMS04", hpo_cfg=None, eval_factory="m:f").checkpoint_dir is None
    spec = EvalSpec(dataset="PeMS04", checkpoint_dir="/tmp/ck")
    assert spec.checkpoint_dir == "/tmp/ck"


def test_render_model_card_weights_line():
    """模型卡"复现"节: 有 weights_file 加权重存档行, 无则不加 (旧行为)。"""
    from darwin_st.leaderboard import LeaderboardRow, render_model_card
    row = LeaderboardRow(rank=1, name="darwin-st/abc", kind="ours", mae=18.5,
                         is_ours=True, signature="abc12345",
                         genotype={"blocks": [], "hidden": 64}, hp={}, synth_ops=[])
    card = render_model_card(row, "PeMS04", "AGCRN", 18.40, weights_file="model.pt")
    assert "权重存档" in card and "model.pt" in card
    card_wo = render_model_card(row, "PeMS04", "AGCRN", 18.40)
    assert "权重存档" not in card_wo
