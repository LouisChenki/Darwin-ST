"""E5 基线同炉复现跑批入口 (Baseline Reproduction Runner)。

对四个文献基线 (dcrnn/gwnet/agcrn/stid) 在与演化模型完全相同的炉子里复现:
同一 prepare.py 切分/scaler/邻接, 同一 masked_mae 归一化训练损失 + Adam +
clip 5.0 (train_one 语义), 同一 P.evaluate 真实尺度 masked 指标分别评 val/test。

用法:
    BASELINE=stid DATASET=PeMS04 DEVICE=cuda:0 python -u scripts/run_baseline.py
    BASELINE=dcrnn SEEDS=0,1,2 OUT=results/dcrnn_pems04.json python -u scripts/run_baseline.py

环境变量:
    BASELINE    必需: 注册表内任一名 (dcrnn/gwnet/agcrn/stid/ha/ar/fc_lstm)
    DATASET     默认 PeMS04 (protocol.PROFILES 任一已登记数据集)
    DEVICE      默认 cuda:0
    EPOCHS      训练轮数上限 (默认 100; 早停 patience=10, min_delta=0.001 相对)
    SEED        单 seed 默认 42; SEEDS=逗号列表覆盖为多 seed 模式 (逐个重训)
    OUT         结果 JSON 输出路径 (默认只打印不落盘)

输出 JSON 字段风格与 scripts/eval_test.py 的 summarize_results 一致:
val/test 的 mae/rmse/mape + offset_test_minus_val + test_mae_mean/std + n_seeds。
"""

from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from darwin_st.baselines import BASELINES as _REG
BASELINE_NAMES = tuple(sorted(_REG))

# baseline_registry.py 中对应的已发表条目名 (AGCRN 未登记 → None, 不臆造)
_PUBLISHED_KEY = {"dcrnn": "DCRNN", "gwnet": "GraphWaveNet", "agcrn": None, "stid": "STID"}


def _env(k: str, d: str) -> str:
    return os.environ.get(k, d)


def parse_seeds(seeds_env: str, seed_default: int = 42) -> list[int]:
    """SEEDS 逗号列表优先; 空则回退单 SEED (纯函数, 可测)。"""
    seeds = [int(s) for s in seeds_env.split(",") if s.strip()]
    return seeds if seeds else [seed_default]


def summarize_results(results: list[dict], baseline: str, dataset: str) -> dict:
    """多 seed 结果汇总 (纯函数, 可测): test MAE mean±std + 偏移统计。

    字段风格与 scripts/eval_test.py summarize_results 一致 (val/test/offset/n_seeds)。
    """
    import statistics
    val_maes = [r["val"]["mae"] for r in results]
    test_maes = [r["test"]["mae"] for r in results]
    offsets = [r["offset_test_minus_val"] for r in results]
    return {
        "baseline": baseline, "dataset": dataset, "n_seeds": len(results),
        "results": results,
        "val_mae_mean": statistics.mean(val_maes),
        "test_mae_mean": statistics.mean(test_maes),
        "test_mae_std": statistics.stdev(test_maes) if len(test_maes) > 1 else 0.0,
        "offset_mean": statistics.mean(offsets),
        "offset_std": statistics.stdev(offsets) if len(offsets) > 1 else 0.0,
    }


def main() -> int:
    name = _env("BASELINE", "").lower()
    if name not in BASELINE_NAMES:
        print(f"错误: BASELINE 须为 {BASELINE_NAMES} 之一, 实为 {name!r}", file=sys.stderr)
        return 2

    dataset = _env("DATASET", "PeMS04")
    device = _env("DEVICE", "cuda:0")
    epochs = int(_env("EPOCHS", "100"))
    out_path = _env("OUT", "")
    seeds = parse_seeds(os.environ.get("SEEDS", ""), int(_env("SEED", "42")))

    import numpy as np
    import torch

    from darwin_st.baseline_registry import get_baseline_metric
    from darwin_st.baselines import build_baseline
    from darwin_st.baselines.train_baseline import DEFAULT_HPS, train_baseline
    from darwin_st.data import prepare as P
    from darwin_st.data.protocol import get_profile

    profile = get_profile(dataset)
    data_dir = P.prepare_dataset(dataset)
    adj = P.load_adj(data_dir)
    if adj is None and name in ("dcrnn", "gwnet"):
        print(f"⚠️ {data_dir} 无邻接矩阵 (adj.npy), {name} 退化为自环/纯自适应图")
    # 同炉统一超参 (四基线同一优化器配置, 保证可比; 见 train_baseline.DEFAULT_HPS)
    hps = dict(DEFAULT_HPS)
    batch_size = int(hps.get("batch_size", 64))

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model = build_baseline(name, num_nodes=profile.num_nodes,
                               in_channels=profile.num_channels,
                               seq_len_in=profile.seq_len_in,
                               seq_len_out=profile.seq_len_out, adj=adj)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        ckpt = os.path.join("/tmp", f"baseline_{name}_{dataset}_s{seed}.pt")
        for p in (ckpt, ckpt + ".meta.json"):
            if os.path.exists(p):
                os.remove(p)               # sidecar 语义: 删旧保证本次权重是本次的

        best_val, info = train_baseline(
            model, hps, data_dir, profile, adj, device,
            max_epochs=epochs, checkpoint_path=ckpt)
        print(f"[seed {seed}] 训练完成: best val MAE={best_val:.4f} "
              f"(epochs={info['n_epochs']}, params={n_params:,})")

        # 加载 best-val 权重, val/test 双口径评测
        eval_model = build_baseline(name, num_nodes=profile.num_nodes,
                                    in_channels=profile.num_channels,
                                    seq_len_in=profile.seq_len_in,
                                    seq_len_out=profile.seq_len_out, adj=adj)
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu", weights_only=True)
            eval_model.load_state_dict(sd)
        else:                              # 训练全程 NaN/从未刷新 best → 兜底评当前权重
            print(f"[seed {seed}] ⚠️ checkpoint 未生成 (best 未刷新), 评测训练末态权重")
            eval_model.load_state_dict(
                {k: v.detach().cpu() for k, v in model.state_dict().items()})
        eval_model.to(device)

        met_val = P.evaluate(eval_model, data_dir, "val", batch_size, device=device,
                             null_val=profile.null_val)
        met_test = P.evaluate(eval_model, data_dir, "test", batch_size, device=device,
                              null_val=profile.null_val)
        rec = {"seed": seed, "n_params": n_params, "n_epochs": info["n_epochs"],
               "retrained_val_mae": best_val,
               "val": met_val, "test": met_test,
               "offset_test_minus_val": met_test["mae"] - met_val["mae"]}
        results.append(rec)
        print(f"[seed {seed}] val MAE={met_val['mae']:.4f} | test MAE={met_test['mae']:.4f} "
              f"| RMSE={met_test['rmse']:.3f} | MAPE={met_test['mape'] * 100:.2f}% "
              f"| test−val={rec['offset_test_minus_val']:+.4f}")

    summary = summarize_results(results, name, dataset)
    print("\n=== 汇总 ===")
    print(f"{name} @ {dataset}: test MAE = {summary['test_mae_mean']:.4f} ± "
          f"{summary['test_mae_std']:.4f} ({len(results)} seeds)")
    pub_key = _PUBLISHED_KEY.get(name)
    pub = get_baseline_metric(profile.name, pub_key, "mae") if pub_key else None
    if pub is not None:
        print(f"文献参考 (baseline_registry, {pub_key}): MAE={pub} "
              f"— 同炉数字与文献同档即复现合格")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
