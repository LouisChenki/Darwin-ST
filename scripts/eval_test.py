"""E1 口径对齐评测 (test-set evaluation) —— val 与 test 同协议评测 + 偏移统计。

投稿地基: 项目历史成绩全部是验证集 (val) MAE, 文献 baseline 报的是 test MAE,
混排在 IJGIS 审稿是死穴。本脚本对模型卡 (genotype + hparams) 做:
  1) 复权 (有 .pt 直接用; 无则按卡内超参重训, 权重落临时 checkpoint);
  2) 用同一 data 管线 (data.prepare.evaluate, masked 指标, inverse 回真实尺度,
     12 horizons 平均, split 不变) 分别评测 val 与 test;
  3) 输出 val/test 双口径 + 偏移量 (test−val), 供榜单重排与"选择集过拟合"分析。

用法 (服务器, 需模型卡目录含 genotype.json/hparams.json/operators/):
    MODEL_DIR=leaderboard/models/PeMS04/05_mae18.35_f93f4598 \
        python -u scripts/eval_test.py
    # 多 seed (E2 复用): SEEDS=0,1,2,3,4 逐个重训
    SEEDS=0,1,2,3,4 MODEL_DIR=... python -u scripts/eval_test.py
    # 已有权重只评测 (不重训): CKPT=/path/to/x.pt
    CKPT=/path/to/x.pt MODEL_DIR=... python -u scripts/eval_test.py

环境变量:
    MODEL_DIR   必需, 模型卡目录 (含 genotype.json / hparams.json; operators/ 可选)
    DATASET     默认 PeMS04 (模型卡数据集)
    DEVICE      默认 cuda:0
    EPOCHS      重训轮数上限 (默认 60, 与长跑 MAX_EPOCHS 一致)
    SEED        单 seed 默认 42; SEEDS=逗号列表覆盖为多 seed 模式 (逐个重训)
    EARLY_STOP  早停耐心 (默认 0=关, 与长跑配置解耦; 要复现长跑行为可设 3)
    CKPT        已有 .pt 权重路径 (非空则跳过训练直接评测)
    OUT         结果 JSON 输出路径 (默认打印不落盘)
"""

from __future__ import annotations

import json
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _env(k: str, d: str) -> str:
    return os.environ.get(k, d)


def summarize_results(results: list[dict], model_dir: str, dataset: str) -> dict:
    """多 seed 结果汇总 (纯函数, 可测): test MAE mean±std + 偏移统计。"""
    import statistics
    test_maes = [r["test"]["mae"] for r in results]
    offsets = [r["offset_test_minus_val"] for r in results]
    return {
        "model_dir": model_dir, "dataset": dataset, "n_seeds": len(results),
        "results": results,
        "test_mae_mean": statistics.mean(test_maes),
        "test_mae_std": statistics.stdev(test_maes) if len(test_maes) > 1 else 0.0,
        "offset_mean": statistics.mean(offsets),
        "offset_std": statistics.stdev(offsets) if len(offsets) > 1 else 0.0,
    }


def main() -> int:
    model_dir = os.environ.get("MODEL_DIR", "")
    if not model_dir or not os.path.isdir(model_dir):
        print(f"错误: MODEL_DIR 不存在: {model_dir!r}", file=sys.stderr)
        return 2

    import numpy as np
    import torch

    from darwin_st.creation.registry import OperatorRegistry
    from darwin_st.data import prepare as P
    from darwin_st.data.protocol import get_profile
    from darwin_st.optim.train import train_one
    from darwin_st.search.genotype import Genotype

    dataset = _env("DATASET", "PeMS04")
    device = _env("DEVICE", "cuda:0")
    epochs = int(_env("EPOCHS", "60"))
    early_stop = int(_env("EARLY_STOP", "0"))
    ckpt_env = _env("CKPT", "")
    out_path = _env("OUT", "")

    # ---- 1) 模型卡: genotype + hparams + 算子注册 ----
    with open(os.path.join(model_dir, "genotype.json"), encoding="utf-8") as f:
        geno = Genotype.from_dict(json.load(f))
    with open(os.path.join(model_dir, "hparams.json"), encoding="utf-8") as f:
        hps = json.load(f)
    ops_dir = os.path.join(model_dir, "operators")
    if os.path.isdir(ops_dir):
        reg = OperatorRegistry(persist_dir=ops_dir)
        loaded = reg.load_persisted()          # synth_*/aux_* 按前缀注入全局注册表
        print(f"[算子] 模型卡自带算子注册: {loaded}")
    # 兜底: 模型卡可能没装齐 genotype 引用的全部算子 (历史卡缺失合成算子) ——
    # 校验失败则从动态算子库 (DYNAMIC_OPS_DIR, 默认缓存 dynamic_ops/) 全量兜底加载
    try:
        geno.validate()
    except ValueError as e:
        dyn_dir = os.environ.get("DYNAMIC_OPS_DIR") or os.path.join(
            _env("DARWIN_ST_CACHE", os.path.expanduser("~/.cache/darwin-st")), "dynamic_ops")
        print(f"[算子] 模型卡算子不齐 ({e}), 从动态算子库兜底: {dyn_dir}")
        loaded2 = OperatorRegistry(persist_dir=dyn_dir).load_persisted()
        print(f"[算子] 动态算子库注册 {len(loaded2)} 个")
        geno.validate()                        # 再不过就真炸
    print(f"[模型卡] {model_dir} | {dataset} | 超参 {hps}")

    profile = get_profile(dataset)
    data_dir = P.prepare_dataset(dataset)
    adj = P.load_adj(data_dir)

    seeds_env = os.environ.get("SEEDS", "")
    seeds = [int(s) for s in seeds_env.split(",") if s.strip()] if seeds_env else \
        [int(_env("SEED", "42"))]

    results = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # ---- 2) 复权: 有 ckpt 直接用; 否则按卡内超参重训 (best-val 权重落临时盘) ----
        if ckpt_env:
            ckpt_path = ckpt_env
            trace_best = None
        else:
            ckpt_path = os.path.join("/tmp", f"eval_test_{geno.signature()[:12]}_s{seed}.pt")
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)           # sidecar 语义: 删旧保证本次权重是本次的
            best_val, trace = train_one(
                geno, hps, data_dir, profile, adj, device,
                max_epochs=epochs, early_stop_patience=early_stop,
                checkpoint_path=ckpt_path,
                checkpoint_meta={"signature": geno.signature(), "dataset": dataset})
            trace_best = best_val
            print(f"[seed {seed}] 训练完成: best val MAE={best_val:.4f} "
                  f"(epochs={trace.n_epochs_run})")
        if not os.path.exists(ckpt_path):
            print(f"[seed {seed}] 错误: checkpoint 不存在 {ckpt_path}", file=sys.stderr)
            return 1

        # ---- 3) 加载 best-val 权重, val/test 双口径评测 ----
        from darwin_st.search.builder import build_model
        model = build_model(geno, num_nodes=profile.num_nodes,
                            in_channels=profile.num_channels,
                            seq_len_in=profile.seq_len_in, seq_len_out=profile.seq_len_out,
                            adj=adj, dropout=float(hps.get("dropout", 0.0)),
                            num_heads=hps.get("num_heads")).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        batch_size = int(hps.get("batch_size", 64))
        met_val = P.evaluate(model, data_dir, "val", batch_size, device=device,
                             null_val=profile.null_val)
        met_test = P.evaluate(model, data_dir, "test", batch_size, device=device,
                              null_val=profile.null_val)
        rec = {"seed": seed, "retrained_val_mae": trace_best,
               "val": met_val, "test": met_test,
               "offset_test_minus_val": met_test["mae"] - met_val["mae"]}
        results.append(rec)
        print(f"[seed {seed}] val MAE={met_val['mae']:.4f} | test MAE={met_test['mae']:.4f} "
              f"| RMSE={met_test['rmse']:.3f} | MAPE={met_test['mape']:.2f}% "
              f"| test−val={rec['offset_test_minus_val']:+.4f}")

    # ---- 4) 汇总 ----
    summary = summarize_results(results, model_dir, dataset)
    print("\n=== 汇总 ===")
    print(f"test MAE = {summary['test_mae_mean']:.4f} ± {summary['test_mae_std']:.4f} "
          f"({len(results)} seeds)")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
