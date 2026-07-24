"""排行榜导出入口 (Leaderboard Exporter) —— 把实验成果汇成 git 可展示的静态榜单 + 可复现存档。

读三数据源 → 调 src/darwin_st/leaderboard.py 纯核心聚合渲染 → 落盘:
  - memory.db (MemoryStore.list_trials): 本项目跑出的 KEEP 实验 (架构/超参/MAE)。
  - registry persist_dir (dynamic_ops/*.json + *.py): synth 跨域合成算子的元数据 + 代码。
  - baseline_registry: 已发表 baseline/SOTA 锚点。

输出目录结构 (默认 leaderboard/, 可 OUT_DIR 覆盖):
    leaderboard/
      README.md                      # 总览: 各数据集最佳本项目 vs SOTA
      <dataset>.md                   # 单数据集榜单 (baseline+本项目混排)
      models/<dataset>/<rank>_mae<x>_<sig>/
        genotype.json  hparams.json  model_card.md
        model.pt + model.meta.json          # 若 CHECKPOINT_DIR 里有对应签名的权重存档
        operators/<synth_op>.py + .json   # 若用了合成算子 (复现需要)

用法 (导出本地/拷回的 db):
    MEMORY_DB=/path/memory_creation.db OPS_DIR=/path/dynamic_ops \
    OUT_DIR=leaderboard DATASETS=PeMS04 python scripts/export_leaderboard.py

环境变量 (均有默认):
    MEMORY_DB(必需有效路径), OPS_DIR(synth 算子持久目录), OUT_DIR(默认 leaderboard),
    DATASETS(逗号分隔, 默认全 baseline_registry 数据集), RUN_TAG(隔离某次实验, 默认不隔离全收),
    THRESHOLD(入榜门槛 MAE; 默认=该数据集最弱 baseline; "none"=不设限全收 KEEP),
    CHECKPOINT_DIR(权重存档目录; 设了则把对应签名的 model.pt+sidecar 一并归档)。
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import sys

from darwin_st.baseline_registry import BASELINE_METRICS, PROTOCOL_NOTES, get_sota
from darwin_st.leaderboard import (
    build_leaderboard,
    render_index_md,
    render_leaderboard_md,
    render_model_card,
    select_our_models,
    weakest_baseline_mae,
)
from darwin_st.memory.store import MemoryStore


def _load_synth_meta(ops_dir: str | None) -> dict[str, dict]:
    """从 registry persist_dir 读所有 synth 算子的 .json 元数据 (纯 JSON, 不 import torch)。

    返回 {reg_name: {composition, source_mechanisms, real_mae, class_name, _code_path, _meta_path}}。
    """
    meta: dict[str, dict] = {}
    if not ops_dir or not os.path.isdir(ops_dir):
        return meta
    for fn in sorted(os.listdir(ops_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(ops_dir, fn)
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        reg = m.get("reg_name") or fn[:-5]
        code_path = os.path.join(ops_dir, fn[:-5] + ".py")
        m["_code_path"] = code_path if os.path.exists(code_path) else None
        m["_meta_path"] = path
        meta[reg] = m
    return meta


def _copy_synth_ops(synth_ops, synth_meta, dest_ops_dir) -> list[str]:
    """把模型用到的 synth 算子 .py + .json 拷进该模型存档的 operators/ (复现需要)。

    返回成功拷贝的算子名 (缺代码的跳过, 不致命)。
    """
    copied = []
    for op in synth_ops:
        m = synth_meta.get(op)
        if not m or not m.get("_code_path"):
            continue
        os.makedirs(dest_ops_dir, exist_ok=True)
        shutil.copy2(m["_code_path"], os.path.join(dest_ops_dir, op + ".py"))
        if m.get("_meta_path"):
            shutil.copy2(m["_meta_path"], os.path.join(dest_ops_dir, op + ".json"))
        copied.append(op)
    return copied


def _find_checkpoint(ckpt_dir: str | None, dataset: str, signature: str | None) -> str | None:
    """在权重存档目录里按文件名找该模型对应的 .pt (命名 <dataset>_<sig[:12]>.pt, 见 train.py)。

    先精确匹配 sig[:12]; 找不到再退 sig[:8] 前缀 glob (手工改存档/旧命名兜底)。无命中 → None。
    """
    if not ckpt_dir or not signature or not os.path.isdir(ckpt_dir):
        return None
    exact = os.path.join(ckpt_dir, f"{dataset}_{signature[:12]}.pt")
    if os.path.exists(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(ckpt_dir, f"{dataset}_{signature[:8]}*.pt")))
    return hits[0] if hits else None


def _copy_checkpoint(ckpt_dir: str | None, dataset: str, signature: str | None,
                     mdir: str) -> str | None:
    """把该模型的权重 .pt + sidecar 拷进归档目录 (model.pt + model.meta.json)。

    返回拷入后的权重文件名 ("model.pt") 供模型卡引用; 无存档 → None (不致命)。
    """
    src = _find_checkpoint(ckpt_dir, dataset, signature)
    if src is None:
        return None
    shutil.copy2(src, os.path.join(mdir, "model.pt"))
    meta_src = src + ".meta.json"
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, os.path.join(mdir, "model.meta.json"))
    return "model.pt"


def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def export(memory_db: str, out_dir: str, datasets: list[str],
           ops_dir: str | None, run_tag: str | None, threshold_env: str | None,
           ckpt_dir: str | None = None) -> None:
    store = MemoryStore(memory_db)
    synth_meta = _load_synth_meta(ops_dir)
    summaries: list[dict] = []

    try:
        for ds in datasets:
            baselines = BASELINE_METRICS.get(ds, {})
            sota_name, sota_mae = get_sota(ds)
            protocol = PROTOCOL_NOTES.get(ds)

            # 入榜门槛: 默认最弱 baseline (超它即够格); THRESHOLD=none 不设限; 数值则用之
            if threshold_env and threshold_env.lower() == "none":
                threshold = None
            elif threshold_env:
                threshold = float(threshold_env)
            else:
                threshold = weakest_baseline_mae(baselines)

            trials = store.list_trials(dataset=ds, status="KEEP", run_tag=run_tag)
            our = select_our_models(trials, threshold_mae=threshold, run_tag=run_tag)
            rows = build_leaderboard(ds, our, baselines, sota_name, sota_mae)

            # 榜单页
            _write(os.path.join(out_dir, f"{ds}.md"),
                   render_leaderboard_md(ds, rows, protocol_note=protocol,
                                         sota_name=sota_name, sota_mae=sota_mae))

            # 每个本项目模型: 存档目录 (genotype/hparams/model_card/operators)
            n_ours = 0
            best_ours_mae = None
            for r in rows:
                if not r.is_ours:
                    continue
                n_ours += 1
                if best_ours_mae is None:
                    best_ours_mae = r.mae   # rows 已按 MAE 升序, 第一个本项目即最佳
                mdir = os.path.join(out_dir, r.model_dir)
                os.makedirs(mdir, exist_ok=True)
                _write(os.path.join(mdir, "genotype.json"),
                       json.dumps(r.genotype, ensure_ascii=False, indent=2))
                _write(os.path.join(mdir, "hparams.json"),
                       json.dumps(r.hp, ensure_ascii=False, indent=2))
                if r.synth_ops:
                    _copy_synth_ops(r.synth_ops, synth_meta, os.path.join(mdir, "operators"))
                weights_file = _copy_checkpoint(ckpt_dir, ds, r.signature, mdir)
                _write(os.path.join(mdir, "model_card.md"),
                       render_model_card(r, ds, sota_name, sota_mae,
                                         synth_meta=synth_meta, protocol_note=protocol,
                                         weights_file=weights_file))

            summaries.append({
                "dataset": ds, "sota_name": sota_name, "sota_mae": sota_mae,
                "best_ours_mae": best_ours_mae, "n_ours": n_ours,
                "n_baselines": len(baselines),
            })
            vs = (f"{best_ours_mae - sota_mae:+.2f}" if (best_ours_mae is not None
                  and sota_mae is not None) else "—")
            print(f"[{ds}] 本项目入榜 {n_ours} 个; 最佳 MAE="
                  f"{best_ours_mae if best_ours_mae is not None else '—'} (vs SOTA {vs}); "
                  f"门槛={'none' if threshold is None else f'{threshold:.2f}'}")
    finally:
        store.close()

    _write(os.path.join(out_dir, "README.md"), render_index_md(summaries))
    print(f"\n✅ 导出完成 → {out_dir}/  (README.md + {len(datasets)} 榜单页 + models/)")


def main():
    memory_db = os.environ.get("MEMORY_DB")
    if not memory_db or not os.path.exists(memory_db):
        print(f"✗ MEMORY_DB 未设或不存在: {memory_db!r}\n"
              f"  用法: MEMORY_DB=/path/memory_creation.db python scripts/export_leaderboard.py",
              file=sys.stderr)
        sys.exit(1)
    out_dir = os.environ.get("OUT_DIR", "leaderboard")
    ops_dir = os.environ.get("OPS_DIR")  # synth 算子持久目录 (没有则模型卡无算子代码链接)
    ckpt_dir = os.environ.get("CHECKPOINT_DIR")  # 权重存档目录 (没有则归档不含 model.pt)
    run_tag = os.environ.get("RUN_TAG")  # None = 不隔离, 全收
    threshold_env = os.environ.get("THRESHOLD")
    ds_env = os.environ.get("DATASETS")
    datasets = ([d.strip() for d in ds_env.split(",") if d.strip()]
                if ds_env else list(BASELINE_METRICS.keys()))

    print(f"=== 排行榜导出 ===\n db={memory_db} ops_dir={ops_dir} out={out_dir} "
          f"datasets={datasets} run_tag={run_tag or '(全部)'} ckpt_dir={ckpt_dir}")
    export(memory_db, out_dir, datasets, ops_dir, run_tag, threshold_env, ckpt_dir)


if __name__ == "__main__":
    main()
