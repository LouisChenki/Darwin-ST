"""自治研究运行入口 (Autonomous Research Runner) —— P2 点火。

把 orchestrator + 真实 eval_fn + memory 接起来, 在真实数据上跑自治优化循环。
这是整套系统第一次真正"自己跑起来"的入口。

用法 (服务器):
    DARWIN_ST_CACHE=/root/autodl-tmp/darwin-st-cache GITHUB_MIRROR=https://gh-proxy.com/ \
    DATASET=PeMS04 MAX_ROUNDS=3 N_GPUS=4 HPO_TRIALS=4 MAX_EPOCHS=12 \
    python scripts/run_autoresearch.py

环境变量 (均有默认):
    DATASET, RUN_TAG, MAX_ROUNDS, MAX_EVALS, N_GPUS, POP_SIZE,
    HPO_TRIALS, MAX_EPOCHS, MEMORY_DB, TARGET_MAE, SPATIAL, TEMPORAL,
    CHECKPOINT_DIR(权重存档目录, 默认空=关闭), CKPT_KEEP(存档保留个数, 默认20)
"""

from __future__ import annotations

import os
import sys
import time

import torch

# 行缓冲 stdout: 否则 piped/重定向时全缓冲, 看不到实时轮次进度 (服务器实跑教训)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from darwin_st.optim.hpo import HPOConfig
from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
from darwin_st.optim.scheduler import EvalSpec
from darwin_st.optim.train import make_eval_fn
from darwin_st.memory.store import MemoryStore
from darwin_st.search.genotype import random_genotype


def _env_int(key, default):
    v = os.environ.get(key)
    return int(v) if v else default


def _env_float_or_none(key):
    v = os.environ.get(key)
    return float(v) if v else None


def main():
    ds = os.environ.get("DATASET", "PeMS04")
    run_tag = os.environ.get("RUN_TAG", f"exp/{ds.lower()}-auto")
    n_gpus = _env_int("N_GPUS", torch.cuda.device_count() or 1)
    # 防多 worker 并发训练时 CPU 线程过订阅 (208 vCPU 上 torch 默认 ~100 线程 × n_gpus worker
    # → thrashing 卡死)。按 worker 数限制每训练的 intra-op 线程。
    from darwin_st.optim.train import limit_cpu_threads
    limit_cpu_threads(n_gpus)
    pop = _env_int("POP_SIZE", 8)
    max_rounds = _env_int("MAX_ROUNDS", 3)
    max_evals = _env_int("MAX_EVALS", 0) or None
    hpo_trials = _env_int("HPO_TRIALS", 4)
    max_epochs = _env_int("MAX_EPOCHS", 12)
    target_mae = _env_float_or_none("TARGET_MAE")
    mem_db = os.environ.get("MEMORY_DB", os.path.join(
        os.environ.get("DARWIN_ST_CACHE", "."), "memory.db"))

    print(f"=== Darwin-ST 自治研究: {ds} ===")
    print(f"GPU={n_gpus} 种群={pop} 轮数={max_rounds} HPO_trials/arch={hpo_trials} "
          f"max_epochs={max_epochs} device_count={torch.cuda.device_count()}")

    hpo_cfg = HPOConfig(n_trials=hpo_trials, max_epochs=max_epochs,
                        min_resource=2, reduction_factor=3, n_startup_trials=2)

    # 初始创新点占位: 用 adaptive 图 + tcn + 节点嵌入起步(实际创新点由用户/Tier2 注入)
    base = random_genotype(depth=2, spatial=os.environ.get("SPATIAL", "adaptive"),
                           temporal=os.environ.get("TEMPORAL", "tcn"),
                           fusion="residual", hidden=64)

    cfg = OrchestratorConfig(
        dataset=ds, run_tag=run_tag, population_size=pop, tournament_size=min(4, pop),
        max_rounds=max_rounds, max_evals=max_evals, target_mae=target_mae, seed=0,
    )

    mem = MemoryStore(mem_db)
    # 权重存档 (默认关): CHECKPOINT_DIR 非空 → 刷新纪录的模型权重落盘, 结束时 prune 留 top-K
    ckpt_dir = os.environ.get("CHECKPOINT_DIR", "") or None
    ckpt_keep = _env_int("CKPT_KEEP", 20)
    eval_fn = make_eval_fn(ds, hpo_cfg=hpo_cfg, checkpoint_dir=ckpt_dir)
    # 进程后端: worker 自建 eval_fn (避闭包 pickle), 真正 4 卡并行 (绕 GIL)
    backend = os.environ.get("BACKEND", "auto")
    eval_spec = EvalSpec(dataset=ds, hpo_cfg=hpo_cfg, checkpoint_dir=ckpt_dir)

    def on_round(state):
        b = state.best_mae
        gap = (b - state.sota_mae) if (state.sota_mae and b < float("inf")) else None
        print(f"[round {state.rounds}] evals={state.evals} KEEP={state.n_keep} "
              f"DISCARD={state.n_discard} CRASH={state.n_crash} best_MAE="
              f"{b:.3f} SOTA({state.sota_name})={state.sota_mae}"
              + (f" gap={gap:+.3f}" if gap is not None else ""))

    orch = Orchestrator(cfg, eval_fn, devices=n_gpus, memory=mem,
                        base_genotype=base, on_round=on_round,
                        eval_spec=eval_spec, backend=backend)

    t0 = time.time()
    state = orch.run()
    dt = time.time() - t0

    print(f"\n=== 结束: {state.stop_reason} ({dt:.0f}s) ===")
    print(f"评估 {state.evals} 个架构: KEEP={state.n_keep} DISCARD={state.n_discard} CRASH={state.n_crash}")
    print(f"最优 MAE={state.best_mae:.3f} | SOTA({state.sota_name})={state.sota_mae} | 超越={state.beat_sota}")
    print(f"档案 coverage={orch.archive.coverage():.2f} ({len(orch.archive)}/{orch.archive.TOTAL_CELLS}) "
          f"QD={orch.archive.qd_score():.3f}")
    if state.best_genotype is not None:
        bg = state.best_genotype
        print(f"最优架构: depth={bg.depth} hidden={bg.hidden} adj={bg.adj_mode} "
              f"emb_node={bg.embedding.use_node} blocks="
              f"{[(b.spatial_op, b.temporal_op, b.fusion) for b in bg.blocks]}")
    # 权重存档收尾: 只留 top-K 最优 (按 sidecar val_mae), 防长跑撑盘。失败不致命。
    if ckpt_dir:
        from darwin_st.optim.checkpoint import prune_to_top_k
        try:
            pruned = prune_to_top_k(ckpt_dir, keep=ckpt_keep)
            print(f"[checkpoint] 权重保留 top-{ckpt_keep}, 清理 {len(pruned)} 个 → {ckpt_dir}")
        except Exception as e:
            print(f"[checkpoint] prune 失败 ({type(e).__name__}: {e}), 不致命")
    mem.close()


if __name__ == "__main__":
    main()
