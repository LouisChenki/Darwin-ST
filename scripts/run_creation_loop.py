"""Tier-2 创造闭环端到端实跑: 真实 DeepSeek 创造 + 真实 PeMS04 训练评测。

完整闭环在真实环境跑: orchestrator(进化+HPO+调度)+ CreationLoop(DeepSeek/Aider 合成)
+ 真实 PeMS04 + GPU。停滞触发跨域创造 → 合成算子真实训练 → 看真实 MAE。

用法 (服务器, 需 source darwin-secrets.env):
    source /root/autodl-tmp/darwin-secrets.env
    DARWIN_ST_CACHE=... GITHUB_MIRROR=... python scripts/run_creation_loop.py

环境变量: DATASET / MAX_ROUNDS / N_GPUS / POP_SIZE / HPO_TRIALS / MAX_EPOCHS /
          STAGNATION(默认2, 调小快速触发创造)
"""

from __future__ import annotations

import os
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import torch

from darwin_st.optim.hpo import HPOConfig
from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
from darwin_st.optim.train import make_eval_fn
from darwin_st.memory.store import MemoryStore
from darwin_st.search.genotype import Genotype, STBlock

from darwin_st.knowledge import all_seed_mechanisms
from darwin_st.knowledge.embedding import SentenceTransformerEmbedder
from darwin_st.knowledge.graph_store import Neo4jGraphStore, InMemoryGraphStore
from darwin_st.creation import (CreationLoop, OperatorRegistry, OperatorSynthesizer,
                                SynthesisConfig)
from darwin_st.creation.llm import OpenAICompatLLM
from darwin_st.creation.aider_backend import AiderBackend, AiderConfig


def _env_int(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def main():
    ds = os.environ.get("DATASET", "PeMS04")
    n_gpus = _env_int("N_GPUS", torch.cuda.device_count() or 1)
    pop = _env_int("POP_SIZE", 6)
    max_rounds = _env_int("MAX_ROUNDS", 6)
    hpo_trials = _env_int("HPO_TRIALS", 3)
    max_epochs = _env_int("MAX_EPOCHS", 10)
    stagnation = _env_int("STAGNATION", 2)
    cache = os.environ.get("DARWIN_ST_CACHE", ".")

    print(f"=== Tier-2 创造闭环端到端实跑: {ds} ===")
    print(f"GPU={n_gpus} 种群={pop} 轮数={max_rounds} HPO/arch={hpo_trials} epochs={max_epochs} "
          f"停滞触发={stagnation}")

    # --- 知识图谱 (跨域检索) ---
    embedder = SentenceTransformerEmbedder()
    try:
        store = Neo4jGraphStore(embedder)
        if not store.all_mechanisms():
            for m in all_seed_mechanisms():
                store.add_mechanism(m)
        print(f"[知识] Neo4j 机制数: {len(store.all_mechanisms())}")
    except Exception as e:
        print(f"[知识] Neo4j 不可用({e}), 回退内存图")
        store = InMemoryGraphStore(embedder)
        for m in all_seed_mechanisms():
            store.add_mechanism(m)

    # --- 创造层 (真实 DeepSeek + Aider) ---
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    llm = OpenAICompatLLM()
    backend = AiderBackend(AiderConfig(model=f"deepseek/{model}"))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=3, temperature=0.8),
                                code_backend=backend)
    registry = OperatorRegistry(persist_dir=os.path.join(cache, "dynamic_ops"))
    registry.load_persisted()  # 加载之前合成的算子

    mem = MemoryStore(os.path.join(cache, "memory_creation.db"))
    cloop = CreationLoop(store, embedder, synth, registry, memory=mem)

    # --- 优化引擎 (真实训练) ---
    hpo_cfg = HPOConfig(n_trials=hpo_trials, max_epochs=max_epochs,
                        min_resource=2, reduction_factor=3, n_startup_trials=2)
    eval_fn = make_eval_fn(ds, hpo_cfg=hpo_cfg)
    base = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)  # 故意弱基线, 逼出创造

    cfg = OrchestratorConfig(dataset=ds, run_tag=f"exp/{ds.lower()}-creation",
                             population_size=pop, tournament_size=min(3, pop),
                             max_rounds=max_rounds, target_mae=0.0,  # 不可达 → 靠 max_rounds 停
                             warmup_keep=pop * 2, seed=0)

    def on_round(state):
        b = state.best_mae
        n_creat = sum(1 for h in state.history if h.get("event") == "creation")
        print(f"[round {state.rounds}] evals={state.evals} KEEP={state.n_keep} "
              f"best_MAE={b if b<1e9 else 'inf'} 创造次数={n_creat}")

    orch = Orchestrator(cfg, eval_fn, devices=n_gpus, memory=mem, base_genotype=base,
                        on_round=on_round, creation_loop=cloop)
    orch.archive.stagnation_patience = stagnation  # 快速触发创造

    t0 = time.time()
    state = orch.run()
    dt = time.time() - t0

    print(f"\n=== 结束: {state.stop_reason} ({dt:.0f}s) ===")
    print(f"评估 {state.evals}: KEEP={state.n_keep} DISCARD={state.n_discard} CRASH={state.n_crash}")
    print(f"最优 MAE={state.best_mae if state.best_mae<1e9 else 'inf'}")

    # 创造事件
    creations = [h for h in state.history if h.get("event") == "creation"]
    print(f"\n=== 创造事件 ({len(creations)} 次) ===")
    for c in creations:
        print(f"  • 合成算子 {c['operator']} 解决瓶颈: {c['bottleneck']}")
    print(f"注入算子库: {registry.registered_names()}")

    # 合成算子是否被评估 + 表现
    synth_trials = []
    for row in mem.conn.execute(
            "SELECT genotype_json, val_mae, status FROM experiments WHERE genotype_json LIKE '%synth_%'").fetchall():
        synth_trials.append((row[1], row[2]))
    print(f"\n=== 合成算子的真实评测 ({len(synth_trials)} 个含 synth 算子的试验) ===")
    for mae, status in synth_trials:
        print(f"  • MAE={mae} status={status}")

    # insights
    print(f"\n=== 融合经验 (memory insights) ===")
    for ins in mem.get_insights(ds)[:5]:
        print(f"  • [{ins['confidence']}] {ins['insight_text'][:120]}")

    store.close(); mem.close()
    print("\n=== 端到端创造闭环完成 ===")


if __name__ == "__main__":
    main()
