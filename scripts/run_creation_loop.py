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
from darwin_st.optim.scheduler import EvalSpec
from darwin_st.optim.train import make_eval_fn
from darwin_st.memory.store import MemoryStore
from darwin_st.search.genotype import Genotype, STBlock

from darwin_st.knowledge import all_seed_mechanisms
from darwin_st.knowledge.embedding import SentenceTransformerEmbedder
from darwin_st.knowledge.graph_store import Neo4jGraphStore, InMemoryGraphStore
from darwin_st.knowledge.qc import mechanisms_from_cards
from darwin_st.creation import (CreationLoop, OperatorRegistry, OperatorSynthesizer,
                                SynthesisConfig)
from darwin_st.creation.llm import OpenAICompatLLM
from darwin_st.creation.aider_backend import AiderBackend, AiderConfig


def _env_int(k, d):
    v = os.environ.get(k)
    return int(v) if v else d


def _load_mechanisms(cache):
    """按 KB_SOURCE 选知识源: cards=615 定稿库 / seeds=16 种子卡。

    KB_SOURCE=cards (默认): 从 mechanism_cards.json 读 → mechanisms_from_cards 重建。
    KB_SOURCE=seeds: 16 种子卡 (旧行为, 便于对比时一行切回)。
    路径可 KB_CARDS 覆盖, 默认仓库内 data/mechanism_cards.json。
    """
    import json

    src = os.environ.get("KB_SOURCE", "cards")
    if src == "seeds":
        return all_seed_mechanisms(), "seeds-16"
    default_cards = os.path.join(os.path.dirname(__file__), "..", "src", "darwin_st",
                                 "knowledge", "data", "mechanism_cards.json")
    path = os.environ.get("KB_CARDS", default_cards)
    cards = json.load(open(path, encoding="utf-8")).get("mechanisms", [])
    mechs = mechanisms_from_cards(cards)
    return mechs, f"cards-{len(mechs)}"


def _domain_dist(mechs):
    from collections import Counter
    return dict(sorted(Counter(m.origin_domain for m in mechs).items()))


def main():
    ds = os.environ.get("DATASET", "PeMS04")
    n_gpus = _env_int("N_GPUS", torch.cuda.device_count() or 1)
    from darwin_st.optim.train import limit_cpu_threads
    limit_cpu_threads(n_gpus)  # 防多 worker 并发训练 CPU 线程过订阅 (见 train.limit_cpu_threads)
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
    mechs, kb_tag = _load_mechanisms(cache)
    kb_reload = os.environ.get("KB_RELOAD", "0") == "1"
    print(f"[知识] 源={kb_tag} 域分布={_domain_dist(mechs)}")
    embedder = SentenceTransformerEmbedder()
    try:
        store = Neo4jGraphStore(embedder)
        existing = len(store.all_mechanisms())
        # 残留旧库 (如上次的 16 种子) 与本次期望不符, 或显式 KB_RELOAD → 清空重灌,
        # 否则"验证了 615"是假的 (检索到的是旧库)。
        if kb_reload or existing != len(mechs):
            with store.driver.session() as s:
                s.run("MATCH (n) WHERE n:Mechanism OR n:Precondition DETACH DELETE n")
            for m in mechs:
                store.add_mechanism(m)
            print(f"[知识] Neo4j 清空重灌: {existing} → {len(store.all_mechanisms())}")
        else:
            print(f"[知识] Neo4j 已有 {existing} 机制, 数量匹配, 复用 (KB_RELOAD=1 强制重灌)")
    except Exception as e:
        print(f"[知识] Neo4j 不可用({e}), 回退内存图")
        store = InMemoryGraphStore(embedder)
        for m in mechs:
            store.add_mechanism(m)
        print(f"[知识] 内存图机制数: {len(store.all_mechanisms())}")

    # --- 创造层 (真实 DeepSeek + Aider) ---
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    llm = OpenAICompatLLM()
    backend = AiderBackend(AiderConfig(model=f"deepseek/{model}"))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2, temperature=0.9),
                                code_backend=backend)
    registry = OperatorRegistry(persist_dir=os.path.join(cache, "dynamic_ops"))
    registry.load_persisted()  # 加载之前合成的算子

    mem = MemoryStore(os.environ.get("MEMORY_DB", os.path.join(cache, "memory_creation.db")))
    from darwin_st.creation import CreationConfig
    n_hypo = _env_int("N_HYPOTHESES", 4)
    cloop = CreationLoop(store, embedder, synth, registry, memory=mem,
                         config=CreationConfig(n_hypotheses=n_hypo), llm=llm)

    # --- 优化引擎 (真实训练) ---
    hpo_cfg = HPOConfig(n_trials=hpo_trials, max_epochs=max_epochs,
                        min_resource=2, reduction_factor=3, n_startup_trials=2)
    eval_fn = make_eval_fn(ds, hpo_cfg=hpo_cfg)
    # 进程后端: worker 自建 eval_fn + 从 persist_dir 重载 synth 算子 (spawn 子进程丢进程全局 SPATIAL_OPS)
    backend = os.environ.get("BACKEND", "auto")
    eval_spec = EvalSpec(dataset=ds, hpo_cfg=hpo_cfg,
                         synth_persist_dir=os.path.join(cache, "dynamic_ops"))
    base = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)  # 故意弱基线, 逼出创造

    cfg = OrchestratorConfig(dataset=ds,
                             run_tag=os.environ.get("RUN_TAG", f"exp/{ds.lower()}-creation"),
                             population_size=pop, tournament_size=min(3, pop),
                             max_rounds=max_rounds, target_mae=0.0,  # 不可达 → 靠 max_rounds 停
                             warmup_keep=pop * 2, seed=0)

    def on_round(state):
        b = state.best_mae
        n_creat = sum(1 for h in state.history if h.get("event") == "creation")
        print(f"[round {state.rounds}] evals={state.evals} KEEP={state.n_keep} "
              f"best_MAE={b if b<1e9 else 'inf'} 创造次数={n_creat}")

    orch = Orchestrator(cfg, eval_fn, devices=n_gpus, memory=mem, base_genotype=base,
                        on_round=on_round, creation_loop=cloop,
                        eval_spec=eval_spec, backend=backend)
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
        print(f"  • 合成算子 {c.get('operators')} (成功{c.get('n_success')}个) 解决瓶颈: {c['bottleneck']}")
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
