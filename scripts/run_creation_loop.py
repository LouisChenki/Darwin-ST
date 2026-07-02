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


def resolve_warmstart_base(mem, dataset, scope):
    """RESUME 作用域匹配暖启动: 只在 (dataset, space_version) 有历史 KEEP 时返回最优 genotype。

    不匹配/新作用域 → 返回 None → orchestrator 走 seed_genotypes 冷启动公平 bootstrap。
    这是 Stage2 稀释 bug 的直接修复: 换代 (space_version 变) 时旧代最优查不到 → 不暖启动旧局部最优,
    且新代 synth 目录为空 → 变异池不被旧算子碾压, 新原子算子公平竞争。
    抽为纯函数便于单测 (见 test_scope / test_memory_store)。
    """
    try:
        prev = mem.best_so_far(dataset, metric="val_mae", space_version=scope.space_version)
    except Exception:
        return None
    if prev and prev.get("genotype"):
        from darwin_st.search.genotype import Genotype
        return Genotype.from_dict(prev["genotype"])
    return None


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
    stagnation = _env_int("STAGNATION", 8)   # 停滞耐心 (修衔接节奏: 别太小导致高频创造过早收敛)
    cache = os.environ.get("DARWIN_ST_CACHE", ".")

    # 实验作用域 (数据集 + 搜索空间代际): 隔离 synth 目录 / RESUME 暖启动 / trial 落库。
    # space_version 默认自动哈希 (内置算子集变→自动新代→干净隔离); SPACE_VERSION 环境变量可覆盖。
    from darwin_st.scope import ExperimentScope
    scope = ExperimentScope.resolve(ds)

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
    registry = OperatorRegistry(persist_dir=scope.synth_dir(cache))
    n_loaded = registry.load_persisted()  # 加载**本作用域**之前合成的算子 (换代/换数据集→空目录冷场)
    print(f"[作用域] {scope.dataset}/{scope.space_version} synth_dir={scope.synth_dir(cache)} "
          f"载回 {len(n_loaded)} 算子")

    mem = MemoryStore(os.environ.get("MEMORY_DB", os.path.join(cache, "memory_creation.db")))
    from darwin_st.creation import CreationConfig
    n_hypo = _env_int("N_HYPOTHESES", 4)
    seed_hpo_trials = _env_int("SEED_HPO_TRIALS", 20)   # 创造 seed 专项大 HPO (AlphaEvolve 式深评)
    cloop = CreationLoop(store, embedder, synth, registry, memory=mem,
                         config=CreationConfig(n_hypotheses=n_hypo, seed_hpo_trials=seed_hpo_trials),
                         llm=llm)

    # --- 优化引擎 (真实训练) ---
    hpo_cfg = HPOConfig(n_trials=hpo_trials, max_epochs=max_epochs,
                        min_resource=2, reduction_factor=3, n_startup_trials=2)
    eval_fn = make_eval_fn(ds, hpo_cfg=hpo_cfg)
    # 进程后端: worker 自建 eval_fn + 从 persist_dir 重载 synth 算子 (spawn 子进程丢进程全局 SPATIAL_OPS)
    backend = os.environ.get("BACKEND", "auto")
    eval_spec = EvalSpec(dataset=ds, hpo_cfg=hpo_cfg,
                         synth_persist_dir=scope.synth_dir(cache))   # 同 registry 单一源, 主/worker 一致
    base = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)  # 故意弱基线, 逼出创造
    # RESUME=1 (看门狗重启续跑用): **作用域匹配**暖启动 —— 只从本 (dataset, space_version) 的最优 KEEP
    # 暖启动。换代/换数据集 (space_version 变) 时查不到旧代最优 → base 留弱基线 → 冷启动公平 bootstrap
    # (修 Stage2 稀释 bug: 不暖启动旧代局部最优, 且本作用域 synth 目录为空, 新算子不被旧算子碾压)。
    if os.environ.get("RESUME", "0") == "1":
        warm = resolve_warmstart_base(mem, ds, scope)
        if warm is not None:
            base = warm
            print(f"[续跑] 作用域匹配暖启动 base (scope={ds}/{scope.space_version})")
        else:
            print(f"[续跑] 本作用域 ({ds}/{scope.space_version}) 无历史 → 冷启动公平 seed (新代/新数据集)")

    # target_mae: 默认 0.0 不可达(靠 max_rounds 停); 24h 冲 SOTA 设 TARGET_MAE=17.80 程序化停。
    target_mae = float(os.environ.get("TARGET_MAE", "0.0"))
    cfg = OrchestratorConfig(dataset=ds,
                             run_tag=os.environ.get("RUN_TAG", f"exp/{ds.lower()}-creation"),
                             space_version=scope.space_version,   # trial 落库带代际, best_so_far 隔离
                             population_size=pop, tournament_size=min(3, pop),
                             max_rounds=max_rounds, target_mae=target_mae,
                             warmup_keep=pop * 2, seed=0,
                             creation_refine_rounds=_env_int("REFINE_ROUNDS", 4),  # 创造后精修窗口
                             # 距离自适应停滞耐心 (Gap-Annealed): 远 SOTA 快创造, 近 SOTA 重精修
                             adaptive_patience=os.environ.get("ADAPTIVE_PATIENCE", "0") == "1",
                             patience_base=_env_int("PATIENCE_BASE", 3),
                             patience_k=float(os.environ.get("PATIENCE_K", "5.0")),
                             patience_cap=_env_int("PATIENCE_CAP", 12))

    def on_round(state):
        b = state.best_mae
        n_creat = sum(1 for h in state.history if h.get("event") == "creation")
        gap = (b - state.sota_mae) if (state.sota_mae and b < 1e9) else None
        pat = orch.archive.stagnation_patience
        print(f"[round {state.rounds}] evals={state.evals} KEEP={state.n_keep} "
              f"best_MAE={b if b<1e9 else 'inf'} "
              + (f"gap={gap:+.2f} " if gap is not None else "")
              + f"patience={pat} 创造次数={n_creat}")

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
