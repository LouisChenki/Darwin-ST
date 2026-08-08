"""Tier-2 创造闭环端到端实跑: 真实 DeepSeek 创造 + 真实 PeMS04 训练评测。

完整闭环在真实环境跑: orchestrator(进化+HPO+调度)+ CreationLoop(DeepSeek/Aider 合成)
+ 真实 PeMS04 + GPU。停滞触发跨域创造 → 合成算子真实训练 → 看真实 MAE。

用法 (服务器, 需 source darwin-secrets.env):
    source /root/autodl-tmp/darwin-secrets.env
    DARWIN_ST_CACHE=... GITHUB_MIRROR=... python scripts/run_creation_loop.py

环境变量: DATASET / MAX_ROUNDS / N_GPUS / POP_SIZE / HPO_TRIALS / MAX_EPOCHS /
          STAGNATION(默认2, 调小快速触发创造) /
          CHECKPOINT_DIR(权重存档目录, 默认空=关闭) / CKPT_KEEP(存档保留个数, 默认20) /
          USE_PROXY(B3 粗筛开关, 默认1) / PROXY_DEVICE(proxy 短训设备, 默认 cuda:0) /
          DEEPSEEK_MODEL(主模型, 默认 deepseek-v4-flash (新版已超 pro preview)) /
          DEEPSEEK_FAST_MODEL(B5 双模型路由的便宜快模型, 默认 deepseek-v4-flash;
          置空 → 不单建 fast_llm, 全部假设走强模型) /
          REFLECT(反思巩固开关, 默认1) / REFLECT_EVERY(每 N 条新履历反思一批, 默认20) /
          INSIGHTS_CAPACITY(insights 表容量上限, 默认50) /
          AUX_CREATION(B7 辅助任务创造通道开关, 默认1: 检索命中自监督/掩码族机制时
          造自监督辅助损失模块挂 aux_op 槽, 而非架构算子; 0 → 全部走原算子通道)
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
from darwin_st.knowledge.embedding import (HashEmbedder, SentenceTransformerEmbedder,
                                           sentence_transformers_available, warn_hash_fallback)
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
    # 真语义嵌入缺失时显式回退 Hash + 醒目警告: 否则 ModuleNotFoundError 会在下面的
    # Neo4j try/except 里被误报成 "Neo4j 不可用", 且内存图路径再抛一次, 用户不知真因。
    if sentence_transformers_available():
        embedder = SentenceTransformerEmbedder()
    else:
        warn_hash_fallback("run_creation_loop 生产检索")
        embedder = HashEmbedder(dim=256)
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
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")   # 默认 flash (新版已超 pro preview; 2026-08 用户拍板)
    llm = OpenAICompatLLM()
    # B5 双模型路由: 第 0 个假设用强模型 (质量锚点), 其余用便宜快模型 (成本);
    # fast 全败自动 escalation 回强模型。DEEPSEEK_FAST_MODEL 置空 → 全部强模型 (行为同 B5 前)。
    fast_model = os.environ.get("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash")
    fast_llm = OpenAICompatLLM(model=fast_model) if fast_model else None
    print(f"[创造] 强模型={model} fast模型={fast_model or '(关闭, 全部强模型)'}")
    backend = AiderBackend(AiderConfig(model=f"deepseek/{model}"))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2, temperature=0.9),
                                code_backend=backend, fast_llm=fast_llm)
    registry = OperatorRegistry(persist_dir=scope.synth_dir(cache))
    n_loaded = registry.load_persisted()  # 加载**本作用域**之前合成的算子 (换代/换数据集→空目录冷场)
    print(f"[作用域] {scope.dataset}/{scope.space_version} synth_dir={scope.synth_dir(cache)} "
          f"载回 {len(n_loaded)} 算子")

    mem_db = os.environ.get("MEMORY_DB", os.path.join(cache, "memory_creation.db"))
    mem = MemoryStore(mem_db)
    from darwin_st.creation import CreationConfig
    from darwin_st.creation.creation_archive import CreationArchive
    # B1 创造档案 (履历本): 默认放 memory db 同目录, CREATION_ARCHIVE 可覆盖
    archive_path = os.environ.get(
        "CREATION_ARCHIVE",
        os.path.join(os.path.dirname(mem_db) or ".", "creation_archive.jsonl"))
    print(f"[创造档案] {archive_path}")
    n_hypo = _env_int("N_HYPOTHESES", 4)
    seed_hpo_trials = _env_int("SEED_HPO_TRIALS", 20)   # 创造 seed 专项大 HPO (AlphaEvolve 式深评)
    ccfg = CreationConfig(n_hypotheses=n_hypo, seed_hpo_trials=seed_hpo_trials,
                          use_proxy=os.environ.get("USE_PROXY", "1") == "1",
                          # B7 辅助任务创造通道 (默认开): 命中自监督/掩码族机制 → 造 aux 损失而非架构算子
                          enable_aux_creation=os.environ.get("AUX_CREATION", "1") == "1")

    # B3 评估中间档 (proxy 粗筛): 合成 seed 先在主进程短训打分, 前 proxy_top_k 才给深评大 HPO,
    # 其余小预算浅评 —— 单轮创造 GPU 成本减半以上。只绑主进程一个设备 (PROXY_DEVICE 默认 cuda:0);
    # 进程后端 worker 不需要 proxy (proxy 只在主进程创造时用)。USE_PROXY=0 关闭 (行为同 B3 前)。
    # 避免与 worker 争卡: worker 进程后端占满全部 N_GPUS 时 proxy 与 worker 同卡竞争易 OOM
    # (线上实证: 一轮 4 候选 proxy_mae 全 inf, 粗筛形同虚设) —— 建议 N_GPUS 留一卡
    # (如 4 卡机 N_GPUS=3, PROXY_DEVICE=cuda:3) 或 proxy 失败时按 inf 降级 (已有 [B3] 日志)。
    proxy_fn = None
    if ccfg.use_proxy:
        from darwin_st.data import prepare as P
        from darwin_st.data.protocol import get_profile
        from darwin_st.optim.train import proxy_eval_genotype
        proxy_device = os.environ.get("PROXY_DEVICE", "cuda:0")
        _proxy_profile = get_profile(ds)
        _proxy_data_dir = P.prepare_dataset(ds)     # 与 make_eval_fn 同一缓存, 幂等
        _proxy_adj = P.load_adj(_proxy_data_dir)

        def proxy_fn(g):
            return proxy_eval_genotype(g, _proxy_data_dir, _proxy_profile, _proxy_adj,
                                       proxy_device, epochs=ccfg.proxy_epochs,
                                       max_train_batches=ccfg.proxy_max_batches)
        print(f"[B3] proxy 粗筛开: device={proxy_device} epochs={ccfg.proxy_epochs} "
              f"batches/epoch={ccfg.proxy_max_batches} top_k={ccfg.proxy_top_k} "
              f"小预算={ccfg.small_hpo_trials}")

    cloop = CreationLoop(store, embedder, synth, registry, memory=mem,
                         config=ccfg,
                         llm=llm, archive=CreationArchive(archive_path),
                         proxy_fn=proxy_fn)

    # 反思巩固 (Reflection Consolidation): 攒够 REFLECT_EVERY 条新履历触发一次 LLM 反思,
    # 蒸馏条件式教训进 insights 表 (Hermes 容量 INSIGHTS_CAPACITY 上限), 并回注合成/诊断 prompt。
    # 用强模型低温 (0.3) 反思; REFLECT=0 关闭 (行为同接入前, prompt 逐字节不变)。
    if os.environ.get("REFLECT", "1") == "1":
        from darwin_st.creation.creation_archive import CreationArchive as _CA  # 同一路径同一档案
        from darwin_st.creation.reflection import ReflectionConfig, ReflectionLoop
        rcfg = ReflectionConfig(reflect_every_records=_env_int("REFLECT_EVERY", 20),
                                insights_capacity=_env_int("INSIGHTS_CAPACITY", 50))
        cloop.reflection = ReflectionLoop(mem, _CA(archive_path), llm, rcfg)
        print(f"[反思] 开: 每 {rcfg.reflect_every_records} 条新履历反思一批, "
              f"insights 容量 {rcfg.insights_capacity}")

    # 权重存档 (默认关): CHECKPOINT_DIR 非空 → 训练中刷新纪录的模型权重落盘该目录,
    # 脚本结束时 prune_to_top_k 只留最优 CKPT_KEEP 个防撑盘。worker 进程直接写共享盘。
    ckpt_dir = os.environ.get("CHECKPOINT_DIR", "") or None
    ckpt_keep = _env_int("CKPT_KEEP", 20)

    # --- 优化引擎 (真实训练) ---
    hpo_cfg = HPOConfig(n_trials=hpo_trials, max_epochs=max_epochs,
                        min_resource=2, reduction_factor=3, n_startup_trials=2,
                        early_stop_patience=_env_int("EARLY_STOP_PATIENCE", 0),  # >0 启用收敛早停
                        early_stop_min_delta=float(os.environ.get("EARLY_STOP_MIN_DELTA", "0.001")))
    eval_fn = make_eval_fn(ds, hpo_cfg=hpo_cfg, checkpoint_dir=ckpt_dir)
    # 进程后端: worker 自建 eval_fn + 从 persist_dir 重载 synth 算子 (spawn 子进程丢进程全局 SPATIAL_OPS)
    backend = os.environ.get("BACKEND", "auto")
    eval_spec = EvalSpec(dataset=ds, hpo_cfg=hpo_cfg,
                         synth_persist_dir=scope.synth_dir(cache),   # 同 registry 单一源, 主/worker 一致
                         checkpoint_dir=ckpt_dir)
    base = Genotype(blocks=[STBlock("gcn", "tcn")], hidden=64)  # 非 RESUME 时的弱基线 (逼出创造)
    # RESUME=1 (看门狗重启续跑用): **作用域匹配**暖启动 —— 只从本 (dataset, space_version) 的最优 KEEP
    # 暖启动。换代/换数据集 (space_version 变) 时查不到旧代最优 → base=None → 走 seed_genotypes 冷启动
    # (公平 bootstrap: 深度 [2,3,4] 播种 + 多样算子。修 Stage2 稀释 + 深度坍缩两个 bug)。
    #
    # 铁律 (曾踩坑): base **非 None 会让 evolution 走暖启动分支** (base+变异, 见 _init_seeds),
    # 绕过 seed_genotypes → 深度播种失效, 全从 depth-1 弱基线变异坍缩浅层。冷启动必须 base=None。
    # 创造仍会触发 (靠 state.best_genotype 停滞判定, 与 base 无关), 故 base=None 不影响 Tier-2。
    if os.environ.get("RESUME", "0") == "1":
        warm = resolve_warmstart_base(mem, ds, scope)
        if warm is not None:
            base = warm
            print(f"[续跑] 作用域匹配暖启动 base (scope={ds}/{scope.space_version})")
        else:
            base = None   # 真冷启动: 走 seed_genotypes ([2,3,4] 深度播种 + 多样算子)
            print(f"[续跑] 本作用域 ({ds}/{scope.space_version}) 无历史 → 冷启动公平 seed (base=None, 深度[2,3,4]播种)")

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
                             patience_cap=_env_int("PATIENCE_CAP", 12),
                             # 距离自适应 HPO trials (Gap-Annealed HPO Depth): 远/冷启动少调快筛, 近 SOTA 深调
                             adaptive_hpo_trials=os.environ.get("ADAPTIVE_HPO_TRIALS", "0") == "1",
                             hpo_trials_base=_env_int("HPO_TRIALS_BASE", 4),
                             hpo_trials_cap=_env_int("HPO_TRIALS_CAP", 12),
                             hpo_trials_k=float(os.environ.get("HPO_TRIALS_K", "2.0")),
                             hpo_trials_eps=float(os.environ.get("HPO_TRIALS_EPS", "0.3")))

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

    # 权重存档收尾: 只留 top-K 最优 (按 sidecar val_mae), 防长跑撑盘。失败不致命。
    if ckpt_dir:
        from darwin_st.optim.checkpoint import prune_to_top_k
        try:
            pruned = prune_to_top_k(ckpt_dir, keep=ckpt_keep)
            print(f"[checkpoint] 权重保留 top-{ckpt_keep}, 清理 {len(pruned)} 个 → {ckpt_dir}")
        except Exception as e:
            print(f"[checkpoint] prune 失败 ({type(e).__name__}: {e}), 不致命")

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
