#!/usr/bin/env python
"""P3-b 建图 pipeline: 从 500 篇文献语料构建跨域机制知识库 (机制卡)。

设计哲学 (docs/TIER2_DESIGN_DECISIONS.md §4,§9): 以【正交可组合的基元机制】为中心,
价值在互斥性(基元不重叠) + 全面性(覆盖 ST/NLP/CV/Graph 四域)。机制卡严格符合
src/darwin_st/knowledge/ontology.py 的 Mechanism schema。

pipeline 形态 (确定性骨架 + 可迭代的抽取逻辑):
  load_corpus → [迭代循环: extract → post-process → self-critique → 改 prompt → 重跑]
              → 灌库 (Neo4j 优先, 回退 InMemory→JSON) + 人类可读 JSON 复核稿。

抽取策略 (用户拍板):
  - 摘要起步: 第一遍只用 metadata.abstract (快/省)。
  - 按需回退 PDF: ST 核心域默认回退解析 Method 段; 跨域(NLP/CV/Graph)摘要不足才回退。
  - LLM = DeepSeek (复用 darwin_st.creation.llm.OpenAICompatLLM, key 从 DEEPSEEK_API_KEY 读)。

自我改进闭环 (无"好卡"先验, 一次抽不会好):
  每轮抽 → 去重/归一/覆盖统计 → LLM 扮质检员读【去重后的库】诊断互斥性/全面性/词表 →
  把诊断注入下一轮抽取 prompt 的"避免清单 / 必抽清单" → 重跑, 直到机制数稳定且 4 域覆盖。

确定性后处理在 darwin_st.knowledge.extraction (纯函数, 已单测)。本脚本负责 IO + 编排 + LLM。

用法:
    export DEEPSEEK_API_KEY=...            # 必需 (找不到则只跑 dry-run 演示后处理)
    export DEEPSEEK_MODEL=deepseek-v4-flash   # 用户要求 flash 加速; 默认见下
    python scripts/build_kg.py                # 全量 500 篇, 4 轮迭代
    python scripts/build_kg.py --limit 40     # 小批量试跑验证 prompt
    python scripts/build_kg.py --dry-run      # 不调 LLM, 仅演示后处理/灌库骨架 (用 mock 卡)
    python scripts/build_kg.py --rounds 3 --no-pdf-fallback

产物:
  - <out_dir>/mechanism_cards.json     人类可读机制卡库 (供用户复核)
  - <out_dir>/coverage.json            每轮覆盖/自评诊断
  - Neo4j (若可用) 或 <out_dir>/graph_store.json  灌库结果
默认 out_dir = src/darwin_st/knowledge/data/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- 让脚本在 repo 根直接跑 (src layout) ---
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from darwin_st.knowledge.ontology import (  # noqa: E402
    DOMAINS,
    FUNCTION_VOCAB,
    PRECONDITION_VOCAB,
    Mechanism,
)
from darwin_st.knowledge.extraction import (  # noqa: E402
    coverage_report,
    dedup_mechanisms,
    parse_mechanism_cards,
)
from darwin_st.knowledge.seeds import all_seed_mechanisms  # noqa: E402

# 默认语料路径 (可 --corpus 覆盖)
DEFAULT_CORPUS = "/Users/chenkaiqi/Documents/codes/codex/deep-research-spatiotemporal/corpus"
DEFAULT_OUT = str(_REPO / "src" / "darwin_st" / "knowledge" / "data")

# DeepSeek flash 模型名 (用户要求 flash 加速)。DeepSeek 当前 flash 档型号;
# 可被 DEEPSEEK_MODEL 环境变量覆盖。创造层默认是 deepseek-v4-pro, 这里抽取用 flash。
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 摘要信息不足的判据: 短于此 → 触发 PDF 回退 (ST 域); 跨域域阈值更低。
ABSTRACT_MIN_LEN = 600
ST_DOMAINS = {"spatio-temporal", "spatiotemporal", "st"}

# 一次 LLM 抽取喂几篇 (批处理省 token; 太大则单卡质量降)。
BATCH_SIZE = 4


# ===========================================================================
# 1. 语料加载
# ===========================================================================


def load_corpus(corpus_dir: str, limit: int | None = None) -> list[dict]:
    """读 metadata.jsonl → list[dict]。每条含 id/title/abstract/domain/pdf_path 等。"""
    meta_path = os.path.join(corpus_dir, "metadata.jsonl")
    rows: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]
    return rows


def needs_pdf_fallback(row: dict, no_pdf: bool) -> bool:
    """判定是否回退 PDF 方法段。ST 核心域默认回退; 跨域域仅摘要太短才回退。"""
    if no_pdf:
        return False
    abstract = (row.get("abstract") or "").strip()
    domain = (row.get("domain") or "").lower()
    if domain in ST_DOMAINS:
        return True  # 核心 ST 默认补 PDF
    return len(abstract) < ABSTRACT_MIN_LEN  # 跨域: 信息不足才补


def extract_pdf_methods(pdf_path: str, max_chars: int = 6000) -> str:
    """抽 PDF 的 Method/Approach 节文本 (pypdf 优先, pdfplumber 兜底)。失败返回 ""。"""
    if not pdf_path or not os.path.exists(pdf_path):
        return ""
    text = ""
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:12])
    except Exception:
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages[:12])
        except Exception:
            return ""
    return _slice_method_section(text, max_chars)


def _slice_method_section(text: str, max_chars: int) -> str:
    """启发式截取 Method/Approach/Model 节 (找小节标题, 取其后若干字符)。"""
    if not text:
        return ""
    low = text.lower()
    markers = ["method", "approach", "model architecture", "proposed", "our model",
               "methodology", "framework"]
    pos = -1
    for mk in markers:
        i = low.find(mk)
        if i >= 0:
            pos = i
            break
    if pos < 0:
        pos = 0
    return text[pos : pos + max_chars]


# ===========================================================================
# 2. LLM 抽取 (prompt 可迭代)
# ===========================================================================

EXTRACT_SYSTEM = """你是时空预测 AutoML 系统的【跨域机制抽取员】。任务: 从论文文本里抽出\
【正交可组合的基元机制 (primitive mechanism)】, 不是整个模型, 不是案例。

什么是一个"机制卡":
- 一个【正交基元】: 如"掩码自编码""膨胀因果卷积""状态空间模型""对比学习""旋转位置编码"\
"傅里叶/小波频域分解""混合专家(MoE)""可逆实例归一化(RevIN)""patch化分块""残差连接"。
- 反例(不要抽成一张卡): "PatchTST""Informer""STGCN" 这类完整模型 —— 要把它们拆成内部基元。
- 粒度标准: 与种子卡一致——一个机制是一句"抽象功能"能概括、能独立移植到别的域的东西。

每张卡输出这些字段 (JSON):
  name              规范英文 snake_case 名 (如 masked_autoencoding), 用学界通名, 不要自造。
  abstract_function 一句【域无关】的目的描述 (会被向量化做跨域检索; 不许出现具体模型/数据集名)。
  preconditions     从受控词表里选 (1-3 个): {VOCAB}
  causal_behavior   结构→功能的因果链 (为什么这个结构能达成那个功能)。
  function_tags     从受控功能词表里选 (0-2 个): {FUNC}
  math_structure    核心数学/算子 (一行公式或伪代码)。
  consequences      代价/权衡 (算力/内存/不稳定等)。
  origin_domain     从 {DOMAINS} 里选该机制的"发源域"。
  abstraction_level "concept"(高层跨域基元) 或 "variant"(某基元的具体变体)。
  anti_patterns     1-2 条雷区 (可空)。

硬约束:
- 只抽真正的【可移植基元】。一篇论文通常贡献 0-2 个新基元; 多数是在复用已有基元——\
  复用的基元也可抽 (跨域全面性靠它), 但 abstract_function 要写成域无关的。
- preconditions 必须从给定受控词表里挑, 不要自造前提。
- 严格输出 JSON: {{"mechanisms": [ {{...}}, ... ]}}。无机制则输出 {{"mechanisms": []}}。"""


def build_extract_prompt(avoid: list[str], must_have: list[str]) -> str:
    """组装 system prompt; 把上一轮自评诊断作为 avoid/must_have 注入 (迭代自改进)。"""
    p = EXTRACT_SYSTEM.format(
        VOCAB=", ".join(sorted(PRECONDITION_VOCAB)),
        FUNC=", ".join(sorted(FUNCTION_VOCAB)),
        DOMAINS=", ".join(sorted(DOMAINS)),
    )
    if avoid:
        p += ("\n\n【上一轮发现的重复/过细机制, 本轮避免重复抽取或合并为更基元的形式】:\n- "
              + "\n- ".join(avoid[:20]))
    if must_have:
        p += ("\n\n【应当存在却尚未抽到的重要基元, 本轮若文本涉及务必抽出】:\n- "
              + "\n- ".join(must_have[:20]))
    return p


def make_user_prompt(batch: list[dict]) -> str:
    """把一批论文 (摘要 [+ PDF 方法段]) 拼成 user 消息。"""
    parts = []
    for r in batch:
        seg = [f"### 论文 {r.get('id','?')}: {r.get('title','')}",
               f"域: {r.get('domain','')}",
               f"摘要: {(r.get('abstract') or '').strip()}"]
        if r.get("_method_text"):
            seg.append("方法段(节选): " + r["_method_text"])
        parts.append("\n".join(seg))
    return ("从以下论文抽取正交基元机制卡 (严格 JSON)。同一基元在多篇出现时只抽一次、写成域无关形式。\n\n"
            + "\n\n".join(parts))


def extract_batch(llm, batch: list[dict], avoid, must_have) -> list[Mechanism]:
    system = build_extract_prompt(avoid, must_have)
    user = make_user_prompt(batch)
    try:
        resp = llm.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.3, max_tokens=4096,
        )
    except Exception as e:  # 单批失败不致命, 记日志继续
        print(f"  [warn] LLM 抽取批失败: {e}", flush=True)
        return []
    prov = ",".join(r.get("id", "") for r in batch)
    return parse_mechanism_cards(resp, provenance=prov)


# ===========================================================================
# 3. 自我质检 (LLM 扮质检员读去重后的库, 不读原文)
# ===========================================================================

CRITIC_SYSTEM = """你是跨域机制知识库的【质检员】。下面给你【当前去重后的机制卡库】(只看卡, 不看原文)。
按机制库哲学诊断 (库价值 = 互斥/正交 + 覆盖 ST/NLP/CV/Graph 四域全面性):
1. 互斥性: 哪些卡其实是同一基元的重复/重叠? 哪些卡粒度太细(是别的基元的变体, 应合并)或太粗(是整个模型, 应拆)?
2. 全面性: 是否缺了明显该有的经典基元? 参考清单(不限于此): 掩码自编码, 状态空间模型(SSM/Mamba),\
 对比学习, 膨胀因果卷积, 小波/傅里叶频域, 混合专家(MoE), 可逆实例归一化(RevIN), patch分块,\
 旋转位置编码, 图扩散卷积, 自适应邻接, 自注意力, 残差连接, 序列分解。
3. 各域(ST/NLP/CV/Graph)覆盖是否均衡?

只输出 JSON:
{"avoid": ["<应合并/避免重复的机制名或描述>", ...], "must_have": ["<缺失应补的基元名>", ...],
 "verdict": "<一句话总体诊断>"}"""


def self_critique(llm, mechs: list[Mechanism]) -> dict:
    """LLM 读去重后机制卡库 → 返回 {avoid, must_have, verdict}。LLM 不可用时返回空诊断。"""
    cards = [{"name": m.name, "abstract_function": m.abstract_function,
              "origin_domain": m.origin_domain, "abstraction_level": m.abstraction_level,
              "preconditions": m.preconditions} for m in mechs]
    user = "当前机制卡库 (共 %d 张):\n%s" % (
        len(cards), json.dumps(cards, ensure_ascii=False, indent=1))
    try:
        resp = llm.chat(
            [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.2, max_tokens=2048,
        )
    except Exception as e:
        print(f"  [warn] 自评失败: {e}", flush=True)
        return {"avoid": [], "must_have": [], "verdict": "self-critique unavailable"}
    from darwin_st.knowledge.extraction import _extract_json  # 复用解析

    data = _extract_json(resp) or {}
    return {
        "avoid": [str(x) for x in (data.get("avoid") or [])],
        "must_have": [str(x) for x in (data.get("must_have") or [])],
        "verdict": str(data.get("verdict", "")),
    }


# ===========================================================================
# 4. 灌库
# ===========================================================================


def build_store(out_dir: str):
    """Neo4j 优先, 不可用回退 InMemory。返回 (store, backend_name, embedder)。"""
    from darwin_st.knowledge.embedding import HashEmbedder

    # 嵌入: 服务器用 SentenceTransformer (真语义), 本地默认 Hash (零依赖)。
    use_st = os.environ.get("KG_EMBEDDER", "").lower() in ("st", "sentence", "sbert")
    if use_st:
        from darwin_st.knowledge.embedding import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder()
    else:
        embedder = HashEmbedder()

    try:
        import socket
        socket.create_connection(("localhost", 7687), timeout=1.0).close()
        from darwin_st.knowledge.graph_store import Neo4jGraphStore
        store = Neo4jGraphStore(embedder)
        return store, "neo4j", embedder
    except Exception:
        from darwin_st.knowledge.graph_store import InMemoryGraphStore
        return InMemoryGraphStore(embedder), "inmemory", embedder


def ingest(store, mechs: list[Mechanism]) -> tuple[int, list[str]]:
    """灌库 (validate 失败的卡跳过并记录)。返回 (成功数, 跳过原因)。"""
    ok, skipped = 0, []
    for m in mechs:
        try:
            store.add_mechanism(m)
            ok += 1
        except Exception as e:
            skipped.append(f"{m.name}: {e}")
    return ok, skipped


def dump_json(mechs: list[Mechanism], path: str) -> None:
    payload = {"n": len(mechs), "mechanisms": [m.to_dict() for m in mechs]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def dump_inmemory_store(store, path: str) -> None:
    """InMemory 后端持久化到 JSON (含 embedding, 供复跑加载)。"""
    data = {"mechanisms": [m.to_dict() for m in store.all_mechanisms()]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===========================================================================
# 5. 编排: 迭代自改进主循环
# ===========================================================================


def run_extraction_round(llm, corpus, avoid, must_have, no_pdf) -> list[Mechanism]:
    """一轮全量抽取: 分批喂 LLM, 收集草稿卡。"""
    drafts: list[Mechanism] = []
    # 预取 PDF 方法段 (按需)
    for r in corpus:
        if needs_pdf_fallback(r, no_pdf) and "_method_text" not in r:
            r["_method_text"] = extract_pdf_methods(r.get("pdf_path", ""))
    for i in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[i : i + BATCH_SIZE]
        drafts.extend(extract_batch(llm, batch, avoid, must_have))
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"    ...抽取进度 {min(i+BATCH_SIZE,len(corpus))}/{len(corpus)} "
                  f"草稿卡={len(drafts)}", flush=True)
    return drafts


def iterate(llm, corpus, seed_names, rounds, no_pdf, out_dir) -> tuple[list[Mechanism], list[dict]]:
    """自我改进闭环。返回 (最终机制卡, 每轮诊断日志)。"""
    avoid: list[str] = []
    must_have: list[str] = []
    history: list[dict] = []
    prev_n = -1
    final: list[Mechanism] = []

    for rnd in range(1, rounds + 1):
        print(f"\n=== 迭代第 {rnd}/{rounds} 轮: 抽取 ===", flush=True)
        drafts = run_extraction_round(llm, corpus, avoid, must_have, no_pdf)
        merged, removed = dedup_mechanisms(drafts, seed_names=seed_names)
        cov = coverage_report(merged)
        print(f"  草稿 {len(drafts)} → 去重后 {len(merged)} (去掉/合并 {removed}); "
              f"域分布 {cov.by_domain}", flush=True)

        print(f"=== 迭代第 {rnd} 轮: 自我质检 ===", flush=True)
        critique = self_critique(llm, merged)
        print(f"  诊断: {critique['verdict']} | 待避免 {len(critique['avoid'])} | "
              f"待补 {len(critique['must_have'])}", flush=True)

        history.append({"round": rnd, "n_drafts": len(drafts), "n_after_dedup": len(merged),
                        "removed": removed, "coverage": cov.as_dict(), "critique": critique})
        final = merged
        # 收敛判据: 机制数稳定(±5%) 且 4 域(ST/NLP/CV/Graph)都覆盖 且无强烈待补信号
        four = {"ST", "NLP", "CV", "GraphLearning"}
        covered4 = four.issubset(set(cov.by_domain))
        stable = prev_n > 0 and abs(len(merged) - prev_n) <= max(2, prev_n * 0.05)
        if rnd >= 2 and stable and covered4 and len(critique["must_have"]) <= 1:
            print(f"  ✓ 收敛 (机制数稳定 {prev_n}→{len(merged)}, 四域覆盖, 无显著缺口)", flush=True)
            break
        prev_n = len(merged)
        avoid, must_have = critique["avoid"], critique["must_have"]
        # 每轮存一份覆盖快照
        with open(os.path.join(out_dir, "coverage.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    return final, history


# ===========================================================================
# dry-run: 无 LLM 时演示后处理 + 灌库骨架 (用种子卡 + 几张合成草稿)
# ===========================================================================


def _mock_drafts() -> list[Mechanism]:
    """造几张含重复/别名前提的草稿, 演示去重+归一 (不触网)。"""
    return [
        Mechanism(name="masked_autoencoder_module", abstract_function="遮盖输入并重建以学冗余结构",
                  preconditions=["redundancy", "unlabeled"], causal_behavior="遮盖→重建→学冗余",
                  origin_domain="cv"),
        Mechanism(name="reversible_instance_norm", abstract_function="可逆实例归一化抵消分布漂移",
                  preconditions=["non_stationarity"], causal_behavior="归一化→预测→反归一化",
                  origin_domain="time_series", math_structure="x'=(x-μ)/σ; ŷ=ŷ'*σ+μ"),
        Mechanism(name="patching", abstract_function="把序列切成子块作为 token 降低序列长度",
                  preconditions=["long_range", "redundancy"], causal_behavior="分块→token→省算力",
                  origin_domain="nlp"),
    ]


# ===========================================================================
# main
# ===========================================================================


def main():
    ap = argparse.ArgumentParser(description="P3-b 跨域机制知识库建图 pipeline")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="只取前 N 篇 (小批量试跑)")
    ap.add_argument("--rounds", type=int, default=4, help="自我改进最大轮数")
    ap.add_argument("--no-pdf-fallback", action="store_true", help="禁用 PDF 回退 (纯摘要)")
    ap.add_argument("--include-seeds", action="store_true", default=True,
                    help="把 16 张种子卡并入最终库 (默认 True)")
    ap.add_argument("--dry-run", action="store_true", help="不调 LLM, 演示后处理+灌库骨架")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    seeds = all_seed_mechanisms()
    seed_names = {m.name for m in seeds}

    if args.dry_run:
        print("=== DRY-RUN: 演示后处理 + 灌库骨架 (无 LLM) ===", flush=True)
        drafts = _mock_drafts()
        merged, removed = dedup_mechanisms(drafts, seed_names=seed_names)
        mechs = (seeds if args.include_seeds else []) + merged
        cov = coverage_report(mechs)
        print(f"草稿 {len(drafts)} → 去重 {len(merged)} (去 {removed}); "
              f"并入种子后共 {len(mechs)}; 域分布 {cov.by_domain}", flush=True)
    else:
        # --- 真实 LLM 抽取 ---
        if not os.environ.get("DEEPSEEK_API_KEY"):
            print("[FATAL] 未找到 DEEPSEEK_API_KEY 环境变量。pipeline 已就绪, 请设置 key 后重跑:",
                  flush=True)
            print("        export DEEPSEEK_API_KEY=...; export DEEPSEEK_MODEL=deepseek-v4-flash",
                  flush=True)
            print("        (或先 python scripts/build_kg.py --dry-run 验证后处理/灌库链路)",
                  flush=True)
            sys.exit(2)
        from darwin_st.creation.llm import OpenAICompatLLM

        llm = OpenAICompatLLM(model=DEFAULT_MODEL)
        print(f"=== 加载语料 {args.corpus} (limit={args.limit}) ===", flush=True)
        corpus = load_corpus(args.corpus, limit=args.limit)
        print(f"  共 {len(corpus)} 篇; 模型={DEFAULT_MODEL}", flush=True)

        t0 = time.time()
        extracted, history = iterate(llm, corpus, seed_names, args.rounds,
                                     args.no_pdf_fallback, args.out_dir)
        mechs = (seeds if args.include_seeds else []) + extracted
        cov = coverage_report(mechs)
        print(f"\n=== 抽取完成 ({time.time()-t0:.0f}s): 自动卡 {len(extracted)} + "
              f"种子 {len(seeds) if args.include_seeds else 0} = {len(mechs)} ===", flush=True)

    # --- 灌库 ---
    store, backend, _ = build_store(args.out_dir)
    ok, skipped = ingest(store, mechs)
    print(f"\n=== 灌库 [{backend}]: 成功 {ok}/{len(mechs)}; 跳过 {len(skipped)} ===", flush=True)
    for s in skipped[:10]:
        print(f"  [skip] {s}", flush=True)

    # 人类可读复核稿 (始终写)
    cards_path = os.path.join(args.out_dir, "mechanism_cards.json")
    dump_json([m for m in mechs], cards_path)
    print(f"  机制卡复核稿 → {cards_path}", flush=True)
    if backend == "inmemory":
        gp = os.path.join(args.out_dir, "graph_store.json")
        dump_inmemory_store(store, gp)
        print(f"  内存图持久化 → {gp}", flush=True)
    store.close()

    cov = coverage_report(mechs)
    print(f"\n=== 最终覆盖 ===\n  总数 {cov.n_total} | 域 {cov.by_domain} | "
          f"层 {cov.by_abstraction} | 未用前提 {cov.unused_preconditions}", flush=True)


if __name__ == "__main__":
    main()
