#!/usr/bin/env python
"""P3 检索验收: 把定稿机制库 (mechanism_cards.json) 灌进检索栈, 跑跨域类比检索。

与 build_knowledge_graph.py 的区别: 那个只灌 16 张种子卡 + 要 Neo4j+SentenceTransformer;
本脚本灌**全部 615 张定稿卡**, 默认走本地零依赖栈 (InMemoryGraphStore + HashEmbedder),
无需 Neo4j / sentence-transformers。

检索栈两条路 (retrieval.find_cross_domain_analogy):
  - MAC: 瓶颈嵌入 → 向量召回 (Hash 嵌入下语义弱, 但能跑)
  - FAC: 瓶颈 → 前提分解 → 前提图遍历拉"共享前提的跨域机制" (不依赖嵌入质量, 是结构桥)
本地用 Hash 主要验 FAC + 全链路连通; 真实语义 (SentenceTransformer/Neo4j) 留服务器。

用法:
    python scripts/verify_retrieval.py                 # 本地 Hash 栈, 全部瓶颈
    REAL_EMBED=1 python scripts/verify_retrieval.py    # 若装了 sentence-transformers 用真实语义
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from darwin_st.knowledge.embedding import HashEmbedder, warn_hash_fallback  # noqa: E402
from darwin_st.knowledge.graph_store import InMemoryGraphStore  # noqa: E402
from darwin_st.knowledge.qc import mechanisms_from_cards  # noqa: E402
from darwin_st.knowledge.retrieval import find_cross_domain_analogy  # noqa: E402

DEFAULT_CARDS = str(_REPO / "src" / "darwin_st" / "knowledge" / "data" / "mechanism_cards.json")

# 真实 ST 瓶颈 (覆盖不同前提, 验证跨域检索能否拉回对的互补机制集)
BOTTLENECKS = [
    "模型对长时间跨度的预测误差大,长程时序依赖捕获不好",
    "交通流量数据有很强的冗余和周期性,但标注稀缺,想用自监督预训练",
    "不同路口的行为差异大异质性强,且节点身份难以区分",
    "深层图卷积导致节点表示过平滑,空间信息丢失",
    "训练分布和测试分布不一致,存在分布漂移和非平稳",
    "需要多尺度建模,不同尺度的时序模式差异大",
]


def build_local_store(cards):
    """灌 615 张卡到 InMemoryGraphStore (Hash 嵌入)。返回 (store, embedder, 灌入数, 跳过)。"""
    embedder = HashEmbedder(dim=256)
    store = InMemoryGraphStore(embedder)
    mechs = mechanisms_from_cards(cards)
    ok, skipped = 0, []
    for m in mechs:
        try:
            store.add_mechanism(m)  # 内部 validate, 非法卡会抛
            ok += 1
        except Exception as e:
            skipped.append(f"{m.name}: {e}")
    return store, embedder, ok, skipped


def main():
    cards_path = os.environ.get("CARDS", DEFAULT_CARDS)
    use_real = os.environ.get("REAL_EMBED", "0") == "1"

    cards = json.load(open(cards_path, encoding="utf-8")).get("mechanisms", [])
    print(f"=== 检索验收: 灌 {len(cards)} 张定稿卡 ===", flush=True)

    if use_real:
        try:
            from darwin_st.knowledge.embedding import SentenceTransformerEmbedder
            embedder = SentenceTransformerEmbedder()
            store = InMemoryGraphStore(embedder)
            mechs = mechanisms_from_cards(cards)
            for m in mechs:
                store.add_mechanism(m)
            ok, skipped = len(mechs), []
            print(f"  embedding: SentenceTransformer(真实语义), 维度 {embedder.dim}", flush=True)
        except Exception as e:
            warn_hash_fallback(f"REAL_EMBED=1 但真实语义加载失败: {e}")
            store, embedder, ok, skipped = build_local_store(cards)
            print(f"  embedding: Hash(占位), 维度 {embedder.dim}", flush=True)
    else:
        store, embedder, ok, skipped = build_local_store(cards)
        print(f"  embedding: Hash(占位), 维度 {embedder.dim}", flush=True)

    print(f"  灌入 {ok} 张; 跳过 {len(skipped)}", flush=True)
    for s in skipped[:10]:
        print(f"    [skip] {s}", flush=True)

    all_mechs = store.all_mechanisms()
    from collections import Counter
    dom = Counter(m.origin_domain for m in all_mechs)
    print(f"  库内域分布: {dict(sorted(dom.items()))}", flush=True)

    # === 跨域检索验收 ===
    print("\n" + "=" * 64, flush=True)
    print("跨域检索验收 (强制非 ST 域, 找互补机制集)", flush=True)
    print("=" * 64, flush=True)
    n_empty = 0
    for b in BOTTLENECKS:
        res = find_cross_domain_analogy(b, store, embedder, target_domain="ST",
                                        cross_domain_only=True, max_results=3)
        print(f"\n🔍 瓶颈: {b}", flush=True)
        print(f"   分解前提: {res.target_preconditions}", flush=True)
        print(f"   覆盖前提: {sorted(res.covered_preconditions)}", flush=True)
        print(f"   跨域互补机制集 ({len(res.mechanisms)} 个):", flush=True)
        if not res.mechanisms:
            n_empty += 1
            print("      (空 —— 未检出跨域机制!)", flush=True)
        for r in res.rationale:
            print(f"      • {r}", flush=True)
        # 验证: 选出的机制确实非 ST 域
        bad = [m.name for m in res.mechanisms if m.origin_domain == "ST"]
        if bad:
            print(f"      ⚠️ 含 ST 域机制 (应跨域): {bad}", flush=True)

    print(f"\n=== 验收完成: {len(BOTTLENECKS)} 个瓶颈, {n_empty} 个空结果 ===", flush=True)
    store.close()


if __name__ == "__main__":
    main()
