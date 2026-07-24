"""P3 验收脚本: 把种子机制卡灌进 Neo4j(真 embedding), 跑跨域检索验收。

用法 (服务器):
    DARWIN_ST_CACHE=... GITHUB_MIRROR=... python scripts/build_knowledge_graph.py

需 neo4j 运行中(bolt://localhost:7687) + sentence-transformers 已装。
验收门: 给定 ST 瓶颈, 能否跨域检索出正确的互补机制集(如 STD-MAE 场景拉回掩码自编码)。
"""

from __future__ import annotations

import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from darwin_st.knowledge import all_seed_mechanisms
from darwin_st.knowledge.embedding import (HashEmbedder, SentenceTransformerEmbedder,
                                           sentence_transformers_available, warn_hash_fallback)
from darwin_st.knowledge.graph_store import Neo4jGraphStore
from darwin_st.knowledge.retrieval import find_cross_domain_analogy


def main():
    use_real = os.environ.get("REAL_EMBED", "1") == "1"
    if use_real and not sentence_transformers_available():
        # 默认要真语义但包未装: 醒目警告后回退 Hash (不静默, 否则 embedder.dim 处裸抛 ModuleNotFoundError)
        warn_hash_fallback("REAL_EMBED=1 但 sentence-transformers 未安装")
        use_real = False
    print(f"=== Darwin-ST 知识图谱构建 + 跨域检索验收 ===")
    print(f"embedding: {'SentenceTransformer(真实语义)' if use_real else 'Hash(占位)'}")

    embedder = SentenceTransformerEmbedder() if use_real else HashEmbedder(dim=256)
    # 触发模型加载 + 报维度
    print(f"embedding 维度: {embedder.dim}")

    store = Neo4jGraphStore(embedder)

    # 清空旧机制(幂等重建)
    with store.driver.session() as s:
        s.run("MATCH (n) WHERE n:Mechanism OR n:Precondition DETACH DELETE n")

    # 灌种子卡
    seeds = all_seed_mechanisms()
    for m in seeds:
        store.add_mechanism(m)
    print(f"已灌入 {len(seeds)} 张机制卡到 Neo4j")
    print(f"Neo4j 中机制数: {len(store.all_mechanisms())}")

    # === 验收: 几个真实 ST 瓶颈的跨域检索 ===
    bottlenecks = [
        "模型对长时间跨度的预测误差大,长程时序依赖捕获不好",
        "交通流量数据有很强的冗余和周期性,但标注稀缺,想用自监督预训练",
        "不同路口的行为差异大异质性强,且节点身份难以区分",
        "深层图卷积导致节点表示过平滑,空间信息丢失",
    ]

    print("\n" + "=" * 64)
    print("跨域检索验收 (强制非 ST 域, 真实语义 embedding)")
    print("=" * 64)
    for b in bottlenecks:
        res = find_cross_domain_analogy(b, store, embedder, target_domain="ST",
                                        cross_domain_only=True, max_results=3)
        print(f"\n🔍 瓶颈: {b}")
        print(f"   前提: {res.target_preconditions}")
        print(f"   跨域互补机制集:")
        for r in res.rationale:
            print(f"      • {r}")

    store.close()
    print("\n=== 验收完成 ===")


if __name__ == "__main__":
    main()
