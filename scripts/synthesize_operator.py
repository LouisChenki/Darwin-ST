"""Tier-2 算子合成验收: 真实 DeepSeek 把跨域机制融合成新算子。

端到端: ST 瓶颈 → P3-a 跨域检索互补机制集 → FusionRequest →
        DeepSeek plan-then-code 合成 → 验证 harness 全门 → 报告。

用法 (服务器, 需 source darwin-secrets.env 注入 DEEPSEEK_API_KEY):
    source /root/autodl-tmp/darwin-secrets.env
    DARWIN_ST_CACHE=... python scripts/synthesize_operator.py
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
from darwin_st.knowledge.graph_store import Neo4jGraphStore, InMemoryGraphStore
from darwin_st.knowledge.retrieval import find_cross_domain_analogy
from darwin_st.creation import FusionRequest, OperatorSynthesizer, SynthesisConfig
from darwin_st.creation.llm import OpenAICompatLLM
from darwin_st.creation.aider_backend import AiderBackend, AiderConfig


def main():
    bottleneck = os.environ.get(
        "BOTTLENECK",
        "模型对长时间跨度预测误差大,长程时序依赖捕获不好;且标注稀缺想用自监督")

    print("=== Tier-2 算子合成验收 (真实 DeepSeek) ===")
    print(f"瓶颈: {bottleneck}\n")

    # 1) 跨域检索互补机制集 (用 Neo4j; 失败回退内存)
    # 真语义嵌入缺失时显式回退 Hash + 醒目警告 (检索质量退化, 但合成流程仍可跑通)
    if sentence_transformers_available():
        embedder = SentenceTransformerEmbedder()
    else:
        warn_hash_fallback("synthesize_operator 跨域检索")
        embedder = HashEmbedder(dim=256)
    try:
        store = Neo4jGraphStore(embedder)
        if not store.all_mechanisms():
            for m in all_seed_mechanisms():
                store.add_mechanism(m)
        print(f"[检索] Neo4j 机制数: {len(store.all_mechanisms())}")
    except Exception as e:
        print(f"[检索] Neo4j 不可用({e}), 回退内存图")
        store = InMemoryGraphStore(embedder)
        for m in all_seed_mechanisms():
            store.add_mechanism(m)

    res = find_cross_domain_analogy(bottleneck, store, embedder, target_domain="ST", max_results=3)
    print(f"[检索] 前提: {res.target_preconditions}")
    print(f"[检索] 跨域互补机制集:")
    for r in res.rationale:
        print(f"        • {r}")
    if not res.mechanisms:
        print("未检索到机制, 退出"); return

    # 2) 组装融合请求
    req = FusionRequest(
        bottleneck=bottleneck,
        target_preconditions=res.target_preconditions,
        mechanisms=res.mechanisms,
        baseline_operator="dilated_causal_convolution",
    )

    # 3) 真实 DeepSeek 合成 (plan 用 LLM, 代码用 Aider 沙箱)
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
    aider_model = f"deepseek/{model}"
    print(f"\n[合成] 计划用 {model}, 代码用 Aider({aider_model}) 沙箱写入...")
    llm = OpenAICompatLLM()
    backend = AiderBackend(AiderConfig(model=aider_model))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=4, temperature=0.8),
                                code_backend=backend)
    result = synth.synthesize(req, needs_adj=False)

    print(f"\n=== 合成结果 ===")
    print(f"成功: {result.success} | 尝试次数: {result.attempts}")
    if result.plan:
        print(f"融合计划: {result.plan.operator_name} | 组合方式: {result.plan.composition}")
        print(f"  源机制: {result.plan.source_mechanisms}")
        print(f"  理由: {result.plan.rationale[:200]}")
    if result.success:
        print(f"\n✅ 合成出算子 '{result.operator.name}' 并通过全部验证门!")
        print(f"--- 算子代码 ---\n{result.operator.code}")
    else:
        print(f"\n❌ 合成失败: {result.last_error[:400]}")

    store.close()
    print("\n=== 验收完成 ===")


if __name__ == "__main__":
    main()
