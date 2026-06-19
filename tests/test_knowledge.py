"""knowledge 包的正确性回归测试 (本地 InMemory+Hash 后端, 无服务器/无模型下载)。

验证机制本体 + 嵌入 + 图存储 + 正式跨域检索的契约:
  - 种子卡全部合法
  - 受控词表约束生效
  - embedding_text 只含抽象功能+前提(不含表面/域)
  - InMemoryGraphStore: 增删/向量搜/前提图遍历
  - find_cross_domain_analogy: 跨域过滤 + 互补集覆盖 + FAC 前提桥
"""

from __future__ import annotations

import numpy as np
import pytest

from darwin_st.knowledge import (
    HashEmbedder,
    InMemoryGraphStore,
    Mechanism,
    all_seed_mechanisms,
    find_cross_domain_analogy,
)
from darwin_st.knowledge.ontology import Evidence
from darwin_st.knowledge.retrieval import decompose_to_preconditions


# ---------------------------------------------------------------------------
# 本体 + 种子卡
# ---------------------------------------------------------------------------


def test_all_seed_mechanisms_valid():
    ms = all_seed_mechanisms()
    assert len(ms) >= 15
    for m in ms:
        m.validate()  # 不抛即合法


def test_seed_covers_multiple_domains():
    ms = all_seed_mechanisms()
    domains = {m.origin_domain for m in ms}
    assert len(domains) >= 6  # 跨域素材


def test_invalid_precondition_rejected():
    with pytest.raises(ValueError):
        Mechanism(name="bad", abstract_function="x", preconditions=["not_a_real_precond"],
                  causal_behavior="y").validate()


def test_invalid_domain_rejected():
    with pytest.raises(ValueError):
        Mechanism(name="bad", abstract_function="x", preconditions=[],
                  causal_behavior="y", origin_domain="Mars").validate()


def test_embedding_text_excludes_surface():
    """embedding_text 只含抽象功能+前提, 不含 math_structure/origin_domain。"""
    m = Mechanism(name="t", abstract_function="抽象功能XYZ", preconditions=["long_range_dependency"],
                  causal_behavior="因果", math_structure="表面数学ABC", origin_domain="CV")
    text = m.embedding_text()
    assert "抽象功能XYZ" in text
    assert "long_range_dependency" in text
    assert "表面数学ABC" not in text  # 表面描述不进 embedding
    assert "CV" not in text            # 域不进 embedding


# ---------------------------------------------------------------------------
# 嵌入 + 图存储
# ---------------------------------------------------------------------------


def test_hash_embedder_normalized():
    emb = HashEmbedder(dim=64)
    v = emb.embed(["长程依赖 long range"])
    assert v.shape == (1, 64)
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-5  # L2 归一化


def test_store_add_and_get():
    store = InMemoryGraphStore(HashEmbedder(dim=64))
    for m in all_seed_mechanisms():
        store.add_mechanism(m)
    assert len(store.all_mechanisms()) == len(all_seed_mechanisms())
    assert store.get_mechanism("masked_autoencoding") is not None


def test_store_vector_search_excludes_domain():
    store = InMemoryGraphStore(HashEmbedder(dim=128))
    for m in all_seed_mechanisms():
        store.add_mechanism(m)
    q = store.embedder.embed(["长程时序依赖"])[0]
    hits = store.vector_search(q, top_k=5, exclude_domain="ST")
    assert all(m.origin_domain != "ST" for m, _ in hits)  # 跨域过滤生效


def test_store_precondition_graph():
    """FAC 前提图: 沿 long_range_dependency 应拉到多个跨域机制。"""
    store = InMemoryGraphStore(HashEmbedder(dim=64))
    for m in all_seed_mechanisms():
        store.add_mechanism(m)
    shared = store.mechanisms_sharing_precondition("long_range_dependency", exclude_domain="ST")
    names = {m.name for m in shared}
    # 状态空间模型 / 自注意力 都标了 long_range_dependency
    assert "state_space_model" in names
    assert len(names) >= 2


def test_incremental_add_mechanism():
    """增量丰富: 新机制卡可随时加入 (优化中 Aider 融合产物回写的接口)。"""
    store = InMemoryGraphStore(HashEmbedder(dim=64))
    store.add_mechanism(all_seed_mechanisms()[0])
    n0 = len(store.all_mechanisms())
    new = Mechanism(name="fused_novel_op", abstract_function="融合产生的新机制",
                    preconditions=["long_range_dependency"], causal_behavior="...",
                    origin_domain="ST")
    store.add_mechanism(new)
    assert len(store.all_mechanisms()) == n0 + 1
    assert store.get_mechanism("fused_novel_op") is not None


# ---------------------------------------------------------------------------
# 瓶颈分解
# ---------------------------------------------------------------------------


def test_decompose_long_range():
    pre = decompose_to_preconditions("模型对长时间跨度的预测误差大,长程依赖捕获不好")
    assert "long_range_dependency" in pre


def test_decompose_multi_precondition():
    pre = decompose_to_preconditions("数据冗余且标签稀缺,有周期性")
    assert "redundant_structure" in pre
    assert "label_scarcity" in pre
    assert "temporal_periodicity" in pre


# ---------------------------------------------------------------------------
# 正式跨域检索 (核心)
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    s = InMemoryGraphStore(HashEmbedder(dim=256))
    for m in all_seed_mechanisms():
        s.add_mechanism(m)
    return s


def test_find_cross_domain_returns_non_st(store):
    """跨域检索: 强制返回非 ST 域机制。"""
    res = find_cross_domain_analogy("长程时序依赖捕获不好", store, store.embedder,
                                    target_domain="ST", cross_domain_only=True)
    assert len(res.mechanisms) >= 1
    assert all(m.origin_domain != "ST" for m in res.mechanisms)


def test_find_cross_domain_pulls_ssm_for_long_range(store):
    """长程瓶颈应跨域拉回管长程的机制 (SSM/注意力/膨胀卷积皆合理)。

    注: state_space_model 与 dilated_causal_convolution 都标了 long_range_dependency+
    sequential_order, 覆盖度相同时由语义相似度打破平局。两者都是正确的跨域答案。
    """
    res = find_cross_domain_analogy("长程时序依赖捕获不好,序列很长", store, store.embedder)
    names = {m.name for m in res.mechanisms}
    # 应拉回至少一个"管长程依赖"的跨域机制
    long_range_mechs = {"state_space_model", "dilated_causal_convolution", "self_attention"}
    assert names & long_range_mechs, f"未拉回任何长程机制, 实得 {names}"
    # 且确实覆盖了 long_range_dependency 前提
    assert "long_range_dependency" in res.covered_preconditions


def test_find_cross_domain_complementary_set(store):
    """多前提瓶颈应返回【互补集】覆盖多个前提 (STD-MAE 场景)。"""
    res = find_cross_domain_analogy(
        "交通数据冗余且周期性强,但标签稀缺,想自监督预训练", store, store.embedder, max_results=4)
    # 应覆盖 label_scarcity + redundant_structure (掩码自编码) 等多个前提
    assert "label_scarcity" in res.covered_preconditions
    assert "redundant_structure" in res.covered_preconditions
    names = {m.name for m in res.mechanisms}
    assert "masked_autoencoding" in names  # 正是 STD-MAE 跨域来源


def test_find_cross_domain_fac_bridge_beats_weak_semantics(store):
    """FAC 关键性质: 即使语义相似度弱, 前提共享的机制也能被拉到 (结构桥 > 表面相似)。"""
    # 用一个语义上不含机制术语、但前提明确的瓶颈
    res = find_cross_domain_analogy("节点身份难以区分", store, store.embedder)
    # node_indistinguishability 前提应桥接到 adaptive_adjacency 等
    names = {m.name for m in res.mechanisms}
    assert any(n in names for n in ("adaptive_adjacency",))


def test_find_cross_domain_allow_same_domain(store):
    """关闭跨域过滤时, 同域机制也可返回。"""
    res = find_cross_domain_analogy("节点身份难以区分", store, store.embedder,
                                    cross_domain_only=False)
    assert len(res.mechanisms) >= 1
