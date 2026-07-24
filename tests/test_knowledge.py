"""knowledge 包的正确性回归测试 (本地 InMemory+Hash 后端, 无服务器/无模型下载)。

验证机制本体 + 嵌入 + 图存储 + 正式跨域检索的契约:
  - 种子卡全部合法
  - 受控词表约束生效
  - embedding_text 只含抽象功能+前提(不含表面/域)
  - InMemoryGraphStore: 增删/向量搜/前提图遍历
  - find_cross_domain_analogy: 跨域过滤 + 互补集覆盖 + FAC 前提桥
  - Neo4j 后端 evidence/anti_patterns/inferred 字段保真 (集成测试默认跳过:
    需 DARWIN_ST_NEO4J_TEST=1 且 bolt://localhost:7687 可达)
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from darwin_st.knowledge import (
    HashEmbedder,
    InMemoryGraphStore,
    Mechanism,
    all_seed_mechanisms,
    find_cross_domain_analogy,
)
from darwin_st.knowledge.graph_store import Neo4jGraphStore, _dicts_to_evidence, _evidence_to_dicts
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


def test_override_preconditions_bypasses_keyword_match(store):
    """override_preconditions 直接驱动 FAC, 绕过 bottleneck 字符串的关键词匹配。

    瓶颈字符串故意'无关键词'(不含长程/冗余等线索), 但 override 指定 label_scarcity+
    redundant_structure → 仍应拉回 masked_autoencoding (LLM 直出前提词绕过脆弱 decompose)。
    """
    res = find_cross_domain_analogy(
        "模型表现不够好需要改进", store, store.embedder,   # 无线索词, decompose 会落空
        override_preconditions=["label_scarcity", "redundant_structure"])
    assert "label_scarcity" in res.target_preconditions
    assert "redundant_structure" in res.target_preconditions
    names = {m.name for m in res.mechanisms}
    assert "masked_autoencoding" in names


def test_override_preconditions_filters_illegal(store):
    """override 含非法词 → 过滤; 全非法 → 退回关键词分解 (不崩)。"""
    # 一个合法 + 一个非法 → 只保留合法的
    res = find_cross_domain_analogy(
        "随便什么", store, store.embedder,
        override_preconditions=["long_range_dependency", "not_a_vocab_word"])
    assert res.target_preconditions == ["long_range_dependency"]
    # 全非法 → 退回对 bottleneck 的关键词分解 (不报错)
    res2 = find_cross_domain_analogy(
        "长程时序依赖捕获不好", store, store.embedder,
        override_preconditions=["bogus1", "bogus2"])
    assert "long_range_dependency" in res2.target_preconditions


# ---------------------------------------------------------------------------
# Neo4j 后端: evidence/anti_patterns/inferred 字段保真
# ---------------------------------------------------------------------------


def _rich_mechanism(name: str = "rich_mech") -> Mechanism:
    """三字段全填的机制卡: evidence 每条字段填全, anti_patterns 多条, inferred=False(已复核)。"""
    return Mechanism(
        name=name, abstract_function="测试用抽象功能",
        preconditions=["long_range_dependency"], causal_behavior="因果链",
        math_structure="数学结构", origin_domain="TimeSeries",
        evidence=[Evidence("PeMS04", "MAE", "去掉后 MAE 18.29→21.65 (+18%)", "high", "STID arXiv:2208.05233"),
                  Evidence("METR-LA", "MAE", "提升 3%")],
        anti_patterns=["掩码率不当→预训练信号过弱", "大图 N² 开销"],
        inferred=False,
    )


def test_evidence_dicts_roundtrip():
    """Evidence↔dict 序列化辅助函数: 转换后字段全保真 (不依赖 Neo4j 服务)。"""
    m = _rich_mechanism()
    dicts = _evidence_to_dicts(m.evidence)
    assert all(set(d) == {"dataset", "metric", "delta", "grade", "source"} for d in dicts)
    assert _dicts_to_evidence(dicts) == m.evidence
    # 容错: None/空/非 dict 项不崩
    assert _dicts_to_evidence(None) == []
    assert _dicts_to_evidence([{"dataset": "D"}, "junk"]) == [Evidence("D", "", "")]


def test_neo4j_row_to_mech_restores_nested_fields():
    """_row_to_mech: evidence(JSON 字符串)/anti_patterns/inferred 读回保真 (不依赖 Neo4j 服务)。"""
    m = _rich_mechanism()
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)   # 跳过 __init__, 不连服务只测纯转换
    rec = {
        "name": m.name, "abstract_function": m.abstract_function,
        "preconditions": m.preconditions, "causal_behavior": m.causal_behavior,
        "math_structure": m.math_structure, "origin_domain": m.origin_domain,
        # 与 add_mechanism 写入相同的形态: evidence 为 JSON 字符串
        "evidence": json.dumps(_evidence_to_dicts(m.evidence), ensure_ascii=False),
        "anti_patterns": m.anti_patterns, "inferred": m.inferred,
    }
    got = store._row_to_mech(rec)
    assert got.evidence == m.evidence
    assert got.anti_patterns == m.anti_patterns
    assert got.inferred is False


def test_neo4j_row_to_mech_backward_compat():
    """向后兼容: 旧库节点缺三属性 → 默认 evidence=[]/anti_patterns=[]/inferred=True(待核)。"""
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)
    got = store._row_to_mech({"name": "old_mech", "abstract_function": "x", "preconditions": []})
    assert got.evidence == []
    assert got.anti_patterns == []
    assert got.inferred is True   # 旧卡视为待人工复核, 与 ontology 默认语义一致


class _FakeDriver:
    """假 neo4j driver: 捕获 auth, session 空转 (不依赖真实 Neo4j 服务)。"""

    captured: dict = {}

    def __init__(self, uri, auth=None, **kw):
        _FakeDriver.captured = {"uri": uri, "auth": auth}

    def session(self):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            class _S:
                def run(self, *a, **k):
                    return None
            yield _S()
        return _ctx()

    def close(self):
        pass


def _patch_neo4j_module(monkeypatch):
    """把假 neo4j 模块注进 sys.modules (Neo4jGraphStore.__init__ 内 from neo4j import ...)。"""
    import sys
    import types
    fake = types.ModuleType("neo4j")
    fake.GraphDatabase = types.SimpleNamespace(driver=_FakeDriver)
    monkeypatch.setitem(sys.modules, "neo4j", fake)


def test_neo4j_password_from_env(monkeypatch):
    """密码默认读 NEO4J_PASSWORD (构造期一次): env 命中用 env; 缺省回旧默认值兼容。"""
    _patch_neo4j_module(monkeypatch)
    monkeypatch.setenv("NEO4J_PASSWORD", "env_secret")
    Neo4jGraphStore(HashEmbedder(dim=8))
    assert _FakeDriver.captured["auth"] == ("neo4j", "env_secret")
    # env 缺失 → 旧默认值 (向后兼容)
    monkeypatch.delenv("NEO4J_PASSWORD")
    Neo4jGraphStore(HashEmbedder(dim=8))
    assert _FakeDriver.captured["auth"] == ("neo4j", "darwin_st_2024")


def test_neo4j_password_explicit_overrides_env(monkeypatch):
    """显式 password 参数优先于 env (调用方明确指定时不被环境变量抢)。"""
    _patch_neo4j_module(monkeypatch)
    monkeypatch.setenv("NEO4J_PASSWORD", "env_secret")
    Neo4jGraphStore(HashEmbedder(dim=8), password="explicit_pw")
    assert _FakeDriver.captured["auth"] == ("neo4j", "explicit_pw")


def _neo4j_reachable() -> bool:
    """集成测试判据: 显式设 DARWIN_ST_NEO4J_TEST 且 bolt://localhost:7687 短超时可达。"""
    if not os.environ.get("DARWIN_ST_NEO4J_TEST"):
        return False
    try:
        from neo4j import GraphDatabase
        d = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "darwin_st_2024"),
                                 connection_timeout=2)
        d.verify_connectivity()
        d.close()
        return True
    except Exception:
        return False


_NEO4J_OK = _neo4j_reachable()


@pytest.mark.skipif(not _NEO4J_OK, reason="无 Neo4j 服务 (需 DARWIN_ST_NEO4J_TEST=1 且本地 7687 可达)")
def test_neo4j_add_get_roundtrip():
    """集成: 真实 Neo4j add→get 往返, evidence/anti_patterns/inferred 全保真。"""
    store = Neo4jGraphStore(HashEmbedder(dim=64))
    m = _rich_mechanism(name="test_neo4j_roundtrip_tmp")
    try:
        store.add_mechanism(m)
        got = store.get_mechanism(m.name)
        assert got is not None
        assert got.evidence == m.evidence
        assert got.anti_patterns == m.anti_patterns
        assert got.inferred is False
    finally:
        with store.driver.session() as s:   # 清理测试节点, 不污染库
            s.run("MATCH (m:Mechanism {name:$n}) DETACH DELETE m", n=m.name)
        store.close()
