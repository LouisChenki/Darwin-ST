"""机制图存储 (Mechanism Graph Store) —— 可插拔后端: 内存 / Neo4j。

存机制节点 + 前提节点 + (Mechanism)-[:HAS_PRECONDITION]->(Precondition) 边,
并持有每个机制的 embedding(MAC 用)。两后端同一接口:
  - InMemoryGraphStore: 纯 Python(networkx 可选, 这里用 dict), 本地/测试, 零服务依赖。
  - Neo4jGraphStore: 服务器真 Neo4j, 持久化 + 规模化。

支持 add_mechanism(增量丰富留口 —— 优化中 Aider 融合出的新算子可回写成新机制卡)。

MAC: vector_search(query_emb) → 按余弦召回候选机制。
FAC: mechanisms_sharing_precondition(p) → 沿前提图拉同前提的机制(跨域桥)。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from darwin_st.knowledge.embedding import Embedder, cosine_sim_matrix
from darwin_st.knowledge.ontology import Mechanism

__all__ = ["GraphStore", "InMemoryGraphStore", "Neo4jGraphStore"]


class GraphStore(Protocol):
    """图存储接口。"""

    def add_mechanism(self, mech: Mechanism) -> None: ...
    def all_mechanisms(self) -> list[Mechanism]: ...
    def get_mechanism(self, name: str) -> Mechanism | None: ...
    def vector_search(self, query_emb: np.ndarray, top_k: int, exclude_domain: str | None = None) -> list[tuple[Mechanism, float]]: ...
    def mechanisms_sharing_precondition(self, precondition: str, exclude_domain: str | None = None) -> list[Mechanism]: ...
    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# 内存后端 (本地/测试)
# ---------------------------------------------------------------------------


class InMemoryGraphStore:
    """纯 Python 内存图存储。机制卡 + embedding + 前提倒排索引。"""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._mechs: dict[str, Mechanism] = {}
        self._emb: dict[str, np.ndarray] = {}          # name -> 向量
        self._by_precond: dict[str, set[str]] = {}     # precondition -> {mech name}

    def add_mechanism(self, mech: Mechanism) -> None:
        mech.validate()
        self._mechs[mech.name] = mech
        self._emb[mech.name] = self.embedder.embed([mech.embedding_text()])[0]
        for p in mech.preconditions:
            self._by_precond.setdefault(p, set()).add(mech.name)

    def all_mechanisms(self) -> list[Mechanism]:
        return list(self._mechs.values())

    def get_mechanism(self, name: str) -> Mechanism | None:
        return self._mechs.get(name)

    def vector_search(self, query_emb: np.ndarray, top_k: int, exclude_domain: str | None = None) -> list[tuple[Mechanism, float]]:
        names = [n for n in self._mechs
                 if exclude_domain is None or self._mechs[n].origin_domain != exclude_domain]
        if not names:
            return []
        mat = np.vstack([self._emb[n] for n in names])
        sims = cosine_sim_matrix(query_emb[None, :], mat)[0]   # [len(names)]
        order = np.argsort(-sims)[:top_k]
        return [(self._mechs[names[i]], float(sims[i])) for i in order]

    def mechanisms_sharing_precondition(self, precondition: str, exclude_domain: str | None = None) -> list[Mechanism]:
        out = []
        for name in self._by_precond.get(precondition, set()):
            m = self._mechs[name]
            if exclude_domain is None or m.origin_domain != exclude_domain:
                out.append(m)
        return out

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Neo4j 后端 (服务器)
# ---------------------------------------------------------------------------


class Neo4jGraphStore:
    """Neo4j 后端。机制/前提节点 + HAS_PRECONDITION 边; embedding 存为节点属性。

    MAC 检索: 取全部机制 embedding 在 Python 端算余弦(种子规模小, 够用;
    规模化后可换 Neo4j 原生向量索引)。
    """

    def __init__(self, embedder: Embedder, uri: str = "bolt://localhost:7687",
                 user: str = "neo4j", password: str = "darwin_st_2024"):
        from neo4j import GraphDatabase

        self.embedder = embedder
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._ensure_constraints()

    def _ensure_constraints(self) -> None:
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT mech_name IF NOT EXISTS FOR (m:Mechanism) REQUIRE m.name IS UNIQUE")
            s.run("CREATE CONSTRAINT precond_name IF NOT EXISTS FOR (p:Precondition) REQUIRE p.name IS UNIQUE")

    def add_mechanism(self, mech: Mechanism) -> None:
        mech.validate()
        emb = self.embedder.embed([mech.embedding_text()])[0].tolist()
        with self.driver.session() as s:
            s.run(
                """
                MERGE (m:Mechanism {name: $name})
                SET m.abstract_function=$af, m.causal_behavior=$cb, m.math_structure=$ms,
                    m.origin_domain=$dom, m.abstraction_level=$lvl, m.consequences=$cons,
                    m.provenance=$prov, m.embedding=$emb, m.preconditions=$preconds,
                    m.function_tags=$ftags
                """,
                name=mech.name, af=mech.abstract_function, cb=mech.causal_behavior,
                ms=mech.math_structure, dom=mech.origin_domain, lvl=mech.abstraction_level,
                cons=mech.consequences, prov=mech.provenance, emb=emb,
                preconds=mech.preconditions, ftags=mech.function_tags,
            )
            for p in mech.preconditions:
                s.run(
                    """
                    MERGE (p:Precondition {name: $p})
                    WITH p
                    MATCH (m:Mechanism {name: $name})
                    MERGE (m)-[:HAS_PRECONDITION]->(p)
                    """,
                    p=p, name=mech.name,
                )

    def _row_to_mech(self, rec) -> Mechanism:
        return Mechanism(
            name=rec["name"], abstract_function=rec.get("abstract_function", ""),
            preconditions=rec.get("preconditions", []) or [],
            causal_behavior=rec.get("causal_behavior", ""),
            function_tags=rec.get("function_tags", []) or [],
            math_structure=rec.get("math_structure", ""),
            consequences=rec.get("consequences", ""),
            origin_domain=rec.get("origin_domain", "ST"),
            abstraction_level=rec.get("abstraction_level", "concept"),
            provenance=rec.get("provenance", ""),
        )

    def all_mechanisms(self) -> list[Mechanism]:
        with self.driver.session() as s:
            rows = s.run("MATCH (m:Mechanism) RETURN m").data()
        return [self._row_to_mech(r["m"]) for r in rows]

    def get_mechanism(self, name: str) -> Mechanism | None:
        with self.driver.session() as s:
            rec = s.run("MATCH (m:Mechanism {name:$n}) RETURN m", n=name).single()
        return self._row_to_mech(rec["m"]) if rec else None

    def vector_search(self, query_emb: np.ndarray, top_k: int, exclude_domain: str | None = None) -> list[tuple[Mechanism, float]]:
        clause = "WHERE m.origin_domain <> $excl" if exclude_domain else ""
        with self.driver.session() as s:
            rows = s.run(f"MATCH (m:Mechanism) {clause} RETURN m", excl=exclude_domain).data()
        if not rows:
            return []
        mechs = [self._row_to_mech(r["m"]) for r in rows]
        mat = np.vstack([np.array(r["m"]["embedding"], dtype=np.float32) for r in rows])
        sims = cosine_sim_matrix(query_emb[None, :], mat)[0]
        order = np.argsort(-sims)[:top_k]
        return [(mechs[i], float(sims[i])) for i in order]

    def mechanisms_sharing_precondition(self, precondition: str, exclude_domain: str | None = None) -> list[Mechanism]:
        clause = "AND m.origin_domain <> $excl" if exclude_domain else ""
        with self.driver.session() as s:
            rows = s.run(
                f"""
                MATCH (m:Mechanism)-[:HAS_PRECONDITION]->(p:Precondition {{name:$p}})
                WHERE 1=1 {clause}
                RETURN m
                """,
                p=precondition, excl=exclude_domain,
            ).data()
        return [self._row_to_mech(r["m"]) for r in rows]

    def close(self) -> None:
        self.driver.close()
