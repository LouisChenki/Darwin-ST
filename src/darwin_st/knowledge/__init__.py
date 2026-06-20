"""知识层: 机制中心本体 + 跨域类比检索 (GraphRAG)。

P3 知识系统。目标 = 跨域机制库(不只 ST 技巧), 让 LLM 做跨域类比迁移。
详见 docs/TIER2_RESEARCH_FINDINGS.md §3-4。
"""

from darwin_st.knowledge.ontology import (
    Mechanism,
    Precondition,
    Evidence,
    DOMAINS,
    PRECONDITION_VOCAB,
    FUNCTION_VOCAB,
)
from darwin_st.knowledge.seeds import all_seed_mechanisms, SEED_MECHANISMS
from darwin_st.knowledge.embedding import HashEmbedder, SentenceTransformerEmbedder
from darwin_st.knowledge.graph_store import InMemoryGraphStore, Neo4jGraphStore
from darwin_st.knowledge.retrieval import (
    RetrievalResult,
    cross_domain_analogy,
    find_cross_domain_analogy,
    decompose_to_preconditions,
)
from darwin_st.knowledge.extraction import (
    parse_mechanism_cards,
    normalize_preconditions,
    normalize_domain,
    dedup_mechanisms,
    coverage_report,
    CoverageReport,
)

__all__ = [
    "Mechanism",
    "Precondition",
    "Evidence",
    "DOMAINS",
    "PRECONDITION_VOCAB",
    "FUNCTION_VOCAB",
    "all_seed_mechanisms",
    "SEED_MECHANISMS",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "InMemoryGraphStore",
    "Neo4jGraphStore",
    "RetrievalResult",
    "cross_domain_analogy",
    "find_cross_domain_analogy",
    "decompose_to_preconditions",
    "parse_mechanism_cards",
    "normalize_preconditions",
    "normalize_domain",
    "dedup_mechanisms",
    "coverage_report",
    "CoverageReport",
]
