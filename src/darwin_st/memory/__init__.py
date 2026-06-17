"""记忆层: 结构化试验记忆 (SQLite) + ExpeL 式经验蒸馏。

把过往优化的成功/失败经验结构化记录, 防止无效优化。详见 docs/BLUEPRINT.md §3。
"""

from darwin_st.memory.store import Trial, MemoryStore, compute_signature
from darwin_st.memory.reflect import (
    build_experience_context,
    rank_by_relevance,
    summarize_graveyard,
    propose_insight_candidates,
)

__all__ = [
    "Trial",
    "MemoryStore",
    "compute_signature",
    "build_experience_context",
    "rank_by_relevance",
    "summarize_graveyard",
    "propose_insight_candidates",
]
