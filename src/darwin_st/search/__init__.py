"""搜索层: 架构基因型、算子库、genotype→nn.Module 编译器。

P2 AutoML 的核心: 把架构表示为可变异的离散 genotype, 解码为真实 PyTorch 模型,
供进化式 NAS 搜索。详见 docs/BLUEPRINT.md §4。
"""

from darwin_st.search.operators import (
    SPATIAL_OPS,
    TEMPORAL_OPS,
    build_spatial_op,
    build_temporal_op,
)
from darwin_st.search.embeddings import (
    SpatialNodeEmbedding,
    TimeOfDayEmbedding,
    DayOfWeekEmbedding,
    STEmbedding,
)
from darwin_st.search.genotype import (
    STBlock,
    EmbeddingConfig,
    Genotype,
    mutate,
    random_genotype,
)
from darwin_st.search.builder import STModel, build_model, count_params

__all__ = [
    "SPATIAL_OPS",
    "TEMPORAL_OPS",
    "build_spatial_op",
    "build_temporal_op",
    "SpatialNodeEmbedding",
    "TimeOfDayEmbedding",
    "DayOfWeekEmbedding",
    "STEmbedding",
    "STBlock",
    "EmbeddingConfig",
    "Genotype",
    "mutate",
    "random_genotype",
    "STModel",
    "build_model",
    "count_params",
]
