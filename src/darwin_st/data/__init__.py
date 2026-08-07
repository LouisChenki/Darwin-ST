"""数据层: 标准协议数据管道、masked 评测指标、邻接矩阵构建。

本层是 P0「可信评测地基」的核心——所有逻辑均为**设备无关**(CPU/GPU 结果一致),
可在本地用 pytest 验证正确性。详见 docs/BLUEPRINT.md §2。
"""

from darwin_st.data.metrics import (
    masked_mae,
    masked_huber,
    masked_rmse,
    masked_mape,
    compute_all_metrics,
)
from darwin_st.data.protocol import (
    DatasetProfile,
    PROFILES,
    get_profile,
    chronological_split,
    ZScoreScaler,
)
from darwin_st.data.adjacency import (
    gaussian_kernel,
    build_adj_from_distance_csv,
    load_adj_pkl,
    load_adj_csv,
    symmetric_normalize,
    random_walk_normalize,
    build_adjacency,
)
from darwin_st.data.prepare import (
    generate_windows,
    prepare_dataset,
    load_split,
    load_scaler,
    load_adj,
    evaluate,
)

__all__ = [
    "masked_mae",
    "masked_huber",
    "masked_rmse",
    "masked_mape",
    "compute_all_metrics",
    "DatasetProfile",
    "PROFILES",
    "get_profile",
    "chronological_split",
    "ZScoreScaler",
    "gaussian_kernel",
    "build_adj_from_distance_csv",
    "load_adj_pkl",
    "load_adj_csv",
    "symmetric_normalize",
    "random_walk_normalize",
    "build_adjacency",
    "generate_windows",
    "prepare_dataset",
    "load_split",
    "load_scaler",
    "load_adj",
    "evaluate",
]
