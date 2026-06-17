"""Darwin-ST: 时空预测模型自动化研究 Agent Skill 的核心 Python 包。"""

from darwin_st.baseline_registry import (
    BASELINE_METRICS,
    get_baseline_metric,
    get_sota,
    list_baselines,
)

__version__ = "0.1.0"

__all__ = [
    "BASELINE_METRICS",
    "get_baseline_metric",
    "get_sota",
    "list_baselines",
]
