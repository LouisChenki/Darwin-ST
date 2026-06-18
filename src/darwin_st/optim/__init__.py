"""优化层: 内层 HPO (Optuna TPE+ASHA) + 并行调度 + 外层进化编排。

P2 AutoML 的执行引擎。详见 docs/P2_ALGORITHM_DESIGN.md §4。
"""

from darwin_st.optim.hpo import (
    HPOConfig,
    HPOResult,
    suggest_hps,
    make_study,
    optimize_architecture,
)
from darwin_st.optim.scheduler import EvalResult, GPUScheduler, resolve_devices
from darwin_st.optim.archive import (
    Elite,
    MAPElitesArchive,
    behavior_descriptor,
    cell_index,
)
from darwin_st.optim.orchestrator import OrchestratorConfig, RunState, Orchestrator

__all__ = [
    "HPOConfig",
    "HPOResult",
    "suggest_hps",
    "make_study",
    "optimize_architecture",
    "EvalResult",
    "GPUScheduler",
    "resolve_devices",
    "Elite",
    "MAPElitesArchive",
    "behavior_descriptor",
    "cell_index",
    "OrchestratorConfig",
    "RunState",
    "Orchestrator",
]
