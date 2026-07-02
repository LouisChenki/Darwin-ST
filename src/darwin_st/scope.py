"""实验作用域 (Experiment Scope) —— 隔离键: (数据集, 搜索空间代际)。

为什么存在: 系统有三份持久状态 (memory DB / synth 算子注册表 / 评测 worker),
它们必须对「当前实验属于哪个数据集的哪代搜索空间」达成一致, 否则跨实验记忆会污染:
  - 换数据集 (PeMS04→METR-LA): PeMS04 的 synth 算子/最优架构不该泄漏到 METR-LA。
  - 换搜索空间代 (Stage1→2→3): 旧代 synth 算子会在变异池里数量碾压新原子算子
    (Stage2 稀释 bug); 旧代最优会把进化暖启动进旧局部最优。

设计 (见 plan serene-churning-goblet.md):
  - space_version 默认自动来自 operators.builtin_op_signature() —— 内置算子集/融合集一变
    哈希就变 → 新代自动落独立作用域, 零人工, 结构性杜绝污染。
  - SPACE_VERSION 环境变量可覆盖成可读名 (如 "stage2-dag"), 便于人读日志/目录。
  - 只负责 space_version 派生 + synth 目录路径; 不管 DB 路径 (仍走 MEMORY_DB) / config /
    知识图谱 (那是全局领域知识, 跨代复用)。依赖方向 scope→operators, operators 绝不反向 import。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["ExperimentScope"]


@dataclass(frozen=True)
class ExperimentScope:
    """一次实验的隔离作用域。不可变 (frozen) → 可安全跨模块传递/pickle 到 worker。"""

    dataset: str          # 目标数据集 (PeMS04 / PeMS08 / METR-LA / PEMS-BAY)
    space_version: str    # 搜索空间代际标识 (内置算子集哈希, 或 SPACE_VERSION 覆盖)

    def synth_dir(self, cache_root: str) -> str:
        """本作用域的 synth 算子持久化目录: dynamic_ops/{dataset}/{space_version}/。

        主进程 (registry) 与评测 worker (EvalSpec.synth_persist_dir) 必须都用此路径,
        单一真源 → 主/worker 一致 (旧代码三处各拼 dynamic_ops 正是 bug 结构性成因)。
        """
        return os.path.join(cache_root, "dynamic_ops", self.dataset, self.space_version)

    @classmethod
    def resolve(cls, dataset: str, space_version: str | None = None) -> "ExperimentScope":
        """构造作用域。space_version 优先级: 显式传参 > SPACE_VERSION 环境变量 > 自动哈希。"""
        if space_version is None:
            # 函数内 import 避免顶层 scope→operators→... 潜在环 (operators 不 import scope)
            from darwin_st.search.operators import builtin_op_signature
            space_version = os.environ.get("SPACE_VERSION") or builtin_op_signature()
        return cls(dataset=dataset, space_version=space_version)
