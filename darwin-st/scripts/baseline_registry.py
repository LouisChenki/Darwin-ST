"""基线指标库 —— 薄壳 (Thin Re-export Shell)。

⚠️ 单一数据源在 `src/darwin_st/baseline_registry.py`。本文件只是 Agent Skill 发布层
(`darwin-st/scripts/`) 的转发壳, 让 SKILL.md 里 `python3 scripts/baseline_registry.py`
之类的调用仍可用, 同时**避免数字双份漂移**。

如需修改基线数字, 请改核心包那一份, 不要改这里。
"""

from __future__ import annotations

import os
import sys

# 优先用已安装的 darwin_st 包; 未安装则把仓库 src/ 加入路径后再导入。
try:
    from darwin_st.baseline_registry import (  # type: ignore
        BASELINE_METRICS,
        PROTOCOL_NOTES,
        get_baseline_metric,
        get_sota,
        list_baselines,
    )
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _src = os.path.join(_repo_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from darwin_st.baseline_registry import (  # type: ignore
        BASELINE_METRICS,
        PROTOCOL_NOTES,
        get_baseline_metric,
        get_sota,
        list_baselines,
    )

__all__ = [
    "BASELINE_METRICS",
    "PROTOCOL_NOTES",
    "get_baseline_metric",
    "get_sota",
    "list_baselines",
]


if __name__ == "__main__":
    # 便于 agent 直接 `python3 scripts/baseline_registry.py [数据集]` 速查锚点
    ds = sys.argv[1] if len(sys.argv) > 1 else "PeMS04"
    print(f"=== {ds} 基线锚点 (MAE, 越小越强) ===")
    print(f"协议口径: {PROTOCOL_NOTES.get(ds, 'N/A')}")
    ranked = list_baselines(ds, "mae")
    if not ranked:
        print(f"未知数据集 '{ds}'. 可选: {list(BASELINE_METRICS)}")
    else:
        for name, mae in ranked.items():
            print(f"  {name:14s} MAE={mae}")
        sota_name, sota_val = get_sota(ds)
        print(f"\n🎯 当前 SOTA: {sota_name} (MAE={sota_val}) —— 超越此值即达成终极目标")
