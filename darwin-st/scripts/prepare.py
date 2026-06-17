"""时空预测数据准备 —— 薄壳 (Thin Re-export Shell)。

⚠️ 单一实现在 `src/darwin_st/data/prepare.py`。本文件只是 Agent Skill 发布层
(`darwin-st/scripts/`) 的转发壳, 保留 SKILL.md 里 `DATASET=PeMS04 python3 scripts/prepare.py`
的用法, 同时避免与核心包实现重复漂移。

旧版本 (无 masked 指标 / 无 test 集 / 无邻接矩阵 / 占位符 URL) 已被核心包重写取代,
详见 docs/ANALYSIS.md 与 docs/BLUEPRINT.md §2。如需修改, 请改核心包。
"""

from __future__ import annotations

import os
import sys

try:
    from darwin_st.data import prepare as _prepare  # type: ignore
except ModuleNotFoundError:
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _src = os.path.join(_repo_root, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from darwin_st.data import prepare as _prepare  # type: ignore

# 转发核心包的公开接口
prepare_dataset = _prepare.prepare_dataset
load_split = _prepare.load_split
load_scaler = _prepare.load_scaler
load_adj = _prepare.load_adj
evaluate = _prepare.evaluate
generate_windows = _prepare.generate_windows

__all__ = [
    "prepare_dataset",
    "load_split",
    "load_scaler",
    "load_adj",
    "evaluate",
    "generate_windows",
]


if __name__ == "__main__":
    # 用法: DATASET=PeMS04 python3 scripts/prepare.py
    ds = os.environ.get("DATASET", "PeMS04")
    print(f"准备时空预测数据 (Prepare Spatio-Temporal Data): {ds}")
    prepare_dataset(ds)
    print("就绪 (Ready)! 数据管道搭建完成。")
