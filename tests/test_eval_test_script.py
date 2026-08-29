"""scripts/eval_test.py 的纯函数测试 (E1 口径对齐评测的汇总逻辑)。"""

from __future__ import annotations

import importlib.util
import os


def _load_eval_test():
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", "eval_test.py")
    spec = importlib.util.spec_from_file_location("eval_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rec(seed, test_mae, val_mae):
    return {"seed": seed, "retrained_val_mae": val_mae,
            "val": {"mae": val_mae, "rmse": 0, "mape": 0},
            "test": {"mae": test_mae, "rmse": 0, "mape": 0},
            "offset_test_minus_val": test_mae - val_mae}


def test_summarize_single_seed_std_zero():
    mod = _load_eval_test()
    s = mod.summarize_results([_rec(42, 18.40, 18.35)], "m", "PeMS04")
    assert s["n_seeds"] == 1
    assert abs(s["test_mae_mean"] - 18.40) < 1e-9
    assert s["test_mae_std"] == 0.0                      # 单 seed 不报假方差
    assert abs(s["offset_mean"] - 0.05) < 1e-9


def test_summarize_multi_seed_mean_std():
    mod = _load_eval_test()
    recs = [_rec(0, 18.40, 18.35), _rec(1, 18.50, 18.35), _rec(2, 18.45, 18.40)]
    s = mod.summarize_results(recs, "m", "PeMS04")
    assert s["n_seeds"] == 3
    assert abs(s["test_mae_mean"] - 18.45) < 1e-9        # 均值
    assert s["test_mae_std"] > 0                          # 多 seed 出 std
    # 偏移 = test−val: [0.05, 0.15, 0.05] → mean 0.0833
    assert abs(s["offset_mean"] - (0.25 / 3)) < 1e-6
    assert len(s["results"]) == 3                         # 原始记录全保留 (可追溯)
