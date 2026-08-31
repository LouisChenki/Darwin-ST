"""E13 搜索效率曲线 (best-so-far vs 评估预算) —— 创造闭环 vs 纯进化的样本效率证据。

IJGIS Claim-6 的硬指标: 创造闭环不只是终点好, 而是同评估数下全程更优。
从记忆库直接画 best-so-far 曲线 (x 轴评估数 / GPU·h 双口径)。

用法:
    python scripts/efficiency_curves.py --db 纯进化=frozen-runs/e13/stage2_clean.db \
        --db v2=frozen-runs/e13/stage2_sota.db --db v3=frozen-runs/e13/tier2_v1.db \
        --db v4=frozen-runs/b7v1/b7_v1.db --out leaderboard/assets/efficiency_curves.png
"""

from __future__ import annotations

import argparse
import sqlite3
import sys


def best_so_far(path: str) -> tuple[list[float], list[float]]:
    """读 db → (累计评估数轴, best-so-far MAE 轴) 与 (累计 GPU·h 轴, ...)。只取 KEEP 且有限 MAE。"""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cols = [r[1] for r in db.execute("PRAGMA table_info(experiments)").fetchall()]
        mae_col = "val_mae" if "val_mae" in cols else "mae"
        rows = db.execute(
            f"SELECT {mae_col}, wall_seconds FROM experiments "
            f"WHERE status='KEEP' AND {mae_col} IS NOT NULL ORDER BY rowid").fetchall()
    finally:
        db.close()
    xs, ys, xh = [], [], []
    best = float("inf")
    cum_h = 0.0
    for i, (mae, ws) in enumerate(rows, 1):
        if mae is None:
            continue
        best = min(best, mae)
        cum_h += (ws or 0) / 3600
        xs.append(i)
        ys.append(best)
        xh.append(cum_h)
    return xs, ys, xh


def _pick_cjk_font() -> str | None:
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans GB", "PingFang SC", "PingFang HK", "Microsoft YaHei",
                 "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"):
        if name in available:
            return name
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="E13 搜索效率曲线 (best-so-far vs 评估预算)")
    ap.add_argument("--db", action="append", required=True,
                    help="标签=db路径 (可重复, 曲线顺序即图例顺序)")
    ap.add_argument("--out", required=True, help="输出 PNG 路径")
    ap.add_argument("--summary", default="", help="汇总 markdown 输出路径 (可选)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    font = _pick_cjk_font()
    if font:
        plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    table_rows = []
    for spec in args.db:
        label, _, path = spec.partition("=")
        xs, ys, xh = best_so_far(path)
        if not xs:
            print(f"警告: {label} ({path}) 无有效评估, 跳过", file=sys.stderr)
            continue
        ax1.plot(xs, ys, linewidth=1.8, label=label)
        ax2.plot(xh, ys, linewidth=1.8, label=label)
        table_rows.append((label, len(xs), round(xh[-1], 1), round(ys[-1], 4)))

    for ax, xlabel in ((ax1, "评估次数"), (ax2, "GPU·h (卡时)")):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("best-so-far val MAE")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    ax1.set_title("样本效率: 同评估数比 best-so-far")
    ax2.set_title("成本效率: 同 GPU·h 比 best-so-far")
    fig.suptitle("Darwin-ST 搜索效率曲线 (E13)", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"曲线已写入 {args.out}")

    md = ["# E13 搜索效率汇总", "",
          "| 臂 | 评估数 | 总 GPU·h | 终点 best-so-far |", "|---|---:|---:|---:|"]
    for label, n, gh, best in table_rows:
        md.append(f"| {label} | {n} | {gh} | {best} |")
    text = "\n".join(md)
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"汇总已写入 {args.summary}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
