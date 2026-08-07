"""方法版本榜构建入口 (Version Board Builder) —— 生成排行榜新门面: 版本主榜 + 成绩演进趋势图。

读 leaderboard/versions.json (每大版本一条最佳记录) + baseline_registry (已发表 baseline/SOTA)
→ 调 src/darwin_st/leaderboard.py 纯核心算位次/渲染 → 落盘:
  - leaderboard/README.md: 门面 (一句话说明 + 版本主榜 + 趋势图 + 当前最佳 + 完整存档链接)。
  - leaderboard/assets/pems04_version_trend.png: 成绩随版本演进趋势图
    (仿 artificialanalysis "Intelligence, Over Time": 反转 y 轴 + SOTA 参考线)。

纯函数核 (parse_versions / build_version_rows / render_version_*) 在 src/darwin_st/leaderboard.py,
本脚本只做 IO 与绘图。默认幂等可重跑。

用法:
    uv run python scripts/build_version_board.py                     # 全默认
    uv run python scripts/build_version_board.py --out /tmp/trend.png  # 只改趋势图输出
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date

from darwin_st.baseline_registry import BASELINE_METRICS, get_sota
from darwin_st.leaderboard import (
    VersionRecord,
    build_version_rows,
    parse_versions,
    render_version_readme_md,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 趋势图上的水平参考线 (已发表强 baseline; 数值从 baseline_registry 取, 保持同源)
REF_BASELINES = ["STD-MAE", "STAEformer", "STID"]


def load_versions(path: str, dataset: str) -> list[VersionRecord]:
    """读 versions.json, 解析并按数据集过滤, 按日期/版本升序返回。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = [r for r in parse_versions(data) if r.dataset == dataset]
    return sorted(records, key=lambda r: (r.date, r.version))


# --------------------------------------------------------------------------
# 趋势图 (脚本层绘图, 非纯函数核)
# --------------------------------------------------------------------------

def _pick_cjk_font() -> str | None:
    """找一个可用的中文字体 (无则返回 None → 图表全用英文标签, 避免豆腐块)。"""
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Hiragino Sans GB", "PingFang SC", "PingFang HK", "Microsoft YaHei",
                 "Noto Sans CJK SC", "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS"):
        if name in available:
            return name
    return None


def plot_version_trend(
    records: list[VersionRecord],
    baselines: dict[str, dict],
    out_path: str,
    dataset: str,
) -> None:
    """绘制"成绩随版本演进"趋势图并写入 out_path (白底/浅网格/反转 y 轴/dpi>=150/约1200×700)。

    - 折线 + 数据点标注 "版本号 + MAE"; x 轴为真实日期。
    - REF_BASELINES 三条水平虚线参考线, 名称标注在图右侧。
    - 中文字体缺失时自动切英文标签。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无显示环境
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    cjk_font = _pick_cjk_font()
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    txt = {
        "title": (f"{dataset} 方法版本成绩演进" if cjk_font
                  else f"{dataset} — Best MAE by Method Version"),
        "ylabel": "Best MAE（越低越好）" if cjk_font else "Best MAE (lower is better)",
        "ours": "本项目版本最佳" if cjk_font else "Darwin-ST version best",
    }

    xs = [date.fromisoformat(r.date) for r in records]
    ys = [r.best_mae for r in records]

    fig, ax = plt.subplots(figsize=(8, 4.667), dpi=150)  # 150dpi → 约 1200×700
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 水平虚线参考线 (已发表强 baseline), 名称标在图右缘外
    ref_colors = ["#c0392b", "#e67e22", "#7f8c8d"]
    for i, name in enumerate(REF_BASELINES):
        mae = baselines.get(name, {}).get("mae")
        if mae is None:
            continue
        ax.axhline(mae, ls="--", lw=1.2, color=ref_colors[i % len(ref_colors)], alpha=0.8, zorder=1)
        ax.text(1.005, mae, f"{name} {mae:.2f}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=8.5, color=ref_colors[i % len(ref_colors)])

    # 本项目版本折线 + 数据点 (贴近参考线的点把标注放到点下方, 避免与虚线/右侧标签相碰)
    ax.plot(xs, ys, "-o", color="#1f6feb", lw=2, ms=7, zorder=3, label=txt["ours"])
    ref_maes = [baselines[n]["mae"] for n in REF_BASELINES
                if baselines.get(n, {}).get("mae") is not None]
    for x, y, r in zip(xs, ys, records):
        mae_txt = f"{y:.3f}".rstrip("0").rstrip(".")  # 与榜单 _fmt_mae3 同口径
        near_ref = any(abs(y - rm) < 0.2 for rm in ref_maes)
        ax.annotate(f"{r.version}  {mae_txt}",
                    (x, y), textcoords="offset points",
                    xytext=(0, -18) if near_ref else (0, 10),
                    ha="center", fontsize=9, fontweight="bold", color="#1f6feb", zorder=4)

    ax.invert_yaxis()  # MAE 越低越好 → 低者在上
    ax.set_title(txt["title"], fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(txt["ylabel"], fontsize=10)
    ax.grid(True, color="#e5e5e5", lw=0.8, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.margins(x=0.08, y=0.12)
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    fig.subplots_adjust(right=0.84, left=0.1, top=0.9, bottom=0.13)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="构建方法版本榜 (README 门面 + 趋势图)")
    ap.add_argument("--versions", default=os.path.join(REPO_ROOT, "leaderboard", "versions.json"),
                    help="版本记录 JSON 路径")
    ap.add_argument("--dataset", default="PeMS04", help="数据集 (versions.json 按此过滤)")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "leaderboard", "assets",
                                                  "pems04_version_trend.png"),
                    help="趋势图输出 PNG 路径")
    ap.add_argument("--readme", default=os.path.join(REPO_ROOT, "leaderboard", "README.md"),
                    help="门面 README 输出路径")
    args = ap.parse_args()

    records = load_versions(args.versions, args.dataset)
    baselines = BASELINE_METRICS.get(args.dataset, {})
    sota_name, sota_mae = get_sota(args.dataset)
    rows = build_version_rows(records, baselines, sota_mae)

    # 趋势图 (可重复生成)
    plot_version_trend(records, baselines, args.out, args.dataset)

    # 门面 README (趋势图路径相对 README 所在目录)
    trend_rel = os.path.relpath(args.out, os.path.dirname(args.readme))
    md = render_version_readme_md(args.dataset, rows, sota_name, sota_mae,
                                  trend_image=trend_rel)
    os.makedirs(os.path.dirname(args.readme), exist_ok=True)
    with open(args.readme, "w", encoding="utf-8") as f:
        f.write(md)

    best = rows[-1] if rows else None
    print(f"✅ 版本榜构建完成: {len(rows)} 个版本 → {args.readme} + {args.out}")
    if best is not None:
        print(f"   当前最佳: {best.record.version} MAE={best.record.best_mae} "
              f"(榜单位次 {best.rank}/{best.board_size}, vs SOTA "
              f"{f'{best.vs_sota:+.2f}' if best.vs_sota is not None else '—'})")


if __name__ == "__main__":
    main()
