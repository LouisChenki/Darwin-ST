"""E12 成本核算 (cost report) —— 从记忆库 + 动作账本统计 GPU·h 与 LLM 调用量。

IJGIS Claim-6 ("自治化的成本可接受") 的定量支撑。纯统计, 不占 GPU。

统计口径:
  - GPU·h: experiments.wall_seconds 求和 / 3600 (单评估墙钟; 多卡并行时为"卡时",
    即 ∑卡×小时, 与"卡·天"换算 ÷24)。
  - LLM 调用: 按动作类型统计 (tier2_actions.jsonl 账本); 每次合成动作含
    n_hypotheses 次假设生成 (履历本记录), 精炼/反思各 1 次。token 量未逐次记录
    (历史未留), 报告中显式标注为"调用次数"而非 token —— 不虚构精度。

用法:
    python scripts/cost_report.py --db frozen-runs/b7v1/b7_v1.db \
        --ledger frozen-runs/b7v1/creation_archive_b7v2.jsonl.tier2_actions.jsonl \
        --archive frozen-runs/b7v1/creation_archive_b7v1.jsonl --out report.md
    # 多个 db (多版本全程): --db 可重复
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter


def _db_stats(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        n = db.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
        by_status = dict(db.execute(
            "SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall())
        gpu_h = (db.execute(
            "SELECT COALESCE(SUM(wall_seconds),0) FROM experiments").fetchone()[0] or 0) / 3600
        by_run = db.execute(
            "SELECT run_tag, COUNT(*), COALESCE(SUM(wall_seconds),0)/3600 "
            "FROM experiments GROUP BY run_tag").fetchall()
        best = db.execute(
            "SELECT MIN(val_mae) FROM experiments WHERE status='KEEP'").fetchone()[0]
    finally:
        db.close()
    return {"db": path, "evals": n, "by_status": by_status, "gpu_hours": round(gpu_h, 1),
            "best_val_mae": best, "by_run": [(r[0], r[1], round(r[2], 1)) for r in by_run]}


def _ledger_stats(path: str) -> dict:
    if not os.path.exists(path):
        return {"ledger": path, "exists": False}
    events = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    started = Counter()
    terminal = Counter()
    for ev in events:
        if ev.get("event") == "started":
            started[ev.get("action_type", "?")] += 1
        elif ev.get("event") == "terminal":
            terminal[(ev.get("final_action_type") or ev.get("action_type") or "?",
                      ev.get("terminal_status", "?"))] += 1
    return {"ledger": path, "exists": True, "started": dict(started),
            "terminal": {f"{k[0]}|{k[1]}": v for k, v in terminal.items()}}


def _archive_stats(path: str) -> dict:
    """履历本: LLM 假设调用量估计 (每条记录 = 一次假设的成败)。"""
    if not os.path.exists(path):
        return {"archive": path, "exists": False}
    n = sum(1 for l in open(path, encoding="utf-8") if l.strip())
    return {"archive": path, "exists": True, "creation_records": n,
            "note": "每条记录≈一次假设合成的 LLM 调用 (含失败; token 量历史未逐次记录)"}


def main() -> int:
    ap = argparse.ArgumentParser(description="成本核算: GPU·h + LLM 调用量 (IJGIS Claim-6)")
    ap.add_argument("--db", action="append", required=True, help="memory db (可重复)")
    ap.add_argument("--ledger", default="", help="tier2_actions.jsonl 路径")
    ap.add_argument("--archive", default="", help="creation_archive.jsonl 路径")
    ap.add_argument("--out", default="", help="输出 markdown 路径 (缺省打印)")
    args = ap.parse_args()

    lines = ["# 成本核算报告 (cost report)", ""]
    lines.append("## GPU 成本 (按评估墙钟求和, 卡时口径)")
    lines.append("")
    lines.append("| db | 评估数 | KEEP | CRASH | DISCARD | GPU·h | best val MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    total_h = 0.0
    for p in args.db:
        st = _db_stats(p)
        total_h += st["gpu_hours"]
        lines.append(f"| `{os.path.basename(p)}` | {st['evals']} | "
                     f"{st['by_status'].get('KEEP', 0)} | {st['by_status'].get('CRASH', 0)} | "
                     f"{st['by_status'].get('DISCARD', 0)} | {st['gpu_hours']} | "
                     f"{st['best_val_mae']:.4f} |")
    lines.append("")
    lines.append(f"**合计 GPU·h: {round(total_h, 1)} (≈ {round(total_h / 24, 1)} 卡·天)**")
    lines.append("")

    if args.ledger:
        ls = _ledger_stats(args.ledger)
        lines.append("## Tier-2 动作与 LLM 调用 (账本)")
        lines.append("")
        if ls["exists"]:
            lines.append(f"- 动作启动: {ls['started']}")
            lines.append(f"- 动作终态: {ls['terminal']}")
        else:
            lines.append(f"- 账本不存在: {args.ledger}")
        lines.append("")
    if args.archive:
        ar = _archive_stats(args.archive)
        lines.append("## 创造履历 (LLM 假设调用量估计)")
        lines.append("")
        if ar["exists"]:
            lines.append(f"- 创造履历记录: {ar['creation_records']} 条 ({ar['note']})")
        lines.append("")

    text = "\n".join(lines)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"报告已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
