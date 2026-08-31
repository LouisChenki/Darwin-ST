"""E11 防退化守卫统计 —— 验证门拦截 + CRASH 构成 + 典型案例 (纯日志/履历分析)。

IJGIS 审稿人大概率问"LLM 生成的算子会不会作弊/退化 (reward hacking)"。
本脚本用历史创造履历 + 记忆库产出"防线真实工作量"的证据表:
  - 各验证门拦截次数 (shape/forward/static_leak/...), 分版本;
  - 记忆库 CRASH 率与 fail_reason 构成;
  - 典型案例 (被拦下的具体算子 + 错误摘要)。

用法:
    python scripts/gate_stats.py --archive v4=frozen-runs/b7v1/creation_archive_b7v1.jsonl \
        --archive v2在跑=frozen-runs/e11/creation_archive_b7v2.jsonl \
        --db frozen-runs/b7v1/b7_v1.db --out paper-results/e11/gate_stats.md
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict


def _archive_stats(path: str) -> dict:
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    gates = Counter(r.get("gate", "unknown") for r in recs)
    comps = Counter(r.get("composition", "?") for r in recs)
    # 典型案例: 被拦的记录 (gate != all), 带算子名与错误摘要
    blocked = [r for r in recs if r.get("gate") not in ("all", None)]
    cases = [{"operator": r.get("operator_name") or "(未命名)", "gate": r.get("gate"),
              "error": (r.get("error") or "")[:160]} for r in blocked[:5]]
    adopted = [r for r in recs if r.get("adopted") is not None]
    return {"n": len(recs), "gates": dict(gates), "compositions": dict(comps),
            "n_blocked": len(blocked), "cases": cases,
            "n_outcome": len(adopted), "n_keep": sum(1 for r in adopted if r["adopted"])}


def _db_stats(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        by_status = dict(db.execute(
            "SELECT status, COUNT(*) FROM experiments GROUP BY status").fetchall())
        cols = [r[1] for r in db.execute("PRAGMA table_info(experiments)").fetchall()]
        reasons = Counter()
        if "fail_reason" in cols:
            for fr, n in db.execute(
                    "SELECT fail_reason, COUNT(*) FROM experiments WHERE status='CRASH' "
                    "GROUP BY fail_reason").fetchall():
                reasons[fr or "?"] = n
    finally:
        db.close()
    return {"by_status": by_status, "crash_reasons": dict(reasons)}


def main() -> int:
    ap = argparse.ArgumentParser(description="E11 防退化守卫统计 (验证门拦截 + CRASH 构成)")
    ap.add_argument("--archive", action="append", default=[],
                    help="标签=履历 jsonl (可重复)")
    ap.add_argument("--db", action="append", default=[], help="memory db (可重复)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    lines = ["# E11 防退化守卫统计", ""]
    lines.append("## 验证门拦截 (创造履历, 按版本)")
    lines.append("")
    lines.append("| 版本 | 履历数 | 过门 (all) | 被拦 | 拦截率 | 拦截门分布 |")
    lines.append("|---|---:|---:|---:|---:|---|")
    all_cases: list[tuple[str, dict]] = []
    for spec in args.archive:
        label, _, path = spec.partition("=")
        st = _archive_stats(path)
        n_blocked = st["n_blocked"]
        gate_dist = ", ".join(f"{g}×{n}" for g, n in
                              sorted(st["gates"].items()) if g != "all") or "—"
        lines.append(f"| {label} | {st['n']} | {st['gates'].get('all', 0)} | "
                     f"{n_blocked} | {n_blocked / max(st['n'], 1):.1%} | {gate_dist} |")
        for c in st["cases"]:
            all_cases.append((label, c))
    lines.append("")

    if args.db:
        lines.append("## 记忆库 CRASH 构成 (训练侧失败)")
        lines.append("")
        lines.append("| db | KEEP | CRASH | DISCARD | CRASH 率 | 主要 fail_reason |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for p in args.db:
            st = _db_stats(p)
            bs = st["by_status"]
            total = sum(bs.values())
            cr = bs.get("CRASH", 0)
            top = "; ".join(f"{(r or '?').splitlines()[0][:60]}×{n}" for r, n in
                            sorted(st["crash_reasons"].items(), key=lambda x: -x[1])[:3]) or "—"
            lines.append(f"| `{os.path.basename(p)}` | {bs.get('KEEP', 0)} | {cr} | "
                         f"{bs.get('DISCARD', 0)} | {cr / max(total, 1):.1%} | {top} |")
        lines.append("")

    lines.append("## 被拦典型案例 (防线的真实工作量)")
    lines.append("")
    if all_cases:
        for label, c in all_cases:
            lines.append(f"- **[{label}] {c['operator']}** (门={c['gate']}): {c['error']}")
    else:
        lines.append("- (样本内无被拦记录)")
    lines.append("")
    lines.append("> 注: 零初始化分支 α 的学习轨迹当前未逐点落盘, 本轮不虚构; "
                 "如需该证据, 补一次小规模带 α 记录的训练对照 (见 plan E11 备注)。")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
