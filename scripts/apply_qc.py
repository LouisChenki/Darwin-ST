#!/usr/bin/env python
"""P3-b 质检结果 apply —— 按 qc_report.json 的裁决改 mechanism_cards.json (640→615)。

**改库是不可逆操作** (mechanism_cards.json 在 gitignore 的 data/ 内, git 兜不了底),
故多重防护:
  - 默认 dry-run: 只打印计划, 不写库。必须显式 --apply 才真改。
  - 强制备份: 真 apply 前写 mechanism_cards.bak.json (已存在则拒绝, 防二次运行冲掉备份)。
  - 种子保护铁律: 任何动作触及种子卡内容一律跳过 (qc_apply.plan_actions 保证)。

确定性逻辑在 darwin_st.knowledge.qc_apply (纯函数, 已单测)。本脚本只做 IO + 打印。

用法:
    python scripts/apply_qc.py                 # dry-run, 只看计划
    python scripts/apply_qc.py --apply         # 真改库 (先备份)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from darwin_st.knowledge.qc import mechanisms_from_cards  # noqa: E402
from darwin_st.knowledge.qc_apply import (  # noqa: E402
    apply_plan,
    plan_actions,
    seed_protected_names,
)

DATA = _REPO / "src" / "darwin_st" / "knowledge" / "data"
DEFAULT_CARDS = str(DATA / "mechanism_cards.json")
DEFAULT_REPORT = str(DATA / "qc_report.json")


def _coverage_line(cards: list[dict]) -> str:
    from collections import Counter
    dom = Counter(c.get("origin_domain", "ST") for c in cards)
    lvl = Counter(c.get("abstraction_level", "concept") for c in cards)
    return f"{len(cards)} 张 | 域 {dict(sorted(dom.items()))} | 层 {dict(lvl)}"


def main():
    ap = argparse.ArgumentParser(description="P3-b 质检结果 apply (默认 dry-run)")
    ap.add_argument("--cards", default=DEFAULT_CARDS)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--apply", action="store_true",
                    help="真改库 (默认只 dry-run 打印计划)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    cards = json.load(open(args.cards, encoding="utf-8")).get("mechanisms", [])
    report = json.load(open(args.report, encoding="utf-8"))
    verdicts = report.get("verdicts", [])
    # 空前提死卡 (从 rule_flags 的 empty_preconditions 取)
    rule_flags = report.get("rule_flags", {})
    empty_pre = [n for n, fs in rule_flags.items() if "empty_preconditions" in fs]

    protected = seed_protected_names()
    plan = plan_actions(cards, verdicts, protected, empty_precondition_drops=empty_pre)

    print(f"=== 质检 apply 计划 ({'DRY-RUN' if not args.apply else 'APPLY'}) ===", flush=True)
    print(f"  种子保护名单: {len(protected)} 张", flush=True)
    print(f"  改前: {_coverage_line(cards)}", flush=True)
    print(f"\n  合并 {len(plan.merges)} 张 (源→目标):", flush=True)
    for src, tgt in plan.merges:
        print(f"    {src} → {tgt}", flush=True)
    print(f"\n  删除 {len(plan.drops)} 张:", flush=True)
    for n in plan.drops:
        print(f"    {n}", flush=True)
    print(f"\n  跳过 {len(plan.skipped)} 张 (含种子保护):", flush=True)
    for n, reason in plan.skipped:
        print(f"    {n}: {reason}", flush=True)

    new_cards, stats = apply_plan(cards, plan)
    print(f"\n  改后预览: {_coverage_line(new_cards)}", flush=True)
    print(f"  统计: {stats}", flush=True)

    if not args.apply:
        print("\n[dry-run] 未改库。确认无误后加 --apply 真改。", flush=True)
        return

    # --- 真改库: 强制备份 ---
    bak = args.cards.replace(".json", ".bak.json")
    if os.path.exists(bak):
        print(f"\n✗ 备份已存在 {bak} —— 拒绝覆盖 (防冲掉上次备份)。"
              f"\n  如确认要重跑, 先手动移走该备份。", flush=True)
        sys.exit(1)
    with open(bak, "w", encoding="utf-8") as f:
        json.dump({"mechanisms": cards}, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ 原库已备份 → {bak}", flush=True)

    with open(args.cards, "w", encoding="utf-8") as f:
        json.dump({"n": len(new_cards), "mechanisms": new_cards}, f,
                  ensure_ascii=False, indent=2)
    print(f"  ✓ 改后库写回 → {args.cards}", flush=True)

    # 校验: 重建确认无破坏
    mechs = mechanisms_from_cards(new_cards)
    print(f"  ✓ 重建校验: {len(mechs)} 张可重建为 Mechanism", flush=True)


if __name__ == "__main__":
    main()
