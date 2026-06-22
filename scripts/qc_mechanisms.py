#!/usr/bin/env python
"""P3-b 机制库质检 (QC) 报告生成器 —— DeepSeek 当质检员, 只读不改库。

全量建图产出 ~624 张自动卡, 颗粒度偏细 (44% variant)。本脚本让 DeepSeek 逐组判断
keep/merge/split/drop, 产出**建议性、人类可读**的质检报告供用户定夺。**不改 mechanism_cards.json**。

立场 (用户决策): 保留细粒度。variant 多是"基元+论文创新"(gravity_informed_attention 等),
默认 KEEP, 只标真问题: 纯同义改名(merge)/整模型未拆(split)/空泛无信息(drop)。

形态 (确定性内核 darwin_st.knowledge.qc + 本脚本 IO/LLM 编排):
  读 mechanism_cards.json → cluster_by_concept (按中心词聚类) → 大组分批喂 DeepSeek
  → parse_verdicts (截断容错) → 汇总 → 写 qc_report.md (人读) + qc_report.json (机读, 供后续 apply)。

用法:
    export DEEPSEEK_API_KEY=...
    export DEEPSEEK_MODEL=deepseek-v4-flash
    python scripts/qc_mechanisms.py                 # 全量质检
    python scripts/qc_mechanisms.py --limit 50      # 小批 (前 N 张卡) 试跑看裁决质量
    python scripts/qc_mechanisms.py --dry-run       # 不调 LLM, 只出聚类+规则标记+统计 (验证链路)

产物 (默认 out-dir = src/darwin_st/knowledge/data/):
  - qc_report.md     人类可读质检报告 (合并/拆分/删除建议 + 规则标记 + 聚类一览)
  - qc_report.json   机器可读裁决 (供后续 apply 步骤)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# --- 让脚本在 repo 根直接跑 (src layout) ---
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from darwin_st.knowledge.qc import (  # noqa: E402
    CRITIC_QC_SYSTEM,
    cluster_by_concept,
    mechanisms_from_cards,
    parse_verdicts,
    render_report,
    rule_based_flags,
)

DEFAULT_CARDS = str(_REPO / "src" / "darwin_st" / "knowledge" / "data" / "mechanism_cards.json")
DEFAULT_OUT = str(_REPO / "src" / "darwin_st" / "knowledge" / "data")
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

# 一个 LLM 批最多塞多少张卡 (大组如 attention 83 张需拆批; 同组多批共享 concept 标签)
BATCH_MAX = 25


def _card_brief(m) -> dict:
    """喂质检员的精简卡 (省 token, 只给判同义/拆分/删除所需字段)。"""
    return {
        "name": m.name,
        "abstract_function": m.abstract_function,
        "preconditions": m.preconditions,
        "math_structure": (m.math_structure or "")[:160],
        "origin_domain": m.origin_domain,
        "abstraction_level": m.abstraction_level,
    }


def _batches(group: list, size: int):
    """把一个聚类组切成 ≤size 的批。"""
    for i in range(0, len(group), size):
        yield group[i : i + size]


def critique_group(llm, head: str, batch: list) -> list[dict]:
    """让 DeepSeek 质检一批 (同中心词) 卡, 返回裁决列表。失败返回空 (不致命)。"""
    cards = [_card_brief(m) for m in batch]
    user = (f"中心词 '{head}' 下的 {len(cards)} 张机制卡 (判断哪些是同义改名应合并、"
            f"哪些是整模型应拆、哪些空泛应删; 其余 keep):\n"
            f"{json.dumps(cards, ensure_ascii=False, indent=1)}")
    try:
        resp = llm.chat(
            [{"role": "system", "content": CRITIC_QC_SYSTEM},
             {"role": "user", "content": user}],
            temperature=0.2, max_tokens=8192,
        )
    except Exception as e:
        print(f"  [warn] 质检批失败 (head={head}): {e}", flush=True)
        return []
    return parse_verdicts(resp)


def main():
    ap = argparse.ArgumentParser(description="P3-b 机制库质检报告生成器 (只读)")
    ap.add_argument("--cards", default=DEFAULT_CARDS, help="mechanism_cards.json 路径")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="只质检前 N 张卡 (小批试跑)")
    ap.add_argument("--dry-run", action="store_true",
                    help="不调 LLM, 只出聚类+规则标记+统计 (验证链路)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    # --- 读卡 (只读) ---
    payload = json.load(open(args.cards, encoding="utf-8"))
    cards = payload.get("mechanisms", payload if isinstance(payload, list) else [])
    mechs = mechanisms_from_cards(cards)
    if args.limit:
        mechs = mechs[: args.limit]
    print(f"=== 质检 {len(mechs)} 张卡 (源: {args.cards}) ===", flush=True)

    clusters = cluster_by_concept(mechs)
    multi = {h: g for h, g in clusters.items() if len(g) > 1}
    flags = rule_based_flags(mechs)
    print(f"  聚成 {len(clusters)} 组 ({len(multi)} 组多卡); 规则标记 {len(flags)} 张", flush=True)

    # --- LLM 质检 (dry-run 跳过) ---
    verdicts: list[dict] = []
    if args.dry_run:
        print("  [dry-run] 跳过 LLM, 仅出聚类/规则/统计", flush=True)
    else:
        from darwin_st.creation.llm import OpenAICompatLLM

        llm = OpenAICompatLLM(model=DEFAULT_MODEL)
        print(f"  质检员模型: {DEFAULT_MODEL}", flush=True)
        t0 = time.time()
        # 只质检多卡组 (单卡组本就正交, 默认 keep, 省额度); 大组拆批
        n_batches = sum(max(1, (len(g) + BATCH_MAX - 1) // BATCH_MAX) for g in multi.values())
        done = 0
        for head, group in multi.items():
            for batch in _batches(group, BATCH_MAX):
                verdicts.extend(critique_group(llm, head, batch))
                done += 1
                if done % 5 == 0 or done == n_batches:
                    print(f"    ...质检进度 {done}/{n_batches} 批", flush=True)
        print(f"  质检完成 ({time.time()-t0:.0f}s): {len(verdicts)} 条裁决", flush=True)

    # --- 渲染 + 写报告 (只读库, 只写报告) ---
    os.makedirs(args.out_dir, exist_ok=True)
    md = render_report(mechs, clusters, verdicts, flags)
    md_path = os.path.join(args.out_dir, "qc_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    json_path = os.path.join(args.out_dir, "qc_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"n_cards": len(mechs), "n_clusters": len(clusters),
                   "verdicts": verdicts, "rule_flags": flags}, f,
                  ensure_ascii=False, indent=2)
    print(f"\n=== 报告 → {md_path} ===", flush=True)
    print(f"=== 机读裁决 → {json_path} ===", flush=True)

    # --- 控制台摘要 ---
    from collections import Counter
    vc = Counter(v["verdict"] for v in verdicts)
    if verdicts:
        print(f"裁决: keep {vc.get('keep',0)} / merge {vc.get('merge',0)} / "
              f"split {vc.get('split',0)} / drop {vc.get('drop',0)}", flush=True)


if __name__ == "__main__":
    main()
