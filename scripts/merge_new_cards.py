#!/usr/bin/env python
"""B6 知识库扩充: 合并 new_cards_2026.json 进 mechanism_cards.json (615→631)。

**改库是不可逆操作** (mechanism_cards.json 在 gitignore 的 data/ 内, git 兜不了底),
防护同 apply_qc.py:
  - 默认 dry-run: 只打印计划, 不写库。必须显式 --apply 才真改。
  - 强制备份: 真 apply 前写 mechanism_cards.pre_b6.bak.json (已存在则拒绝,
    防冲掉备份; 注意别与 apply_qc.py 的 mechanism_cards.bak.json 混淆)。
  - 逐卡校验: 每张新卡重建为 Mechanism 并 validate() (受控词表/抽象函数非空)。
  - 重名检查: 新卡之间、新卡与旧库之间重名 → 报错列出并拒绝合并。
  - 语义骨架去重: 用 extraction._name_key 检查新卡不与旧卡/彼此撞骨架。

另: 调研素材中有 4 个机制其载体卡**已在库中** (B3 自动抽取, 证据为空),
不新建卡, 只把核实过的 evidence 回填进现有卡 (见 EVIDENCE_BACKFILL):
  - meta_node_parameter        ← HimNet (B6 素材 #4 heterogeneity_meta_parameter 同机制)
  - irregular_spatial_patching ← PatchSTG (B6 素材 #8, 同名同论文)
  - linear_attention           ← STGformer 效率证据 (B6 素材 #20, 同机制)
  - reversible_instance_normalization ← OpenCity (B6 素材 #19 instance_norm_anti_shift 同机制)

用法:
    python scripts/merge_new_cards.py           # dry-run, 只看计划
    python scripts/merge_new_cards.py --apply   # 真合并 (先备份)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))

from darwin_st.knowledge.extraction import _name_key  # noqa: E402
from darwin_st.knowledge.ontology import Evidence, Mechanism  # noqa: E402

DATA = _REPO / "src" / "darwin_st" / "knowledge" / "data"
DEFAULT_CARDS = str(DATA / "mechanism_cards.json")
DEFAULT_NEW = str(DATA / "new_cards_2026.json")
BACKUP = str(DATA / "mechanism_cards.pre_b6.bak.json")

# 回填到现有卡的核实证据 (只追加, 绝不覆盖现有字段)。
# 判定标准: 同一机制同一论文/同一核心结构 → 不新建卡, 回填证据。
EVIDENCE_BACKFILL: dict[str, list[dict]] = {
    "meta_node_parameter": [
        {
            "dataset": "PeMS04",
            "metric": "MAE",
            "delta": "HimNet 18.14 (异质性嵌入经 meta-learner 生成节点/时间特定参数)",
            "grade": "moderate",
            "source": "HimNet KDD'24 arXiv:2405.10800",
        }
    ],
    "irregular_spatial_patching": [
        {
            "dataset": "大尺度交通图 (万级传感器)",
            "metric": "训练吞吐",
            "delta": "PatchSTG KDTree 均衡 patch + patch 内深度/跨 patch 广度注意力交替, 大尺度图约 10× 提速",
            "grade": "moderate",
            "source": "PatchSTG KDD'25 arXiv:2412.09972",
        }
    ],
    "linear_attention": [
        {
            "dataset": "8600 节点大图 (PEMS 系)",
            "metric": "推理速度",
            "delta": "STGformer 配核化线性注意力比 STAEformer 快约 100×",
            "grade": "moderate",
            "source": "STGformer arXiv:2410.00385",
        }
    ],
    "reversible_instance_normalization": [
        {
            "dataset": "多城市交通基准",
            "metric": "MAE",
            "delta": "OpenCity 配逐实例归一化抗漂移, zero-shot 与全数据 SOTA 差距 <8%",
            "grade": "high",
            "source": "OpenCity arXiv:2408.10269",
        }
    ],
}

_MECH_FIELDS = {f.name for f in dataclasses.fields(Mechanism)}
_EVID_FIELDS = {f.name for f in dataclasses.fields(Evidence)}


def mech_from_card(c: dict) -> Mechanism:
    """dict 卡 → Mechanism (处理 evidence 子对象; 过滤 schema 外字段)。"""
    d = {k: v for k, v in c.items() if k in _MECH_FIELDS}
    d["evidence"] = [
        Evidence(**{k: v for k, v in e.items() if k in _EVID_FIELDS})
        for e in d.get("evidence", [])
        if isinstance(e, dict)
    ]
    return Mechanism(**d)


def validate_new_cards(new_cards: list[dict]) -> list[str]:
    """逐卡重建 + validate()。返回错误列表 (空=全过)。"""
    errors = []
    for c in new_cards:
        try:
            mech_from_card(c).validate()
        except Exception as e:
            errors.append(f"{c.get('name', '?')}: {e}")
    return errors


def check_name_clashes(old_cards: list[dict], new_cards: list[dict]) -> list[str]:
    """重名检查: 新卡内部重复 + 新卡与旧库重名。返回冲突描述列表。"""
    clashes = []
    old_names = {c["name"] for c in old_cards}
    for name, cnt in Counter(c["name"] for c in new_cards).items():
        if cnt > 1:
            clashes.append(f"新卡内部重名 ×{cnt}: {name}")
    for c in new_cards:
        if c["name"] in old_names:
            clashes.append(f"与旧库重名: {c['name']}")
    return clashes


def check_skeleton_clashes(old_cards: list[dict], new_cards: list[dict]) -> list[str]:
    """语义骨架检查 (_name_key): 新卡与旧卡撞骨架、新卡彼此撞骨架。"""
    clashes = []
    old_keys: dict[str, list[str]] = {}
    for c in old_cards:
        old_keys.setdefault(_name_key(c["name"]), []).append(c["name"])
    seen_new: dict[str, str] = {}
    for c in new_cards:
        k = _name_key(c["name"])
        if k in old_keys:
            clashes.append(f"{c['name']} 骨架撞旧卡 {old_keys[k]}")
        if k in seen_new:
            clashes.append(f"{c['name']} 骨架撞新卡 {seen_new[k]}")
        seen_new[k] = c["name"]
    return clashes


def apply_backfill(old_cards: list[dict]) -> tuple[list[dict], list[str]]:
    """把 EVIDENCE_BACKFILL 的证据追加进现有卡 (已存在按 (dataset,metric,source) 跳过)。

    只追加 evidence, 不动其他任何字段。返回 (新列表, 回填日志)。
    """
    log = []
    for c in old_cards:
        adds = EVIDENCE_BACKFILL.get(c["name"])
        if not adds:
            continue
        have = {(e.get("dataset"), e.get("metric"), e.get("source")) for e in c.get("evidence", [])}
        n = 0
        for e in adds:
            key = (e["dataset"], e["metric"], e["source"])
            if key not in have:
                c.setdefault("evidence", []).append(dict(e))
                have.add(key)
                n += 1
        log.append(f"{c['name']}: +{n} 条 evidence")
    return old_cards, log


def _coverage_line(cards: list[dict]) -> str:
    dom = Counter(c.get("origin_domain", "ST") for c in cards)
    lvl = Counter(c.get("abstraction_level", "concept") for c in cards)
    return f"{len(cards)} 张 | 域 {dict(sorted(dom.items()))} | 层 {dict(lvl)}"


def main():
    ap = argparse.ArgumentParser(description="B6 新机制卡合并 (默认 dry-run)")
    ap.add_argument("--cards", default=DEFAULT_CARDS)
    ap.add_argument("--new", default=DEFAULT_NEW)
    ap.add_argument("--apply", action="store_true",
                    help="真改库 (默认只 dry-run 打印计划)")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    old_raw = json.load(open(args.cards, encoding="utf-8"))
    old_orig = old_raw.get("mechanisms", [])          # 原库快照 (备份用, 绝不被回填污染)
    old_cards = json.loads(json.dumps(old_orig))      # 深拷贝, 供回填就地修改
    new_cards = json.load(open(args.new, encoding="utf-8")).get("mechanisms", [])

    print(f"=== B6 合并计划 ({'DRY-RUN' if not args.apply else 'APPLY'}) ===", flush=True)
    print(f"  旧库: {_coverage_line(old_cards)}", flush=True)
    print(f"  新卡: {len(new_cards)} 张 ← {args.new}", flush=True)

    # --- 关卡 1: 新卡逐张 validate ---
    errors = validate_new_cards(new_cards)
    if errors:
        print(f"\n✗ {len(errors)} 张新卡 validate 失败, 拒绝合并:", flush=True)
        for e in errors:
            print(f"    {e}", flush=True)
        sys.exit(1)
    print(f"\n  ✓ {len(new_cards)} 张新卡全部通过 validate (受控词表/抽象函数)", flush=True)

    # --- 关卡 2: 重名检查 ---
    clashes = check_name_clashes(old_cards, new_cards)
    if clashes:
        print(f"\n✗ {len(clashes)} 处重名, 拒绝合并:", flush=True)
        for e in clashes:
            print(f"    {e}", flush=True)
        sys.exit(1)
    print("  ✓ 无重名 (新卡内部 / 新卡 vs 旧库)", flush=True)

    # --- 关卡 3: 语义骨架去重检查 ---
    skel = check_skeleton_clashes(old_cards, new_cards)
    if skel:
        print(f"\n✗ {len(skel)} 处语义骨架冲突, 拒绝合并:", flush=True)
        for e in skel:
            print(f"    {e}", flush=True)
        sys.exit(1)
    print("  ✓ 无语义骨架冲突 (_name_key)", flush=True)

    # --- 证据回填 (4 个已有机制卡) ---
    old_cards, backfill_log = apply_backfill(old_cards)
    print(f"\n  证据回填 {len(backfill_log)} 张现有卡 (不新建):", flush=True)
    for line in backfill_log:
        print(f"    {line}", flush=True)

    merged = old_cards + new_cards
    print(f"\n  合并后预览: {_coverage_line(merged)}", flush=True)

    # 最终重建校验 (全部 merged 卡都能重建为合法 Mechanism)
    n_ok, rebuild_errors = 0, []
    for c in merged:
        try:
            mech_from_card(c).validate()
            n_ok += 1
        except Exception as e:
            rebuild_errors.append(f"{c.get('name', '?')}: {e}")
    if rebuild_errors:
        print(f"\n✗ 合并后 {len(rebuild_errors)} 张卡重建校验失败, 拒绝写库:", flush=True)
        for e in rebuild_errors[:10]:
            print(f"    {e}", flush=True)
        sys.exit(1)
    print(f"  ✓ 重建校验: {n_ok} 张可重建为合法 Mechanism", flush=True)

    if not args.apply:
        print("\n[dry-run] 未改库。确认无误后加 --apply 真合并。", flush=True)
        return

    # --- 真改库: 强制备份 (已存在则拒绝) ---
    if os.path.exists(BACKUP):
        print(f"\n✗ 备份已存在 {BACKUP} —— 拒绝覆盖 (防冲掉上次备份)。"
              f"\n  如确认要重跑, 先手动移走该备份。", flush=True)
        sys.exit(1)
    with open(BACKUP, "w", encoding="utf-8") as f:
        json.dump({"n": len(old_orig), "mechanisms": old_orig},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ 原库已备份 → {BACKUP}", flush=True)

    with open(args.cards, "w", encoding="utf-8") as f:
        json.dump({"n": len(merged), "mechanisms": merged}, f,
                  ensure_ascii=False, indent=2)
    print(f"  ✓ 合并库写回 → {args.cards} (n={len(merged)})", flush=True)

    print("\n=== B6 合并完成 ===", flush=True)
    print(f"  新增 {len(new_cards)} 张新卡 + 回填 {len(backfill_log)} 张现有卡", flush=True)
    print(f"  总数: {_coverage_line(merged)}", flush=True)


if __name__ == "__main__":
    main()
