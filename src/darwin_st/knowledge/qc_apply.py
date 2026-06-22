"""P3-b 质检结果 apply 的确定性内核 —— 无 IO, 可单测。

把 qc_report.json 的裁决编译成确定性动作计划并执行, 改 mechanism_cards.json
(640→615)。立场: merge=删源并把源的证据/前提并入目标; drop/split=删卡 (split 按
用户决策当 drop)。

**种子保护铁律**: 种子卡 (seeds.all_seed_mechanisms) 是人工精校基准, 任何动作若
触及种子卡内容 (merge 源是种子 / drop 种子) 一律跳过并记录。种子可以是 merge 目标
(继承别人的证据), 但永不被删、永不被并入别人。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["ApplyPlan", "seed_protected_names", "plan_actions", "apply_plan"]


def seed_protected_names() -> set[str]:
    """运行时从种子卡取受保护名单 (不硬编码)。"""
    from darwin_st.knowledge.seeds import all_seed_mechanisms

    return {m.name for m in all_seed_mechanisms()}


@dataclass
class ApplyPlan:
    """确定性动作计划。merges=[(src,tgt)], drops=[name], skipped=[(name,reason)]。"""

    merges: list[tuple[str, str]] = field(default_factory=list)
    drops: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (f"merges={len(self.merges)} drops={len(self.drops)} "
                f"skipped={len(self.skipped)}")


def plan_actions(cards: list[dict], verdicts: list[dict],
                 protected: set[str], empty_precondition_drops: list[str] | None = None) -> ApplyPlan:
    """把裁决编译成确定性动作计划 (纯函数, 不改 cards)。

    规则:
      - merge & src∉protected & tgt 存在 & src≠tgt → merges
      - merge & src∈protected → skipped(种子受保护)
      - merge & tgt 不存在/src==tgt → skipped(无效)
      - drop|split & name∉protected → drops (split 当 drop)
      - drop|split & name∈protected → skipped
      - empty_precondition_drops 里的卡(若∉protected且在库)→ drops(去重)
    name 已不在库 → skipped(幂等)。
    """
    names = {c.get("name") for c in cards}
    plan = ApplyPlan()
    seen_drops: set[str] = set()

    for v in verdicts:
        name = v.get("name", "")
        verdict = v.get("verdict", "keep")
        if name not in names:
            plan.skipped.append((name, "已不在库"))
            continue
        if verdict == "merge":
            tgt = v.get("merge_into", "")
            if name in protected:
                plan.skipped.append((name, "种子卡受保护, 不被合并"))
            elif not tgt or tgt not in names:
                plan.skipped.append((name, f"合并目标 '{tgt}' 不存在"))
            elif tgt == name:
                plan.skipped.append((name, "合并目标即自身"))
            else:
                plan.merges.append((name, tgt))
        elif verdict in ("drop", "split"):
            if name in protected:
                plan.skipped.append((name, "种子卡受保护, 不被删除"))
            elif name not in seen_drops:
                plan.drops.append(name)
                seen_drops.add(name)
        # keep: 无动作

    # 空前提死卡 (从 rule_flags 来) 并入 drops
    for name in (empty_precondition_drops or []):
        if name in protected:
            plan.skipped.append((name, "种子卡受保护(空前提但不删)"))
        elif name in names and name not in seen_drops:
            plan.drops.append(name)
            seen_drops.add(name)

    return plan


def _merge_card_dict(tgt: dict, src: dict) -> None:
    """把 src 卡的有用信息并入 tgt (dict 层, 等价 extraction._merge_into)。

    并入: preconditions / function_tags / anti_patterns (去重) + evidence (按
    dataset+metric+source 去重) + math_structure (tgt 空才取 src)。**不动 tgt 的
    name/abstract_function/origin_domain/abstraction_level** (保 tgt 身份)。
    """
    for key in ("preconditions", "function_tags", "anti_patterns"):
        merged = list(tgt.get(key, []) or [])
        for x in src.get(key, []) or []:
            if x not in merged:
                merged.append(x)
        tgt[key] = merged
    have = {(e.get("dataset"), e.get("metric"), e.get("source"))
            for e in (tgt.get("evidence") or [])}
    ev = list(tgt.get("evidence", []) or [])
    for e in src.get("evidence", []) or []:
        kkey = (e.get("dataset"), e.get("metric"), e.get("source"))
        if kkey not in have:
            ev.append(e)
            have.add(kkey)
    tgt["evidence"] = ev
    if not (tgt.get("math_structure") or "").strip() and (src.get("math_structure") or "").strip():
        tgt["math_structure"] = src["math_structure"]


def apply_plan(cards: list[dict], plan: ApplyPlan) -> tuple[list[dict], dict]:
    """执行计划。返回 (新卡列表, 统计)。不改入参 cards (返回新列表)。"""
    by_name = {c.get("name"): dict(c) for c in cards}  # 浅拷贝每张卡
    merged_n = dropped_n = 0

    # 先 merge: 把源并入目标, 标记源待删
    to_remove: set[str] = set()
    for src, tgt in plan.merges:
        if src in by_name and tgt in by_name:
            _merge_card_dict(by_name[tgt], by_name[src])
            to_remove.add(src)
            merged_n += 1

    # 再 drop
    for name in plan.drops:
        if name in by_name:
            to_remove.add(name)
            dropped_n += 1

    # 用 by_name (含 merge 后的修改) 重建, 保原序; 剔除待删
    new_cards = [by_name[c.get("name")] for c in cards if c.get("name") not in to_remove]
    stats = {
        "before": len(cards),
        "after": len(new_cards),
        "merged": merged_n,
        "dropped": dropped_n,
        "skipped": len(plan.skipped),
    }
    return new_cards, stats
