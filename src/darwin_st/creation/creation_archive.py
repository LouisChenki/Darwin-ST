"""创造档案 (Creation Archive) —— B1 履历本回读 (ADAS 式)。

治"每轮创造失忆重启": 把每次创造的方案/代码/成败记成 jsonl 履历本, 下一轮合成时
把【成功先例 + 失败教训】喂回 LLM prompt (见 synthesizer.build_plan_many_prompt 的
exemplars 注入)。

三个接线点:
  - 写入: CreationLoop.maybe_create 对每个 SynthesisResult 无论成败 record()。
  - 读取: CreationLoop 下一轮合成前 exemplar_context() → synthesize_many(exemplars=...)。
  - 回填: Orchestrator._digest 把创造 seed 的实测 MAE/结局 update_outcome() 回写。

纯逻辑模块 (jsonl 持久化, 原子重写), 无 LLM/torch 依赖, 可独立全测。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields

__all__ = ["CreationRecord", "CreationArchive"]


@dataclass
class CreationRecord:
    """一条创造履历 (一次融合假设的记录, 成败都记)。

    gate: "all" = 合成验证全门通过 (成功); 否则为失败门名 (shape/plan/unknown 等)。
    seed_val_mae / adopted 为回填字段: 创造 seed 真实评测后由 orchestrator 回写
    (None = 尚未评测 / 评测无效)。
    """

    round_idx: int                       # 第几轮创造 (CreationLoop._round_idx)
    created_at: str                      # ISO8601 时间戳
    bottleneck: str = ""                 # 当时诊断出的瓶颈
    preconditions: list[str] = field(default_factory=list)   # 检索目标前提
    mechanisms: list[str] = field(default_factory=list)      # 参与融合的跨域机制名
    operator_name: str = ""              # 成功=注册名(synth_*); 失败=计划名或空
    composition: str = ""                # 4 组合文法之一
    rationale: str = ""                  # 融合理由 (plan)
    code: str = ""                       # 成功算子的实现代码 (失败为空)
    gate: str = "unknown"                # "all" 或失败门名
    error: str | None = None             # 失败时的 last_error
    seed_signature: str | None = None    # 待评 genotype 签名 (回填识别用)
    seed_val_mae: float | None = None    # 回填: seed 实测 MAE
    adopted: bool | None = None          # 回填: seed 评测结局 (KEEP=True)
    proxy_mae: float | None = None       # B3 proxy 粗筛短训分 (未走 proxy 路径/旧履历行为 None)
    proxy_error: str | None = None       # B3 proxy 失败原因 (异常摘要或 "inf"; 成功/未走 proxy 为 None)
    # v2 动作账本关联 (可选; 旧履历行为 None —— 动作粒度统计自 v2 起可用, 不追溯换算)
    action_id: str | None = None         # 所属 Tier-2 动作 (一次动作多假设共用一个 id)
    action_seq: int | None = None        # 动作序号 (顺序权威, 见 action_ledger)
    run_tag: str | None = None           # 产生该履历的运行 (v1/v2 共用 archive 时区分来源)
    # v2 精炼提出时刻记录 (F3 熔断判据; 提出时刻的父版本信息, 防时间穿越)
    refine_family: str | None = None     # 精炼目标族 (非精炼为 None)
    parent_operator: str | None = None   # 父版本注册名
    parent_mae_at_proposal: float | None = None  # 提出时刻父版本实测 MAE


class CreationArchive:
    """创造履历的 jsonl 档案。小文件, 整体读/原子写。"""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # 写入 / 回填
    # ------------------------------------------------------------------

    def record(self, rec: CreationRecord) -> None:
        """追加一条履历 (jsonl 一行)。"""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

    def update_outcome(self, operator_name: str, seed_signature: str | None,
                       seed_val_mae: float | None, adopted: bool) -> bool:
        """按 operator_name 找最近一条成功记录, 回填实测结局。返回是否命中。

        小文件整体重写, 原子写 (临时文件 + rename), 防中途崩溃留半文件。
        """
        records = self._load()
        for rec in reversed(records):          # 最近一条优先
            if rec.gate == "all" and rec.operator_name == operator_name:
                rec.seed_signature = seed_signature
                rec.seed_val_mae = seed_val_mae
                rec.adopted = adopted
                self._rewrite(records)
                return True
        return False

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------

    def top_successes(self, k: int) -> list[CreationRecord]:
        """成功履历 top-k: 有实测 MAE 的按 MAE 升序; 无 MAE 的按时间近者优先、排后。"""
        succ = [r for r in self._load() if r.gate == "all"]
        with_mae = [r for r in succ if _finite(r.seed_val_mae)]
        without_mae = [r for r in succ if not _finite(r.seed_val_mae)]
        with_mae.sort(key=lambda r: r.seed_val_mae)
        without_mae.sort(key=lambda r: r.created_at, reverse=True)
        return (with_mae + without_mae)[:k]

    def recent_failures(self, k: int) -> list[CreationRecord]:
        """失败履历 top-k: 按时间倒序 (最近教训优先)。"""
        fails = [r for r in self._load() if r.gate != "all"]
        fails.sort(key=lambda r: r.created_at, reverse=True)
        return fails[:k]

    def exemplar_context(self, k_success: int = 4, k_failure: int = 3,
                         max_code_lines: int = 60) -> dict:
        """prompt 就绪的履历上下文: 成功先例 (含截断代码+实测 MAE) + 失败教训。"""
        successes = []
        for r in self.top_successes(k_success):
            successes.append({
                "operator_name": r.operator_name,
                "composition": r.composition,
                "rationale": r.rationale,
                "seed_val_mae": r.seed_val_mae,
                "code": _truncate_code(r.code, max_code_lines),
            })
        failures = []
        for r in self.recent_failures(k_failure):
            failures.append({
                "operator_name": r.operator_name,
                "composition": r.composition,
                "rationale": r.rationale,
                "gate": r.gate,
                "error": (r.error or "")[:300],     # 错误摘要截 300 字
            })
        return {"successes": successes, "failures": failures}

    def is_empty(self) -> bool:
        return not self._load()

    # ------------------------------------------------------------------
    # 内部: 读 / 写
    # ------------------------------------------------------------------

    def _load(self) -> list[CreationRecord]:
        """整体读出全部履历。文件不存在 = 空档案; 坏行跳过 (容错不崩)。"""
        if not os.path.exists(self.path):
            return []
        known = {f.name for f in fields(CreationRecord)}
        records: list[CreationRecord] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    records.append(CreationRecord(**{k: v for k, v in d.items() if k in known}))
                except Exception:
                    continue            # 坏行跳过
        return records

    def _rewrite(self, records: list[CreationRecord]) -> None:
        """整体重写 (原子: 同目录临时文件 + os.replace)。"""
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _truncate_code(code: str, max_lines: int) -> str:
    """代码截断到 max_lines 行并标注 (防履历代码撑爆 prompt)。"""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    return "\n".join(lines[:max_lines]) + f"\n# ... [截断] 共 {len(lines)} 行, 仅示前 {max_lines} 行"
