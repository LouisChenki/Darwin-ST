"""结构化试验记忆 (Experiment Memory Store) —— SQLite 读写接口。

把过往优化经验结构化, 让演化系统「记得住、查得到、绕得开」:
  - record_trial:     落库一次完整试验 (基因型/超参/多seed指标/状态/死因/谱系)
  - best_so_far:      取当前最优 (供「是否超越 SOTA」判定)
  - query_graveyard:  按签名查已知失败 → 硬阻断重复无效优化
  - nearest_experiments: 按行为描述子检索最近邻历史 → 新变异借鉴经验
  - record_lineage / get_children: 谱系关系 (支撑进化式 NAS)
  - get_stats:        KEEP/DISCARD/CRASH 统计 (供 analysis 与监控)

设计 (见 docs/BLUEPRINT.md §3):
  - signature = genotype+hp 的稳定哈希 (排序键后 JSON), 用于 Graveyard 秒级查重
  - 纯 stdlib sqlite3, 零额外依赖; 时间戳由调用方传入 (确定性, 不用 SQLite 时间函数)
  - 多 worker 并发: WAL 模式 + 短事务
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Any

__all__ = ["Trial", "MemoryStore", "compute_signature"]

_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

VALID_STATUS = {"KEEP", "DISCARD", "CRASH", "PRUNED"}


def compute_signature(genotype: dict, hp: dict | None = None) -> str:
    """genotype+hp 的稳定哈希 (查重键)。

    对 dict 做有序 JSON 序列化再 sha1, 保证「内容相同 → 签名相同」(键序无关)。
    """
    payload = {"genotype": genotype, "hp": hp or {}}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


@dataclass
class Trial:
    """一次试验的输入记录 (落库前的结构化载体)。"""

    run_tag: str
    dataset: str
    genotype: dict
    status: str                                  # KEEP / DISCARD / CRASH / PRUNED
    created_at: str                              # ISO8601, 由调用方传入
    hp: dict = field(default_factory=dict)

    val_mae: float | None = None
    val_mae_std: float | None = None
    val_rmse: float | None = None
    val_mape: float | None = None
    test_mae: float | None = None
    num_seeds: int = 1

    fail_reason: str | None = None
    behavior_descriptor: dict = field(default_factory=dict)

    num_params: int | None = None
    wall_seconds: float | None = None
    peak_mem_gb: float | None = None

    parent_id: int | None = None
    commit_hash: str | None = None

    def __post_init__(self) -> None:
        if self.status not in VALID_STATUS:
            raise ValueError(f"status 非法: {self.status} (应为 {VALID_STATUS})")

    @property
    def signature(self) -> str:
        return compute_signature(self.genotype, self.hp)


class MemoryStore:
    """SQLite 试验记忆库。可用作上下文管理器 (with MemoryStore(path) as m: ...)。"""

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with open(_SCHEMA_PATH, encoding="utf-8") as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    # -- 上下文管理 --
    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.conn.close()

    # -- 写 --
    def record_trial(self, trial: Trial) -> int:
        """落库一次试验, 返回新行 id。若有 parent_id 则自动补一条 lineage 边。"""
        cur = self.conn.execute(
            """
            INSERT INTO experiments (
                run_tag, dataset, signature, genotype_json, hp_json,
                val_mae, val_mae_std, val_rmse, val_mape, test_mae, num_seeds,
                status, fail_reason, behavior_descriptor,
                num_params, wall_seconds, peak_mem_gb, parent_id, commit_hash, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                trial.run_tag, trial.dataset, trial.signature,
                json.dumps(trial.genotype, ensure_ascii=False),
                json.dumps(trial.hp, ensure_ascii=False),
                trial.val_mae, trial.val_mae_std, trial.val_rmse, trial.val_mape,
                trial.test_mae, trial.num_seeds,
                trial.status, trial.fail_reason,
                json.dumps(trial.behavior_descriptor, ensure_ascii=False),
                trial.num_params, trial.wall_seconds, trial.peak_mem_gb,
                trial.parent_id, trial.commit_hash, trial.created_at,
            ),
        )
        new_id = int(cur.lastrowid)
        if trial.parent_id is not None:
            self.conn.execute(
                "INSERT INTO lineage (parent_id, child_id, mutation_op, mutation_detail) VALUES (?,?,?,?)",
                (trial.parent_id, new_id, None, None),
            )
        self.conn.commit()
        return new_id

    def record_lineage(
        self, parent_id: int, child_id: int, mutation_op: str | None = None, mutation_detail: str | None = None
    ) -> None:
        """显式登记一条变异边 (附变异算子与细节)。"""
        self.conn.execute(
            "INSERT INTO lineage (parent_id, child_id, mutation_op, mutation_detail) VALUES (?,?,?,?)",
            (parent_id, child_id, mutation_op, mutation_detail),
        )
        self.conn.commit()

    # -- 读 --
    def get_trial(self, trial_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM experiments WHERE id=?", (trial_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def best_so_far(self, dataset: str, metric: str = "val_mae") -> dict | None:
        """取该数据集 KEEP 实验中指标最小 (最优) 的一条 —— 供超越 SOTA 判定。"""
        if metric not in ("val_mae", "val_rmse", "val_mape", "test_mae"):
            raise ValueError(f"不支持的 metric: {metric}")
        row = self.conn.execute(
            f"""
            SELECT * FROM experiments
            WHERE dataset=? AND status='KEEP' AND {metric} IS NOT NULL
            ORDER BY {metric} ASC LIMIT 1
            """,
            (dataset,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def query_graveyard(self, genotype: dict, hp: dict | None = None) -> dict | None:
        """按签名查是否为已知失败配置 (status in CRASH/DISCARD)。

        命中则返回该失败记录 (供 Agent 跳过, 不浪费算力重跑); 否则 None。
        """
        sig = compute_signature(genotype, hp)
        row = self.conn.execute(
            """
            SELECT * FROM experiments
            WHERE signature=? AND status IN ('CRASH','DISCARD')
            ORDER BY id DESC LIMIT 1
            """,
            (sig,),
        ).fetchone()
        return _row_to_dict(row) if row else None

    def seen_signature(self, genotype: dict, hp: dict | None = None) -> bool:
        """该 genotype+hp 是否被试过 (任意状态)。"""
        sig = compute_signature(genotype, hp)
        row = self.conn.execute(
            "SELECT 1 FROM experiments WHERE signature=? LIMIT 1", (sig,)
        ).fetchone()
        return row is not None

    def nearest_experiments(
        self, behavior_descriptor: dict, dataset: str | None = None, k: int = 5,
        status: str | None = "KEEP",
    ) -> list[dict]:
        """按行为描述子的结构化距离, 检索最近邻历史试验 (默认只看 KEEP 的成功经验)。

        距离 = 描述子键值的简单不匹配/数值差综合 (无外部依赖)。用于让新变异
        借鉴「与我相似的架构当年表现如何」。
        """
        clauses, params = [], []
        if dataset is not None:
            clauses.append("dataset=?")
            params.append(dataset)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(f"SELECT * FROM experiments {where}", params).fetchall()

        scored = []
        for row in rows:
            d = _row_to_dict(row)
            dist = _descriptor_distance(behavior_descriptor, d["behavior_descriptor"])
            scored.append((dist, d))
        scored.sort(key=lambda t: t[0])
        return [d for _, d in scored[:k]]

    def list_trials(
        self, dataset: str | None = None, status: str | None = None,
        run_tag: str | None = None,
    ) -> list[dict]:
        """列出实验行 (供排行榜导出等离线聚合)。可按 dataset/status/run_tag 过滤。

        与 best_so_far 不同: 返回**全部**匹配行 (不取最优), 已解析 genotype/hp。
        """
        clauses, params = [], []
        for col, val in (("dataset", dataset), ("status", status), ("run_tag", run_tag)):
            if val is not None:
                clauses.append(f"{col}=?")
                params.append(val)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM experiments {where} ORDER BY id", params
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_children(self, parent_id: int) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT e.*, l.mutation_op, l.mutation_detail
            FROM lineage l JOIN experiments e ON e.id = l.child_id
            WHERE l.parent_id=?
            """,
            (parent_id,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get_stats(self, dataset: str | None = None) -> dict[str, int]:
        """各结局计数 {KEEP, DISCARD, CRASH, PRUNED, total}。"""
        where, params = ("WHERE dataset=?", (dataset,)) if dataset else ("", ())
        rows = self.conn.execute(
            f"SELECT status, COUNT(*) AS n FROM experiments {where} GROUP BY status", params
        ).fetchall()
        stats = {s: 0 for s in VALID_STATUS}
        for r in rows:
            stats[r["status"]] = r["n"]
        stats["total"] = sum(stats[s] for s in VALID_STATUS)
        return stats

    # -- 洞察 (ExpeL) --
    def add_insight(
        self, insight_text: str, created_at: str, dataset: str | None = None,
        condition: str | None = None, evidence_ids: list[int] | None = None, confidence: float = 0.5,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO insights (dataset, condition, insight_text, evidence_ids, confidence, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (dataset, condition, insight_text, json.dumps(evidence_ids or []), confidence, created_at),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get_insights(self, dataset: str | None = None) -> list[dict]:
        """取适用洞察 (该数据集专属 + 通用), 按置信度降序。"""
        if dataset is None:
            rows = self.conn.execute(
                "SELECT * FROM insights ORDER BY confidence DESC"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM insights WHERE dataset=? OR dataset IS NULL ORDER BY confidence DESC",
                (dataset,),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["evidence_ids"] = json.loads(d["evidence_ids"])
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------

_JSON_FIELDS = {"genotype_json": "genotype", "hp_json": "hp", "behavior_descriptor": "behavior_descriptor"}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """sqlite Row → dict, 并把 JSON 文本字段解析回 Python 对象。"""
    d = dict(row)
    for raw_key, parsed_key in _JSON_FIELDS.items():
        if raw_key in d and d[raw_key] is not None:
            d[parsed_key] = json.loads(d[raw_key])
            if raw_key != parsed_key:
                d.pop(raw_key)
    return d


def _descriptor_distance(a: dict, b: dict) -> float:
    """两个行为描述子的简单结构化距离 (无外部依赖)。

    - 数值键: 归一化绝对差
    - 类别键: 不等记 1
    - 仅一方有的键: 记 1 (惩罚维度缺失)
    距离越小越相似。
    """
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    total = 0.0
    for k in keys:
        if k in a and k in b:
            va, vb = a[k], b[k]
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                denom = max(abs(va), abs(vb), 1.0)
                total += abs(va - vb) / denom
            else:
                total += 0.0 if va == vb else 1.0
        else:
            total += 1.0
    return total / len(keys)
