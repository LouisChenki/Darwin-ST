"""反思巩固环节 (Reflection Consolidation) —— 语义记忆层。

把"没人读的 insights 表"复活为真正的语义记忆层。三个机制的混合:
  - Generative Agents (arXiv:2304.03442) 反思: 攒够 N 条新履历才触发一次 LLM 反思
    (不是每轮都反思 —— 低频才有信噪比, 且省 token), 从结构化记录蒸馏高层教训;
  - ExpeL (arXiv:2308.10144) 集合维护: 教训库不是只增不改的流水账, 每个操作是
    ADD/EDIT/UPVOTE/DOWNVOTE 之一, 相近教训合并 (EDIT) 而非无限新增, 被反复验证的
    教训升权, 被证伪的降权直至删除;
  - Hermes 容量上限: insights 表有硬容量, 超上限先按 confidence×recency 淘汰再新增,
    保证注入 prompt 的永远是最可信的一小撮。

数据流 (全部走 store 的 SQL, LLM 只产操作 JSON, 绝不直接碰 db):
  履历本新增 N 条 → build_reflection_prompt (新记录+墓地死因+现有 insights+方向轨迹)
  → llm.chat (temperature 0.3 低随机) → parse_reflection_ops (容错解析 JSON 数组)
  → validate_ops (证据/引用/去重校验, 相似 ADD 改判 EDIT) → apply_ops (落库+置信度维护)
  → 超容量淘汰 → render_insights_block 注入下游 prompt (synthesizer/diagnosis)。

纯函数核 (build/parse/validate/render) 无 LLM/DB 依赖可全测; ReflectionLoop 是薄编排壳,
所有外部依赖 (store/archive/llm) 注入 → MockLLM 可端到端测。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from darwin_st.memory.reflect import summarize_graveyard

__all__ = [
    "ReflectionConfig",
    "ReflectionLoop",
    "build_reflection_prompt",
    "parse_reflection_ops",
    "validate_ops",
    "apply_ops",
    "render_insights_block",
    "load_direction_history",
]

VALID_OPS = ("ADD", "EDIT", "UPVOTE", "DOWNVOTE")


@dataclass
class ReflectionConfig:
    """反思环节全部拍板参数 (构造可覆盖)。默认值见各字段注释。"""

    reflect_every_records: int = 20   # 自上次反思起履历本新增 N 条才触发 (行数游标)
    insights_capacity: int = 50       # insights 表硬容量: 超上限时本批先巩固/淘汰再新增
    max_new_lessons: int = 5          # 每批新增教训 (ADD) 上限
    conf_init: float = 0.5            # ADD 初始置信度
    conf_up: float = 0.10             # UPVOTE 步进
    conf_cap: float = 0.95            # 置信度上限 (封顶)
    conf_down: float = 0.15           # DOWNVOTE 步进 (罚得比奖得重, 防坏账滞留)
    conf_delete_below: float = 0.30   # DOWNVOTE 后跌破此线即删行
    prompt_top_n: int = 3             # 注入 prompt 的教训条数
    lesson_max_chars: int = 120       # 单条教训渲染上限
    prompt_max_chars: int = 500       # 经验块总长上限 (超再截尾)
    max_tokens: int = 16384           # 反思 LLM 输出预算 (推理模型按 reasoning+输出总量给; 空响应自动加倍
                                      # 到 32768。实测 API 接受 65536(flash), 余额充足给足 —— 不再省 token)
    max_existing_insights: int = 15   # 反思输入展示的现有教训上限 (实测 36 条近似旧教训诱发推理模型
                                      # 24k reasoning 空转 (finish_reason=length, content 空), 故截断+从略注记)


# ---------------------------------------------------------------------------
# prompt 组装 (纯函数)
# ---------------------------------------------------------------------------

def _build_system(cfg: ReflectionConfig) -> str:
    """system: 任务定义 + 操作规范。无 persona (同 diagnosis 的文献结论)。"""
    return (
        "你在为一个时空预测架构搜索系统维护【经验教训库】。"
        "给你一批最近的创造履历 (结构化记录)、失败死因统计、现有教训和历史诊断方向轨迹, "
        "请把这些原始记录【反思蒸馏】成可复用的条件式教训, 并以操作列表的形式维护教训库。\n\n"
        "只输出一个 JSON 数组 (不要其它文字), 每个元素是一个操作对象:\n"
        '{"op": "ADD"|"EDIT"|"UPVOTE"|"DOWNVOTE", "condition": "条件式前提", '
        '"text": "条件→教训", "evidence_ids": [履历行号...], '
        '"insight_id": <EDIT/UPVOTE/DOWNVOTE 必填>}\n\n'
        "硬约束:\n"
        "- 只基于给定的结构化记录提炼, 不得引入记录之外的知识或臆造数字。\n"
        "- 每条教训必须挂证据: evidence_ids 非空, 且只能引用给定履历的行号。\n"
        "- 教训必须是条件式 (明确何种条件下成立/该怎么做), 禁止无条件空话 "
        "(如「要注意稳定性」这类放之四海皆准的废话)。\n"
        f"- 每批最多 {cfg.max_new_lessons} 条 ADD; 与现有教训相近的优先 EDIT 而非 ADD。\n"
        "- 现有教训被新证据再次验证 → UPVOTE; 被新证据证伪 → DOWNVOTE; "
        "表述不准/需并入新证据 → EDIT。\n"
        "- EDIT/UPVOTE/DOWNVOTE 必须填 insight_id (现有教训的编号)。"
    )


def _build_user(new_records: list[dict], graveyard_summary: str,
                existing_insights: list[dict], direction_history: list[dict],
                cfg: ReflectionConfig) -> str:
    """user: 分块小标题; 新记录带行号 (证据 id 就是这些行号)。"""
    rec_lines = []
    for r in new_records:
        seed = r.get("seed_val_mae")
        seed_s = f"实测MAE={seed:.4f}" if isinstance(seed, (int, float)) else "未评测"
        err = (r.get("error") or "")[:200]
        err_s = f" | 错误: {err}" if err else ""
        adopted = r.get("adopted")
        adopted_s = {True: "KEEP", False: "DISCARD/CRASH"}.get(adopted, "未回填")
        rec_lines.append(
            f"- 行#{r.get('row_id')} | 轮{r.get('round_idx')} | {r.get('operator_name') or '(未命名)'}"
            f" | 组合={r.get('composition') or '?'} | 门={r.get('gate')}"
            f" | proxy={r.get('proxy_mae')} | {seed_s}({adopted_s})"
            f" | 瓶颈: {r.get('bottleneck', '')}"
            f" | 机制: {','.join(r.get('mechanisms') or [])}{err_s}")
    rec_block = "\n".join(rec_lines) or "(无新记录)"

    if existing_insights:
        # 截断展示 (近似旧教训过多会诱发推理模型逐条比对空转, 见 cfg.max_existing_insights 注释)
        shown = existing_insights[: cfg.max_existing_insights]
        ins_lines = []
        for ins in shown:
            ev = ",".join(str(e) for e in (ins.get("evidence_ids") or []))
            ins_lines.append(
                f"- insight_id={ins.get('id')} | 置信度={ins.get('confidence', 0):.2f}"
                f" | 条件: {ins.get('condition') or '(无)'}"
                f" | {ins.get('insight_text', '')} | 证据: [{ev}]")
        omitted = len(existing_insights) - len(shown)
        if omitted > 0:
            ins_lines.append(f"- ...(另有 {omitted} 条旧教训从略, 不必逐条复核; 可经 ADD/EDIT 触及相同主题)")
        ins_block = "\n".join(ins_lines)
    else:
        ins_block = "(教训库为空, 本批只能 ADD)"

    hist_lines = []
    for h in direction_history[-10:]:
        mae = h.get("result_mae")
        mae_s = f"MAE≈{mae:.2f}" if isinstance(mae, (int, float)) else "MAE未知"
        precs = ", ".join(h.get("preconditions") or [])
        hist_lines.append(
            f"- 轮{h.get('round')}: 方向[{precs or h.get('bottleneck', '')}] → {mae_s}"
            f" | 信用分={h.get('score', 0.0):g}")
    hist_block = "\n".join(hist_lines) or "(无方向轨迹)"

    return (
        f"### 新增创造履历 (行号即可引用为 evidence_ids)\n{rec_block}\n\n"
        f"### 失败死因统计 (墓地聚合)\n{graveyard_summary or '(无失败记录)'}\n\n"
        f"### 现有教训 (可 EDIT/UPVOTE/DOWNVOTE, 引用 insight_id)\n{ins_block}\n\n"
        f"### 历史诊断方向轨迹\n{hist_block}\n\n"
        f"### 输出 (严格 JSON 数组, 无其它文字)\n"
        '[{"op":"ADD","condition":"...","text":"...","evidence_ids":[行号...]}, ...]'
    )


def build_reflection_prompt(new_records: list[dict], graveyard_summary: str,
                            existing_insights: list[dict], direction_history: list[dict],
                            cfg: ReflectionConfig) -> list[dict]:
    """组装反思 prompt (system+user)。纯函数, 无 LLM/DB 依赖。"""
    return [{"role": "system", "content": _build_system(cfg)},
            {"role": "user", "content": _build_user(new_records, graveyard_summary,
                                                    existing_insights, direction_history, cfg)}]


# ---------------------------------------------------------------------------
# 解析 + 校验 (纯函数)
# ---------------------------------------------------------------------------

def parse_reflection_ops(text: str) -> list[dict]:
    """容错解析 LLM 输出为操作列表。复用 synthesizer.extract_json_array 的思路
    (优先抓 [...] 数组, 退化为单 {..} 对象包列表), 但先整段 json.loads,
    且数组匹配只命中内嵌字段数组 (如 "evidence_ids":[3]) 时继续退化到对象匹配。
    非 dict 元素在此过滤。完全找不到 → ValueError。
    """
    def _as_ops(obj) -> list[dict] | None:
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [op for op in obj if isinstance(op, dict)]
        return None

    try:                                       # 干净输出: 整段即 JSON (空数组 = 合法零操作)
        ops = _as_ops(json.loads(text))
        if ops is not None:
            return ops
    except Exception:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)  # 主格式: JSON 数组 (贪婪抓最外层)
    if m:
        try:
            ops = _as_ops(json.loads(m.group(0)))
            if ops:                            # 无 dict 元素 = 误中内嵌字段数组, 继续退化
                return ops
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)  # 退化: 单对象 (含内嵌数组字段)
    if m:
        try:
            ops = _as_ops(json.loads(m.group(0)))
            if ops is not None:
                return ops
        except Exception:
            pass
    raise ValueError("响应中未找到 JSON 数组或对象")


def _normalize_text(s: str) -> str:
    """文本规范化 (去空白/标点, 小写) —— 相似判定的第一步。"""
    return re.sub(r"[\s\W_]+", "", (s or "").lower())


def _text_tokens(s: str) -> set[str]:
    """文本词集: 拉丁词干 + CJK 单字 (中文教训无空格分词, 单字重叠已够判相似)。"""
    s = (s or "").lower()
    toks = set(re.findall(r"[a-z0-9]+", s))
    toks |= set(re.findall(r"[一-鿿]", s))
    return toks


def _is_similar(a: str, b: str) -> bool:
    """两条教训文本是否高度相似: 规范化后互为子串, 或词集 Jaccard ≥ 0.6。"""
    na, nb = _normalize_text(a), _normalize_text(b)
    if na and nb and (na in nb or nb in na):
        return True
    ta, tb = _text_tokens(a), _text_tokens(b)
    if not ta or not tb:
        return False
    j = len(ta & tb) / len(ta | tb)
    return j >= 0.6


def _clip_conf(x: float) -> float:
    """confidence 裁剪到 [0, 1]。"""
    return min(1.0, max(0.0, x))


def _parse_evidence(raw, valid_record_ids: set[int]) -> list[int] | None:
    """evidence_ids 归一化为 int 列表并校验 ⊆ valid_record_ids。非法返回 None。"""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[int] = []
    for e in raw:
        if isinstance(e, bool):
            return None
        if isinstance(e, float):
            if not float(e).is_integer():
                return None
            e = int(e)
        if not isinstance(e, int) or e not in valid_record_ids:
            return None
        out.append(e)
    return out


def validate_ops(ops: list[dict], valid_record_ids: set[int],
                 existing_insights: list[dict], cfg: ReflectionConfig
                 ) -> tuple[list[dict], list[dict]]:
    """校验 LLM 操作列表, 返回 (accepted, rejected)。拒因随 rejected 返回。

    规则:
      - op ∈ {ADD, EDIT, UPVOTE, DOWNVOTE}, 否则拒;
      - evidence_ids 必须非空且 ⊆ valid_record_ids (空/越界即拒) —— 无证据的教训不收;
      - EDIT/UPVOTE/DOWNVOTE 的 insight_id 必须存在于现有教训, 否则拒;
      - ADD 与现有教训文本高度相似 (规范化子串 / 词集重叠) → 强制改判 EDIT
        (insight_id 取最相似者, 新文本与原证据并入);
      - ADD 条数超 cfg.max_new_lessons → 溢出拒 (收前 N 条, prompt 已声明上限);
      - ADD/EDIT 的 text 必须非空; confidence 一律裁剪 [0, 1]。
    """
    valid_ids = {int(i) for i in valid_record_ids}
    by_id = {int(ins["id"]): ins for ins in existing_insights}
    accepted: list[dict] = []
    rejected: list[dict] = []
    n_add = 0

    for op in ops:
        if not isinstance(op, dict):
            rejected.append({"op": op, "reason": "操作不是 JSON 对象"})
            continue
        action = str(op.get("op", "")).upper()
        if action not in VALID_OPS:
            rejected.append({"op": op, "reason": f"非法 op: {op.get('op')!r}"})
            continue

        evidence = _parse_evidence(op.get("evidence_ids"), valid_ids)
        if evidence is None:
            rejected.append({"op": op, "reason": "evidence_ids 含未知/非法履历行号"})
            continue
        if not evidence:
            rejected.append({"op": op, "reason": "evidence_ids 为空 (无证据不收)"})
            continue

        clean = {"op": action, "evidence_ids": evidence}

        if action == "ADD":
            text = str(op.get("text") or "").strip()
            if not text:
                rejected.append({"op": op, "reason": "ADD 缺少 text"})
                continue
            if n_add >= cfg.max_new_lessons:
                rejected.append({"op": op, "reason": f"超过每批 ADD 上限 ({cfg.max_new_lessons})"})
                continue
            # LLM 可选带 confidence → 裁剪 [0, 1]; 缺省走 cfg.conf_init (apply 时再裁)
            conf_raw = op.get("confidence")
            if isinstance(conf_raw, (int, float)) and not isinstance(conf_raw, bool):
                clean["confidence"] = _clip_conf(float(conf_raw))
            # 相似改判 EDIT: 取规范化重叠度最高/首个命中的现有教训
            similar = next((ins for ins in existing_insights
                            if _is_similar(text, str(ins.get("insight_text") or ""))), None)
            if similar is not None:
                clean["op"] = "EDIT"
                clean["insight_id"] = int(similar["id"])
                clean["text"] = text
                clean["condition"] = str(op.get("condition") or similar.get("condition") or "")
                clean["_auto_edit"] = True     # 标记: 相似强制改判 (观测/日志用)
            else:
                n_add += 1
                clean["text"] = text
                clean["condition"] = str(op.get("condition") or "")
            accepted.append(clean)
            continue

        # EDIT / UPVOTE / DOWNVOTE: 必须引用存在的 insight_id
        raw_id = op.get("insight_id")
        if isinstance(raw_id, float) and float(raw_id).is_integer():
            raw_id = int(raw_id)
        if not isinstance(raw_id, int) or isinstance(raw_id, bool) or raw_id not in by_id:
            rejected.append({"op": op, "reason": f"insight_id 缺失或不存在: {op.get('insight_id')!r}"})
            continue
        clean["insight_id"] = raw_id
        if action == "EDIT":
            text = str(op.get("text") or "").strip()
            if not text:
                rejected.append({"op": op, "reason": "EDIT 缺少 text"})
                continue
            clean["text"] = text
            clean["condition"] = str(op.get("condition") or by_id[raw_id].get("condition") or "")
        accepted.append(clean)

    return accepted, rejected


# ---------------------------------------------------------------------------
# 落库 (纯 SQL, 经 store; 无 LLM)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_evidence(old: list, new: list[int]) -> list[int]:
    """并入证据 id 去重保序 (旧在前保 recency)。"""
    out = list(old or [])
    for e in new:
        if e not in out:
            out.append(e)
    return out


def _fetch_insight(store, insight_id: int) -> dict | None:
    row = store.conn.execute(
        "SELECT id, dataset, condition, insight_text, evidence_ids, confidence, created_at "
        "FROM insights WHERE id=?", (insight_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["evidence_ids"] = json.loads(d["evidence_ids"])
    return d


def apply_ops(store, accepted: list[dict], cfg: ReflectionConfig) -> dict:
    """把校验通过的操作落到 insights 表 (全经 store 的 SQL, LLM 不碰 db)。

    ADD=insert (conf_init); EDIT=更新文本/条件+并入 evidence_ids;
    UPVOTE=conf+conf_up 封顶 conf_cap (并入新 evidence_ids);
    DOWNVOTE=conf−conf_down, 跌破 conf_delete_below 即删行。
    返回 {added, edited, upvoted, downvoted, deleted} 计数。
    """
    stats = {"added": 0, "edited": 0, "upvoted": 0, "downvoted": 0, "deleted": 0}
    for op in accepted:
        action = op["op"]
        if action == "ADD":
            store.add_insight(
                insight_text=op["text"], created_at=_now_iso(),
                dataset=None, condition=op.get("condition") or None,
                evidence_ids=op["evidence_ids"],
                confidence=op.get("confidence", _clip_conf(cfg.conf_init)))
            stats["added"] += 1
            continue

        ins = _fetch_insight(store, op["insight_id"])
        if ins is None:                      # 并发被删 → 跳过 (不崩)
            continue
        if action == "EDIT":
            store.update_insight(
                ins["id"], text=op["text"], condition=op.get("condition") or ins["condition"],
                evidence_ids=_merge_evidence(ins["evidence_ids"], op["evidence_ids"]))
            stats["edited"] += 1
        elif action == "UPVOTE":
            store.update_insight(
                ins["id"], confidence=min(ins["confidence"] + cfg.conf_up, cfg.conf_cap),
                evidence_ids=_merge_evidence(ins["evidence_ids"], op["evidence_ids"]))
            stats["upvoted"] += 1
        elif action == "DOWNVOTE":
            new_conf = _clip_conf(ins["confidence"] - cfg.conf_down)
            if new_conf < cfg.conf_delete_below:
                store.delete_insight(ins["id"])
                stats["deleted"] += 1
            else:
                store.update_insight(ins["id"], confidence=new_conf)
                stats["downvoted"] += 1
    return stats


# ---------------------------------------------------------------------------
# prompt 注入渲染 (纯函数)
# ---------------------------------------------------------------------------

def render_insights_block(insights: list[dict], cfg: ReflectionConfig) -> str:
    """把 top 教训渲染成 prompt 经验块。按 confidence 降序 (并列取新) 取 prompt_top_n 条。

    格式:
        【历史经验(反思蒸馏)】
        1. [0.85] 条件X → 教训Y (依据 #id1,#id2)
    单条教训文本截 lesson_max_chars; 总块超 prompt_max_chars 再截尾。
    空列表返回 "" —— 调用方拼 prompt 时空串=不注入, 保证旧 prompt 逐字节不变。
    """
    if not insights:
        return ""
    ordered = sorted(insights,
                     key=lambda i: (i.get("confidence", 0.0), i.get("id", 0)), reverse=True)
    lines = ["【历史经验(反思蒸馏)】"]
    for rank, ins in enumerate(ordered[: cfg.prompt_top_n], 1):
        text = str(ins.get("insight_text") or "")[: cfg.lesson_max_chars]
        cond = str(ins.get("condition") or "通用")
        ev = ",".join(f"#{e}" for e in (ins.get("evidence_ids") or []))
        ev_s = f" (依据 {ev})" if ev else ""
        lines.append(f"{rank}. [{ins.get('confidence', 0.0):.2f}] {cond} → {text}{ev_s}")
    block = "\n".join(lines)
    if len(block) > cfg.prompt_max_chars:
        block = block[: cfg.prompt_max_chars].rstrip() + "…"
    return block


# ---------------------------------------------------------------------------
# 方向轨迹读取 (creation_loop direction_state.json, 二选一的读文件分支)
# ---------------------------------------------------------------------------

def load_direction_history(state_path: str | None) -> list[dict]:
    """读 creation_loop 的 direction_state.json 取 history 列表。文件缺/坏 → []。"""
    if not state_path or not os.path.exists(state_path):
        return []
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
        history = data.get("history")
        return [h for h in history if isinstance(h, dict)] if isinstance(history, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 编排壳
# ---------------------------------------------------------------------------

class ReflectionLoop:
    """反思巩固编排: 攒够新履历 → LLM 反思 → 校验 → 落库 → 容量维护。

    store:   MemoryStore (insights 表读写, 墓地聚合);
    archive: CreationArchive (履历本, 按行数游标取增量);
    llm:     LLMClient (MockLLM 可测); temperature 0.3 低随机;
    direction_history: 可选注入 (creation_loop._history); 未注入则读 direction_state.json
    (与履历本同目录, 二选一); 都没有 → 空轨迹。
    cursor: 行数游标起点; None (默认) = 构造时快照当前履历总数, 只对增量反思 (在线语义);
    显式给 0 → 从头全量反思 (离线 bootstrap 用, 见 scripts/reflect_consolidate.py)。
    """

    def __init__(self, store, archive, llm, cfg: ReflectionConfig | None = None,
                 log_fn=print, direction_history: list[dict] | None = None,
                 direction_state_path: str | None = None, cursor: int | None = None):
        self.store = store
        self.archive = archive
        self.llm = llm
        self.cfg = cfg or ReflectionConfig()
        self.log_fn = log_fn
        self._direction_history = direction_history
        if direction_state_path is not None:
            self._direction_state_path = direction_state_path
        elif archive is not None:
            self._direction_state_path = os.path.join(
                os.path.dirname(os.path.abspath(archive.path)), "direction_state.json")
        else:
            self._direction_state_path = None
        # 行数游标: None → 启动快照 (只对增量反思); 显式值 → 从该位置起
        self._cursor = self._count_records() if cursor is None else cursor

    # ------------------------------------------------------------------

    def _count_records(self) -> int:
        try:
            return len(self.archive._load()) if self.archive is not None else 0
        except Exception:
            return 0

    def _load_records(self) -> list[dict]:
        if self.archive is None:
            return []
        from dataclasses import asdict
        try:
            return [asdict(r) for r in self.archive._load()]
        except Exception:
            return []

    def _get_direction_history(self) -> list[dict]:
        if self._direction_history is not None:
            return list(self._direction_history)
        return load_direction_history(self._direction_state_path)

    def _graveyard_text(self) -> str:
        try:
            counts = summarize_graveyard(self.store)
        except Exception:
            return ""
        return "; ".join(f"{reason}×{n}" for reason, n in counts.items())

    # ------------------------------------------------------------------

    def maybe_reflect(self, force: bool = False) -> dict | None:
        """攒够 cfg.reflect_every_records 条新履历 (或 force) 就反思一批, 否则返回 None。

        流程: 取增量记录 (带行号) → 墓地摘要 + 现有 insights + 方向轨迹 → prompt →
        llm.chat(temperature=0.3) → parse → validate → apply_ops → 容量淘汰 → 游标推进。
        LLM/解析异常返回 {"error": ...} 不抛出 (反思是低频增益, 绝不中止进化)。
        解析/LLM 失败时游标不推进, 下批重试 (记录不丢)。
        """
        total = self._count_records()
        n_new = total - self._cursor
        if not force and n_new < self.cfg.reflect_every_records:
            return None
        if n_new <= 0:
            return None

        all_records = self._load_records()
        new_records = []
        for idx, rec in enumerate(all_records[self._cursor:], start=self._cursor + 1):
            d = dict(rec)
            d["row_id"] = idx                 # 行号 (1 起) = LLM 可引用的证据 id
            new_records.append(d)
        valid_ids = {r["row_id"] for r in new_records}

        existing = self.store.list_insights_by_confidence()
        prompt = build_reflection_prompt(new_records, self._graveyard_text(), existing,
                                         self._get_direction_history(), self.cfg)
        try:
            # 推理模型 (deepseek-v4 系): reasoning trace 先吃 token 才吐 JSON,
            # 预算按 reasoning+输出总量给 (诊断层同款教训, commit 408a4d3); 空响应加倍重试一次
            budget = self.cfg.max_tokens
            text = ""
            for _attempt in range(2):
                text = self.llm.chat(prompt, temperature=0.3, max_tokens=budget)
                if text and text.strip():
                    break
                self.log_fn(f"[反思] LLM 空响应 (max_tokens={budget}), 加倍重试")
                budget *= 2
        except Exception as e:
            self.log_fn(f"[反思] LLM 调用异常, 本批跳过 (游标不动, 下批重试): "
                        f"{type(e).__name__}: {str(e)[:120]}")
            return {"error": f"llm: {type(e).__name__}: {str(e)[:200]}",
                    "n_new_records": n_new}
        try:
            ops = parse_reflection_ops(text)
        except Exception as e:
            self.log_fn(f"[反思] 输出解析失败, 本批跳过 (游标不动, 下批重试): "
                        f"{type(e).__name__}: {str(e)[:120]}")
            return {"error": f"parse: {type(e).__name__}: {str(e)[:200]}",
                    "n_new_records": n_new}

        accepted, rejected = validate_ops(ops, valid_ids, existing, self.cfg)
        apply_stats = apply_ops(self.store, accepted, self.cfg)
        evicted = self._enforce_capacity()
        self._cursor = total                 # 成功批才推进游标

        report = {
            "n_new_records": n_new,
            "n_ops": len(ops),
            "accepted": len(accepted),
            "rejected": [{"op": r.get("op"), "reason": r.get("reason")} for r in rejected],
            "apply": apply_stats,
            "evicted": evicted,
            "insights_total": len(self.store.list_insights_by_confidence()),
        }
        self.log_fn(f"[反思] 新履历 {n_new} 条 → 操作 {len(ops)} 个: "
                    f"接受 {len(accepted)} (新增{apply_stats['added']}/改{apply_stats['edited']}"
                    f"/赞{apply_stats['upvoted']}/踩{apply_stats['downvoted']}"
                    f"/删{apply_stats['deleted']}), 拒绝 {len(rejected)}"
                    + (f", 容量淘汰 {evicted}" if evicted else "")
                    + f", 库现存 {report['insights_total']}")
        return report

    def _enforce_capacity(self) -> int:
        """Hermes 容量上限: 超 insights_capacity 按 confidence×recency 淘汰, 返回淘汰数。

        recency = 最新证据行号 (履历本序, 越大越新; 无证据退化为 insight id)。
        按 (confidence, recency) 升序删到容量内 —— 最低信且最旧的先走。
        """
        insights = self.store.list_insights_by_confidence()
        overflow = len(insights) - self.cfg.insights_capacity
        if overflow <= 0:
            return 0

        def _recency(ins: dict) -> float:
            ev = [e for e in (ins.get("evidence_ids") or []) if isinstance(e, (int, float))]
            return float(max(ev)) if ev else float(ins.get("id", 0))

        victims = sorted(insights, key=lambda i: (i.get("confidence", 0.0), _recency(i)))
        n = 0
        for ins in victims[:overflow]:
            if self.store.delete_insight(int(ins["id"])):
                n += 1
        return n
