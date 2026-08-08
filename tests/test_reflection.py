"""reflection.py (反思巩固环节 Reflection Consolidation) 的测试 —— MockLLM 全链, 无 GPU/网络。

核心契约:
  - prompt: 新记录/墓地/现有 insights/方向轨迹四块齐全 + system 操作规范
  - parse: 合法/非法 JSON 容错 (复用 extract_json_array 思路)
  - validate: 未知 evidence 拒 / 空 evidence 拒 / insight_id 缺失或不存在拒 /
    相似 ADD 改判 EDIT / confidence 裁剪 / ADD 上限 / 非法 op 拒
  - apply_ops: ADD→EDIT→UPVOTE(封顶)→DOWNVOTE→跌破删除 全生命周期
  - 容量: 超 insights_capacity 按 confidence×recency 淘汰
  - maybe_reflect: 游标计数 (不足不触发/补足触发/force) / MockLLM 端到端 (接受+拒绝混合) /
    LLM 与解析异常不中止 (游标不动) / 方向轨迹注入 (参数注入 + direction_state.json 两路)
  - render_insights_block: 排序/截断/空串
  - prompt 注入回归: build_plan_prompt/build_plan_many_prompt/诊断 user 的
    None/空列表逐字节不变 + 非空注入位置正确 (机制上下文 → exemplars → 经验块)
  - orchestrator: _after_digest 在创造检查后调 maybe_reflect, 异常吞掉不中止进化
  - store 增补: update_insight / delete_insight / list_insights_by_confidence
"""

from __future__ import annotations

import json

import pytest

from darwin_st.creation import CreationArchive, CreationRecord, MockLLM
from darwin_st.creation.contracts import FusionRequest
from darwin_st.creation.diagnosis import DiagnosisSummary, _build_user
from darwin_st.creation.reflection import (
    ReflectionConfig,
    ReflectionLoop,
    apply_ops,
    build_reflection_prompt,
    load_direction_history,
    parse_reflection_ops,
    render_insights_block,
    validate_ops,
)
from darwin_st.creation.synthesizer import build_plan_many_prompt, build_plan_prompt
from darwin_st.knowledge import all_seed_mechanisms
from darwin_st.memory.store import MemoryStore


def _rec(op="synth_a", gate="all", mae=18.5, created="2026-01-01T00:00:00+00:00", rnd=0):
    return CreationRecord(round_idx=rnd, created_at=created, bottleneck="瓶颈b",
                          preconditions=["p1"], mechanisms=["m1", "m2"],
                          operator_name=op, composition="additive_residual",
                          rationale="理由r", code="class A: pass", gate=gate,
                          seed_val_mae=mae, adopted=True)


def _req():
    ms = [m for m in all_seed_mechanisms() if m.name in ("state_space_model", "masked_autoencoding")]
    return FusionRequest(bottleneck="长程依赖 + 标签稀缺",
                         target_preconditions=["long_range_dependency", "label_scarcity"],
                         mechanisms=ms)


def _exemplars():
    return {"successes": [{"operator_name": "synth_old", "composition": "additive_residual",
                           "rationale": "旧理由", "seed_val_mae": 18.0,
                           "code": "class A: pass"}],
            "failures": []}


def _insight(iid=1, text="教训A", cond="条件A", conf=0.9, ev=(1, 2)):
    return {"id": iid, "dataset": None, "condition": cond, "insight_text": text,
            "evidence_ids": list(ev), "confidence": conf, "created_at": "t"}


def _make_loop(tmp_path, responder=None, cfg=None, n_records=0, store=None, **kw):
    arch = CreationArchive(str(tmp_path / "a.jsonl"))
    for i in range(n_records):
        arch.record(_rec(op=f"synth_{i}", rnd=i))
    store = store or MemoryStore(":memory:")
    llm = MockLLM(responder or (lambda m: "[]"))
    loop = ReflectionLoop(store, arch, llm, cfg or ReflectionConfig(),
                          log_fn=lambda *a: None, **kw)
    return loop, arch, llm


# ---------------------------------------------------------------------------
# prompt 组装
# ---------------------------------------------------------------------------

def test_build_reflection_prompt_contains_all_blocks():
    cfg = ReflectionConfig()
    records = [{"row_id": 1, "round_idx": 0, "operator_name": "synth_a",
                "composition": "additive_residual", "gate": "all", "proxy_mae": 19.0,
                "seed_val_mae": 18.5, "adopted": True, "bottleneck": "长程依赖不足",
                "mechanisms": ["m1"], "error": None}]
    insights = [_insight(iid=7, text="旧教训X", cond="条件X", conf=0.8, ev=(1,))]
    hist = [{"round": 0, "bottleneck": "b", "preconditions": ["long_range_dependency"],
             "result_mae": 18.5, "score": 1.0}]
    msgs = build_reflection_prompt(records, "nan×3", insights, hist, cfg)

    assert [m["role"] for m in msgs] == ["system", "user"]
    sys_p, user = msgs[0]["content"], msgs[1]["content"]
    # system: 操作规范 + 硬约束
    for token in ("ADD", "EDIT", "UPVOTE", "DOWNVOTE", "evidence_ids", "insight_id",
                  "条件式", "JSON 数组"):
        assert token in sys_p
    assert str(cfg.max_new_lessons) in sys_p          # 每批 ADD 上限写进规范
    # user: 四块齐全
    assert "行#1" in user and "synth_a" in user and "18.5000" in user
    assert "nan×3" in user                            # 墓地摘要
    assert "insight_id=7" in user and "旧教训X" in user  # 现有 insights
    assert "long_range_dependency" in user and "信用分=1" in user  # 方向轨迹


def test_build_reflection_prompt_empty_inputs():
    msgs = build_reflection_prompt([], "", [], [], ReflectionConfig())
    user = msgs[1]["content"]
    assert "(无新记录)" in user and "(无失败记录)" in user
    assert "教训库为空" in user and "(无方向轨迹)" in user


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

def test_parse_ops_valid_array_with_prose():
    ops = parse_reflection_ops('思考过程... [{"op":"ADD","text":"t","evidence_ids":[1]}] 完')
    assert ops == [{"op": "ADD", "text": "t", "evidence_ids": [1]}]


def test_parse_ops_single_object_wrapped():
    ops = parse_reflection_ops('{"op":"UPVOTE","insight_id":2,"evidence_ids":[3]}')
    assert len(ops) == 1 and ops[0]["op"] == "UPVOTE"


def test_parse_ops_invalid_raises():
    with pytest.raises(Exception):
        parse_reflection_ops("完全没有 JSON 的回复")


def test_parse_ops_filters_non_dict_elements():
    ops = parse_reflection_ops('[{"op":"ADD","text":"t","evidence_ids":[1]}, 42, "x"]')
    assert len(ops) == 1


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def test_validate_rejects_unknown_evidence():
    ops = [{"op": "ADD", "text": "某教训", "evidence_ids": [99]}]
    acc, rej = validate_ops(ops, {1, 2}, [], ReflectionConfig())
    assert acc == [] and len(rej) == 1 and "未知" in rej[0]["reason"]


def test_validate_rejects_empty_evidence():
    ops = [{"op": "ADD", "text": "无证据教训", "evidence_ids": []}]
    acc, rej = validate_ops(ops, {1, 2}, [], ReflectionConfig())
    assert acc == [] and "为空" in rej[0]["reason"]


def test_validate_rejects_missing_or_nonexistent_insight_id():
    ops = [
        {"op": "EDIT", "text": "改", "evidence_ids": [1]},                    # 无 insight_id
        {"op": "UPVOTE", "insight_id": 42, "evidence_ids": [1]},              # id 不存在
        {"op": "DOWNVOTE", "insight_id": "x", "evidence_ids": [1]},           # id 非法类型
    ]
    acc, rej = validate_ops(ops, {1, 2}, [_insight(iid=1)], ReflectionConfig())
    assert acc == [] and len(rej) == 3
    assert all("insight_id" in r["reason"] for r in rej)


def test_validate_rejects_illegal_op():
    acc, rej = validate_ops([{"op": "MERGE", "evidence_ids": [1]}], {1}, [], ReflectionConfig())
    assert acc == [] and "非法 op" in rej[0]["reason"]


def test_validate_similar_add_demoted_to_edit():
    existing = [_insight(iid=3, text="当梯度爆炸时应缩小学习率", conf=0.6)]
    ops = [{"op": "ADD", "condition": "梯度爆炸", "evidence_ids": [2],
            "text": "当梯度爆炸时应缩小学习率来稳定训练"}]     # 规范化子串 → 相似
    acc, rej = validate_ops(ops, {1, 2}, existing, ReflectionConfig())
    assert rej == [] and len(acc) == 1
    assert acc[0]["op"] == "EDIT" and acc[0]["insight_id"] == 3
    assert acc[0].get("_auto_edit") is True


def test_validate_dissimilar_add_stays_add():
    existing = [_insight(iid=3, text="当梯度爆炸时应缩小学习率", conf=0.6)]
    ops = [{"op": "ADD", "condition": "proxy inf", "evidence_ids": [2],
            "text": "proxy 全 inf 多半是争卡 OOM, 应换空闲卡"}]
    acc, rej = validate_ops(ops, {1, 2}, existing, ReflectionConfig())
    assert rej == [] and acc[0]["op"] == "ADD" and "insight_id" not in acc[0]


def test_validate_add_cap():
    cfg = ReflectionConfig(max_new_lessons=1)
    ops = [
        {"op": "ADD", "text": "教训一", "evidence_ids": [1]},
        {"op": "ADD", "text": "完全不同的教训二", "evidence_ids": [2]},
    ]
    acc, rej = validate_ops(ops, {1, 2}, [], cfg)
    assert len(acc) == 1 and len(rej) == 1 and "上限" in rej[0]["reason"]


def test_validate_confidence_clipped():
    ops = [{"op": "ADD", "text": "教训", "evidence_ids": [1], "confidence": 1.8}]
    acc, _ = validate_ops(ops, {1}, [], ReflectionConfig())
    assert acc[0]["confidence"] == pytest.approx(1.0)
    ops2 = [{"op": "ADD", "text": "教训", "evidence_ids": [1], "confidence": -0.5}]
    acc2, _ = validate_ops(ops2, {1}, [], ReflectionConfig())
    assert acc2[0]["confidence"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# apply_ops 全生命周期 + confidence 维护
# ---------------------------------------------------------------------------

def test_apply_ops_full_lifecycle():
    store = MemoryStore(":memory:")
    cfg = ReflectionConfig()
    # ADD: conf_init=0.5
    stats = apply_ops(store, [{"op": "ADD", "text": "做法A", "condition": "条件A",
                               "evidence_ids": [1, 2]}], cfg)
    assert stats["added"] == 1
    ins = store.list_insights_by_confidence()[0]
    assert ins["confidence"] == pytest.approx(0.5)
    assert ins["condition"] == "条件A" and ins["evidence_ids"] == [1, 2]
    iid = ins["id"]

    # EDIT: 文本/条件更新 + 证据并入去重保序
    stats = apply_ops(store, [{"op": "EDIT", "insight_id": iid, "text": "做法B",
                               "condition": "条件B", "evidence_ids": [2, 3]}], cfg)
    ins = store.list_insights_by_confidence()[0]
    assert stats["edited"] == 1
    assert ins["insight_text"] == "做法B" and ins["condition"] == "条件B"
    assert ins["evidence_ids"] == [1, 2, 3]

    # UPVOTE: +0.10 并并入新证据
    apply_ops(store, [{"op": "UPVOTE", "insight_id": iid, "evidence_ids": [4]}], cfg)
    ins = store.list_insights_by_confidence()[0]
    assert ins["confidence"] == pytest.approx(0.6) and 4 in ins["evidence_ids"]

    # UPVOTE 封顶 conf_cap=0.95
    for _ in range(10):
        apply_ops(store, [{"op": "UPVOTE", "insight_id": iid, "evidence_ids": [4]}], cfg)
    assert store.list_insights_by_confidence()[0]["confidence"] == pytest.approx(0.95)

    # DOWNVOTE: −0.15
    apply_ops(store, [{"op": "DOWNVOTE", "insight_id": iid, "evidence_ids": [5]}], cfg)
    assert store.list_insights_by_confidence()[0]["confidence"] == pytest.approx(0.80)

    # 连续 DOWNVOTE 跌破 conf_delete_below=0.30 → 删行 (0.80→0.65→0.50→0.35→0.20 删)
    deleted = 0
    for _ in range(4):
        deleted += apply_ops(store, [{"op": "DOWNVOTE", "insight_id": iid,
                                      "evidence_ids": [5]}], cfg)["deleted"]
    assert deleted == 1 and store.list_insights_by_confidence() == []


def test_apply_ops_conf_init_clipped():
    store = MemoryStore(":memory:")
    cfg = ReflectionConfig(conf_init=1.5)          # 构造覆盖: 超界初始值
    apply_ops(store, [{"op": "ADD", "text": "t", "evidence_ids": [1]}], cfg)
    assert store.list_insights_by_confidence()[0]["confidence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 容量上限 (Hermes)
# ---------------------------------------------------------------------------

def test_capacity_eviction_confidence_then_recency(tmp_path):
    loop, _, _ = _make_loop(tmp_path, cfg=ReflectionConfig(insights_capacity=2))
    store = loop.store
    store.add_insight("低信-旧证据", created_at="t", evidence_ids=[1], confidence=0.5)
    store.add_insight("低信-新证据", created_at="t", evidence_ids=[9], confidence=0.5)
    store.add_insight("高信", created_at="t", evidence_ids=[2], confidence=0.9)
    evicted = loop._enforce_capacity()
    assert evicted == 1
    remaining = {i["insight_text"] for i in store.list_insights_by_confidence()}
    assert remaining == {"低信-新证据", "高信"}     # 同置信度淘汰证据更旧的


def test_capacity_enforced_during_reflect(tmp_path):
    cfg = ReflectionConfig(insights_capacity=2, reflect_every_records=1)
    store = MemoryStore(":memory:")
    store.add_insight("低信旧教训", created_at="t", evidence_ids=[1], confidence=0.4)
    store.add_insight("高信旧教训", created_at="t", evidence_ids=[1], confidence=0.9)
    responder = lambda m: json.dumps(
        [{"op": "ADD", "condition": "新条件", "text": "全新教训", "evidence_ids": [1]}])
    loop, _, _ = _make_loop(tmp_path, responder=responder, cfg=cfg, store=store,
                            n_records=1, cursor=0)
    rep = loop.maybe_reflect()
    assert rep is not None and rep["apply"]["added"] == 1
    assert rep["evicted"] == 1                     # 2+1 超容量 2 → 淘汰最低信
    remaining = {i["insight_text"] for i in store.list_insights_by_confidence()}
    assert remaining == {"高信旧教训", "全新教训"}


# ---------------------------------------------------------------------------
# maybe_reflect: 游标 / 端到端 / 异常 / 方向轨迹
# ---------------------------------------------------------------------------

def test_cursor_threshold_and_force(tmp_path):
    cfg = ReflectionConfig(reflect_every_records=5)
    loop, arch, llm = _make_loop(tmp_path, cfg=cfg)      # 构造时 0 条, 游标=0
    for i in range(3):
        arch.record(_rec(op=f"synth_{i}", rnd=i))
    assert loop.maybe_reflect() is None                  # 3 < 5 不触发
    assert llm.call_count == 0
    rep = loop.maybe_reflect(force=True)                 # force 触发
    assert rep is not None and rep["n_new_records"] == 3 and llm.call_count == 1
    assert loop._cursor == 3                             # 游标推进
    for i in range(2):
        arch.record(_rec(op=f"synth_b{i}", rnd=i))
    assert loop.maybe_reflect() is None                  # 增量 2 < 5
    for i in range(3):
        arch.record(_rec(op=f"synth_c{i}", rnd=i))
    rep = loop.maybe_reflect()                           # 增量 5 补足 → 触发
    assert rep is not None and rep["n_new_records"] == 5


def test_cursor_init_skips_backlog(tmp_path):
    """在线语义: 构造时快照为游标, 历史积压不反思, 只对增量反思。"""
    loop, arch, llm = _make_loop(tmp_path, cfg=ReflectionConfig(reflect_every_records=2),
                                 n_records=10)
    assert loop.maybe_reflect() is None                  # 10 条积压不算增量
    arch.record(_rec(op="synth_new", rnd=99))
    arch.record(_rec(op="synth_new2", rnd=100))
    rep = loop.maybe_reflect()
    assert rep is not None and rep["n_new_records"] == 2


def test_end_to_end_mixed_accept_reject(tmp_path):
    store = MemoryStore(":memory:")
    seed_id = store.add_insight("旧教训: 争卡会 OOM", created_at="t",
                                condition="proxy 阶段", evidence_ids=[1], confidence=0.5)
    ops_payload = json.dumps([
        {"op": "ADD", "condition": "proxy 全 inf", "evidence_ids": [1, 2],
         "text": "proxy 全 inf 时换空闲卡重试"},
        {"op": "UPVOTE", "insight_id": seed_id, "evidence_ids": [3]},
        {"op": "ADD", "text": "无证据拒", "evidence_ids": []},
        {"op": "DOWNVOTE", "insight_id": 999, "evidence_ids": [1]},
    ])
    loop, _, _ = _make_loop(tmp_path, responder=lambda m: ops_payload,
                            cfg=ReflectionConfig(reflect_every_records=3),
                            n_records=3, store=store, cursor=0)
    rep = loop.maybe_reflect()
    assert rep is not None and "error" not in rep
    assert rep["accepted"] == 2 and len(rep["rejected"]) == 2
    assert rep["apply"]["added"] == 1 and rep["apply"]["upvoted"] == 1
    insights = {i["insight_text"]: i for i in store.list_insights_by_confidence()}
    assert "proxy 全 inf 时换空闲卡重试" in insights
    assert insights["proxy 全 inf 时换空闲卡重试"]["confidence"] == pytest.approx(0.5)
    # 旧教训被 upvote: 0.5+0.10, 证据并入
    old = insights["旧教训: 争卡会 OOM"]
    assert old["confidence"] == pytest.approx(0.6) and old["evidence_ids"] == [1, 3]


def test_llm_exception_does_not_raise_and_cursor_stays(tmp_path):
    def boom(msgs):
        raise RuntimeError("API 炸了")
    loop, _, _ = _make_loop(tmp_path, responder=boom,
                            cfg=ReflectionConfig(reflect_every_records=2), n_records=2,
                            cursor=0)
    rep = loop.maybe_reflect()
    assert rep is not None and rep["error"].startswith("llm:")
    assert loop._cursor == 0                             # 失败不推进, 下批重试


def test_parse_failure_does_not_raise_and_cursor_stays(tmp_path):
    loop, _, _ = _make_loop(tmp_path, responder=lambda m: "这不是 JSON",
                            cfg=ReflectionConfig(reflect_every_records=2), n_records=2,
                            cursor=0)
    rep = loop.maybe_reflect()
    assert rep is not None and rep["error"].startswith("parse:")
    assert loop._cursor == 0


def test_direction_history_injected_via_param(tmp_path):
    captured = {}

    def spy(msgs):
        captured["user"] = msgs[1]["content"]
        return "[]"

    hist = [{"round": 0, "bottleneck": "b", "preconditions": ["heterogeneity"],
             "result_mae": 18.5, "score": -1.0}]
    loop, _, _ = _make_loop(tmp_path, responder=spy,
                            cfg=ReflectionConfig(reflect_every_records=1),
                            n_records=1, direction_history=hist, cursor=0)
    loop.maybe_reflect()
    assert "heterogeneity" in captured["user"] and "信用分=-1" in captured["user"]


def test_direction_history_read_from_state_file(tmp_path):
    captured = {}

    def spy(msgs):
        captured["user"] = msgs[1]["content"]
        return "[]"

    state = {"temperature": 0.8,
             "history": [{"round": 3, "bottleneck": "b", "preconditions": ["noise_corruption"],
                          "result_mae": 19.1, "score": 0.5}]}
    import os
    # _make_loop 的履历本在 tmp_path/a.jsonl → direction_state.json 默认同目录
    loop, arch, _ = _make_loop(tmp_path, responder=spy,
                               cfg=ReflectionConfig(reflect_every_records=1), n_records=1,
                               cursor=0)
    with open(os.path.join(tmp_path, "direction_state.json"), "w", encoding="utf-8") as f:
        json.dump(state, f)
    loop.maybe_reflect()
    assert "noise_corruption" in captured["user"]


def test_load_direction_history_tolerates_missing_and_broken(tmp_path):
    assert load_direction_history(None) == []
    assert load_direction_history(str(tmp_path / "nope.json")) == []
    bad = tmp_path / "bad.json"
    bad.write_text("{坏 json", encoding="utf-8")
    assert load_direction_history(str(bad)) == []


# ---------------------------------------------------------------------------
# render_insights_block
# ---------------------------------------------------------------------------

def test_render_empty_returns_empty_string():
    assert render_insights_block([], ReflectionConfig()) == ""


def test_render_sorting_and_format():
    insights = [_insight(iid=1, text="低信教训", conf=0.5),
                _insight(iid=2, text="高信教训", cond="条件H", conf=0.9),
                _insight(iid=3, text="中信教训", conf=0.7)]
    block = render_insights_block(insights, ReflectionConfig())
    lines = block.splitlines()
    assert lines[0] == "【历史经验(反思蒸馏)】"
    assert lines[1].startswith("1. [0.90] 条件H → 高信教训")
    assert lines[2].startswith("2. [0.70]")
    assert lines[3].startswith("3. [0.50]")
    assert "(依据 #1,#2)" in lines[1]


def test_render_top_n_limit():
    insights = [_insight(iid=i, text=f"教训{i}", conf=0.9 - i * 0.01) for i in range(1, 6)]
    block = render_insights_block(insights, ReflectionConfig(prompt_top_n=2))
    assert block.count("→") == 2 and "教训3" not in block


def test_render_lesson_truncation():
    insights = [_insight(text="长" * 200)]
    block = render_insights_block(insights, ReflectionConfig())
    # 单条教训截 lesson_max_chars=120
    assert "长" * 121 not in block and "长" * 120 in block


def test_render_block_total_truncation():
    insights = [_insight(iid=i, text=f"教训{'长' * 100}{i}", conf=0.9) for i in range(3)]
    cfg = ReflectionConfig(prompt_max_chars=100)
    block = render_insights_block(insights, cfg)
    assert len(block) <= 101 and block.endswith("…")


# ---------------------------------------------------------------------------
# prompt 注入回归 (逐字节铁律 + 注入位置)
# ---------------------------------------------------------------------------

def test_plan_prompts_byte_identical_without_insights():
    req = _req()
    assert build_plan_prompt(req, insights=None) == build_plan_prompt(req)
    assert build_plan_prompt(req, insights=[]) == build_plan_prompt(req)
    assert build_plan_many_prompt(req, 3, insights=None) == build_plan_many_prompt(req, 3)
    assert build_plan_many_prompt(req, 3, insights=[]) == build_plan_many_prompt(req, 3)


def test_plan_prompt_injects_insights_block_after_exemplars():
    req = _req()
    ins = [_insight(text="注入的教训", cond="注入条件", conf=0.9, ev=(1,))]
    user = build_plan_prompt(req, exemplars=_exemplars(), insights=ins)[1]["content"]
    assert "【历史经验(反思蒸馏)】" in user and "注入的教训" in user
    # 顺序: 机制上下文 → exemplars → 经验块 (末尾)
    assert user.index("瓶颈与跨域机制集") < user.index("【历史创造履历") \
        < user.index("【历史经验(反思蒸馏)】")
    assert user.rstrip().endswith("(依据 #1)")


def test_plan_many_prompt_injects_insights_at_end():
    req = _req()
    ins = [_insight(text="注入的教训", conf=0.9, ev=(1,))]
    user = build_plan_many_prompt(req, 3, insights=ins)[1]["content"]
    assert "【历史经验(反思蒸馏)】" in user
    assert user.index("瓶颈与跨域机制集") < user.index("【历史经验(反思蒸馏)】")


def _summary():
    return DiagnosisSummary(
        labels=["converged_plateau"], best_mae=18.5, best_epoch=9, n_epochs=10,
        sota_gap=1.0, train_loss_segments=(1.0, 0.5, 0.4),
        val_mae_segments=(20.0, 19.0, 18.5), grad_norm_range=(0.1, 3.0),
        convergence_note="末段平台", instability_note="梯度稳定",
        arch_summary="深度2 hidden64", used_spatial=["gcn"], used_temporal=["tcn"])


def test_diagnosis_user_byte_identical_and_inject():
    s = _summary()
    base = _build_user(s, [], 0)
    assert _build_user(s, [], 0, insights=None) == base
    assert _build_user(s, [], 0, insights=[]) == base
    injected = _build_user(s, [], 0, insights=[_insight(text="诊断侧教训", conf=0.9)])
    assert injected != base and "【历史经验(反思蒸馏)】" in injected
    assert "诊断侧教训" in injected and injected.startswith(base[:100])


# ---------------------------------------------------------------------------
# store 增补方法
# ---------------------------------------------------------------------------

def test_store_update_delete_insight():
    store = MemoryStore(":memory:")
    iid = store.add_insight("原文", created_at="t", condition="原条件",
                            evidence_ids=[1], confidence=0.5)
    assert store.update_insight(iid, text="新文", confidence=0.8) is True
    ins = store.list_insights_by_confidence()[0]
    assert ins["insight_text"] == "新文" and ins["confidence"] == pytest.approx(0.8)
    assert ins["condition"] == "原条件" and ins["evidence_ids"] == [1]   # None 字段不动
    assert store.update_insight(iid, evidence_ids=[1, 2]) is True
    assert store.list_insights_by_confidence()[0]["evidence_ids"] == [1, 2]
    assert store.update_insight(999, text="x") is False                  # 未命中
    assert store.update_insight(iid) is False                            # 全 None = 无事可做
    assert store.delete_insight(999) is False
    assert store.delete_insight(iid) is True
    assert store.list_insights_by_confidence() == []


def test_store_list_insights_by_confidence_order_limit_dataset():
    store = MemoryStore(":memory:")
    store.add_insight("通用-低", created_at="t", confidence=0.5)
    store.add_insight("PeMS04-高", created_at="t", dataset="PeMS04", confidence=0.9)
    store.add_insight("PeMS04-同信旧", created_at="t", dataset="PeMS04", confidence=0.7)
    store.add_insight("PeMS04-同信新", created_at="t", dataset="PeMS04", confidence=0.7)
    store.add_insight("METRLA", created_at="t", dataset="METR-LA", confidence=0.99)
    rows = store.list_insights_by_confidence("PeMS04")
    texts = [r["insight_text"] for r in rows]
    # 该数据集专属 + 通用 (不含其它数据集); 置信度降序, 并列取新
    assert texts == ["PeMS04-高", "PeMS04-同信新", "PeMS04-同信旧", "通用-低"]
    top2 = store.list_insights_by_confidence("PeMS04", limit=2)
    assert [r["insight_text"] for r in top2] == ["PeMS04-高", "PeMS04-同信新"]


# ---------------------------------------------------------------------------
# orchestrator 接线: _after_digest 调 maybe_reflect, 异常吞掉
# ---------------------------------------------------------------------------

def test_orchestrator_calls_maybe_reflect_and_swallows_exceptions():
    from darwin_st.optim.orchestrator import Orchestrator, OrchestratorConfig
    from darwin_st.optim.scheduler import EvalResult

    calls = {"reflect": 0}

    class _SpyReflection:
        def maybe_reflect(self, force=False):
            calls["reflect"] += 1
            if calls["reflect"] == 2:
                raise RuntimeError("反思炸了")       # 第二次抛异常 → 必须被吞掉
            return None

    class _StubCreationLoop:
        reflection = _SpyReflection()

        def maybe_create(self, best_genotype, sota_gap=None, run_tag="", dataset="",
                         best_trace=None, best_hps=None):
            from darwin_st.creation.creation_loop import CreationOutcome
            return CreationOutcome(False, bottleneck="b", reason="无产出")

    def const_eval(geno, device):
        return EvalResult(genotype=geno, status="OK", mae=20.0, rmse=30.0, device=device,
                          hps={"lr": 1e-3}, extra={"num_params": 50_000})

    cfg = OrchestratorConfig(population_size=4, tournament_size=2, max_rounds=4, target_mae=0.0)
    orch = Orchestrator(cfg, const_eval, devices=2, creation_loop=_StubCreationLoop())
    orch.archive.stagnation_patience = 99          # 不触发创造, 只看反思钩子
    state = orch.run()
    assert state.rounds == 4                        # 异常未中止进化
    assert calls["reflect"] >= 3                    # 每轮末都调 (含抛异常那次)


def test_maybe_reflect_retries_on_empty_response(tmp_path):
    """推理模型空响应 (reasoning 吃光预算) 时加倍重试, 第二次成功则正常入库。"""
    calls = {"n": 0}

    def responder(m):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""                      # 第一次空响应 (模拟 reasoning 吃光 token)
        return '[{"op": "ADD", "condition": "欠拟合瓶颈", "text": "残差融合有效", "evidence_ids": [1]}]'

    loop, arch, llm = _make_loop(tmp_path, responder=responder, n_records=20, cursor=0)
    rep = loop.maybe_reflect()
    assert calls["n"] == 2                  # 空响应触发了加倍重试
    assert rep is not None and "error" not in rep
    assert rep["apply"]["added"] == 1
    rows = loop.store.list_insights_by_confidence()
    assert len(rows) == 1 and rows[0]["insight_text"] == "残差融合有效"


def test_build_user_caps_existing_insights():
    """现有教训超 cfg.max_existing_insights 时截断展示 + 从略注记 (防推理模型在大量近似旧教训上空转)。"""
    cfg = ReflectionConfig(max_existing_insights=3)
    ins = [_insight(iid=i, text=f"教训{i}", cond=f"条件{i}", conf=0.5) for i in range(1, 8)]
    user = build_reflection_prompt([], "", ins, [], cfg)[1]["content"]
    assert "insight_id=3" in user and "insight_id=4" not in user
    assert "另有 4 条旧教训从略" in user
    user2 = build_reflection_prompt([], "", ins[:2], [], cfg)[1]["content"]
    assert "从略" not in user2
