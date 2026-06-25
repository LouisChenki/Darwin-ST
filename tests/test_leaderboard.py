"""排行榜聚合/渲染纯核心测试 (确定性, 无 IO/LLM/torch 训练)。

落盘那层 (sqlite 读、拷 synth 代码、写 .md) 由 scripts/export_leaderboard.py 负责, 不在此测。
这里锁住聚合逻辑: 筛选/去重/run_tag 隔离/排名/synth 识别/vs-SOTA/Markdown 渲染要素。
"""

from __future__ import annotations

from darwin_st.leaderboard import (
    LeaderboardRow,
    build_leaderboard,
    detect_synth_ops,
    model_dir_name,
    render_index_md,
    render_leaderboard_md,
    render_model_card,
    select_our_models,
    short_sig,
    weakest_baseline_mae,
)

# 取 PeMS04 子集做基准 (真实 baseline_registry 数据形状)
BASELINES = {
    "STGCN": {"mae": 19.57, "rmse": 31.38},
    "ASTGCN": {"mae": 22.93, "rmse": 35.22},      # 最弱 → 入榜门槛
    "STAEformer": {"mae": 18.22, "rmse": 30.18},
    "STD-MAE": {"mae": 17.80, "rmse": 29.83},     # SOTA
}
SOTA_NAME, SOTA_MAE = "STD-MAE", 17.80


def _trial(tid, sig, mae, status="KEEP", run_tag="exp/a", geno=None, **kw):
    d = {"id": tid, "signature": sig, "val_mae": mae, "status": status,
         "run_tag": run_tag, "genotype": geno or {"blocks": [], "hidden": 64},
         "hp": {"lr": 1e-3}, "num_params": 100000}
    d.update(kw)
    return d


# ---- detect_synth_ops ----

def test_detect_synth_ops_finds_prefixed():
    geno = {"blocks": [
        {"spatial_op": "synth_seq_lka_sasf_residual", "temporal_op": "tcn", "fusion": "residual"},
        {"spatial_op": "adaptive", "temporal_op": "tcn", "fusion": "synth_gated_mix"},
    ]}
    ops = detect_synth_ops(geno)
    assert ops == ["synth_seq_lka_sasf_residual", "synth_gated_mix"]


def test_detect_synth_ops_empty_and_none():
    assert detect_synth_ops(None) == []
    assert detect_synth_ops({"blocks": [{"spatial_op": "gcn", "temporal_op": "tcn", "fusion": "seq"}]}) == []


def test_detect_synth_ops_dedup():
    geno = {"blocks": [
        {"spatial_op": "synth_x", "temporal_op": "tcn", "fusion": "seq"},
        {"spatial_op": "synth_x", "temporal_op": "tcn", "fusion": "seq"},
    ]}
    assert detect_synth_ops(geno) == ["synth_x"]


# ---- weakest_baseline_mae / helpers ----

def test_weakest_baseline_is_threshold():
    assert weakest_baseline_mae(BASELINES) == 22.93


def test_weakest_baseline_empty():
    assert weakest_baseline_mae({}) is None
    assert weakest_baseline_mae({"X": {"rmse": 1.0}}) is None  # 无 mae


def test_short_sig_and_dir_name():
    assert short_sig("abcdef1234567890") == "abcdef12"
    assert short_sig(None) == "unknown"
    assert model_dir_name(3, 18.69, "abcdef1234") == "03_mae18.69_abcdef12"


# ---- select_our_models ----

def test_select_filters_non_keep_and_threshold():
    trials = [
        _trial(1, "s1", 18.69),                       # KEEP, 超门槛 → 入
        _trial(2, "s2", 25.0),                        # KEEP 但 > 22.93 门槛 → 出
        _trial(3, "s3", 19.0, status="DISCARD"),      # 非 KEEP → 出
        _trial(4, "s4", float("inf")),                # inf → 出
    ]
    picked = select_our_models(trials, threshold_mae=22.93)
    assert [t["id"] for t in picked] == [1]


def test_select_dedup_by_signature_keeps_best():
    trials = [
        _trial(1, "same", 20.0),
        _trial(2, "same", 18.69),   # 同签名更优 → 留这个
        _trial(3, "same", 19.5),
    ]
    picked = select_our_models(trials, threshold_mae=22.93)
    assert len(picked) == 1 and picked[0]["id"] == 2 and picked[0]["val_mae"] == 18.69


def test_select_run_tag_isolation():
    trials = [
        _trial(1, "s1", 18.69, run_tag="exp/now"),
        _trial(2, "s2", 19.0, run_tag="exp/old"),     # 别的 run → 隔离掉
    ]
    picked = select_our_models(trials, threshold_mae=22.93, run_tag="exp/now")
    assert [t["id"] for t in picked] == [1]
    # 不传 run_tag → 全收
    picked_all = select_our_models(trials, threshold_mae=22.93)
    assert {t["id"] for t in picked_all} == {1, 2}


def test_select_no_threshold_keeps_all_keep():
    trials = [_trial(1, "s1", 18.69), _trial(2, "s2", 99.0)]
    picked = select_our_models(trials, threshold_mae=None)
    assert len(picked) == 2
    assert [t["id"] for t in picked] == [1, 2]   # 仍按 mae 升序


def test_select_sorted_ascending():
    trials = [_trial(1, "s1", 20.0), _trial(2, "s2", 18.69), _trial(3, "s3", 19.5)]
    picked = select_our_models(trials, threshold_mae=22.93)
    assert [t["val_mae"] for t in picked] == [18.69, 19.5, 20.0]


# ---- build_leaderboard ----

def test_build_merges_and_ranks():
    ours = [
        _trial(81, "sigLKA01", 18.69,
               geno={"blocks": [{"spatial_op": "synth_seq_lka_sasf_residual",
                                 "temporal_op": "tcn", "fusion": "residual"}], "hidden": 128}),
    ]
    rows = build_leaderboard("PeMS04", ours, BASELINES, SOTA_NAME, SOTA_MAE)
    # 5 行 (4 baseline + 1 ours), 全局按 MAE 升序
    assert len(rows) == 5
    maes = [r.mae for r in rows]
    assert maes == sorted(maes)
    # SOTA 17.80 第 1, 本项目 18.69 应排在 STAEformer(18.22) 之后、STGCN(19.57) 之前 → 第 3
    ours_row = [r for r in rows if r.is_ours][0]
    assert ours_row.rank == 3
    assert ours_row.kind == "ours"
    assert ours_row.synth_ops == ["synth_seq_lka_sasf_residual"]
    assert ours_row.vs_sota == round(18.69 - 17.80, 2) or abs(ours_row.vs_sota - 0.89) < 1e-9
    assert ours_row.model_dir == "models/PeMS04/03_mae18.69_sigLKA01"
    # SOTA 行标 kind=sota
    sota_row = [r for r in rows if r.kind == "sota"][0]
    assert sota_row.name == "STD-MAE" and sota_row.rank == 1


def test_build_marks_sota_uniquely():
    rows = build_leaderboard("PeMS04", [], BASELINES, SOTA_NAME, SOTA_MAE)
    assert sum(1 for r in rows if r.kind == "sota") == 1
    assert sum(1 for r in rows if r.kind == "baseline") == 3


# ---- 渲染 ----

def test_render_leaderboard_md_has_key_elements():
    ours = [_trial(81, "sigLKA01", 18.69,
                   geno={"blocks": [{"spatial_op": "synth_seq_lka_sasf_residual",
                                     "temporal_op": "tcn", "fusion": "residual"}], "hidden": 128})]
    rows = build_leaderboard("PeMS04", ours, BASELINES, SOTA_NAME, SOTA_MAE)
    md = render_leaderboard_md("PeMS04", rows, protocol_note="12步平均 MAE",
                               sota_name=SOTA_NAME, sota_mae=SOTA_MAE)
    assert "# PeMS04 排行榜" in md
    assert "STD-MAE" in md and "17.80" in md
    assert "18.69" in md
    assert "darwin-st/sigLKA01" in md
    assert "models/PeMS04/03_mae18.69_sigLKA01/" in md   # 存档链接
    assert "12步平均 MAE" in md
    assert "🏆 SOTA" in md


def test_render_model_card_full():
    row = build_leaderboard(
        "PeMS04",
        [_trial(81, "sigLKA01", 18.69, val_rmse=30.5, test_mae=19.1,
                geno={"blocks": [{"spatial_op": "synth_seq_lka_sasf_residual",
                                  "temporal_op": "tcn", "fusion": "residual"}],
                      "hidden": 128, "adj_mode": "sym",
                      "embedding": {"use_node": True, "node_dim": 64}})],
        BASELINES, SOTA_NAME, SOTA_MAE,
    )[2]  # 本项目那行 (rank 3)
    synth_meta = {"synth_seq_lka_sasf_residual": {
        "composition": "LKA 大核注意力 + SASF + 零初始化残差",
        "source_mechanisms": ["large_kernel_attention(CV)", "scale_aware_fusion"],
        "real_mae": 18.69}}
    card = render_model_card(row, "PeMS04", SOTA_NAME, SOTA_MAE, synth_meta=synth_meta,
                             protocol_note="12步平均 MAE")
    assert "模型卡" in card
    assert "synth_seq_lka_sasf_residual" in card
    assert "LKA 大核注意力" in card
    assert "large_kernel_attention(CV)" in card
    assert "距 SOTA 差 +0.89" in card
    assert "genotype.json" in card and "hparams.json" in card
    assert "operators/synth_seq_lka_sasf_residual.py" in card


def test_render_model_card_pure_tier1():
    """无 synth 算子的纯进化模型: 不应渲染跨域算子节, 应注明纯 Tier-1。"""
    row = LeaderboardRow(rank=2, name="darwin-st/abc", kind="ours", mae=19.0,
                         is_ours=True, signature="abc12345",
                         genotype={"blocks": [{"spatial_op": "adaptive", "temporal_op": "tcn",
                                               "fusion": "seq"}], "hidden": 64},
                         hp={"lr": 1e-3}, synth_ops=[])
    card = render_model_card(row, "PeMS04", SOTA_NAME, SOTA_MAE)
    assert "纯 Tier-1 进化架构" in card
    assert "🔬 跨域合成算子" not in card


def test_render_card_beats_sota_verdict():
    row = LeaderboardRow(rank=1, name="darwin-st/win", kind="ours", mae=17.50,
                         is_ours=True, signature="win00001",
                         genotype={"blocks": [], "hidden": 64}, hp={}, synth_ops=[])
    card = render_model_card(row, "PeMS04", SOTA_NAME, SOTA_MAE)
    assert "超越 SOTA" in card


def test_render_index_md():
    summaries = [
        {"dataset": "PeMS04", "sota_name": "STD-MAE", "sota_mae": 17.80,
         "best_ours_mae": 18.69, "n_ours": 1},
        {"dataset": "PeMS08", "sota_name": "STGformer", "sota_mae": 13.41,
         "best_ours_mae": None, "n_ours": 0},
    ]
    md = render_index_md(summaries)
    assert "# Darwin-ST 排行榜" in md
    assert "PeMS04" in md and "18.69" in md
    assert "+0.89" in md
    assert "[PeMS04](PeMS04.md)" in md
    assert "PeMS08" in md   # 即便无本项目模型也列出


def test_render_index_beats_sota_marker():
    summaries = [{"dataset": "PeMS04", "sota_name": "STD-MAE", "sota_mae": 17.80,
                  "best_ours_mae": 17.50, "n_ours": 1}]
    md = render_index_md(summaries)
    assert "🎉" in md   # 超越标记
