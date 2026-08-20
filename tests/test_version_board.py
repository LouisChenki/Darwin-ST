"""方法版本榜纯核心测试 (确定性, 无 IO/LLM)。

锁住版本榜逻辑: versions.json 解析 / 与 baseline 混排位次 (同分 baseline 在前) /
vs-SOTA 差 / 日期排序 / 空列表容错 / Markdown 渲染要素。
落盘与绘图 (scripts/build_version_board.py) 不在此测。
"""

from __future__ import annotations

from darwin_st.leaderboard import (
    VersionRecord,
    build_version_rows,
    parse_versions,
    render_version_board_md,
    render_version_readme_md,
)

# 仿 PeMS04 头部 baseline 形状 (真实 registry 数值子集)
BASELINES = {
    "STD-MAE": {"mae": 17.80, "rmse": 29.83},     # SOTA
    "STGformer": {"mae": 17.89, "rmse": 30.21},
    "HimNet": {"mae": 18.14, "rmse": 30.02},
    "STAEformer": {"mae": 18.22, "rmse": 30.18},
    "STID": {"mae": 18.35, "rmse": 29.85},
    "PDFormer": {"mae": 18.36, "rmse": 30.03},
}
SOTA_NAME, SOTA_MAE = "STD-MAE", 17.80


def _rec(version, mae, date="2026-06-25", **kw):
    return VersionRecord(version=version, date=date, best_mae=mae, dataset="PeMS04", **kw)


# ---- parse_versions ----

def test_parse_versions_roundtrip():
    data = [{"version": "v3", "date": "2026-08-07", "best_mae": 18.356,
             "tag": "acceptable+18.356", "title": "Tier-2 v1 方法升级",
             "retrospective": "docs/TIER2_V1_RETROSPECTIVE.md",
             "dataset": "PeMS04", "model_dir": "models/PeMS04/06_mae18.36_6bcc97b9"},
            {"version": "v0", "date": "2026-06-25", "best_mae": 20.67,
             "tag": None, "title": "纯 NAS + HPO", "retrospective": None,
             "dataset": "PeMS04"}]
    recs = parse_versions(data)
    assert len(recs) == 2
    v3 = recs[0]
    assert v3.version == "v3" and v3.best_mae == 18.356 and v3.dataset == "PeMS04"
    assert v3.tag == "acceptable+18.356"
    assert v3.model_dir == "models/PeMS04/06_mae18.36_6bcc97b9"
    # 缺省可选字段容错
    minimal = parse_versions([{"version": "v9", "date": "2026-09-01",
                               "best_mae": 19.0, "dataset": "PeMS08"}])[0]
    assert minimal.tag is None and minimal.title == "" and minimal.retrospective is None
    assert minimal.model_dir is None


# ---- build_version_rows ----

def test_build_version_rows_rank_against_baselines():
    """v3 best 18.356 在 PeMS04 头部集群中应排第 6 (STID 18.35 之后, PDFormer 18.36 之前)。"""
    rows = build_version_rows([_rec("v3", 18.356, date="2026-08-07")], BASELINES, SOTA_MAE)
    assert len(rows) == 1
    r = rows[0]
    assert r.rank == 6
    assert r.board_size == 7          # 6 baseline + 该版本
    assert abs(r.vs_sota - 0.556) < 1e-9


def test_build_version_rows_tie_goes_after_baseline():
    """与 baseline 同分时 baseline 在前 (与 build_leaderboard 口径一致)。"""
    rows = build_version_rows([_rec("vx", 18.35)], BASELINES, SOTA_MAE)   # 同分 STID
    assert rows[0].rank == 6          # 5 个 <= 18.35 (含 STID) → 第 6


def test_build_version_rows_beats_all_and_weakest():
    rows = build_version_rows([_rec("vSOTA", 17.50), _rec("vweak", 99.0)], BASELINES, SOTA_MAE)
    by_ver = {r.record.version: r for r in rows}
    assert by_ver["vSOTA"].rank == 1
    assert by_ver["vSOTA"].vs_sota < 0                    # 超越 SOTA
    assert by_ver["vweak"].rank == 7                      # 殿底 (6 baseline + 它)


def test_build_version_rows_sorted_by_date():
    versions = [_rec("v3", 18.356, date="2026-08-07"),
                _rec("v0", 20.67, date="2026-06-25"),
                _rec("v2", 18.349, date="2026-07-01")]
    rows = build_version_rows(versions, BASELINES, SOTA_MAE)
    assert [r.record.version for r in rows] == ["v0", "v2", "v3"]


def test_build_version_rows_empty_and_no_sota():
    assert build_version_rows([], BASELINES, SOTA_MAE) == []
    rows = build_version_rows([_rec("v0", 20.67)], BASELINES, None)
    assert rows[0].vs_sota is None and rows[0].rank == 7   # 无 SOTA 仍算位次


def test_build_version_rows_no_baselines():
    rows = build_version_rows([_rec("v0", 20.67)], {}, None)
    assert rows[0].rank == 1 and rows[0].board_size == 1


# ---- 渲染 ----

def _rows():
    versions = [
        _rec("v0", 20.67, date="2026-06-25", title="纯 NAS + HPO",
             retrospective="docs/SERVER_VALIDATION.md"),
        _rec("v2", 18.349, date="2026-07-01", tag="v-sota-pems04-18.35",
             retrospective="docs/SOTA_RETROSPECTIVE.md",
             model_dir="models/PeMS04/05_mae18.35_f93f4598"),
        _rec("v3", 18.356, date="2026-08-07", tag="acceptable+18.356",
             retrospective="docs/TIER2_V1_RETROSPECTIVE.md",
             model_dir="models/PeMS04/06_mae18.36_6bcc97b9"),
    ]
    return build_version_rows(versions, BASELINES, SOTA_MAE)


def test_render_version_board_md_key_elements():
    md = render_version_board_md("PeMS04", _rows(), sota_name=SOTA_NAME, sota_mae=SOTA_MAE)
    assert "## 方法版本榜 (PeMS04)" in md
    assert "**v0**" in md and "**v3**" in md                    # 版本行加粗
    assert "18.349" in md and "18.356" in md                  # 3 位小数可区分
    assert "20.67" in md                                      # 尾零去除
    assert "**6** / 7" in md                                  # 位次醒目
    assert "`acceptable+18.356`" in md and "`v-sota-pems04-18.35`" in md
    assert "[复盘](../docs/TIER2_V1_RETROSPECTIVE.md)" in md   # README 在 leaderboard/ 下, docs 链接上跳一级
    assert "STD-MAE" in md and "17.80" in md                  # SOTA 参照说明
    assert "[PeMS04.md](PeMS04.md)" in md                     # 完整存档指引
    assert "纯 NAS + HPO" in md                               # 一句话版本说明


def test_render_version_board_md_empty():
    md = render_version_board_md("PeMS04", [], sota_name=SOTA_NAME, sota_mae=SOTA_MAE)
    assert "## 方法版本榜 (PeMS04)" in md                     # 空表也渲染表头, 不炸
    assert "| 版本 |" in md


def test_render_version_readme_md_key_elements():
    md = render_version_readme_md("PeMS04", _rows(), SOTA_NAME, SOTA_MAE,
                                  trend_image="assets/pems04_version_trend.png")
    assert md.startswith("# Darwin-ST 排行榜")
    assert "## 方法版本榜 (PeMS04)" in md
    assert "![PeMS04 方法版本成绩演进](assets/pems04_version_trend.png)" in md
    assert "## 当前最佳" in md
    # 当前最佳 = 全版本 min(best_mae) (v2 18.349 < v3 18.356), 不假设"最新=最佳"
    sec = md[md.index("## 当前最佳"):]
    assert "**v2**" in sec and "18.349" in sec and "**5** / 7" in sec
    assert "[models/PeMS04/05_mae18.35_f93f4598](models/PeMS04/05_mae18.35_f93f4598/)" in sec
    assert "06_mae18.36_6bcc97b9" not in sec                  # 非最佳版本的模型卡不进"当前最佳"节
    assert "`acceptable+18.356`" in md
    assert "[PeMS04.md](PeMS04.md)" in md
    assert "[docs/](../docs/)" in md


def test_render_version_readme_md_no_trend_no_card():
    rows = build_version_rows([_rec("v0", 20.67)], BASELINES, SOTA_MAE)
    md = render_version_readme_md("PeMS04", rows, SOTA_NAME, SOTA_MAE, trend_image=None)
    assert "成绩随版本演进" not in md                         # 无图则省略该节
    assert "模型卡" not in md                                 # 无 model_dir 不渲染链接
    assert "## 当前最佳" in md and "20.67" in md
