"""排行榜聚合与渲染 (Leaderboard) —— 把实验成果汇成可复现、可展示的静态榜单。

每个数据集一张榜: 混排 baseline_registry 的已发表 baseline/SOTA 与本项目跑出的"优秀模型",
按 MAE 升序。"优秀"的定义(用户明确"不一定要达到SOTA"): **跑赢至少最弱的已发表 baseline**
即够格入榜存档(过滤掉冒烟/退化的垃圾架构, 但保留像 18.69 这种逼近 SOTA 的跨域合成成果)。

本模块是**纯函数核心**(无 IO / 无 LLM / 确定性), 可 pytest 全测:
  - select_our_models: 从 memory 行筛 KEEP + 超阈值 + (可选)run_tag 隔离, 按签名去重留最优。
  - build_leaderboard: 合并 baseline + 本项目, 排名, 标类型(baseline/sota/ours), 算 vs-SOTA 差。
  - detect_synth_ops: 从 genotype 认出用到的 synth_ 跨域合成算子。
  - render_*: 渲成 Markdown(榜单表 / 模型卡 / 总览索引)。

另含**方法版本榜**(仿 artificialanalysis 门面: 每个大版本只上榜其最佳模型, 展示方法演进):
  - VersionRecord/VersionRow + parse_versions: versions.json 的记录结构与解析。
  - build_version_rows: 版本最佳与已发表 baseline 混排, 算榜单位次与 vs-SOTA 差。
  - render_version_board_md / render_version_readme_md: 渲版本主榜表 / 门面 README。

文件落盘(sqlite 读、synth 代码拷贝、写 .md)由 scripts/export_leaderboard.py 这层薄壳负责;
版本榜的 versions.json 读取、趋势图绘制、README 落盘由 scripts/build_version_board.py 负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from darwin_st.creation.registry import SYNTH_PREFIX

__all__ = [
    "LeaderboardRow",
    "select_our_models",
    "build_leaderboard",
    "detect_synth_ops",
    "weakest_baseline_mae",
    "short_sig",
    "model_dir_name",
    "render_leaderboard_md",
    "render_model_card",
    "render_index_md",
    "VersionRecord",
    "VersionRow",
    "parse_versions",
    "build_version_rows",
    "render_version_board_md",
    "render_version_readme_md",
]


@dataclass
class LeaderboardRow:
    """榜单一行 (baseline / SOTA / 本项目模型 统一表示)。"""

    rank: int
    name: str                       # 模型名 (baseline 名 或 本项目 short_sig)
    kind: str                       # "baseline" | "sota" | "ours"
    mae: float
    rmse: float | None = None
    vs_sota: float | None = None    # mae - sota_mae (正=落后 SOTA, 负=超越)
    num_params: int | None = None
    is_ours: bool = False
    model_dir: str | None = None    # 本项目模型的存档相对目录 (baseline 为 None)
    # 以下仅本项目模型有 (供模型卡渲染)
    trial_id: int | None = None
    signature: str | None = None
    genotype: dict | None = None
    hp: dict | None = None
    synth_ops: list[str] = field(default_factory=list)
    test_mae: float | None = None
    val_rmse: float | None = None
    created_at: str | None = None
    run_tag: str | None = None


def detect_synth_ops(genotype: dict | None) -> list[str]:
    """从 genotype 认出用到的 synth_ 跨域合成算子 (去重保序)。

    合成算子注册为 synth_ 前缀的 spatial op; 也扫 temporal/fusion 兜底 (未来扩展)。
    """
    if not genotype:
        return []
    found: list[str] = []
    for block in genotype.get("blocks", []):
        for key in ("spatial_op", "temporal_op", "fusion"):
            op = block.get(key)
            if isinstance(op, str) and op.startswith(SYNTH_PREFIX) and op not in found:
                found.append(op)
    return found


def weakest_baseline_mae(baselines: dict[str, dict]) -> float | None:
    """最弱(最大 MAE)的已发表 baseline 分 —— 默认入榜门槛 (超它即够格存档)。"""
    maes = [e["mae"] for e in baselines.values() if e.get("mae") is not None]
    return max(maes) if maes else None


def short_sig(signature: str | None, n: int = 8) -> str:
    return (signature or "unknown")[:n]


def model_dir_name(rank: int, mae: float, signature: str | None) -> str:
    """本项目模型存档目录名: <rank>_<mae>_<short_sig>, 排名零填充便于排序。"""
    return f"{rank:02d}_mae{mae:.2f}_{short_sig(signature)}"


def select_our_models(
    trials: list[dict],
    threshold_mae: float | None,
    *,
    run_tag: str | None = None,
) -> list[dict]:
    """从 memory 实验行筛出够格入榜的本项目模型。

    规则:
      - 只取 status=KEEP 且 val_mae 有限。
      - threshold_mae 给定时只留 val_mae < threshold (超 baseline 就录; None=不设限全收)。
      - run_tag 给定时只留该 run (隔离历史实验, 见 deferred 的 run_tag 混淆问题); None=全收。
      - **按 signature 去重**: 同架构多次评估只留 val_mae 最小的一条 (最优代表)。
    返回按 val_mae 升序的实验行列表 (原 dict + 解析好的 genotype/hp)。
    """
    best_by_sig: dict[str, dict] = {}
    for t in trials:
        if t.get("status") != "KEEP":
            continue
        mae = t.get("val_mae")
        if mae is None or not _finite(mae):
            continue
        if threshold_mae is not None and mae >= threshold_mae:
            continue
        if run_tag is not None and t.get("run_tag") != run_tag:
            continue
        sig = t.get("signature") or ""
        prev = best_by_sig.get(sig)
        if prev is None or mae < prev["val_mae"]:
            best_by_sig[sig] = t
    return sorted(best_by_sig.values(), key=lambda r: r["val_mae"])


def build_leaderboard(
    dataset: str,
    our_models: list[dict],
    baselines: dict[str, dict],
    sota_name: str | None,
    sota_mae: float | None,
) -> list[LeaderboardRow]:
    """合并 baseline + 本项目模型成一张排好序的榜单。

    - baseline/SOTA 来自 baseline_registry (sota_name 那条标 kind='sota', 余 'baseline')。
    - 本项目模型标 kind='ours', is_ours=True, 带 genotype/hp/synth_ops/model_dir。
    - 全部按 MAE 升序; rank 从 1; vs_sota = mae - sota_mae。
    """
    rows: list[LeaderboardRow] = []

    # baseline / SOTA 行
    for name, entry in baselines.items():
        mae = entry.get("mae")
        if mae is None:
            continue
        rows.append(LeaderboardRow(
            rank=0, name=name,
            kind="sota" if name == sota_name else "baseline",
            mae=mae, rmse=entry.get("rmse"),
            vs_sota=(mae - sota_mae) if sota_mae is not None else None,
            is_ours=False,
        ))

    # 本项目模型行
    for t in our_models:
        mae = t["val_mae"]
        geno = t.get("genotype") or {}
        synth = detect_synth_ops(geno)
        rows.append(LeaderboardRow(
            rank=0, name="darwin-st",   # 渲染时按 rank 补 short_sig
            kind="ours", mae=mae, rmse=t.get("val_rmse"),
            vs_sota=(mae - sota_mae) if sota_mae is not None else None,
            num_params=t.get("num_params"), is_ours=True,
            trial_id=t.get("id"), signature=t.get("signature"),
            genotype=geno, hp=t.get("hp") or {}, synth_ops=synth,
            test_mae=t.get("test_mae"), val_rmse=t.get("val_rmse"),
            created_at=t.get("created_at"), run_tag=t.get("run_tag"),
        ))

    # 升序排名 (MAE 小在前; 同分 baseline 在前以稳定)
    rows.sort(key=lambda r: (r.mae, 0 if not r.is_ours else 1))
    for i, r in enumerate(rows, start=1):
        r.rank = i
        if r.is_ours:
            dir_name = model_dir_name(r.rank, r.mae, r.signature)
            r.model_dir = f"models/{dataset}/{dir_name}"
            r.name = f"darwin-st/{short_sig(r.signature)}"
    return rows


# --------------------------------------------------------------------------
# 渲染 (纯字符串, 无 IO)
# --------------------------------------------------------------------------

def _fmt(x: float | None, prec: int = 2) -> str:
    return f"{x:.{prec}f}" if isinstance(x, (int, float)) else "—"


def _fmt_delta(x: float | None) -> str:
    if x is None:
        return "—"
    if abs(x) < 1e-9:
        return "±0.00 (=SOTA)"
    return f"{x:+.2f}"


def _kind_label(kind: str) -> str:
    return {"sota": "🏆 SOTA", "baseline": "baseline", "ours": "**本项目**"}.get(kind, kind)


def _params_str(n: int | None) -> str:
    if not n:
        return "—"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def render_leaderboard_md(
    dataset: str, rows: list[LeaderboardRow], protocol_note: str | None = None,
    sota_name: str | None = None, sota_mae: float | None = None,
) -> str:
    """渲染单数据集榜单 Markdown (混排表, 高亮本项目行)。"""
    lines = [f"# {dataset} 排行榜", ""]
    if sota_name and sota_mae is not None:
        lines.append(f"**当前 SOTA**: {sota_name} (MAE {sota_mae:.2f})  ")
    n_ours = sum(1 for r in rows if r.is_ours)
    lines.append(f"**本项目入榜模型**: {n_ours} 个  ")
    if protocol_note:
        lines.append(f"**评测协议**: {protocol_note}  ")
    lines += ["", "| # | 模型 | 类型 | MAE | RMSE | vs SOTA | 参数量 | 存档 |",
              "|---:|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        name = f"**{r.name}**" if r.is_ours else r.name
        link = f"[📂]({r.model_dir}/)" if (r.is_ours and r.model_dir) else "—"
        lines.append(
            f"| {r.rank} | {name} | {_kind_label(r.kind)} | {_fmt(r.mae)} | "
            f"{_fmt(r.rmse)} | {_fmt_delta(r.vs_sota)} | {_params_str(r.num_params)} | {link} |"
        )
    lines += ["", "> 类型说明: 🏆 SOTA / baseline 来自已发表文献 (corrected literature numbers); "
              "**本项目** 为 Darwin-ST 进化 NAS + LLM 跨域创造自动产出, 含可复现架构与算子代码。", ""]
    return "\n".join(lines)


def render_model_card(
    row: LeaderboardRow, dataset: str, sota_name: str | None, sota_mae: float | None,
    synth_meta: dict[str, dict] | None = None, protocol_note: str | None = None,
    weights_file: str | None = None,
) -> str:
    """渲染单个本项目模型的人读模型卡 (架构 + 超参 + 用到的跨域算子 + 性能 + 谱系)。

    weights_file: 归档目录里的权重文件名 (如 "model.pt", 由导出层拷入 checkpoint 存档后传入);
    给定时在"复现"一节加一行权重存档说明。None = 无权重存档 (旧行为)。
    """
    synth_meta = synth_meta or {}
    g = row.genotype or {}
    blocks = g.get("blocks", [])
    emb = g.get("embedding", {})

    lines = [f"# 模型卡: {row.name}", "",
             f"- **数据集**: {dataset}",
             f"- **排名**: 第 {row.rank} 名" + (f" / 库内 KEEP 实验" if row.is_ours else ""),
             f"- **val MAE**: {_fmt(row.mae)}" + (f"  (test MAE {_fmt(row.test_mae)})" if row.test_mae else ""),
             f"- **val RMSE**: {_fmt(row.val_rmse)}",
             f"- **参数量**: {_params_str(row.num_params)}",
             f"- **实验 id**: {row.trial_id}   **签名**: `{row.signature}`",
             f"- **run_tag**: {row.run_tag}   **时间**: {row.created_at}"]
    if sota_mae is not None:
        delta = row.mae - sota_mae
        verdict = "**超越 SOTA** 🎉" if delta < 0 else f"距 SOTA 差 {delta:+.2f}"
        lines.append(f"- **vs SOTA** ({sota_name} {sota_mae:.2f}): {verdict}")
    if protocol_note:
        lines.append(f"- **评测协议**: {protocol_note}")

    # 架构
    lines += ["", "## 架构", "",
              f"- 深度 (blocks): {len(blocks)}   hidden: {g.get('hidden')}   邻接: {g.get('adj_mode')}",
              "- 时空块:"]
    for i, b in enumerate(blocks):
        mark = " 🔬" if any(str(b.get(k, '')).startswith(SYNTH_PREFIX)
                            for k in ('spatial_op', 'temporal_op', 'fusion')) else ""
        lines.append(f"  {i}. spatial=`{b.get('spatial_op')}` temporal=`{b.get('temporal_op')}` "
                     f"fusion=`{b.get('fusion')}`{mark}")
    if emb:
        lines.append(f"- 身份嵌入: node={emb.get('use_node')}({emb.get('node_dim')}) "
                     f"tod={emb.get('use_tod')}({emb.get('tod_dim')}) "
                     f"dow={emb.get('use_dow')}({emb.get('dow_dim')})")
    if g.get("protected"):
        lines.append(f"- 创新点保护区: {g.get('protected')}")

    # 跨域合成算子 (项目核心创新)
    if row.synth_ops:
        lines += ["", "## 🔬 跨域合成算子 (LLM 创造)", "",
                  "本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:"]
        for op in row.synth_ops:
            meta = synth_meta.get(op, {})
            lines.append(f"\n### `{op}`")
            if meta.get("composition"):
                lines.append(f"- **组合方式**: {meta['composition']}")
            if meta.get("source_mechanisms"):
                lines.append(f"- **源机制 (跨域)**: {', '.join(meta['source_mechanisms'])}")
            if meta.get("real_mae") is not None:
                lines.append(f"- **该算子实测 MAE**: {_fmt(meta['real_mae'])}")
            lines.append(f"- **算子代码**: [`operators/{op}.py`](operators/{op}.py)")
    else:
        lines += ["", "## 算子", "", "纯 Tier-1 进化架构 (未用 LLM 合成算子)。"]

    # 超参
    if row.hp:
        lines += ["", "## 超参 (HPO 择优)", "", "```json"]
        import json as _json
        lines.append(_json.dumps(row.hp, ensure_ascii=False, indent=2))
        lines.append("```")

    lines += ["", "## 复现", "",
              "- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`",
              "- 超参: [`hparams.json`](hparams.json)"]
    if weights_file:
        lines.append(f"- 权重存档: [`{weights_file}`]({weights_file}) "
                     "(训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)")
    if row.synth_ops:
        lines.append("- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建")
    lines.append("")
    return "\n".join(lines)


def render_index_md(dataset_summaries: list[dict]) -> str:
    """渲染总览 README: 各数据集最佳本项目模型 vs SOTA 一览。

    dataset_summaries: [{dataset, sota_name, sota_mae, best_ours_mae, best_ours_dir,
                         n_ours, n_baselines}]
    """
    lines = ["# Darwin-ST 排行榜", "",
             "自治时空预测研究系统的成果存档: 每个数据集一张榜, 混排已发表 baseline/SOTA 与本项目"
             "(进化 NAS + LLM 跨域创造) 自动产出的模型。本项目模型均附可复现架构 + 超参 + 合成算子代码。", "",
             "## 总览", "",
             "| 数据集 | SOTA | 本项目最佳 | vs SOTA | 入榜数 | 榜单 |",
             "|---|---|---:|---:|---:|---|"]
    for s in dataset_summaries:
        ds = s["dataset"]
        sota = f"{s['sota_name']} {s['sota_mae']:.2f}" if s.get("sota_mae") is not None else "—"
        best = _fmt(s.get("best_ours_mae"))
        if s.get("best_ours_mae") is not None and s.get("sota_mae") is not None:
            d = s["best_ours_mae"] - s["sota_mae"]
            vs = f"{d:+.2f}" + (" 🎉" if d < 0 else "")
        else:
            vs = "—"
        lines.append(f"| {ds} | {sota} | {best} | {vs} | {s.get('n_ours', 0)} | [{ds}]({ds}.md) |")
    lines += ["", "---", "",
              "> baseline/SOTA 为 corrected literature numbers (见 `baseline_registry.py`); "
              "本项目 MAE 为真实 masked 评测、按各数据集协议口径 (见各榜单页)。", ""]
    return "\n".join(lines)


def _finite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


# --------------------------------------------------------------------------
# 方法版本榜 (Version Board) —— 每个大版本只上榜其最佳模型, 展示方法演进
# --------------------------------------------------------------------------

@dataclass
class VersionRecord:
    """方法版本记录 (leaderboard/versions.json 一条)。

    - version/date/best_mae/dataset 必填; dataset 预留多数据集扩展。
    - tag: 该版本的 git tag 或 commit (无则 None)。
    - retrospective: 对应 docs/ 复盘文档的仓库相对路径 (无则 None)。
    - model_dir: 该版本最佳模型在完整存档里的相对目录 (有存档则填, 供"当前最佳"链接)。
    """

    version: str
    date: str
    best_mae: float
    dataset: str
    tag: str | None = None
    title: str = ""
    retrospective: str | None = None
    model_dir: str | None = None


@dataclass
class VersionRow:
    """版本榜单一行: 版本记录 + 与 baseline 混排后的位次 + vs SOTA 差。"""

    record: VersionRecord
    rank: int                 # 该版本最佳在与 baseline 混排榜上的位次 (从 1)
    board_size: int           # 混排榜总行数 (baseline 数 + 1)
    vs_sota: float | None     # best_mae - sota_mae (正=落后 SOTA)


def parse_versions(data: list[dict]) -> list[VersionRecord]:
    """把 versions.json 的 dict 列表解析成 VersionRecord (容错: 缺省字段取默认)。"""
    records = []
    for d in data:
        records.append(VersionRecord(
            version=str(d["version"]), date=str(d["date"]),
            best_mae=float(d["best_mae"]), dataset=str(d["dataset"]),
            tag=d.get("tag"), title=str(d.get("title", "")),
            retrospective=d.get("retrospective"), model_dir=d.get("model_dir"),
        ))
    return records


def build_version_rows(
    versions: list[VersionRecord],
    baselines: dict[str, dict],
    sota_mae: float | None,
) -> list[VersionRow]:
    """算每个版本的榜单位次与 vs-SOTA 差, 按日期升序返回 (演进叙事序)。

    位次口径: 把**该版本最佳单独**与已发表 baseline 混排 (不同版本互不占位),
    同分 baseline 在前 (与 build_leaderboard 一致) —— 即
    rank = 1 + 满足 baseline.mae <= best_mae 的 baseline 数。
    空版本列表返回 []; sota_mae=None 时 vs_sota 为 None。
    """
    baseline_maes = [e["mae"] for e in baselines.values() if e.get("mae") is not None]
    rows = []
    for rec in sorted(versions, key=lambda r: (r.date, r.version)):
        rank = 1 + sum(1 for m in baseline_maes if m <= rec.best_mae)
        rows.append(VersionRow(
            record=rec, rank=rank, board_size=len(baseline_maes) + 1,
            vs_sota=(rec.best_mae - sota_mae) if sota_mae is not None else None,
        ))
    return rows


def _fmt_mae3(x: float) -> str:
    """MAE 格式化: 至多 3 位小数并去尾零 (18.349/18.356 可区分, 20.67 不补零)。"""
    return f"{x:.3f}".rstrip("0").rstrip(".")


def render_version_board_md(
    dataset: str,
    rows: list[VersionRow],
    sota_name: str | None = None,
    sota_mae: float | None = None,
) -> str:
    """渲染方法版本主榜 Markdown 表 (AA 风: 列少而精, 名次醒目, 版本行加粗)。

    列: 版本(含一句话说明) / 日期 / Best MAE / vs SOTA / 榜单位次 / Tag / 复盘。
    """
    lines = [f"## 方法版本榜 ({dataset})", ""]
    if sota_name and sota_mae is not None:
        lines.append(f"参照 SOTA: **{sota_name}** (MAE {sota_mae:.2f}); MAE 越低越好。")
        lines.append("")
    lines += ["| 版本 | 日期 | Best MAE | vs SOTA | 榜单位次 | Tag | 复盘 |",
              "|---|---|---:|---:|---:|---|---|"]
    for r in rows:
        rec = r.record
        ver = f"**{rec.version}**"
        if rec.title:
            ver += f"<br>{rec.title}"
        tag = f"`{rec.tag}`" if rec.tag else "—"
        # README 落在 leaderboard/ 下, 仓库根相对路径 (docs/...) 需上跳一级才可在 GitHub 解析
        retro = (f"[复盘](../{rec.retrospective})" if rec.retrospective and not rec.retrospective.startswith("../")
                 else (f"[复盘]({rec.retrospective})" if rec.retrospective else "—"))
        lines.append(
            f"| {ver} | {rec.date} | **{_fmt_mae3(rec.best_mae)}** | "
            f"{_fmt_delta(r.vs_sota)} | **{r.rank}** / {r.board_size} | {tag} | {retro} |"
        )
    lines += ["",
              f"> 位次口径: 该版本最佳单独与已发表 baseline 混排 (不同版本互不占位); "
              f"完整模型存档见 [{dataset}.md]({dataset}.md)。", "",
              "> ⚠️ **指标口径声明**: 本项目版本行 Best MAE 为**验证集 (val) MAE** "
              "(split 6/2/2, masked metric, 12 horizons 平均); 已发表 baseline/SOTA 数值为**文献报告的 test MAE**。"
              "两者口径不同, 混排位次仅供演进参照, 不构成严格可比的排名结论。", ""]
    return "\n".join(lines)


def render_version_readme_md(
    dataset: str,
    rows: list[VersionRow],
    sota_name: str | None,
    sota_mae: float | None,
    *,
    trend_image: str | None = None,
    archive_md: str | None = None,
) -> str:
    """渲染版本榜门面 README: 一句话说明 + 版本主榜 + 趋势图 + 当前最佳 + 存档/文档链接。

    trend_image: 趋势图相对路径 (如 assets/pems04_version_trend.png); None 则省略该节。
    archive_md: 完整模型存档榜单文件名 (默认 f"{dataset}.md")。
    """
    archive_md = archive_md or f"{dataset}.md"
    lines = ["# Darwin-ST 排行榜", "",
             "自治时空预测研究系统 (进化 NAS + LLM 跨域创造) 的方法版本榜: "
             "每个大版本只上榜其最佳模型, 展示方法一版版变强的演进主线。", ""]

    board = render_version_board_md(dataset, rows, sota_name=sota_name, sota_mae=sota_mae)
    lines.append(board)

    if trend_image:
        lines += ["## 成绩随版本演进", "",
                  f"![{dataset} 方法版本成绩演进]({trend_image})", ""]

    # 当前最佳: 取全版本 min(best_mae) —— 版本不一定单调变强 (如 v4 路由失效收关),
    # 不能假设"最新=最佳"。同分时取日期更早者 (先达到者占优)。
    if rows:
        cur = min(rows, key=lambda r: (r.record.best_mae, r.record.date))
        lines += ["## 当前最佳", "",
                  f"- **{cur.record.version}** ({cur.record.date}): "
                  f"Best MAE **{_fmt_mae3(cur.record.best_mae)}**, "
                  f"榜单位次 **{cur.rank}** / {cur.board_size}" +
                  (f", 距 SOTA ({sota_name} {sota_mae:.2f}) {_fmt_delta(cur.vs_sota)}"
                   if sota_mae is not None else "")]
        if cur.record.model_dir:
            lines.append(f"- 模型卡: [{cur.record.model_dir}]({cur.record.model_dir}/)")
        if cur.record.tag:
            lines.append(f"- git tag: `{cur.record.tag}`")
        lines.append("")

    lines += ["## 完整存档与设计文档", "",
              f"- 完整模型存档榜 (全部入榜模型 + baseline 混排): [{archive_md}]({archive_md})",
              "- 各版本复盘与设计文档: [docs/](../docs/)", ""]
    return "\n".join(lines)
