"""瓶颈诊断 (Bottleneck Diagnosis) —— 训练动态 → 诊断摘要 → LLM 多样化瓶颈+受控前提词。

解决的真问题 (见 SERVER_VALIDATION.md 连体闭环结论 + deferred): 规则版 diagnose_bottleneck
措辞固定 → 关键词匹配固定前提 → 6 次创造全检索到同一机制 dilated_causal_convolution,
合成的都是它的融合变体, best 卡 18.69 无法突破。根因: **诊断信息源单一 (只有静态架构) +
确定性规则无多样化**。

三层链:
  层1 (train.py): TrainTrace 采集逐 epoch 训练动态 (loss/梯度/收敛标志)。
  层2 (本文件 summarize_trace): 80 epoch 曲线 → <300 token 定量+语义标签摘要 (确定性, 纯标量)。
  层3 (本文件 diagnose_bottleneck_llm): LLM 读摘要 → 诊断瓶颈 + 直出 1-3 个受控前提词驱动跨域检索。

Prompt 工程文献依据 (调研敲定, 纠正了 3 个有害设计):
  - **删 persona** (EMNLP2024 / 2512.05858: "你是专家"对 accuracy-critical 闭集任务无效甚至有害):
    system 直接给任务定义, 不扮角色。
  - **JSON 字段序保 CoT** (Let Me Speak Freely 2408.02442: 强制 JSON 损推理, 根因是字段顺序):
    顺序 reasoning → evidence → diagnosis → preconditions (推理/证据排结论前, 实测 +16%)。
  - **OPRO 轨迹 + in-context 避重做多样化** (OPRO 2309.03409; 温度是最弱杠杆 2405.00492;
    anchoring 顽固 2412.06593): 历轮诊断 (方向, 结果 MAE) 按 MAE 升序排 (最优放末尾最靠近指令),
    加显式 novelty 指令。比"温度 0.7 + 已用机制列表"强。
  - **LLM 直出受控前提词** (2501.12332: 词表放 system + 每词一句 gloss): 绕过脆弱的 decompose
    关键词匹配; 越界词 reject + 过滤 (闭源 API 无 trie masking)。cap 1-3 个。
  - **训练曲线用派生标签 NL 陈述** (LLMTime / 临床时序: 派生标签 >> 原始数字): 层2 的三段斜率 +
    语义标签 + 收敛/稳定 note 以自然语言陈述, 绝不喂原始 80 点曲线。

fallback: 无 key / 坏 JSON / 全非法词 → 调用方 (creation_loop._diagnose) 退回规则版 diagnose_bottleneck。
本模块不导入 torch (只吃 trace dict + genotype), 可 MockLLM 全测。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from darwin_st.knowledge.ontology import PRECONDITION_VOCAB

__all__ = [
    "DiagnosisSummary",
    "BottleneckDiagnosis",
    "summarize_trace",
    "diagnose_bottleneck_llm",
    "PRECONDITION_GLOSS",
]


# 受控前提词的一句话 gloss (放 system prompt; 文献: 每词一句释义提升 LLM 选词准确率)。
# 键必须与 PRECONDITION_VOCAB 完全一致 (test_diagnosis 断言无漂移 —— 避免硬编码"14"的脆弱)。
PRECONDITION_GLOSS: dict[str, str] = {
    "long_range_dependency": "存在长程时间依赖, 远距时刻互相影响 (SSM/注意力的前提)",
    "multi_scale_structure": "信号有多尺度结构, 不同时间/空间粒度并存 (金字塔/膨胀的前提)",
    "spatial_smoothness": "空间上相邻节点取值平滑相关 (扩散/图正则的前提)",
    "temporal_periodicity": "时间上有周期模式 (日/周节律) (周期嵌入的前提)",
    "distribution_shift": "训练与测试/未来分布漂移 (域适应/鲁棒正则的前提)",
    "node_indistinguishability": "节点缺乏可区分身份, 拓扑同构 (身份嵌入的前提)",
    "sparse_interaction": "节点间交互稀疏, 只有少数强连接 (稀疏注意力的前提)",
    "heterogeneity": "节点/模式异质, 行为不一致 (混合专家/元学习的前提)",
    "noise_corruption": "观测含噪声/缺失污染 (去噪/扩散的前提)",
    "redundant_structure": "数据冗余/低秩, 可压缩 (掩码自编码/低秩的前提)",
    "non_euclidean_topology": "非欧几里得图拓扑结构 (图卷积的前提)",
    "sequential_order": "强有序序列, 顺序含因果信息 (因果卷积/递归的前提)",
    "high_dimensionality": "特征高维, 需降维或瓶颈 (压缩/瓶颈的前提)",
    "label_scarcity": "标签稀缺, 监督信号不足 (自监督的前提)",
}


@dataclass
class DiagnosisSummary:
    """训练动态压缩成的诊断摘要 (render 成 prompt ~250 token; 不含原始曲线)。"""

    labels: list[str]                       # 语义标签 (0..N, 如 ["underfitting","early_best"])
    best_mae: float
    best_epoch: int
    n_epochs: int
    sota_gap: float | None
    train_loss_segments: tuple[float, float, float]   # 早/中/末段末值 (归一化尺度)
    val_mae_segments: tuple[float, float, float]       # 早/中/末段末值 (真实尺度)
    grad_norm_range: tuple[float, float]               # (min_mean, max_max)
    convergence_note: str                  # 一句话: "末段仍降5%/已平台/val回升过拟"
    instability_note: str                  # "梯度稳定" / "梯度爆炸N次" / "NaN熔断"
    arch_summary: str                      # genotype 人读摘要
    used_spatial: list[str] = field(default_factory=list)
    used_temporal: list[str] = field(default_factory=list)


@dataclass
class BottleneckDiagnosis:
    """LLM 诊断结果 (解析校验后)。"""

    bottleneck: str                        # 一句话瓶颈 (给 MAC 语义召回)
    preconditions: list[str]               # 1-3 个 ∈ PRECONDITION_VOCAB (给 FAC 图召回)
    reasoning: str = ""                    # 分析过程 (CoT, 引用曲线证据)
    evidence: str = ""                     # 具体训练特征
    direction: str = ""                    # 应引入的机制方向 (自然语言)


# --------------------------------------------------------------------------
# 层2: 信号压缩 (确定性, 纯标量, 无 LLM)
# --------------------------------------------------------------------------

def _seg_values(curve: list[float]) -> tuple[float, float, float]:
    """把曲线三等分, 取每段末值 (早/中/末)。空/短曲线兜底。"""
    if not curve:
        return (0.0, 0.0, 0.0)
    n = len(curve)
    if n < 3:
        v = curve[-1]
        return (curve[0], v, v)
    a, b = n // 3, (2 * n) // 3
    return (curve[a - 1] if a > 0 else curve[0],
            curve[b - 1] if b > 0 else curve[0],
            curve[-1])


def _rel_drop(start: float, end: float) -> float:
    """相对下降率 (start→end), 正=下降。"""
    denom = max(abs(start), 1e-9)
    return (start - end) / denom


def _sign_changes(curve: list[float]) -> int:
    """逐 epoch 差分的符号变化次数 (震荡判据)。"""
    if len(curve) < 3:
        return 0
    diffs = [curve[i + 1] - curve[i] for i in range(len(curve) - 1)]
    changes = 0
    for i in range(len(diffs) - 1):
        if diffs[i] * diffs[i + 1] < 0:
            changes += 1
    return changes


def _arch_summary(genotype) -> tuple[str, list[str], list[str]]:
    """genotype → (人读摘要, used_spatial, used_temporal)。吃 Genotype 或 dict 或 None。"""
    if genotype is None:
        return "(无架构信息)", [], []
    g = genotype.to_dict() if hasattr(genotype, "to_dict") else genotype
    blocks = g.get("blocks", [])
    sp = [b.get("spatial_op") for b in blocks if b.get("spatial_op")]
    tp = [b.get("temporal_op") for b in blocks if b.get("temporal_op")]
    emb = g.get("embedding", {})
    emb_bits = [k for k, on in (("node", emb.get("use_node")), ("tod", emb.get("use_tod")),
                                ("dow", emb.get("use_dow"))) if on]
    summary = (f"深度{len(blocks)} hidden{g.get('hidden')} 邻接{g.get('adj_mode')} "
               f"嵌入[{'+'.join(emb_bits) or '无'}]")
    return summary, sp, tp


def summarize_trace(trace: dict, genotype=None, sota_gap: float | None = None) -> DiagnosisSummary:
    """训练轨迹 (TrainTrace asdict) → DiagnosisSummary。确定性, 三段斜率 + 阈值打标签。

    train_loss 是归一化尺度、val_mae 真实尺度, **不可直接相减**; train-val 背离用"趋势"表达
    (train 仍降而 val 不降 → 过拟)。
    """
    trace = trace or {}
    train_loss = [float(x) for x in trace.get("train_loss", [])]
    val_mae = [float(x) for x in trace.get("val_mae", [])]
    gn_mean = [float(x) for x in trace.get("grad_norm_mean", [])]
    gn_max = [float(x) for x in trace.get("grad_norm_max", [])]
    n = int(trace.get("n_epochs_run", len(val_mae)) or len(val_mae))
    best_epoch = int(trace.get("best_epoch", -1))
    best_mae = float(trace.get("best_mae", val_mae[best_epoch] if (val_mae and 0 <= best_epoch < len(val_mae)) else float("inf")))
    stopped_early = bool(trace.get("stopped_early", False))
    nan_hit = bool(trace.get("nan_hit", False))

    tl_seg = _seg_values(train_loss)
    vm_seg = _seg_values(val_mae)

    labels: list[str] = []
    # 末段 val 相对下降率 (>0 仍降, <0 回升)
    val_tail_drop = _rel_drop(vm_seg[1], vm_seg[2]) if val_mae else 0.0
    tl_tail_drop = _rel_drop(tl_seg[1], tl_seg[2]) if train_loss else 0.0
    frac_best = (best_epoch / max(n - 1, 1)) if (best_epoch >= 0 and n > 1) else 0.0

    # --- 语义标签判定 (确定性阈值规则) ---
    if nan_hit:
        labels.append("nan_instability")
    if stopped_early and val_tail_drop > 0.03:
        labels.append("undertrained")                   # 被时间熔断截断, 末段仍陡降
    if best_epoch >= 0 and frac_best > 0.85 and val_tail_drop > 0.02:
        labels.append("underfitting")                    # best 在末尾且仍在降 → epoch/容量到顶还能再降
    if best_epoch >= 0 and frac_best < 0.3 and n >= 5:
        labels.append("early_best")                      # 很早就到最好, 后面在退化
    if (best_epoch >= 0 and frac_best < 0.7 and val_tail_drop < -0.01
            and tl_tail_drop > 0.01):
        labels.append("overfitting")                     # val 回升而 train 仍降 → 训练-验证背离
    if val_mae and abs(val_tail_drop) < 0.02 and 0.3 <= frac_best <= 0.85:
        labels.append("converged_plateau")               # 末段近乎不变 → 已收敛平台
    if val_mae and (_sign_changes(val_mae) / max(len(val_mae) - 1, 1)) > 0.4:
        labels.append("oscillation")                      # val 频繁上下抖
    if gn_mean and _median(gn_mean[len(gn_mean) // 2:]) < 1e-2:
        labels.append("grad_vanishing")
    if gn_max and max(gn_max) > 50.0:                     # clip 阈值 5.0 的 10 倍
        labels.append("grad_explosion")

    # --- 收敛 / 稳定性 note (自然语言, 给 LLM 定量锚点) ---
    if not val_mae:
        conv = "无 val 曲线 (训练未产出有效评测)"
    elif val_tail_drop > 0.03:
        conv = f"末段 val 仍降 {val_tail_drop*100:.0f}% (还有下降空间, 疑欠拟/欠训)"
    elif val_tail_drop < -0.01:
        conv = f"末段 val 回升 {abs(val_tail_drop)*100:.0f}% (疑过拟, best 在第 {best_epoch} epoch)"
    else:
        conv = "末段 val 基本平台 (已收敛, 提升需换建模能力)"

    if nan_hit:
        instab = "训练中遇 NaN/Inf 熔断 (严重不稳定)"
    elif gn_max and max(gn_max) > 50.0:
        n_explode = sum(1 for x in gn_max if x > 50.0)
        instab = f"梯度爆炸 {n_explode} 次 (max 范数 {max(gn_max):.0f}, 远超 clip 5.0)"
    elif gn_mean and _median(gn_mean[len(gn_mean) // 2:]) < 1e-2:
        instab = "末段梯度范数趋零 (疑梯度消失/学习停滞)"
    else:
        instab = "梯度稳定"

    arch, sp, tp = _arch_summary(genotype)
    gn_lo = min(gn_mean) if gn_mean else 0.0
    gn_hi = max(gn_max) if gn_max else 0.0

    return DiagnosisSummary(
        labels=labels, best_mae=best_mae, best_epoch=best_epoch, n_epochs=n,
        sota_gap=sota_gap, train_loss_segments=tl_seg, val_mae_segments=vm_seg,
        grad_norm_range=(gn_lo, gn_hi), convergence_note=conv, instability_note=instab,
        arch_summary=arch, used_spatial=sp, used_temporal=tp,
    )


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


# --------------------------------------------------------------------------
# 层3: Prompt 工程 + LLM 诊断
# --------------------------------------------------------------------------

def _vocab_block() -> str:
    """受控前提词表 + 每词 gloss (放 system)。从 PRECONDITION_VOCAB 动态生成, 不硬编码数量。"""
    # 稳定顺序 (按 gloss 字典声明序, 已覆盖全 vocab; 漂移由 test 兜底)
    lines = [f"- {w}: {PRECONDITION_GLOSS[w]}" for w in PRECONDITION_GLOSS if w in PRECONDITION_VOCAB]
    return "\n".join(lines)


def _build_system() -> str:
    """system: 无 persona, 直接任务定义 + 词表 gloss + 严格 JSON 字段序 (保 CoT)。"""
    n = len(PRECONDITION_VOCAB)
    return (
        "你在分析一个交通流预测神经网络架构搜索系统的停滞。"
        "给你当前最优架构 + 其训练动态摘要 + 历史尝试轨迹, "
        "请诊断该架构真正的核心瓶颈, 并从受控前提词表中选 1-3 个最匹配的前提词 —— "
        "这些前提词将驱动跨域机制检索, 必须精准。\n\n"
        f"受控前提词表 (只能从这 {n} 个里选, 不得自创):\n{_vocab_block()}\n\n"
        "硬约束:\n"
        "- preconditions 只能是上表中的词, 选 1-3 个。\n"
        "- 必须避开历史尝试已覆盖的方向 —— 本轮换一个新角度 (它们没解决问题)。\n"
        "- 严格输出 JSON, 无任何额外文字。字段顺序固定为: "
        "reasoning(分析过程, 引用训练曲线证据) → evidence(具体训练特征) → "
        "diagnosis(核心瓶颈一句话) → preconditions(1-3个前提词) → direction(应引入的机制方向)。"
    )


def _build_user(summary: DiagnosisSummary, history: list[dict], round_idx: int) -> str:
    """user: 分块小标题; 训练曲线只给三段+派生标签 (NL); 历史按 MAE 升序 (最优末尾)。"""
    tl = summary.train_loss_segments
    vm = summary.val_mae_segments
    gap_line = (f"当前最优落后 SOTA 约 {summary.sota_gap:.2f} MAE。"
                if summary.sota_gap is not None else "(SOTA 差距未知)")

    # OPRO 历史轨迹: 按 result_mae 升序 (最优放末尾, 最靠近指令); 只留有前提词的轮次
    hist_lines = []
    valid = [h for h in history if h.get("preconditions")]
    # MAE 升序 (None 视为最差排前); 取整分桶避免噪声
    valid.sort(key=lambda h: (h.get("result_mae") is None, h.get("result_mae") or 0.0), reverse=True)
    for h in valid[-5:]:                                  # 只留最近/最相关的几轮
        mae = h.get("result_mae")
        mae_s = f"MAE≈{mae:.1f}" if isinstance(mae, (int, float)) else "MAE未知"
        precs = ", ".join(h.get("preconditions", []))
        hist_lines.append(f"- 方向[{precs}] → {mae_s} (未突破)")
    hist_block = ("\n".join(hist_lines) if hist_lines
                  else "(首轮, 无历史) —— 请基于训练动态自由诊断。")

    return (
        f"### 诊断轮次\n第 {round_idx + 1} 轮瓶颈诊断。\n\n"
        f"### 当前最优架构\n{summary.arch_summary}\n"
        f"空间算子: {summary.used_spatial}\n时序算子: {summary.used_temporal}\n\n"
        f"### 训练动态摘要 (派生标签, 非原始曲线)\n"
        f"最优 val MAE: {summary.best_mae:.3f} (第 {summary.best_epoch}/{summary.n_epochs} epoch 达到)\n"
        f"训练 loss 三段(早/中/末, 归一化尺度): {tl[0]:.3f} → {tl[1]:.3f} → {tl[2]:.3f}\n"
        f"val MAE 三段(早/中/末, 真实尺度): {vm[0]:.2f} → {vm[1]:.2f} → {vm[2]:.2f}\n"
        f"梯度范数范围(min均值~max): {summary.grad_norm_range[0]:.3f} ~ {summary.grad_norm_range[1]:.1f}\n"
        f"收敛判断: {summary.convergence_note}\n"
        f"稳定性: {summary.instability_note}\n"
        f"自动检测标签: {summary.labels or ['(无显著特征)']}\n\n"
        f"### 与 SOTA 差距\n{gap_line}\n\n"
        f"### 历史尝试 (按结果排序, 最优在末; 请提出与这些【不同】的新方向)\n{hist_block}\n\n"
        f"### 输出 (严格 JSON, 字段顺序 reasoning→evidence→diagnosis→preconditions→direction)\n"
        f'{{"reasoning":"...","evidence":"...","diagnosis":"...",'
        f'"preconditions":["..."],"direction":"..."}}'
    )


def _parse_diagnosis(text: str) -> BottleneckDiagnosis | None:
    """解析 LLM 输出: json.loads, 容错正则抓第一个 {...}; 校验 preconditions ⊆ VOCAB。

    返回 None 表示解析失败 / 无合法前提词 (调用方退回规则版)。
    """
    if not text:
        return None
    obj = None
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)     # 抓第一个花括号块 (容错 markdown 包裹)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None
    if not isinstance(obj, dict):
        return None

    raw_precs = obj.get("preconditions", [])
    if isinstance(raw_precs, str):
        raw_precs = [raw_precs]
    # 过滤非法词 (闭源 API 无 trie masking, 只能事后过滤) + 去重保序, cap 3
    precs, seen = [], set()
    for p in raw_precs:
        if isinstance(p, str) and p in PRECONDITION_VOCAB and p not in seen:
            precs.append(p)
            seen.add(p)
    precs = precs[:3]
    if not precs:
        return None                                    # 全非法 → 失败, 退规则版

    bottleneck = str(obj.get("diagnosis") or obj.get("bottleneck") or "").strip()
    if not bottleneck:
        bottleneck = "训练动态诊断 (见 reasoning)"
    return BottleneckDiagnosis(
        bottleneck=bottleneck, preconditions=precs,
        reasoning=str(obj.get("reasoning", "")), evidence=str(obj.get("evidence", "")),
        direction=str(obj.get("direction", "")),
    )


def diagnose_bottleneck_llm(summary: DiagnosisSummary, history: list[dict],
                            round_idx: int, llm, temperature: float = 0.7,
                            max_tokens: int = 8192) -> BottleneckDiagnosis | None:
    """LLM 瓶颈诊断: 摘要 + 历史 → 严格 JSON → BottleneckDiagnosis。

    返回 None 表示任何失败 (LLM 异常 / 坏 JSON / 全非法前提词) → 调用方退回规则版。

    **max_tokens 必须给足 (关键, 踩过坑)**: DeepSeek-v4 是推理模型, 内部 reasoning trace 先吃
    token, 之后才吐可见 JSON。可见 JSON 很短 (~200 token), 但 reasoning 可能很长且不定长。
    预算太小 → reasoning 把额度耗尽 → 可见内容为空 → 解析失败静默退回规则版 (诊断多样化失效)。
    实测: max_tokens=10 必空; 1024 在 temperature>0 下概率性返回空; 故默认给 8192 留足 reasoning
    余量。若仍空 (极长 reasoning), 重试一次加倍预算; 再不行才退规则版 (并打日志, 不再静默)。
    """
    if llm is None:
        return None
    messages = [
        {"role": "system", "content": _build_system()},
        {"role": "user", "content": _build_user(summary, history, round_idx)},
    ]
    for attempt, mt in enumerate((max_tokens, max_tokens * 2)):
        try:
            text = llm.chat(messages, temperature=temperature, max_tokens=mt)
        except Exception as e:
            print(f"[诊断] LLM 调用异常 (attempt {attempt}): {type(e).__name__}: {str(e)[:120]}")
            return None
        diag = _parse_diagnosis(text)
        if diag is not None:
            return diag
        # 空/不可解析 → 多半是 reasoning 吃光额度; 重试加倍预算 (仅一次)
        print(f"[诊断] 第 {attempt+1} 次解析失败 (max_tokens={mt}, 返回长度={len(text or '')}), "
              + ("加倍重试" if attempt == 0 else "退回规则版"))
    return None

