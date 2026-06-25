"""瓶颈诊断 (diagnosis.py) 测试 —— summarize_trace 标签 + LLM 诊断解析/校验 + prompt 工程要素。

全程无 torch / 无真实 LLM (MockLLM 注入固定响应)。锁住:
  - summarize_trace: 喂构造曲线 (过拟合/欠拟合/欠训/震荡/梯度爆炸/NaN/平台) 断言标签正确。
  - diagnose_bottleneck_llm: 解析严格 JSON + preconditions ⊆ VOCAB + 非法词过滤 + 坏 JSON→None。
  - prompt 工程: 无 persona / 字段序保 CoT / 历史避重指令 / 词表 gloss 完整。
  - gloss 覆盖整个 PRECONDITION_VOCAB (用户'14词'警觉 → 不硬编码数量, 防漂移)。
"""

from __future__ import annotations

import json

from darwin_st.creation.diagnosis import (
    PRECONDITION_GLOSS,
    BottleneckDiagnosis,
    DiagnosisSummary,
    diagnose_bottleneck_llm,
    summarize_trace,
)
from darwin_st.creation.llm import MockLLM
from darwin_st.knowledge.ontology import PRECONDITION_VOCAB


def _trace(train_loss, val_mae, gn_mean=None, gn_max=None, best_epoch=None,
           stopped_early=False, nan_hit=False):
    n = len(val_mae)
    if best_epoch is None:
        best_epoch = int(min(range(n), key=lambda i: val_mae[i])) if n else -1
    gn_mean = gn_mean or [1.0] * n
    gn_max = gn_max or [2.0] * n
    return {
        "train_loss": train_loss, "val_mae": val_mae,
        "grad_norm_mean": gn_mean, "grad_norm_max": gn_max,
        "n_epochs_run": n, "max_epochs": n, "best_epoch": best_epoch,
        "best_mae": min(val_mae) if val_mae else float("inf"),
        "stopped_early": stopped_early, "nan_hit": nan_hit,
        "lr_schedule": "cosine", "lr": 1e-3,
    }


# ---- gloss 完整性 (防硬编码漂移) ----

def test_gloss_covers_entire_vocab():
    assert set(PRECONDITION_GLOSS) == set(PRECONDITION_VOCAB)


def test_traintrace_is_picklable():
    """跨 spawn 进程铁律: TrainTrace 必须可 pickle (训练在 worker 子进程, trace 要回传主进程)。"""
    import pickle
    from dataclasses import asdict

    from darwin_st.optim.trace import TrainTrace
    t = TrainTrace(train_loss=[1.0, 0.8], val_mae=[30.0, 25.0],
                   grad_norm_mean=[1.0, 0.9], grad_norm_max=[2.0, 1.8],
                   n_epochs_run=2, max_epochs=2, best_epoch=1, best_mae=25.0,
                   final_train_loss=0.8, lr_schedule="cosine", lr=1e-3)
    back = pickle.loads(pickle.dumps(t))
    assert back.best_mae == 25.0 and back.lr_schedule == "cosine"
    # asdict 形态 (实际跨 Optuna user_attr 传的就是这个) 也能喂 summarize_trace
    s = summarize_trace(asdict(t), genotype=None, sota_gap=1.0)
    assert isinstance(s, DiagnosisSummary)


# ---- summarize_trace 标签判定 ----

def test_summarize_overfitting():
    """val 先降后回升 + train 仍降 → overfitting + early_best。"""
    tl = [1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.26, 0.25, 0.24]   # 一路降
    vm = [30, 25, 22, 21, 22, 23, 24, 25, 26, 27]                   # 第3epoch最优后回升
    s = summarize_trace(_trace(tl, vm), genotype=None, sota_gap=3.0)
    assert "overfitting" in s.labels
    assert "回升" in s.convergence_note or "过拟" in s.convergence_note


def test_summarize_underfitting():
    """best 在末尾且末段仍降 → underfitting。"""
    tl = [1.0, 0.9, 0.85, 0.8, 0.78, 0.75, 0.72, 0.70, 0.68, 0.66]
    vm = [30, 28, 26, 25, 24, 23, 22, 21, 20, 19]                  # 单调降, best 在最后
    s = summarize_trace(_trace(tl, vm), sota_gap=1.0)
    assert "underfitting" in s.labels
    assert "仍降" in s.convergence_note


def test_summarize_undertrained():
    """时间熔断截断 + 末段仍陡降 → undertrained。"""
    tl = [1.0, 0.8, 0.6, 0.5]
    vm = [30, 26, 22, 19]                                          # 仍在陡降就被截断
    s = summarize_trace(_trace(tl, vm, stopped_early=True), sota_gap=1.0)
    assert "undertrained" in s.labels


def test_summarize_grad_explosion():
    tl = [1.0, 0.9, 0.8, 0.7, 0.6]
    vm = [30, 28, 27, 26, 25]
    s = summarize_trace(_trace(tl, vm, gn_max=[2.0, 80.0, 5.0, 120.0, 3.0]), sota_gap=2.0)
    assert "grad_explosion" in s.labels
    assert "爆炸" in s.instability_note


def test_summarize_nan_hit():
    s = summarize_trace(_trace([1.0, 0.8], [30, 28], nan_hit=True), sota_gap=2.0)
    assert "nan_instability" in s.labels
    assert "NaN" in s.instability_note


def test_summarize_oscillation():
    tl = [1.0, 0.9, 0.85, 0.82, 0.80, 0.79, 0.78, 0.77]
    vm = [25, 22, 25, 21, 26, 20, 27, 21]                          # 频繁上下抖
    s = summarize_trace(_trace(tl, vm), sota_gap=2.0)
    assert "oscillation" in s.labels


def test_summarize_converged_plateau():
    tl = [1.0, 0.6, 0.4, 0.35, 0.33, 0.32, 0.315, 0.312, 0.311, 0.310]
    # best 在中段 (idx 4, 18.4), 之后基本不变 → 平台 (frac_best≈0.44 ∈ [0.3,0.85])
    vm = [30, 22, 19, 18.5, 18.4, 18.41, 18.42, 18.41, 18.42, 18.41]
    s = summarize_trace(_trace(tl, vm), sota_gap=0.5)
    assert "converged_plateau" in s.labels
    assert "平台" in s.convergence_note


def test_summarize_empty_trace_safe():
    """空轨迹不崩 (训练全 crash 的兜底)。"""
    s = summarize_trace({}, genotype=None, sota_gap=None)
    assert isinstance(s, DiagnosisSummary)
    assert s.labels == [] or isinstance(s.labels, list)


def test_summarize_arch_summary_from_genotype():
    from darwin_st.search.genotype import random_genotype
    g = random_genotype(depth=2, spatial="adaptive", temporal="tcn", hidden=64)
    s = summarize_trace(_trace([1.0, 0.8, 0.6], [30, 25, 22]), genotype=g, sota_gap=2.0)
    assert "adaptive" in s.used_spatial
    assert "tcn" in s.used_temporal
    assert "深度2" in s.arch_summary


# ---- diagnose_bottleneck_llm 解析 + 校验 ----

def _summary():
    return summarize_trace(_trace([1.0, 0.7, 0.5, 0.4, 0.38], [30, 25, 22, 21, 21]),
                           genotype=None, sota_gap=0.89)


def test_llm_diagnosis_parses_strict_json():
    resp = json.dumps({
        "reasoning": "训练 loss 持续下降但 val 末段平台, 说明已收敛, 需新建模能力",
        "evidence": "val 三段 30→22→21, 末段 rel_drop≈0",
        "diagnosis": "模型缺乏多尺度时序建模能力",
        "preconditions": ["multi_scale_structure", "long_range_dependency"],
        "direction": "引入多尺度时序分解",
    }, ensure_ascii=False)
    llm = MockLLM(lambda msgs: resp)
    diag = diagnose_bottleneck_llm(_summary(), history=[], round_idx=0, llm=llm)
    assert isinstance(diag, BottleneckDiagnosis)
    assert diag.preconditions == ["multi_scale_structure", "long_range_dependency"]
    assert "多尺度" in diag.bottleneck
    assert diag.reasoning and diag.direction


def test_llm_diagnosis_filters_illegal_preconditions():
    """非法词被过滤, 只留合法的; 全非法 → None。"""
    resp = json.dumps({"reasoning": "x", "evidence": "y", "diagnosis": "z",
                       "preconditions": ["not_a_real_word", "long_range_dependency", "也不是"],
                       "direction": "d"}, ensure_ascii=False)
    diag = diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: resp))
    assert diag.preconditions == ["long_range_dependency"]


def test_llm_diagnosis_all_illegal_returns_none():
    resp = json.dumps({"diagnosis": "z", "preconditions": ["bogus1", "bogus2"]})
    assert diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: resp)) is None


def test_llm_diagnosis_caps_at_three():
    resp = json.dumps({"diagnosis": "z", "preconditions": [
        "long_range_dependency", "multi_scale_structure", "spatial_smoothness",
        "temporal_periodicity", "heterogeneity"]})
    diag = diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: resp))
    assert len(diag.preconditions) == 3


def test_llm_diagnosis_tolerates_markdown_wrapped_json():
    """LLM 用 ```json ``` 包裹 → 正则抓第一个 {...} 仍解析。"""
    resp = "```json\n" + json.dumps({"diagnosis": "z",
            "preconditions": ["distribution_shift"]}) + "\n```"
    diag = diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: resp))
    assert diag.preconditions == ["distribution_shift"]


def test_llm_diagnosis_bad_json_returns_none():
    assert diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: "完全不是JSON的文本")) is None


def test_llm_diagnosis_llm_exception_returns_none():
    def boom(msgs):
        raise RuntimeError("api down")
    assert diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(boom)) is None


def test_llm_diagnosis_none_llm_returns_none():
    assert diagnose_bottleneck_llm(_summary(), [], 0, None) is None


def test_llm_diagnosis_retries_on_empty_then_succeeds():
    """推理模型 max_tokens 太小返回空 → 加倍预算重试一次 → 第二次成功 (踩过的真 bug 守卫)。"""
    calls = {"n": 0, "max_tokens": []}
    good = json.dumps({"diagnosis": "z", "preconditions": ["long_range_dependency"]})

    def responder(msgs):
        calls["n"] += 1
        return "" if calls["n"] == 1 else good   # 首次空 (reasoning 吃光), 重试成功

    class _MT(MockLLM):
        def chat(self, messages, temperature=0.7, max_tokens=4096):
            calls["max_tokens"].append(max_tokens)
            return self.responder(messages)

    diag = diagnose_bottleneck_llm(_summary(), [], 0, _MT(responder))
    assert diag is not None and diag.preconditions == ["long_range_dependency"]
    assert calls["n"] == 2                          # 重试了一次
    assert calls["max_tokens"][1] == calls["max_tokens"][0] * 2   # 第二次预算加倍


def test_llm_diagnosis_empty_both_attempts_returns_none():
    """两次都空 (极端) → 退回 None (规则版兜底), 不崩。"""
    diag = diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(lambda m: ""))
    assert diag is None


# ---- prompt 工程要素 (文献修正版) ----

def test_prompt_has_no_persona():
    """无 '你是专家' persona framing (EMNLP2024: 对闭集任务无益甚至有害)。

    注: 词表 gloss 里出现"混合专家"(MoE 机制名) 是合法的, 不算 persona;
    检查的是开篇角色扮演句式 (你是.../我是.../扮演...)。
    """
    captured = {}
    def cap(msgs):
        captured["system"] = msgs[0]["content"]
        captured["user"] = msgs[1]["content"]
        return json.dumps({"diagnosis": "z", "preconditions": ["long_range_dependency"]})
    diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(cap))
    sys = captured["system"]
    # 开篇不是角色扮演 (反例: "你是资深...专家")
    assert not sys.lstrip().startswith("你是")
    assert "你是资深" not in sys and "扮演" not in sys and "作为一名" not in sys
    # 任务定义在场
    assert "瓶颈" in sys and "前提词" in sys


def test_prompt_field_order_enforces_cot():
    """字段顺序指令中 reasoning→evidence→diagnosis→preconditions (推理在结论前, 保 CoT)。"""
    captured = {}
    def cap(msgs):
        captured["system"] = msgs[0]["content"]
        return json.dumps({"diagnosis": "z", "preconditions": ["long_range_dependency"]})
    diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(cap))
    sys = captured["system"]
    # 只看"字段顺序固定为"那句之后的子串里的相对顺序 (整 prompt 别处也提 preconditions)
    spec = sys.split("字段顺序固定为", 1)[1]
    assert spec.index("reasoning") < spec.index("evidence") < spec.index("diagnosis") < spec.index("preconditions")


def test_prompt_vocab_gloss_in_system():
    captured = {}
    def cap(msgs):
        captured["system"] = msgs[0]["content"]
        return json.dumps({"diagnosis": "z", "preconditions": ["long_range_dependency"]})
    diagnose_bottleneck_llm(_summary(), [], 0, MockLLM(cap))
    sys = captured["system"]
    # 全部 14 词都带 gloss 出现
    for w in PRECONDITION_VOCAB:
        assert w in sys


def test_prompt_history_novelty_instruction():
    """历史尝试注入 + 显式避重指令 (OPRO 多样化)。"""
    captured = {}
    def cap(msgs):
        captured["user"] = msgs[1]["content"]
        return json.dumps({"diagnosis": "z", "preconditions": ["multi_scale_structure"]})
    history = [
        {"round": 0, "preconditions": ["long_range_dependency"], "result_mae": 18.69},
        {"round": 1, "preconditions": ["long_range_dependency"], "result_mae": 18.69},
    ]
    diagnose_bottleneck_llm(_summary(), history, round_idx=2, llm=MockLLM(cap))
    user = captured["user"]
    assert "long_range_dependency" in user           # 历史方向被注入
    assert "不同" in user                             # 避重指令
    assert "第 3 轮" in user                          # round_idx+1


def test_prompt_history_sorted_best_last():
    """历史按 MAE 升序 (最优放末尾最靠近指令; OPRO)。"""
    captured = {}
    def cap(msgs):
        captured["user"] = msgs[1]["content"]
        return json.dumps({"diagnosis": "z", "preconditions": ["heterogeneity"]})
    history = [
        {"round": 0, "preconditions": ["spatial_smoothness"], "result_mae": 22.0},  # 差
        {"round": 1, "preconditions": ["long_range_dependency"], "result_mae": 18.7},  # 优
    ]
    diagnose_bottleneck_llm(_summary(), history, 2, MockLLM(cap))
    user = captured["user"]
    # 更优的 (18.7 / long_range) 应排在更差的 (22.0 / spatial_smoothness) 之后
    assert user.index("long_range_dependency") > user.index("spatial_smoothness")
