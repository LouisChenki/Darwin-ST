"""B3 评估中间档 (proxy 短训粗筛) 的正确性回归测试。

核心契约:
  - train_one(max_train_batches=N): 每 epoch 训练 batch 数被截断到 N (数据管道不动)
  - proxy_eval_genotype: CPU 玩具数据短训返回有限 MAE
  - CreationLoop._assign_seed_budgets: proxy 升序前 top_k 深评大预算, 其余浅评小预算;
    proxy 异常记 inf 不拖死整轮; 候选数 ≤ top_k / proxy_fn=None / 开关关 → 全部大预算
  - maybe_create 集成: 履历 record 带 proxy_mae, budget 分配正确
无 LLM/GPU/网络 (mock 全链 + 合成小数据)。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from darwin_st.creation import (
    CreationConfig,
    CreationLoop,
    MockLLM,
    OperatorRegistry,
    OperatorSynthesizer,
    SynthesisConfig,
)
from darwin_st.creation.creation_archive import CreationArchive
from darwin_st.data import prepare as P
from darwin_st.knowledge import HashEmbedder, InMemoryGraphStore, all_seed_mechanisms
from darwin_st.optim.train import proxy_eval_genotype, train_one
from darwin_st.search import operators as ops_mod
from darwin_st.search.genotype import Genotype, STBlock, random_genotype


# ---------------------------------------------------------------------------
# proxy_eval_genotype + max_train_batches (CPU 玩具数据, 同 test_train 的 setup)
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """预置一个 tiny 的已处理数据集 (10 节点, 少量样本), 跳过下载。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
    from darwin_st.data.protocol import DatasetProfile, PROFILES
    prof = DatasetProfile(
        name="TINY", num_nodes=10, num_channels=3, target_channel=0,
        split_ratios=(0.6, 0.2, 0.2), seq_len_in=12, seq_len_out=12,
        report_mode="average", report_horizons=tuple(range(1, 13)),
        null_val=0.0, graph_source="distance_csv",
    )
    monkeypatch.setitem(PROFILES, "TINY", prof)

    data_dir = P.data_dir_for("TINY")
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    N, C = 10, 3
    for split, n in (("train", 64), ("val", 24), ("test", 24)):
        x = rng.random((n, 12, N, C)).astype(np.float32)
        y = rng.random((n, 12, N)).astype(np.float32)
        np.save(os.path.join(data_dir, f"{split}_x.npy"), x)
        np.save(os.path.join(data_dir, f"{split}_y.npy"), y)
    np.save(os.path.join(data_dir, "scaler.npy"), np.array([50.0, 20.0]))
    adj = np.eye(N, dtype=np.float32)
    np.save(os.path.join(data_dir, "adj.npy"), adj)
    return prof, data_dir, adj


def test_proxy_eval_genotype_returns_finite_mae(tiny_data):
    """proxy 短训在玩具数据上返回有限 val MAE (真实尺度 masked)。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    mae = proxy_eval_genotype(geno, data_dir, prof, adj, device="cpu",
                              epochs=2, max_train_batches=2, batch_size=16)
    assert np.isfinite(mae) and mae > 0


def test_train_one_max_train_batches_truncates(tiny_data, monkeypatch):
    """max_train_batches=2: 每 epoch 只消费前 2 个训练 batch (全量 64/16=4), epoch 照常评测。"""
    prof, data_dir, adj = tiny_data
    real_load = P.load_split
    counts = {"train": 0}

    class _CountingLoader:                       # 可重迭代包装 (DataLoader 每 epoch 重迭代)
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for b in self.loader:
                counts["train"] += 1
                yield b

    def spy(data_dir_, split, *a, **kw):
        loader = real_load(data_dir_, split, *a, **kw)
        return _CountingLoader(loader) if split == "train" else loader

    monkeypatch.setattr(P, "load_split", spy)
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    mae, trace = train_one(geno, {"lr": 1e-3, "batch_size": 16}, data_dir, prof, adj,
                           device="cpu", max_epochs=3, max_train_batches=2)
    assert np.isfinite(mae)
    assert trace.n_epochs_run == 3            # epoch 数不截 (只截 epoch 内 batch)
    assert counts["train"] == 3 * 2           # 每 epoch 恰好 2 个 batch (全量是 4)


def test_train_one_max_train_batches_none_unchanged(tiny_data, monkeypatch):
    """max_train_batches=None (默认) → 全量 batch, 行为不变 (向后兼容)。"""
    prof, data_dir, adj = tiny_data
    real_load = P.load_split
    counts = {"train": 0}

    class _CountingLoader:
        def __init__(self, loader):
            self.loader = loader

        def __iter__(self):
            for b in self.loader:
                counts["train"] += 1
                yield b

    def spy(data_dir_, split, *a, **kw):
        loader = real_load(data_dir_, split, *a, **kw)
        return _CountingLoader(loader) if split == "train" else loader

    monkeypatch.setattr(P, "load_split", spy)
    geno = random_genotype(depth=1, hidden=16)
    train_one(geno, {"lr": 1e-3, "batch_size": 16}, data_dir, prof, adj,
              device="cpu", max_epochs=2)     # 不传 → 默认 None
    assert counts["train"] == 2 * 4           # 64 样本 / batch 16 = 全量 4 batch/epoch


# ---------------------------------------------------------------------------
# _assign_seed_budgets (mock proxy_fn, 纯逻辑)
# ---------------------------------------------------------------------------


def _bare_loop(cfg: CreationConfig | None = None, proxy_fn=None) -> CreationLoop:
    """无外部依赖的裸 CreationLoop (只测 _assign_seed_budgets / _make_seed_genotype)。"""
    return CreationLoop(None, None, None, None, config=cfg, proxy_fn=proxy_fn)


def _seeds(loop: CreationLoop, n: int) -> list[Genotype]:
    return [loop._make_seed_genotype(None, "gcn") for _ in range(n)]


def test_assign_budgets_top_k_gets_deep_budget():
    """3 候选 top_k=2: proxy 分最低的两个拿大预算, 其余小预算; 返回对齐的 proxy 分。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    scores = iter([0.5, 0.1, 0.3])
    loop = _bare_loop(cfg, proxy_fn=lambda g: next(scores))
    seeds = _seeds(loop, 3)
    proxy_maes = loop._assign_seed_budgets(seeds)
    budgets = [s._seed_meta["hpo_trials"] for s in seeds]
    assert budgets == [3, 20, 20]             # 0.1/0.3 最小 → 深评; 0.5 → 浅评
    assert proxy_maes == [0.5, 0.1, 0.3]      # 与 seeds 原顺序对齐 (履历用)


def test_assign_budgets_proxy_exception_marks_inf_and_survives():
    """单个候选 proxy 崩溃 → 记 inf 排尾拿小预算, 不拖死整轮; 其余照常。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    calls = []

    def proxy(g):
        calls.append(g)
        if len(calls) == 2:                   # 第二个候选崩溃
            raise RuntimeError("mock proxy crash")
        return 0.1

    loop = _bare_loop(cfg, proxy_fn=proxy)
    seeds = _seeds(loop, 3)
    proxy_maes = loop._assign_seed_budgets(seeds)   # 不抛异常
    assert proxy_maes[1] == float("inf")            # 崩溃候选记 inf
    budgets = [s._seed_meta["hpo_trials"] for s in seeds]
    assert budgets == [20, 3, 20]                   # inf 候选浅评, 两个 0.1 深评


def test_assign_budgets_tie_keeps_original_order():
    """proxy 并列 (含 inf 并列) 时稳定排序保持原顺序, 深评名额恰为 top_k。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    scores = iter([float("inf"), float("inf"), 0.1])
    loop = _bare_loop(cfg, proxy_fn=lambda g: next(scores))
    seeds = _seeds(loop, 3)
    loop._assign_seed_budgets(seeds)
    budgets = [s._seed_meta["hpo_trials"] for s in seeds]
    assert budgets == [20, 3, 20]   # 0.1 第一; 两个 inf 并列 → 原顺序靠前者 (idx0) 拿第二个深评名额


def test_assign_budgets_all_big_when_candidates_le_top_k():
    """候选数 ≤ top_k → 全部大预算, proxy_fn 一次不调。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    called = []
    loop = _bare_loop(cfg, proxy_fn=lambda g: called.append(g) or 0.1)
    seeds = _seeds(loop, 2)
    proxy_maes = loop._assign_seed_budgets(seeds)
    assert called == []                                  # proxy 未调
    assert [s._seed_meta["hpo_trials"] for s in seeds] == [20, 20]
    assert proxy_maes == [None, None]


def test_assign_budgets_no_proxy_fn_unchanged():
    """proxy_fn=None → 全部保持 _make_seed_genotype 的大预算 (行为与 B3 前一致)。"""
    loop = _bare_loop(CreationConfig(seed_hpo_trials=20), proxy_fn=None)
    seeds = _seeds(loop, 4)
    proxy_maes = loop._assign_seed_budgets(seeds)
    assert [s._seed_meta["hpo_trials"] for s in seeds] == [20, 20, 20, 20]
    assert proxy_maes == [None] * 4


def test_assign_budgets_use_proxy_off_unchanged():
    """use_proxy=False → 开关关, 全部大预算, proxy_fn 不调。"""
    cfg = CreationConfig(seed_hpo_trials=20, use_proxy=False)
    called = []
    loop = _bare_loop(cfg, proxy_fn=lambda g: called.append(g) or 0.1)
    seeds = _seeds(loop, 4)
    loop._assign_seed_budgets(seeds)
    assert called == []
    assert [s._seed_meta["hpo_trials"] for s in seeds] == [20, 20, 20, 20]


# ---------------------------------------------------------------------------
# proxy 失败可观测/可诊断 (线上实证: 一轮 4 候选全 inf, 无日志可查)
# ---------------------------------------------------------------------------


def test_assign_budgets_exception_logs_warning(capsys):
    """proxy 异常 → 打印 [B3] 警告 (候选简述+异常摘要截断), 失败原因留在 _last_proxy_errors。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    calls = []

    def proxy(g):
        calls.append(g)
        if len(calls) == 2:                   # 第二个候选崩溃
            raise RuntimeError("mock CUDA OOM")
        return 0.1

    loop = _bare_loop(cfg, proxy_fn=proxy)
    seeds = _seeds(loop, 3)
    loop._assign_seed_budgets(seeds)
    out = capsys.readouterr().out
    assert "[B3]" in out and "RuntimeError" in out and "mock CUDA OOM" in out
    assert "gcn" in out                       # 候选简述 (算子名)
    assert loop._last_proxy_errors[0] is None
    assert "mock CUDA OOM" in loop._last_proxy_errors[1]
    assert loop._last_proxy_errors[2] is None


def test_assign_budgets_inf_logs_note(capsys):
    """proxy 返回 inf (训练 NaN/熔断) → 打印 [B3] 说明 (候选名+疑似原因), 记 "inf"。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    scores = iter([0.1, float("inf"), 0.2])
    loop = _bare_loop(cfg, proxy_fn=lambda g: next(scores))
    seeds = _seeds(loop, 3)
    loop._assign_seed_budgets(seeds)
    out = capsys.readouterr().out
    assert "[B3]" in out and "inf" in out
    assert "NaN" in out or "争卡" in out      # 疑似原因说明
    assert loop._last_proxy_errors == [None, "inf", None]


def test_assign_budgets_all_inf_keeps_original_top_k(capsys):
    """全 inf (粗筛形同虚设场景): 预算分配行为不变 —— 稳定排序按原顺序 top-k 深评; 每候选都打日志。"""
    cfg = CreationConfig(seed_hpo_trials=20, small_hpo_trials=3, proxy_top_k=2)
    loop = _bare_loop(cfg, proxy_fn=lambda g: float("inf"))
    seeds = _seeds(loop, 3)
    proxy_maes = loop._assign_seed_budgets(seeds)
    assert proxy_maes == [float("inf")] * 3
    budgets = [s._seed_meta["hpo_trials"] for s in seeds]
    assert budgets == [20, 20, 3]             # 全 inf 并列 → 原顺序前 top_k 深评, 其余浅评
    out = capsys.readouterr().out
    assert out.count("[B3]") == 3             # 每个 inf 候选都留日志


# ---------------------------------------------------------------------------
# maybe_create 集成 (MockLLM + mock proxy_fn + 履历)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_ops():
    before = set(ops_mod.SPATIAL_OPS)
    yield
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before:
            ops_mod.SPATIAL_OPS.pop(k, None)


@pytest.fixture
def store():
    s = InMemoryGraphStore(HashEmbedder(dim=256))
    for m in all_seed_mechanisms():
        s.add_mechanism(m)
    return s


_PLAN_ARRAY_3 = (
    '[{"operator_name": "ProxyOpA", "rationale": "ra", "shared_structure": "s",'
    ' "composition": "additive_residual", "source_mechanisms": ["a"], "expected_effect": "e"},'
    '{"operator_name": "ProxyOpB", "rationale": "rb", "shared_structure": "s",'
    ' "composition": "sequential", "source_mechanisms": ["a"], "expected_effect": "e"},'
    '{"operator_name": "ProxyOpC", "rationale": "rc", "shared_structure": "s",'
    ' "composition": "gated_routed", "source_mechanisms": ["a"], "expected_effect": "e"}]'
)


def _good_code(cls: str) -> str:
    return (f'```python\nimport torch\nimport torch.nn as nn\nclass {cls}(nn.Module):\n'
            '    def __init__(self, channels, num_nodes, **kw):\n'
            '        super().__init__()\n'
            '        self.proj = nn.Linear(channels, channels)\n'
            '        self.alpha = nn.Parameter(torch.zeros(1))\n'
            '    def forward(self, x, adj=None):\n'
            '        return self.proj(x) + self.alpha * x\n'
            '```')


def _loop_with_proxy(store, responses, proxy_fn, archive=None):
    resp = iter(responses)
    llm = MockLLM(lambda msgs: next(resp))
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2))
    # 响应序列按 synthesize_many (单次调用产 N) 编排 → 固定走旧路径 (独立采样默认开)
    return CreationLoop(store, store.embedder, synth, OperatorRegistry(),
                        config=CreationConfig(seed_hpo_trials=20, small_hpo_trials=3,
                                              proxy_top_k=2, independent_sampling=False),
                        archive=archive, proxy_fn=proxy_fn)


def test_maybe_create_proxy_assigns_budgets_and_records(store, tmp_path):
    """3 成功假设 + proxy 粗筛: top_k 深评/其余浅评; 履历 record 带 proxy_mae。"""
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    responses = [_PLAN_ARRAY_3, _good_code("ProxyOpA"), _good_code("ProxyOpB"),
                 _good_code("ProxyOpC")]
    scores = iter([0.3, 0.1, 0.2])            # A/B/C 的 proxy 分
    loop = _loop_with_proxy(store, responses, lambda g: next(scores), archive=arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.success and outcome.n_success == 3
    budgets = [s._seed_meta["hpo_trials"] for s in outcome.seed_genotypes]
    assert budgets == [3, 20, 20]             # B(0.1)/C(0.2) 深评, A(0.3) 浅评
    # 履历: 3 条成功 record, proxy_mae 按算子对齐落盘
    recs = [r for r in arch._load() if r.gate == "all"]
    assert len(recs) == 3
    by_name = {r.operator_name: r for r in recs}
    for name, expect in zip(outcome.operator_names, [0.3, 0.1, 0.2]):
        assert by_name[name].proxy_mae == pytest.approx(expect)


def test_maybe_create_without_proxy_fn_keeps_big_budget(store, tmp_path):
    """无 proxy_fn: 全部 seed 大预算, 履历 proxy_mae=None (行为与 B3 前一致)。"""
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    responses = [_PLAN_ARRAY_3, _good_code("ProxyOpA"), _good_code("ProxyOpB"),
                 _good_code("ProxyOpC")]
    loop = _loop_with_proxy(store, responses, proxy_fn=None, archive=arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.n_success == 3
    budgets = [s._seed_meta["hpo_trials"] for s in outcome.seed_genotypes]
    assert budgets == [20, 20, 20]
    for r in arch._load():
        assert r.proxy_mae is None


def test_creation_record_old_jsonl_without_proxy_mae(tmp_path):
    """向后兼容: 旧 jsonl 行 (无 proxy_mae 字段) 正常加载, proxy_mae 默认 None。"""
    import json
    path = str(tmp_path / "ca.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"round_idx": 0, "created_at": "2026-01-01T00:00:00+00:00",
                            "operator_name": "synth_old", "gate": "all",
                            "composition": "additive_residual"}) + "\n")
    recs = CreationArchive(path)._load()
    assert len(recs) == 1 and recs[0].proxy_mae is None


def test_maybe_create_records_proxy_error(store, tmp_path):
    """履历 record 带 proxy_error: 异常候选记异常摘要, inf 候选记 "inf", 正常候选为 None。"""
    arch = CreationArchive(str(tmp_path / "ca.jsonl"))
    responses = [_PLAN_ARRAY_3, _good_code("ProxyOpA"), _good_code("ProxyOpB"),
                 _good_code("ProxyOpC")]
    calls = []

    def proxy(g):
        calls.append(g)
        if len(calls) == 2:                   # B 异常
            raise RuntimeError("mock CUDA OOM")
        if len(calls) == 3:                   # C 训练 NaN/熔断 → inf
            return float("inf")
        return 0.1                            # A 正常

    loop = _loop_with_proxy(store, responses, proxy, archive=arch)
    outcome = loop.maybe_create(Genotype(blocks=[STBlock("gcn", "tcn")]), sota_gap=3.0)
    assert outcome.n_success == 3
    by_name = {r.operator_name: r for r in arch._load() if r.gate == "all"}
    a, b, c = outcome.operator_names
    assert by_name[a].proxy_error is None
    assert "mock CUDA OOM" in (by_name[b].proxy_error or "")
    assert by_name[c].proxy_error == "inf"
    # proxy_mae 对齐记录不变: A=0.1, B/C=inf
    assert by_name[a].proxy_mae == pytest.approx(0.1)
    assert by_name[b].proxy_mae == float("inf")
    assert by_name[c].proxy_mae == float("inf")


def test_creation_record_old_jsonl_without_proxy_error(tmp_path):
    """向后兼容: 旧 jsonl 行 (无 proxy_error 字段) 正常加载, proxy_error 默认 None。"""
    import json
    path = str(tmp_path / "ca.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"round_idx": 0, "created_at": "2026-01-01T00:00:00+00:00",
                            "operator_name": "synth_old", "gate": "all",
                            "proxy_mae": 0.3}) + "\n")
    recs = CreationArchive(path)._load()
    assert len(recs) == 1 and recs[0].proxy_error is None
