"""B7 第二块: 辅助任务的训练链路接入回归测试 (CPU, 设备无关)。

覆盖五段链路 (契约与防泄漏验证门本身见 test_aux_validation.py, 本文件不测门):
  - builder:   STModel 拆 forward_features/head (forward 逐点一致, 无 aux 行为不变);
               genotype.aux_op 非 None → 从 AUX_OPS 查名挂 model.aux_module; 未注册 → ValueError
  - genotype:  aux_op 单槽 —— to_dict omit-None (无 aux 签名零换代) / 非 None 签名变 /
               validate / mutate aux_toggle/aux_swap; random_mutation 有/无注册 aux 两态
  - registry:  register_aux (validated 门 + aux_ 前缀) + 持久化 + load_persisted 重载进
               AUX_OPS; 与 synth_ 算子同目录共存不串表
  - train_one: 带 aux 训练完成且 trace 记录 aux_lambda/aux_op; λ 真实生效;
               aux NaN → 视同主损失熔断路径 (trace.aux_nan_hit 注明); 无 aux 默认值不变
  - HPO:       aux_lambda 条件采样 (有/无 aux 两态)
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict

import numpy as np
import optuna
import pytest
import torch
import torch.nn as nn

from darwin_st.creation import AuxTaskOperator, AuxTaskPlan, OperatorRegistry
from darwin_st.creation.contracts import FusionPlan, SynthesizedOperator
from darwin_st.data import prepare as P
from darwin_st.data.protocol import DatasetProfile, PROFILES
from darwin_st.optim.hpo import HPOConfig, suggest_hps
from darwin_st.optim.train import train_one
from darwin_st.search import operators as ops_mod
from darwin_st.search.builder import build_model
from darwin_st.search.evolution import random_mutation
from darwin_st.search.genotype import Genotype, mutate, random_genotype

B, T_IN, T_OUT, N, C = 2, 12, 12, 6, 3


# ---------------------------------------------------------------------------
# 玩具辅助任务模块 (契约: __init__(channels, num_nodes, seq_in, seq_out, **kw),
# aux_loss(h, x) -> 标量; h[B,T,N,C] 带梯度, x[B,T_in,N,C_in])。类形式与 code
# 字符串同一逻辑 —— 类直挂 AUX_OPS (builder/train 测试), 字符串走 register_aux exec。
# ---------------------------------------------------------------------------


class _ToyAux(nn.Module):
    """玩具: sensor 掩码重建 —— 随机遮节点维, 线性解码重建原始输入, masked MAE。"""

    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, in_channels=3,
                 mask_ratio=0.25, **kw):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.dec = nn.Linear(channels, in_channels)

    def aux_loss(self, h, x):
        mask = torch.rand_like(x[..., 0]) < self.mask_ratio  # [B,T,N] 节点维掩码
        pred = self.dec(h)                                   # [B,T,N,C_in]
        err = (pred - x).abs()
        if mask.any():
            return err[mask].mean()
        return err.mean()


class _InfAux(nn.Module):
    """反例: aux_loss 恒 inf → 训练时触发 aux NaN 熔断路径。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def aux_loss(self, h, x):
        return (x.sum() / 0.0) * self.w   # inf (张量除零不抛, 得 inf/nan)


_TOY_AUX_CODE = '''
import torch
import torch.nn as nn

class AuxToy(nn.Module):
    """玩具: sensor 掩码重建 (register_aux 持久化/重载测试用)。"""

    def __init__(self, channels, num_nodes, seq_in=12, seq_out=12, in_channels=3,
                 mask_ratio=0.25, **kw):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.dec = nn.Linear(channels, in_channels)

    def aux_loss(self, h, x):
        mask = torch.rand_like(x[..., 0]) < self.mask_ratio
        pred = self.dec(h)
        err = (pred - x).abs()
        if mask.any():
            return err[mask].mean()
        return err.mean()
'''

# 共存测试用的 synth 算子 (与 test_registry.py 同一份: 膨胀卷积 + 零初始化残差)
_SYNTH_CODE = '''
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoexistOp(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=0)
        self.branch = nn.Linear(channels, channels)
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, x, adj=None):
        B, T, N, C = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
        h = F.pad(h, (2, 0))
        h = self.conv(h)
        main = h.view(B, N, C, T).permute(0, 3, 1, 2)
        return main + self.alpha * self.branch(x)
'''


def _aux_op(name="AuxToy", validated=True):
    plan = AuxTaskPlan(
        task_name=name, rationale="掩码重建迫使表征编码空间可解码性", mask_pattern="sensor",
        mask_ratio=0.25, recon_target="raw_input", loss_form="masked_mae",
        decoder_form="linear", source_mechanisms=["masked_autoencoding"],
        expected_effect="同协议 -0.9 MAE 量级")
    return AuxTaskOperator(name=name, code=_TOY_AUX_CODE.replace("AuxToy", name),
                           plan=plan, validated=validated)


def _synth_op(name="CoexistOp", validated=True):
    plan = FusionPlan(operator_name=name, rationale="r", shared_structure="s",
                      composition="additive_residual", source_mechanisms=["a", "b"],
                      expected_effect="e")
    return SynthesizedOperator(name=name, code=_SYNTH_CODE.replace("CoexistOp", name),
                               plan=plan, needs_adj=False, validated=validated)


def _adj():
    a = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        a[i, (i + 1) % N] = a[(i + 1) % N, i] = 1.0
    return a


@pytest.fixture(autouse=True)
def _clean_global_registries():
    """每个测试后清理注入的 aux/synth 算子, 不污染全局 AUX_OPS/SPATIAL_OPS。"""
    before_aux = set(ops_mod.AUX_OPS)
    before_spatial = set(ops_mod.SPATIAL_OPS)
    before_cat = set(ops_mod.OP_CATEGORY)
    yield
    for k in list(ops_mod.AUX_OPS):
        if k not in before_aux:
            ops_mod.AUX_OPS.pop(k, None)
    for k in list(ops_mod.SPATIAL_OPS):
        if k not in before_spatial:
            ops_mod.SPATIAL_OPS.pop(k, None)
    for k in list(ops_mod.OP_CATEGORY):
        if k not in before_cat:
            ops_mod.OP_CATEGORY.pop(k, None)


@pytest.fixture
def tiny_data(tmp_path, monkeypatch):
    """预置 tiny 已处理数据集 (10 节点, 同 test_train.py), 跳过下载。"""
    monkeypatch.setattr(P, "CACHE_DIR", str(tmp_path))
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
    n, c = 10, 3
    for split, cnt in (("train", 64), ("val", 24), ("test", 24)):
        np.save(os.path.join(data_dir, f"{split}_x.npy"),
                rng.random((cnt, 12, n, c)).astype(np.float32))
        np.save(os.path.join(data_dir, f"{split}_y.npy"),
                rng.random((cnt, 12, n)).astype(np.float32))
    np.save(os.path.join(data_dir, "scaler.npy"), np.array([50.0, 20.0]))
    adj = np.eye(n, dtype=np.float32)
    np.save(os.path.join(data_dir, "adj.npy"), adj)
    return prof, data_dir, adj


# ---------------------------------------------------------------------------
# builder: features/head 拆分等价性 + aux 挂载
# ---------------------------------------------------------------------------


def test_forward_equals_head_of_features():
    """拆分等价性: forward == head(forward_features(...)), 逐点一致 (行为不变铁律)。"""
    torch.manual_seed(0)
    geno = random_genotype(depth=2, spatial="gcn", temporal="tcn")
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT,
                    adj=_adj())
    m.eval()
    x = torch.randn(B, T_IN, N, C)
    tod = torch.randint(0, 288, (B, T_IN))
    dow = torch.randint(0, 7, (B, T_IN))
    ref = m(x, tod_idx=tod, dow_idx=dow)                       # 整段 forward
    h = m.forward_features(x, tod_idx=tod, dow_idx=dow)        # 主干表征
    assert h.shape == (B, T_IN, N, geno.hidden)                # [B,T_in,N,hidden]
    out = m.head(h)
    assert out.shape == (B, T_OUT, N)
    assert torch.equal(ref, out)                               # 逐点一致 (非 allclose)


def test_no_aux_model_has_none_aux_module():
    """无 aux 模型: aux_module 默认 None, 主路径零变化。"""
    geno = random_genotype(depth=1)
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT,
                    adj=_adj())
    assert m.aux_module is None


def test_build_model_mounts_aux_and_trainable():
    """genotype.aux_op 已注册 → model.aux_module 非空, 主+aux 联合可训降 loss。"""
    torch.manual_seed(0)
    ops_mod.AUX_OPS["aux_toy"] = _ToyAux
    geno = random_genotype(depth=1, hidden=16)
    geno.aux_op = "aux_toy"
    m = build_model(geno, num_nodes=N, in_channels=C, seq_len_in=T_IN, seq_len_out=T_OUT,
                    adj=_adj())
    assert isinstance(m.aux_module, _ToyAux)
    # aux 参数随 model.parameters() 进优化器 (train_one 的 Adam 自动覆盖)
    aux_param_ids = {id(p) for p in m.aux_module.parameters()}
    assert aux_param_ids & {id(p) for p in m.parameters()}

    x = torch.randn(B, T_IN, N, C)
    y = torch.randn(B, T_OUT, N)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        h = m.forward_features(x)
        pred = m.head(h)
        loss = (pred - y).abs().mean() + 0.1 * m.aux_module.aux_loss(h, x)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]                              # 联合管道在学
    assert m.aux_module.dec.weight.grad is not None            # aux 参数收到梯度


def test_build_model_unregistered_aux_raises():
    """aux_op 未注册 → ValueError, 报错带名字与当前已注册清单。"""
    geno = random_genotype(depth=1)
    geno.aux_op = "aux_ghost"
    with pytest.raises(ValueError, match="aux_ghost"):
        build_model(geno, N, C, T_IN, T_OUT, _adj())


# ---------------------------------------------------------------------------
# genotype: aux_op 单槽 (序列化/签名/校验/变异)
# ---------------------------------------------------------------------------


def test_aux_op_omit_none_keeps_legacy_signature():
    """omit-None: 无 aux 的 genotype to_dict 无 aux_op 键 → 签名与引入该字段前逐字节一致。"""
    g = random_genotype(depth=2, hidden=32)
    d = g.to_dict()
    assert "aux_op" not in d
    # 旧 dict (无 aux_op 键) from_dict → None, 签名不变
    g2 = Genotype.from_dict(d)
    assert g2.aux_op is None
    assert g2.signature() == g.signature()


def test_aux_op_changes_signature_when_set():
    """aux_op 非 None → 进 to_dict → 签名变 (space_version 自动换代, 预期行为)。"""
    g = random_genotype(depth=2, hidden=32)
    g2 = g.copy()
    g2.aux_op = "aux_toy"
    assert g2.to_dict()["aux_op"] == "aux_toy"
    assert g2.signature() != g.signature()
    # 序列化往返
    g3 = Genotype.from_dict(g2.to_dict())
    assert g3.aux_op == "aux_toy"
    assert g3.to_dict() == g2.to_dict()


def test_aux_op_validate_rules():
    """validate: None 或合法标识符过; 非法标识符拒 (注册与否由 build_model 查)。"""
    g = random_genotype(depth=1)
    g.validate()                              # None 过
    g.aux_op = "aux_toy"
    g.validate()                              # 合法标识符过 (未注册也不查)
    g.aux_op = "123 not valid"
    with pytest.raises(ValueError):
        g.validate()


def test_mutate_aux_toggle_on_off():
    """aux_toggle: None→挂 new_op; 已挂→置 None。返回新对象不改原件。"""
    g = random_genotype(depth=1)
    g2 = mutate(g, "aux_toggle", new_op="aux_toy")
    assert g2.aux_op == "aux_toy"
    assert g.aux_op is None                   # 原件不变
    g3 = mutate(g2, "aux_toggle")             # 已挂 → 关闭
    assert g3.aux_op is None


def test_mutate_aux_swap():
    """aux_swap: 已挂可换; 未挂抛错 (先 aux_toggle 开启)。"""
    g = random_genotype(depth=1)
    with pytest.raises(ValueError):
        mutate(g, "aux_swap", new_op="aux_b")
    g2 = mutate(g, "aux_toggle", new_op="aux_a")
    g3 = mutate(g2, "aux_swap", new_op="aux_b")
    assert g3.aux_op == "aux_b"


def test_random_mutation_no_aux_registered_never_picks_aux():
    """无已注册 aux (AUX_OPS 空): aux 变异不可选 —— 多轮采样不崩, 产物 aux_op 恒 None。"""
    rng = random.Random(0)
    g = random_genotype(depth=2)
    for _ in range(30):
        g2 = random_mutation(g, rng)
        assert g2.aux_op is None


def test_random_mutation_aux_toggle_reachable_when_registered():
    """有已注册 aux: aux_toggle 可被加权采样命中 —— 开启(None→aux)产物出现。"""
    ops_mod.AUX_OPS["aux_toy"] = _ToyAux
    rng = random.Random(0)
    g = random_genotype(depth=2)
    seen = {random_mutation(g, rng).aux_op for _ in range(300)}
    assert "aux_toy" in seen


def test_random_mutation_aux_three_states():
    """已挂 aux 的 genotype: 变异能关掉 (→None) 也能换另一个已注册 aux。"""
    ops_mod.AUX_OPS["aux_a"] = _ToyAux
    ops_mod.AUX_OPS["aux_b"] = _ToyAux
    rng = random.Random(1)
    g = mutate(random_genotype(depth=1), "aux_toggle", new_op="aux_a")
    outcomes = {random_mutation(g, rng).aux_op for _ in range(300)}
    assert None in outcomes                   # 关闭
    assert "aux_b" in outcomes                # 换另一个


# ---------------------------------------------------------------------------
# registry: register_aux + 持久化 + 重载 + 与 synth 共存
# ---------------------------------------------------------------------------


def test_register_aux_injects_into_aux_ops():
    reg = OperatorRegistry()
    name = reg.register_aux(_aux_op("AuxRegOp"))
    assert name == "aux_AuxRegOp"             # 未带前缀 → 补 aux_ 前缀
    assert name in ops_mod.AUX_OPS
    assert reg.is_registered_aux(name)
    assert name in reg.registered_aux_names()


def test_register_aux_keeps_existing_prefix():
    """op.name 已带 aux_ 前缀 → 沿用, 不双前缀。"""
    reg = OperatorRegistry()
    name = reg.register_aux(_aux_op("aux_Prefixed"))
    assert name == "aux_Prefixed"
    assert name in ops_mod.AUX_OPS


def test_register_aux_rejects_unvalidated():
    reg = OperatorRegistry()
    with pytest.raises(ValueError):
        reg.register_aux(_aux_op("AuxBad", validated=False))


def test_aux_persist_and_reload(tmp_path):
    """注册 → 持久化 (aux_<name>.py/.json) → 模拟重启 → load_persisted 重载进 AUX_OPS。"""
    reg1 = OperatorRegistry(persist_dir=str(tmp_path))
    name = reg1.register_aux(_aux_op("AuxPersist"))
    assert os.path.exists(os.path.join(tmp_path, name + ".py"))
    assert os.path.exists(os.path.join(tmp_path, name + ".json"))

    ops_mod.AUX_OPS.pop(name, None)           # 模拟重启 (全局注册表清空)

    reg2 = OperatorRegistry(persist_dir=str(tmp_path))
    loaded = reg2.load_persisted()
    assert name in loaded
    assert name in ops_mod.AUX_OPS
    assert name in reg2.registered_aux_names()
    # 重载后的类契约可用: aux_loss(h, x) → 标量
    mod = ops_mod.AUX_OPS[name](channels=8, num_nodes=4, seq_in=12, seq_out=12)
    h = torch.randn(2, 12, 4, 8, requires_grad=True)
    x = torch.randn(2, 12, 4, 3)
    loss = mod.aux_loss(h, x)
    assert loss.dim() == 0
    loss.backward()
    assert h.grad is not None


def test_aux_coexists_with_synth_ops(tmp_path):
    """同一 persist_dir: synth 算子进 SPATIAL_OPS, aux 进 AUX_OPS, 互不串表。"""
    reg1 = OperatorRegistry(persist_dir=str(tmp_path))
    sname = reg1.register(_synth_op())                    # synth_CoexistOp
    aname = reg1.register_aux(_aux_op("AuxCoexist"))      # aux_AuxCoexist

    ops_mod.SPATIAL_OPS.pop(sname, None)
    ops_mod.AUX_OPS.pop(aname, None)

    reg2 = OperatorRegistry(persist_dir=str(tmp_path))
    loaded = reg2.load_persisted()
    assert sname in loaded and aname in loaded
    assert sname in ops_mod.SPATIAL_OPS
    assert aname in ops_mod.AUX_OPS
    assert aname not in ops_mod.SPATIAL_OPS               # 不串表
    assert sname not in ops_mod.AUX_OPS
    # synth 家谱不受 aux 影响 (aux 无版本族语义)
    assert all("aux_" not in fam for fam in reg2.families())


# ---------------------------------------------------------------------------
# train_one: 辅助损失接入
# ---------------------------------------------------------------------------


def _aux_geno(aux_name):
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    geno.aux_op = aux_name
    return geno


def test_train_one_with_aux_runs_and_records(tiny_data):
    """带 aux 玩具训练 2 epoch 完成; trace 记录 aux_lambda (默认 0.1) / aux_op, asdict 可序列化。"""
    prof, data_dir, adj = tiny_data
    ops_mod.AUX_OPS["aux_toy"] = _ToyAux
    mae, trace = train_one(_aux_geno("aux_toy"), {"lr": 1e-3, "batch_size": 16},
                           data_dir, prof, adj, device="cpu", max_epochs=2)
    assert np.isfinite(mae) and mae > 0
    assert trace.n_epochs_run == 2
    assert trace.aux_lambda == 0.1                        # hps 未给 → 默认 0.1
    assert trace.aux_op == "aux_toy"
    assert not trace.nan_hit and not trace.aux_nan_hit
    d = asdict(trace)                                     # Optuna user_attr 路径: 可 dict 化
    assert d["aux_lambda"] == 0.1 and d["aux_op"] == "aux_toy"


def test_train_one_aux_lambda_scales_loss(tiny_data):
    """λ 真实生效: 同种子同架构, λ=1.0 的训练损失显著高于 λ=0.0 (差 = 辅助项量级)。"""
    prof, data_dir, adj = tiny_data
    ops_mod.AUX_OPS["aux_toy"] = _ToyAux
    geno = _aux_geno("aux_toy")
    torch.manual_seed(0)
    _, trace_lo = train_one(geno, {"lr": 1e-3, "batch_size": 16, "aux_lambda": 0.0},
                            data_dir, prof, adj, device="cpu", max_epochs=2)
    torch.manual_seed(0)
    _, trace_hi = train_one(geno, {"lr": 1e-3, "batch_size": 16, "aux_lambda": 1.0},
                            data_dir, prof, adj, device="cpu", max_epochs=2)
    assert trace_lo.aux_lambda == 0.0 and trace_hi.aux_lambda == 1.0
    # λ=0 → loss 数值上等于纯主损失; λ=1 → 主+aux (玩具 aux 量级 ~O(0.5))
    assert trace_hi.train_loss[0] > trace_lo.train_loss[0] + 0.1


def test_train_one_aux_nan_fuses_like_main_nan(tiny_data):
    """aux 前向 inf → 视同主损失 NaN 熔断: 返回 inf, trace.nan_hit + aux_nan_hit 注明 aux 触发。"""
    prof, data_dir, adj = tiny_data
    ops_mod.AUX_OPS["aux_inf"] = _InfAux
    mae, trace = train_one(_aux_geno("aux_inf"), {"lr": 1e-3, "batch_size": 16},
                           data_dir, prof, adj, device="cpu", max_epochs=3)
    assert mae == float("inf")                            # 首 batch 即熔断, best 从未刷新
    assert trace.nan_hit and trace.aux_nan_hit            # 注明是 aux 触发 (区别于主损失 NaN)
    assert trace.aux_op == "aux_inf"
    assert trace.n_epochs_run == 0


def test_train_one_without_aux_trace_defaults(tiny_data):
    """无 aux 对照: trace aux 字段保持默认值 (aux_lambda=0.0 / aux_op=""), 路径不变。"""
    prof, data_dir, adj = tiny_data
    geno = random_genotype(depth=1, spatial="gcn", temporal="tcn", hidden=16)
    mae, trace = train_one(geno, {"lr": 1e-3, "batch_size": 16},
                           data_dir, prof, adj, device="cpu", max_epochs=2)
    assert np.isfinite(mae)
    assert trace.aux_lambda == 0.0 and trace.aux_op == ""
    assert not trace.aux_nan_hit


# ---------------------------------------------------------------------------
# HPO: aux_lambda 条件维度
# ---------------------------------------------------------------------------


def test_suggest_hps_aux_lambda_conditional():
    """aux_lambda 仅当 genotype.aux_op 非 None 时采样 (log-uniform [1e-3, 1e-1]); 无 aux 不采。"""
    cfg = HPOConfig()
    study = optuna.create_study()

    trial = study.ask()
    hps = suggest_hps(trial, random_genotype(depth=1), cfg)
    assert "aux_lambda" not in hps                        # 无 aux → 不采 (不死参数)

    g = random_genotype(depth=1)
    g.aux_op = "aux_toy"
    for _ in range(5):
        trial = study.ask()
        hps = suggest_hps(trial, g, cfg)
        assert 1e-3 <= hps["aux_lambda"] <= 1e-1          # 有 aux → 条件采样
