"""B7 辅助训练任务: 契约 (AuxTaskPlan/AuxTaskOperator) + 防泄漏验证门的回归测试 (CPU, 设备无关)。

核心契约:
  - AuxTaskPlan.validate(): 四个文法字段 ∈ 对应集合; mask_pattern != "none" 时
    mask_ratio ∈ [0.05, 0.5] (none 时忽略); source_mechanisms 非空; task_name 合法标识符
  - validate_aux_operator 7 门: 实例化 / 标量+可微 / NaN / 非平凡 / 参数量 /
    静态防泄漏 (禁词+IO 源码扫描) / 动态防泄漏 (时间打乱对抗测试)
  - 合法 sensor 掩码重建玩具模块过全门; 每门各一个反例被拒
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from darwin_st.creation import (
    AUX_DECODER_FORMS,
    AUX_LOSS_FORMS,
    AUX_MASK_PATTERNS,
    AUX_RECON_TARGETS,
    AuxTaskOperator,
    AuxTaskPlan,
    AuxValidationConfig,
    validate_aux_operator,
)


# ---------------------------------------------------------------------------
# 测试用辅助任务模块 (aux_loss(h, x) -> 标量; h[B,T,N,C] 带梯度, x[B,T,N,C_in])
# 注意: 门 6 会扫描模块类源码, 合法模块的注释/标识符须避开禁词
# (y/label/target/future/..., 见 validation._BANNED_AUX_IDENTIFIERS)
# ---------------------------------------------------------------------------


class _GoodAux(nn.Module):
    """合法玩具: sensor 掩码重建 —— 随机遮节点维, 线性解码重建原始输入, masked MAE。"""

    def __init__(self, channels, num_nodes, in_channels=3, mask_ratio=0.25, **kw):
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


class _ParamFreeAux(nn.Module):
    """无参正则型: 时间平滑正则 (纯塑造表征, 无可学部件也合法)。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()

    def aux_loss(self, h, x):
        return (h[:, 1:] - h[:, :-1]).abs().mean()


class _NonScalarAux(nn.Module):
    """反例: 返回非标量 (留了时间/节点维)。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.dec = nn.Linear(channels, 3)

    def aux_loss(self, h, x):
        return (self.dec(h) - x).abs().mean(dim=0)  # [T,N,C_in], 非标量


class _DetachAux(nn.Module):
    """反例: detach 破图 —— 主干表征 h 收不到梯度。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.dec = nn.Linear(channels, 3)

    def aux_loss(self, h, x):
        return (self.dec(h.detach()) - x).abs().mean()


class _NaNAux(nn.Module):
    """反例: 前向输出 inf。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))  # 0 维, 保持标量输出

    def aux_loss(self, h, x):
        return (x.sum() / 0.0) * self.w  # inf


class _ZeroAux(nn.Module):
    """反例: 恒零输出 (trivial 任务骗正则项)。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def aux_loss(self, h, x):
        return (h.sum() - h.sum()) + self.w * 0.0  # 恒 0, 但计算图连着


class _ConstAux(nn.Module):
    """反例: 常数损失, 与输入无关。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(1.0))

    def aux_loss(self, h, x):
        return self.w ** 2 + (h.sum() - h.sum()) + (x.sum() - x.sum())  # 恒 1


class _StaticLeakAux(nn.Module):
    """反例: 代码区出现禁词标识符 (门 6 从严, 注释里出现同样拒)。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.dec = nn.Linear(channels, 3)

    def aux_loss(self, h, x):
        target = x  # 禁词: 标签语义引用
        return (self.dec(h) - target).abs().mean()


class _IOAux(nn.Module):
    """反例: 携带 IO 调用 (辅助模块不该碰 IO)。"""

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.dec = nn.Linear(channels, 3)

    def aux_loss(self, h, x):
        _ = torch.load  # 仅文本出现即拒 (静态门, 不真执行)
        return (self.dec(h) - x).abs().mean()


class _LeakAux(nn.Module):
    """反例: 作弊型 —— 拿相邻帧(含未来帧)直接当重建答案 (persistence)。

    平滑序列上真实顺序损失异常低, 打乱时间维后显著变差 → 门 7 抓。
    """

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.proj = nn.Linear(channels, channels)

    def aux_loss(self, h, x):
        smooth = (x[:, 1:] - x[:, :-1]).abs().mean()  # 相邻帧一致性: 真实序低, 打乱后高
        return smooth + 0.001 * self.proj(h).abs().mean()


def _boom_factory(channels, num_nodes):
    """反例: 无法实例化的工厂。"""
    raise RuntimeError("构造爆炸")


# exec 生成的类: inspect 取不到源码, 用于门 6 的 skipped / source_code= 路径
_EXEC_AUX_CODE = (
    "import torch\n"
    "import torch.nn as nn\n"
    "class ExecAux(nn.Module):\n"
    "    def __init__(self, channels, num_nodes, **kw):\n"
    "        super().__init__()\n"
    "        self.dec = nn.Linear(channels, 3)\n"
    "    def aux_loss(self, h, x):\n"
    "        return (self.dec(h) - x).abs().mean()\n"
)


def _exec_aux_class():
    ns: dict = {}
    exec(_EXEC_AUX_CODE, ns)
    return ns["ExecAux"]


# ---------------------------------------------------------------------------
# 验证门: 放行好的
# ---------------------------------------------------------------------------


def test_validate_aux_accepts_good():
    """sensor 掩码重建玩具模块过全门。"""
    rep = validate_aux_operator(_GoodAux)
    assert rep.passed, f"好辅助任务被拒: {rep.gate} {rep.reason}"
    assert rep.gate == "all"
    assert rep.num_params > 0
    assert rep.details["static_scan"] == "scanned"          # 模块级类 → inspect 扫到
    assert "probe_loss_real" in rep.details                  # 对抗门探针损失入明细


def test_validate_aux_accepts_param_free():
    """无参正则型 (纯塑造表征) 豁免参数梯度检查, 过全门。"""
    rep = validate_aux_operator(_ParamFreeAux)
    assert rep.passed, f"{rep.gate}: {rep.reason}"


# ---------------------------------------------------------------------------
# 验证门: 每门一个反例
# ---------------------------------------------------------------------------


def test_validate_aux_rejects_uninstantiable():
    rep = validate_aux_operator(_boom_factory)
    assert not rep.passed
    assert rep.gate == "instantiate"


def test_validate_aux_rejects_non_scalar():
    rep = validate_aux_operator(_NonScalarAux)
    assert not rep.passed
    assert rep.gate == "scalar_loss"


def test_validate_aux_rejects_detached():
    rep = validate_aux_operator(_DetachAux)
    assert not rep.passed
    assert rep.gate == "disconnected"   # h 收不到梯度


def test_validate_aux_rejects_nan():
    rep = validate_aux_operator(_NaNAux)
    assert not rep.passed
    assert rep.gate == "nan_forward"


def test_validate_aux_rejects_trivial_zero():
    rep = validate_aux_operator(_ZeroAux)
    assert not rep.passed
    assert rep.gate == "trivial_zero"


def test_validate_aux_rejects_trivial_constant():
    rep = validate_aux_operator(_ConstAux)
    assert not rep.passed
    assert rep.gate == "trivial_constant"


def test_validate_aux_rejects_param_ceiling():
    rep = validate_aux_operator(_GoodAux, AuxValidationConfig(max_params=100))
    assert not rep.passed
    assert rep.gate == "param_ceiling"
    assert rep.num_params > 100


def test_validate_aux_rejects_static_target():
    """代码区出现 `target=` → 静态防泄漏门拒。"""
    rep = validate_aux_operator(_StaticLeakAux)
    assert not rep.passed
    assert rep.gate == "static_leak"
    assert "target" in rep.reason


def test_validate_aux_rejects_static_io():
    """源码含 torch.load → 静态门拒。"""
    rep = validate_aux_operator(_IOAux)
    assert not rep.passed
    assert rep.gate == "static_leak"
    assert "torch.load" in rep.reason


def test_validate_aux_static_scan_skipped_for_exec_class():
    """exec 生成的类取不到源码 → 静态门跳过并记明细, 其余门照跑。"""
    rep = validate_aux_operator(_exec_aux_class())
    assert rep.passed, f"{rep.gate}: {rep.reason}"
    assert rep.details["static_scan"].startswith("skipped")


def test_validate_aux_static_scan_uses_explicit_source_code():
    """显式传 source_code=: 干净源码过, 含禁词源码拒。"""
    cls = _exec_aux_class()
    ok = validate_aux_operator(cls, source_code=_EXEC_AUX_CODE)
    assert ok.passed and ok.details["static_scan"] == "scanned"
    bad = validate_aux_operator(cls, source_code="class ExecAux:\n    future = x\n")
    assert not bad.passed and bad.gate == "static_leak"


def test_validate_aux_rejects_adversarial_leak():
    """persistence 作弊 (相邻帧当答案): 时间打乱后损失显著变差 → 动态门拒。"""
    rep = validate_aux_operator(_LeakAux)
    assert not rep.passed
    assert rep.gate == "adversarial_leak"
    # 明细里能看到 real 显著低于 shuffled
    assert rep.details["probe_loss_real"] < rep.details["probe_loss_shuffled"]


def test_validate_aux_adversarial_margin_configurable():
    """margin 语义: real < shuffled*(1-margin) 才拒。margin 抬高 → 阈值更低 → 门更宽:
    同一作弊模块在默认 0.3 下被拒, 在 margin=0.95 下漏网 (作弊比 ~0.55 不再触发)。"""
    strict = validate_aux_operator(_LeakAux)  # 默认 0.3
    assert not strict.passed and strict.gate == "adversarial_leak"
    lenient = validate_aux_operator(_LeakAux, AuxValidationConfig(adversarial_margin=0.95))
    assert lenient.passed, f"{lenient.gate}: {lenient.reason}"


# ---------------------------------------------------------------------------
# 契约: AuxTaskPlan.validate()
# ---------------------------------------------------------------------------


def _good_plan(**over) -> AuxTaskPlan:
    kw = dict(
        task_name="aux_sensor_mae",
        rationale="掩码重建迫使表征编码空间可解码性, 解标签稀缺瓶颈",
        mask_pattern="sensor",
        mask_ratio=0.25,
        recon_target="raw_input",
        loss_form="masked_mae",
        decoder_form="linear",
        source_mechanisms=["masked_autoencoding"],
        expected_effect="同协议 -0.9 MAE 量级",
    )
    kw.update(over)
    return AuxTaskPlan(**kw)


def test_aux_grammar_sets():
    assert AUX_MASK_PATTERNS == {"sensor", "temporal_patch", "random", "none"}
    assert AUX_RECON_TARGETS == {"raw_input", "hidden"}
    assert AUX_LOSS_FORMS == {"masked_mae", "mse", "huber"}
    assert AUX_DECODER_FORMS == {"linear", "mlp", "light_transformer"}


def test_aux_plan_valid():
    _good_plan().validate()  # 不抛即合法


def test_aux_plan_ratio_boundary_ok():
    _good_plan(mask_ratio=0.05).validate()  # 下界含
    _good_plan(mask_ratio=0.5).validate()   # 上界含


def test_aux_plan_none_pattern_ignores_ratio():
    """mask_pattern=none (非掩码类) 时 ratio 不校验。"""
    _good_plan(mask_pattern="none", mask_ratio=0.9).validate()


def test_aux_plan_bad_mask_pattern():
    with pytest.raises(ValueError):
        _good_plan(mask_pattern="frankenstein").validate()


def test_aux_plan_bad_recon_target():
    with pytest.raises(ValueError):
        _good_plan(recon_target="future_frames").validate()


def test_aux_plan_bad_loss_form():
    with pytest.raises(ValueError):
        _good_plan(loss_form="cross_entropy").validate()


def test_aux_plan_bad_decoder_form():
    with pytest.raises(ValueError):
        _good_plan(decoder_form="giant_gpt").validate()


def test_aux_plan_bad_ratio():
    with pytest.raises(ValueError):
        _good_plan(mask_ratio=0.7).validate()    # 超上界
    with pytest.raises(ValueError):
        _good_plan(mask_ratio=0.01).validate()   # 超下界


def test_aux_plan_empty_mechanisms():
    with pytest.raises(ValueError):
        _good_plan(source_mechanisms=[]).validate()


def test_aux_plan_bad_name():
    with pytest.raises(ValueError):
        _good_plan(task_name="123 not valid").validate()


def test_aux_operator_defaults():
    op = AuxTaskOperator(name="aux_sensor_mae", code="class aux_sensor_mae: ...", plan=_good_plan())
    assert op.validated is False
    assert op.validation_gate == ""
    assert op.real_mae is None
    assert op.plan.task_name == "aux_sensor_mae"
