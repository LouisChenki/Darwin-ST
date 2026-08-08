"""算子验证 harness (Operator Validation Harness) —— 合成算子的硬门, 防 reward hacking。

研究 (docs/TIER2_RESEARCH_FINDINGS.md §7): LLM 合成算子前, 必须过执行接地的硬门
(CodeT 式), 再上昂贵真实训练。这是 FunSearch"评测器是唯一裁判"哲学的落地:
LLM 可以乱写, 但没过门的绝不浪费 GPU。

门 (cheap→expensive 级联):
  1. 实例化: 能 build 成 nn.Module
  2. 形状: forward([B,T,N,C]) → [B,T,N,C] (保持节点维 N, 架构铁律见 docs/P2_ALGORITHM_DESIGN.md)
  3. 可微: gradcheck —— 对输入与参数有有限梯度 (catch 不可微/detach/inplace)
  4. NaN/Inf 门: 输出与梯度均有限
  5. 参数量上限: 不超预算 (防爆显存的巨型算子)
  6. 非平凡: 输出不恒等于输入、不恒为常数 (防退化解骗过弱评测)

全部通过才算"候选合格", 进入真实 masked-MAE 评测。任何门失败 → 拒绝 + 失败原因。
本 harness 设备无关 (CPU 小张量), 可本地全测。

B7 扩展 —— 辅助训练任务验证门 (validate_aux_operator): 创造对象从"架构算子"扩展到
"自监督辅助损失" (STD-MAE 路线)。在算子 6 门同哲学之上加两道防泄漏门:
静态源码扫描 (禁词/IO) + 动态时间打乱对抗测试, 防 aux 任务偷看未来/标签。
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

__all__ = [
    "ValidationConfig", "ValidationReport", "validate_operator",
    "AuxValidationConfig", "AuxValidationReport", "validate_aux_operator",
]


@dataclass
class ValidationConfig:
    B: int = 2
    T: int = 6
    N: int = 5
    C: int = 16
    max_params: int = 2_000_000     # 单算子参数上限
    gradcheck: bool = True
    seed: int = 0


@dataclass
class ValidationReport:
    """验证结果。passed=True 才允许进入真实评测。"""

    passed: bool
    gate: str = ""                  # 失败在哪个门 (passed=True 时为 "all")
    reason: str = ""
    num_params: int = 0
    details: dict = field(default_factory=dict)


# 算子工厂签名: (channels, num_nodes) -> nn.Module, 其 forward(x, adj) -> [B,T,N,C]
OperatorFactory = Callable[..., nn.Module]


def validate_operator(
    factory: OperatorFactory,
    cfg: ValidationConfig | None = None,
    needs_adj: bool = False,
) -> ValidationReport:
    """对一个算子工厂跑全部硬门。返回 ValidationReport。

    factory(channels, num_nodes) 须返回 nn.Module, forward(x, adj) → 同形状。
    needs_adj: 该算子是否需要邻接 (需要则喂一个简单邻接, 否则 adj=None)。
    """
    cfg = cfg or ValidationConfig()
    torch.manual_seed(cfg.seed)
    B, T, N, C = cfg.B, cfg.T, cfg.N, cfg.C

    # 门 1: 实例化
    try:
        op = factory(channels=C, num_nodes=N)
        assert isinstance(op, nn.Module)
    except Exception as e:
        return ValidationReport(False, "instantiate", f"{type(e).__name__}: {e}")

    # 参数量门 (门 5, 提前算)
    n_params = sum(p.numel() for p in op.parameters())
    if n_params > cfg.max_params:
        return ValidationReport(False, "param_ceiling",
                                f"参数量 {n_params} > 上限 {cfg.max_params}", num_params=n_params)

    x = torch.randn(B, T, N, C, dtype=torch.float64, requires_grad=True)
    op = op.double()  # gradcheck 需 double
    adj = None
    if needs_adj:
        a = torch.zeros(N, N, dtype=torch.float64)
        for i in range(N):
            a[i, (i + 1) % N] = a[(i + 1) % N, i] = 1.0
        adj = a

    # 门 2: 形状
    try:
        y = op(x, adj)
    except Exception as e:
        return ValidationReport(False, "forward", f"前向失败 {type(e).__name__}: {e}", num_params=n_params)
    if tuple(y.shape) != (B, T, N, C):
        return ValidationReport(False, "shape",
                                f"输出形状 {tuple(y.shape)} != [B,T,N,C]={(B,T,N,C)} (节点维 N 必须保持)",
                                num_params=n_params)

    # 门 4: NaN/Inf (前向)
    if not torch.isfinite(y).all():
        return ValidationReport(False, "nan_forward", "前向输出含 nan/inf", num_params=n_params)

    # 门 6: 非平凡 (输出不恒等输入、不恒为常数)
    with torch.no_grad():
        if torch.allclose(y, x, atol=1e-8):
            return ValidationReport(False, "trivial_identity", "输出恒等于输入 (平凡)", num_params=n_params)
        if y.std().item() < 1e-9:
            return ValidationReport(False, "trivial_constant", "输出近常数 (退化解)", num_params=n_params)

    # 门 3: 可微 (gradcheck 或退化为有限梯度检查)
    if cfg.gradcheck:
        # 3a: 输入必须真正影响输出 (detach/破图算子在此被抓: 输入梯度为 0)
        x2 = torch.randn(B, T, N, C, dtype=torch.float64, requires_grad=True)
        try:
            out2 = op(x2, adj)
            g_in = torch.autograd.grad(out2.sum(), x2, allow_unused=True)[0]
        except Exception as e:
            return ValidationReport(False, "input_grad", f"求输入梯度失败 {type(e).__name__}: {e}",
                                    num_params=n_params)
        if g_in is None or g_in.abs().sum().item() < 1e-12:
            return ValidationReport(False, "disconnected",
                                    "输入对输出无梯度 (detach/破图, 算子忽略了输入)", num_params=n_params)

        # 3b: gradcheck (数值 vs 解析梯度一致 → 真正可微且数值稳)
        try:
            ok = torch.autograd.gradcheck(lambda inp: op(inp, adj).sum(), (x,),
                                          eps=1e-6, atol=1e-3, rtol=1e-2, raise_exception=False)
            if not ok:
                return ValidationReport(False, "gradcheck", "gradcheck 未通过 (可能不可微/数值不稳)",
                                        num_params=n_params)
        except Exception as e:
            return ValidationReport(False, "gradcheck", f"gradcheck 异常 {type(e).__name__}: {e}",
                                    num_params=n_params)

    # 门 4b: NaN/Inf (反向, 参数梯度)
    try:
        op.zero_grad()
        loss = op(x, adj).sum()
        loss.backward()
        for p in op.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return ValidationReport(False, "nan_backward", "参数梯度含 nan/inf", num_params=n_params)
    except Exception as e:
        return ValidationReport(False, "backward", f"反向失败 {type(e).__name__}: {e}", num_params=n_params)

    return ValidationReport(True, "all", "全部门通过", num_params=n_params,
                            details={"output_std": float(y.std().item())})


# ---------------------------------------------------------------------------
# B7: 辅助训练任务验证门 (Aux Task Validation) —— 自监督辅助损失的硬门 + 防泄漏
# ---------------------------------------------------------------------------


@dataclass
class AuxValidationConfig:
    """辅助任务验证配置。

    玩具尺寸与 validate_operator 不同 (契约: h[B,12,8,64], x[B,12,8,3])。
    不复用 ValidationConfig: gradcheck/seed 字段在此用不到 (seed 走函数参数),
    且 B/T/N/C 默认值全不同, 继承只会留下死字段。
    """

    B: int = 2
    T: int = 12
    N: int = 8
    C: int = 64                     # h 通道数 (主干隐藏维)
    in_channels: int = 3            # x 通道数 (原始输入特征数)
    max_params: int = 2_000_000     # 辅助模块参数上限 (沿用算子默认)
    adversarial_margin: float = 0.3  # 动态防泄漏门边际: real < shuffled*(1-margin) → 拒


@dataclass
class AuxValidationReport:
    """辅助任务验证结果。passed=True 才允许注册 + 接训练器。"""

    passed: bool
    gate: str = ""                  # 失败在哪个门 (passed=True 时为 "all")
    reason: str = ""
    num_params: int = 0
    details: dict = field(default_factory=dict)


# 辅助模块工厂签名: (channels, num_nodes) -> nn.Module, 其 aux_loss(h, x) -> 标量损失
AuxOperatorFactory = Callable[..., nn.Module]

# 门 6 静态防泄漏禁词 (词边界正则: layer 等合法词不误伤。从严 —— 注释/文档字符串中
# 出现同样拒, 误伤由 LLM 重试兜底)
_BANNED_AUX_IDENTIFIERS = (
    "y", "label", "labels", "target", "targets",
    "future", "y_true", "y_pred", "ground_truth",
)
_BANNED_AUX_RE = re.compile(r"\b(" + "|".join(_BANNED_AUX_IDENTIFIERS) + r")\b")

# 门 6b: aux 模块不该碰 IO/网络
_BANNED_AUX_CALLS = (
    "torch.load", "open(", "requests", "urllib", "socket",
    "np.load", "numpy.load", "read_csv", "pickle", "http",
)


def validate_aux_operator(
    factory: AuxOperatorFactory,
    cfg: AuxValidationConfig | None = None,
    *,
    source_code: str | None = None,
    seed: int = 0,
) -> AuxValidationReport:
    """对辅助任务模块工厂跑全部硬门。返回 AuxValidationReport。

    factory(channels=64, num_nodes=8) 须返回 nn.Module, 其 aux_loss(h, x) → 标量损失:
    h 为 [B,T,N,C] 主干隐藏表征 (带梯度), x 为 [B,T_in,N,C_in] 原始输入。签名无 y。

    门序 (执行序按技术依赖微调: NaN 前向检查须在 backward 之前):
      1. instantiate:      玩具尺寸能构造 nn.Module
      2. scalar_loss:      aux_loss 返回标量张量; backward 后参数有非 None 有限梯度,
                           且主干表征 h 收到梯度 (无参模块豁免参数检查 —— 纯正则型合法)
      3. nan_forward:      前向无 nan/inf
      4. non-trivial:      不恒零 (trivial_zero), 且随输入变化 (trivial_constant)
      5. param_ceiling:    参数量 ≤ cfg.max_params
      6. static_leak:      源码文本扫描 (禁词/IO)。source_code 缺省时取
                           inspect.getsource(模块类); 仍取不到 (exec 生成的类) 则跳过
                           并记 details["static_scan"]="skipped ..."
      7. adversarial_leak: 动态对抗门 (启发式, 宁严勿宽)。探针用时间相关的随机游走序列
                           (i.i.d. 输入时间维可交换, 任何任务对打乱都不敏感, 门将形同虚设):
                           打乱 x 时间维得 x', 若 real < shuffled*(1-margin) 判疑似利用
                           时间对齐/未来信息 → 拒。注意: 门在未训练模块上运行, 合法任务
                           尚未学到与 x 的对齐, 对打乱不应显著敏感。
    """
    cfg = cfg or AuxValidationConfig()
    torch.manual_seed(seed)
    B, T, N, C, C_in = cfg.B, cfg.T, cfg.N, cfg.C, cfg.in_channels
    details: dict = {}

    # 门 1: 实例化 (玩具尺寸)
    try:
        op = factory(channels=C, num_nodes=N)
        assert isinstance(op, nn.Module)
    except Exception as e:
        return AuxValidationReport(False, "instantiate", f"{type(e).__name__}: {e}")

    op = op.eval()          # 确定性: 关 dropout 等随机性, 非平凡门需可重复前向
    n_params = sum(p.numel() for p in op.parameters())

    h = torch.randn(B, T, N, C, requires_grad=True)
    x = torch.randn(B, T, N, C_in)

    # 门 2a: aux_loss 返回标量张量
    try:
        loss = op.aux_loss(h, x)
    except Exception as e:
        return AuxValidationReport(False, "forward", f"aux_loss 前向失败 {type(e).__name__}: {e}",
                                   num_params=n_params)
    if not isinstance(loss, torch.Tensor) or loss.dim() != 0:
        got = tuple(loss.shape) if isinstance(loss, torch.Tensor) else type(loss).__name__
        return AuxValidationReport(False, "scalar_loss",
                                   f"aux_loss 须返回标量张量, 实得 {got}", num_params=n_params)

    # 门 3: NaN/Inf (前向)
    if not torch.isfinite(loss):
        return AuxValidationReport(False, "nan_forward", "aux_loss 输出 nan/inf", num_params=n_params)

    # 门 2b: 可微 —— backward 后参数有非 None 有限梯度; h 必须收到梯度 (aux 任务要塑造表征)
    try:
        op.zero_grad()
        loss.backward()
    except Exception as e:
        return AuxValidationReport(False, "backward", f"反向失败 {type(e).__name__}: {e}",
                                   num_params=n_params)
    params = list(op.parameters())
    if params:
        grads = [p.grad for p in params]
        if any(g is not None and not torch.isfinite(g).all() for g in grads):
            return AuxValidationReport(False, "nan_backward", "参数梯度含 nan/inf",
                                       num_params=n_params)
        if not any(g is not None for g in grads):
            return AuxValidationReport(False, "no_param_grad",
                                       "backward 后无任何参数收到梯度 (aux 任务无可学部件/破图)",
                                       num_params=n_params)
    if h.grad is None:
        return AuxValidationReport(False, "disconnected",
                                   "主干表征 h 未收到梯度 (模块内部 detach 或忽略 h): 辅助任务必须塑造表征",
                                   num_params=n_params)
    if not torch.isfinite(h.grad).all():
        return AuxValidationReport(False, "nan_backward", "h 梯度含 nan/inf", num_params=n_params)

    # 门 4: 非平凡 —— 不恒零, 且随输入变化 (防"trivial 任务骗正则项")
    try:
        with torch.no_grad():
            l_a = float(op.aux_loss(torch.randn(B, T, N, C), torch.randn(B, T, N, C_in)))
            l_b = float(op.aux_loss(torch.randn(B, T, N, C), torch.randn(B, T, N, C_in)))
    except Exception as e:
        return AuxValidationReport(False, "scalar_loss",
                                   f"不同输入下 aux_loss 输出形状不稳定/异常 {type(e).__name__}: {e}",
                                   num_params=n_params)
    if not (math.isfinite(l_a) and math.isfinite(l_b)):
        return AuxValidationReport(False, "nan_forward", "非平凡门探针下 aux_loss 非有限",
                                   num_params=n_params)
    if abs(l_a) < 1e-12 and abs(l_b) < 1e-12:
        return AuxValidationReport(False, "trivial_zero", "aux_loss 恒等于 0 (trivial 任务骗正则项)",
                                   num_params=n_params)
    if abs(l_a - l_b) <= 1e-9 * (abs(l_a) + abs(l_b) + 1e-12):
        return AuxValidationReport(False, "trivial_constant",
                                   f"aux_loss 几乎与输入无关 (两次随机输入损失 {l_a:.6g} ≈ {l_b:.6g})",
                                   num_params=n_params)

    # 门 5: 参数量上限
    if n_params > cfg.max_params:
        return AuxValidationReport(False, "param_ceiling",
                                   f"参数量 {n_params} > 上限 {cfg.max_params}", num_params=n_params)

    # 门 6: 静态防泄漏 —— 源码文本扫描
    src = source_code
    if src is None:
        try:
            src = inspect.getsource(type(op))
        except (OSError, TypeError):
            src = None
    if src is None:
        details["static_scan"] = "skipped: 无法获取源码 (exec 生成的类请传 source_code=)"
    else:
        m = _BANNED_AUX_RE.search(src)
        if m:
            return AuxValidationReport(False, "static_leak",
                                       f"源码出现禁词 '{m.group(0)}' (疑似标签/未来信息引用; 从严, 误伤请重命名后重试)",
                                       num_params=n_params, details=details)
        for tok in _BANNED_AUX_CALLS:
            if tok in src:
                return AuxValidationReport(False, "static_leak",
                                           f"源码含 IO/网络调用 '{tok}' (辅助模块不该碰 IO)",
                                           num_params=n_params, details=details)
        details["static_scan"] = "scanned"

    # 门 7: 动态防泄漏 (对抗测试, 启发式 —— 宁严勿宽)
    with torch.no_grad():
        x_adv = torch.cumsum(torch.randn(B, T, N, C_in), dim=1) / math.sqrt(T)  # 时间相关探针
        x_shuf = x_adv[:, torch.randperm(T)]                                     # 打乱时间维
        h_adv = torch.randn(B, T, N, C)
        try:
            l_real = float(op.aux_loss(h_adv, x_adv))
            l_shuf = float(op.aux_loss(h_adv, x_shuf))
        except Exception as e:
            return AuxValidationReport(False, "scalar_loss",
                                       f"对抗探针下 aux_loss 异常 {type(e).__name__}: {e}",
                                       num_params=n_params, details=details)
    details["probe_loss_real"] = l_real
    details["probe_loss_shuffled"] = l_shuf
    if not (math.isfinite(l_real) and math.isfinite(l_shuf)):
        return AuxValidationReport(False, "nan_forward", "对抗探针下 aux_loss 非有限",
                                   num_params=n_params, details=details)
    if l_real < l_shuf * (1.0 - cfg.adversarial_margin):
        return AuxValidationReport(
            False, "adversarial_leak",
            f"打乱 x 时间维后损失显著升高 (real={l_real:.4g} < shuffled={l_shuf:.4g}×(1-{cfg.adversarial_margin})): "
            f"疑似利用时间对齐/未来信息 (启发式对抗门, 宁严勿宽)",
            num_params=n_params, details=details)

    return AuxValidationReport(True, "all", "全部门通过", num_params=n_params, details=details)
