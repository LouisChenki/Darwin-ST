"""算子验证 harness (Operator Validation Harness) —— 合成算子的硬门, 防 reward hacking。

研究 (docs/TIER2_RESEARCH_FINDINGS.md §7): LLM 合成算子前, 必须过执行接地的硬门
(CodeT 式), 再上昂贵真实训练。这是 FunSearch"评测器是唯一裁判"哲学的落地:
LLM 可以乱写, 但没过门的绝不浪费 GPU。

门 (cheap→expensive 级联):
  1. 实例化: 能 build 成 nn.Module
  2. 形状: forward([B,T,N,C]) → [B,T,N,C] (保持节点维 N, architecture-rules 铁律)
  3. 可微: gradcheck —— 对输入与参数有有限梯度 (catch 不可微/detach/inplace)
  4. NaN/Inf 门: 输出与梯度均有限
  5. 参数量上限: 不超预算 (防爆显存的巨型算子)
  6. 非平凡: 输出不恒等于输入、不恒为常数 (防退化解骗过弱评测)

全部通过才算"候选合格", 进入真实 masked-MAE 评测。任何门失败 → 拒绝 + 失败原因。
本 harness 设备无关 (CPU 小张量), 可本地全测。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

__all__ = ["ValidationConfig", "ValidationReport", "validate_operator"]


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
