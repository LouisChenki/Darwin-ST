"""算子注入 (Operator Registry) —— 把合成的算子注入算子库供进化使用。

Tier-2 闭环的关键: 合成并验证过的融合算子, 注册进 operators.py 的算子表
(SPATIAL_OPS), 让 genotype/builder/evolution 能直接用它 —— 搜索空间真正"生长"。

设计:
  - register(): 把 SynthesizedOperator 的类注入 SPATIAL_OPS(签名兼容 forward(x,adj)->[B,T,N,C])。
    合成算子是时空融合体, 当作"空间算子"注册即可被 genotype 引用(它内部自己做时序)。
  - 持久化: 算子代码写入 dynamic_ops 目录, 重启后 load_persisted() 重新注册(存活)。
  - 隔离: 注入的算子带前缀(synth_)区分内置算子; unregister 可移除。

注意(用户拍板): 合成算子进【算子库】给进化用, **不进机制库**(机制库静态只读)。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch.nn as nn

from darwin_st.creation.contracts import SynthesizedOperator
from darwin_st.creation.synthesizer import exec_operator_code
from darwin_st.search import operators as ops_mod
from darwin_st.search.operators import SYNTH_PREFIX  # 单一真源在 operators.py, 此处 re-export 保兼容

__all__ = ["OperatorRegistry", "SYNTH_PREFIX"]


@dataclass
class _RegisteredOp:
    name: str            # 注册名 (synth_<原名>)
    class_name: str      # 代码里的类名
    code: str
    needs_adj: bool


class OperatorRegistry:
    """合成算子的注册表 + 持久化。注入到 search.operators.SPATIAL_OPS。"""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir
        self._registered: dict[str, _RegisteredOp] = {}
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    def register(self, op: SynthesizedOperator) -> str:
        """注入一个合成算子到 SPATIAL_OPS, 返回注册名。要求已 validated。"""
        if not op.validated:
            raise ValueError(f"算子 {op.name} 未过验证, 拒绝注入")
        reg_name = SYNTH_PREFIX + op.name
        cls = exec_operator_code(op.code, op.name)
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise ValueError(f"{op.name} 不是 nn.Module")

        # 注入到 SPATIAL_OPS (物理统一存储; build_op/build_spatial_op 都能取)
        ops_mod.SPATIAL_OPS[reg_name] = _make_factory(cls)
        # 登记类别元数据 → op_category() 据此正确放槽 (spatiotemporal → joint block)
        ops_mod.OP_CATEGORY[reg_name] = getattr(op, "category", "spatiotemporal")
        self._registered[reg_name] = _RegisteredOp(reg_name, op.name, op.code, op.needs_adj)

        if self.persist_dir:
            self._persist(reg_name, op)
        return reg_name

    def unregister(self, reg_name: str) -> None:
        ops_mod.SPATIAL_OPS.pop(reg_name, None)
        ops_mod.OP_CATEGORY.pop(reg_name, None)
        self._registered.pop(reg_name, None)

    def registered_names(self) -> list[str]:
        return list(self._registered)

    def is_registered(self, reg_name: str) -> bool:
        return reg_name in self._registered

    # -- 持久化 --
    def _persist(self, reg_name: str, op: SynthesizedOperator) -> None:
        base = os.path.join(self.persist_dir, reg_name)
        with open(base + ".py", "w") as f:
            f.write(op.code)
        with open(base + ".json", "w") as f:
            json.dump({"reg_name": reg_name, "class_name": op.name,
                       "needs_adj": op.needs_adj, "plan_name": op.plan.operator_name,
                       "composition": op.plan.composition,
                       "source_mechanisms": op.plan.source_mechanisms,
                       "category": getattr(op, "category", "spatiotemporal"),
                       "real_mae": op.real_mae}, f, ensure_ascii=False, indent=2)

    def load_persisted(self) -> list[str]:
        """从 persist_dir 重新加载并注册所有持久化的算子。返回注册名列表。"""
        if not self.persist_dir or not os.path.isdir(self.persist_dir):
            return []
        loaded = []
        for fn in sorted(os.listdir(self.persist_dir)):
            if not fn.endswith(".json"):
                continue
            meta_path = os.path.join(self.persist_dir, fn)
            with open(meta_path) as f:
                meta = json.load(f)
            code_path = os.path.join(self.persist_dir, fn[:-5] + ".py")
            if not os.path.exists(code_path):
                continue
            with open(code_path) as f:
                code = f.read()
            reg_name = meta["reg_name"]
            try:
                cls = exec_operator_code(code, meta["class_name"])
                ops_mod.SPATIAL_OPS[reg_name] = _make_factory(cls)
                ops_mod.OP_CATEGORY[reg_name] = meta.get("category", "spatiotemporal")
                self._registered[reg_name] = _RegisteredOp(
                    reg_name, meta["class_name"], code, meta.get("needs_adj", False))
                loaded.append(reg_name)
            except Exception:
                continue  # 坏的持久算子跳过, 不崩
        return loaded


def _make_factory(cls):
    """把合成算子类包成 build_spatial_op 兼容的工厂。

    build_spatial_op 调 cls(dim, **kw)。合成算子签名是 (channels, num_nodes, **kw),
    需 num_nodes; 但 build_spatial_op 对非 adaptive 算子不传 num_nodes。
    故包一层: 接受 dim + 可选 num_nodes(从 kw 或默认), 调用合成算子。
    """
    class _SynthWrapper(nn.Module):
        def __init__(self, dim, num_nodes=None, **kw):
            super().__init__()
            # 合成算子需要 num_nodes; build_spatial_op 对内置算子不传, 这里给默认
            self.inner = cls(channels=dim, num_nodes=(num_nodes if num_nodes is not None else 1), **kw)

        def forward(self, x, adj=None):
            return self.inner(x, adj)

    return _SynthWrapper
