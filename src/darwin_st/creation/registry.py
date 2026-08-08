"""算子注入 (Operator Registry) —— 把合成的算子注入算子库供进化使用。

Tier-2 闭环的关键: 合成并验证过的融合算子, 注册进 operators.py 的算子表
(SPATIAL_OPS), 让 genotype/builder/evolution 能直接用它 —— 搜索空间真正"生长"。

设计:
  - register(): 把 SynthesizedOperator 的类注入 SPATIAL_OPS(签名兼容 forward(x,adj)->[B,T,N,C])。
    合成算子是时空融合体, 当作"空间算子"注册即可被 genotype 引用(它内部自己做时序)。
  - 持久化: 算子代码写入 dynamic_ops 目录, 重启后 load_persisted() 重新注册(存活)。
  - 隔离: 注入的算子带前缀(synth_)区分内置算子; unregister 可移除。

算子家谱 (B2, FunSearch/ELM 微进化):
  - family: 注册名去掉版本尾缀 (_v<N>); synth_foo 与 synth_foo_v2 同族 synth_foo。
  - register_variant(): 把精炼产物注册为父版同族的新版本 (版本号由 refiner 按 next_version
    命名, 此处校验族一致 + 版本不重复), 家谱只增不删。
  - 持久化: 每族一个 lineage_<family>.json {family, versions:[{reg_name, code_file,
    real_mae(可空), round_idx, created_at, parent_reg_name(可空), version}]}。
  - update_real_mae(): 实测后回填 lineage 与算子 json; family_best 由查询端 (best_in_family)
    按 mae 比较天然得出, 不做物理替换。
  - load_persisted() 兼容旧目录: 无 lineage 文件时每个算子隐式成单版本族。

注意(用户拍板): 合成算子进【算子库】给进化用, **不进机制库**(机制库静态只读)。

B7 扩展 —— 辅助训练任务注册 (register_aux):
  - 创造对象从"架构算子"扩展到"自监督辅助损失模块" (STD-MAE 路线, 契约见 contracts.AuxTaskOperator)。
  - register_aux(): 要求 op.validated=True (过 validate_aux_operator 防泄漏门); 注册名保证
    aux_ 前缀 (op.name 已带则沿用, 未带则补); 注入 operators.AUX_OPS (独立注册表,
    与 synth_ 算子的 SPATIAL_OPS 共存互不干扰)。
  - 持久化: 同一 persist_dir 下 aux_<name>.py + aux_<name>.json (存 plan 设计表 + real_mae 钩子);
    load_persisted() 按文件名前缀分派 (aux_ → AUX_OPS, synth_ → SPATIAL_OPS, lineage_ → 家谱),
    进程后端 worker 经现有 synth_persist_dir 机制自动拿到 aux 算子 (零新链路)。
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import torch.nn as nn

from darwin_st.creation.contracts import AuxTaskOperator, SynthesizedOperator
from darwin_st.creation.synthesizer import exec_operator_code
from darwin_st.search import operators as ops_mod
from darwin_st.search.operators import (  # 单一真源在 operators.py, 此处 re-export 保兼容
    AUX_PREFIX,
    SYNTH_PREFIX,
)

__all__ = ["OperatorRegistry", "SYNTH_PREFIX", "AUX_PREFIX", "family_of", "version_of"]

_VERSION_RE = re.compile(r"_v(\d+)$")


def family_of(reg_name: str) -> str:
    """注册名 → 族名: 去掉版本尾缀 _v<N> (synth_foo_v2 → synth_foo; 无尾缀即自身)。"""
    return _VERSION_RE.sub("", reg_name)


def version_of(reg_name: str) -> int:
    """注册名 → 版本号: 无 _v<N> 尾缀视为 v1 (族内首版)。"""
    m = _VERSION_RE.search(reg_name)
    return int(m.group(1)) if m else 1


@dataclass
class _RegisteredOp:
    name: str            # 注册名 (synth_<原名>)
    class_name: str      # 代码里的类名
    code: str
    needs_adj: bool
    composition: str = ""                        # plan 的组合文法 (精炼沿用用)
    source_mechanisms: list[str] = field(default_factory=list)  # plan 的源机制 (精炼沿用用)


@dataclass
class _RegisteredAux:
    """B7 辅助任务注册条目 (内存态; 持久层为 aux_<name>.py + aux_<name>.json)。"""

    name: str            # 注册名 (aux_ 前缀)
    class_name: str      # 代码里的类名 (= plan.task_name)
    code: str
    plan: dict = field(default_factory=dict)     # AuxTaskPlan asdict (设计表元数据)


class OperatorRegistry:
    """合成算子的注册表 + 持久化 + 算子家谱 (B2)。注入到 search.operators.SPATIAL_OPS。"""

    def __init__(self, persist_dir: str | None = None):
        self.persist_dir = persist_dir
        self._registered: dict[str, _RegisteredOp] = {}
        self._registered_aux: dict[str, _RegisteredAux] = {}   # B7 辅助任务 (独立表)
        self._lineages: dict[str, list[dict]] = {}   # family -> [version 条目, 按版本序]
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------

    def register(self, op: SynthesizedOperator) -> str:
        """注入一个合成算子到 SPATIAL_OPS, 返回注册名。要求已 validated。"""
        reg_name = self._inject(op)
        # 首版算子: 隐式成族 (parent=None, 版本号按名解析, 通常是 v1)
        self._track_lineage(reg_name, parent_reg_name=None, round_idx=None,
                            real_mae=op.real_mae)
        return reg_name

    def register_variant(self, op: SynthesizedOperator, parent_reg_name: str,
                         round_idx: int | None = None) -> str:
        """把精炼产物注册为 parent 同族的新版本 (B2 微进化)。返回注册名。

        族一致性: family_of(synth_+op.name) 必须等于 family_of(parent_reg_name);
        版本不重复: 注册名不得已存在 (refiner 按 next_version 命名保证递增)。
        其余与 register 相同 (注入 SPATIAL_OPS + 持久化), 并写 lineage (parent 指针)。
        """
        if parent_reg_name not in self._registered:
            raise ValueError(f"父算子 {parent_reg_name} 未注册, 无法注册变体")
        fam = family_of(parent_reg_name)
        reg_name = SYNTH_PREFIX + op.name
        if family_of(reg_name) != fam:
            raise ValueError(f"变体 {reg_name} 与父版 {parent_reg_name} 不同族 {fam}")
        if reg_name in self._registered:
            raise ValueError(f"版本 {reg_name} 已存在 (族内版本须递增不重复)")
        reg_name = self._inject(op)
        self._track_lineage(reg_name, parent_reg_name=parent_reg_name,
                            round_idx=round_idx, real_mae=op.real_mae)
        return reg_name

    def _inject(self, op: SynthesizedOperator) -> str:
        """注册公共体: 校验 → exec → 注入 SPATIAL_OPS → 持久化算子文件。"""
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
        self._registered[reg_name] = _RegisteredOp(
            reg_name, op.name, op.code, op.needs_adj,
            composition=op.plan.composition,
            source_mechanisms=list(op.plan.source_mechanisms))

        if self.persist_dir:
            self._persist(reg_name, op)
        return reg_name

    def unregister(self, reg_name: str) -> None:
        ops_mod.SPATIAL_OPS.pop(reg_name, None)
        ops_mod.OP_CATEGORY.pop(reg_name, None)
        self._registered.pop(reg_name, None)
        # 家谱只增不删: lineage 记录保留 (历史版本信息是进化经验)

    def registered_names(self) -> list[str]:
        return list(self._registered)

    def is_registered(self, reg_name: str) -> bool:
        return reg_name in self._registered

    # ------------------------------------------------------------------
    # B7: 辅助训练任务注册 (AUX_OPS 注入 + 持久化)
    # ------------------------------------------------------------------

    def register_aux(self, op: AuxTaskOperator) -> str:
        """把过验证门的辅助任务模块注入 AUX_OPS, 返回注册名 (aux_ 前缀)。

        要求 op.validated=True (过 validate_aux_operator 防泄漏门) —— 未验证拒绝注入。
        注册名: op.name 已带 aux_ 前缀则沿用, 未带则补前缀 (aux_ 前缀惯例,
        genotype.aux_op/build_aux_op 按此名查找)。持久化 aux_<name>.py + aux_<name>.json。
        """
        if not op.validated:
            raise ValueError(f"辅助任务 {op.name} 未过防泄漏验证门, 拒绝注入")
        reg_name = op.name if op.name.startswith(AUX_PREFIX) else AUX_PREFIX + op.name
        cls = exec_operator_code(op.code, op.name)
        if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
            raise ValueError(f"{op.name} 不是 nn.Module")

        # 注入 AUX_OPS (独立注册表, 与 SPATIAL_OPS 的 synth_ 算子共存; 存类本身,
        # 与 SPATIAL_OPS 同风格 —— 实例化契约 channels/num_nodes/seq_in/seq_out 由 build_aux_op 走)
        ops_mod.AUX_OPS[reg_name] = cls
        self._registered_aux[reg_name] = _RegisteredAux(
            reg_name, op.name, op.code, plan=asdict(op.plan))

        if self.persist_dir:
            self._persist_aux(reg_name, op)
        return reg_name

    def registered_aux_names(self) -> list[str]:
        """已注册的辅助任务名 (本 registry 内存态; 全局可挂名单以 AUX_OPS 为准)。"""
        return list(self._registered_aux)

    def is_registered_aux(self, reg_name: str) -> bool:
        return reg_name in self._registered_aux

    def _persist_aux(self, reg_name: str, op: AuxTaskOperator) -> None:
        base = os.path.join(self.persist_dir, reg_name)
        with open(base + ".py", "w") as f:
            f.write(op.code)
        with open(base + ".json", "w") as f:
            json.dump({"reg_name": reg_name, "class_name": op.name,
                       "kind": "aux",                    # 类型标记 (防御: 前缀之外的第二重判别)
                       "plan": asdict(op.plan),
                       "real_mae": op.real_mae}, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 算子家谱 (B2)
    # ------------------------------------------------------------------

    def families(self) -> list[str]:
        """所有已知族名 (含隐式单版本族), 字典序。"""
        return sorted(self._lineages)

    def family_versions(self, family: str) -> list[dict]:
        """族内全部版本条目 (按版本序), 每条附 code/composition/source_mechanisms (内存 enrich)。"""
        return [self._enrich(v) for v in self._lineages.get(family, [])]

    def best_in_family(self, family: str) -> dict | None:
        """族内实测 MAE 最小的版本条目 (enrich 后); 无有限 MAE → None。"""
        vs = [v for v in self._lineages.get(family, []) if _finite(v.get("real_mae"))]
        if not vs:
            return None
        return self._enrich(min(vs, key=lambda v: v["real_mae"]))

    def update_real_mae(self, reg_name: str, mae: float) -> None:
        """回填某版本的实测 MAE: lineage + 算子 json 同步 (orchestrator 评测后调用)。

        只增不改其它字段; 未知注册名静默跳过 (回填是增益, 不中止优化循环)。
        """
        fam = family_of(reg_name)
        hit = False
        for v in self._lineages.get(fam, []):
            if v["reg_name"] == reg_name:
                v["real_mae"] = mae
                hit = True
        if hit:
            self._persist_lineage(fam)
        # 算子 json 的 real_mae 钩子同步 (持久层两处的同一真值)
        if self.persist_dir:
            meta_path = os.path.join(self.persist_dir, reg_name + ".json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path) as f:
                        meta = json.load(f)
                    meta["real_mae"] = mae
                    with open(meta_path, "w") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass            # 回填失败不崩 (增益非必需)

    def _enrich(self, entry: dict) -> dict:
        """版本条目 + 内存里的代码/plan 元数据 (精炼 prompt 需要父代码全文)。"""
        e = dict(entry)
        r = self._registered.get(e["reg_name"])
        e["code"] = r.code if r else None
        e["composition"] = r.composition if r else ""
        e["source_mechanisms"] = list(r.source_mechanisms) if r else []
        e["needs_adj"] = r.needs_adj if r else False
        return e

    def _track_lineage(self, reg_name: str, parent_reg_name: str | None,
                       round_idx: int | None, real_mae: float | None) -> None:
        """把一版算子记入家谱 (幂等: 已存在不重复记) 并持久化 lineage 文件。"""
        fam = family_of(reg_name)
        versions = self._lineages.setdefault(fam, [])
        if any(v["reg_name"] == reg_name for v in versions):
            return
        versions.append({"reg_name": reg_name, "version": version_of(reg_name),
                         "code_file": reg_name + ".py", "real_mae": real_mae,
                         "round_idx": round_idx,
                         "created_at": datetime.now(timezone.utc).isoformat(),
                         "parent_reg_name": parent_reg_name})
        versions.sort(key=lambda v: v["version"])
        self._persist_lineage(fam)

    def _persist_lineage(self, family: str) -> None:
        """原子写 lineage_<family>.json (临时文件 + rename, 防半途崩溃留半文件)。"""
        if not self.persist_dir:
            return
        path = os.path.join(self.persist_dir, f"lineage_{family}.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"family": family, "versions": self._lineages[family]},
                      f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ------------------------------------------------------------------
    # 持久化 (算子本体)
    # ------------------------------------------------------------------

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
        """从 persist_dir 重新加载并注册所有持久化的算子 + 家谱 + B7 辅助任务。返回注册名列表。

        按文件名前缀分派 (与 synth_ 算子共存): aux_*.json → AUX_OPS; 其余算子 json → SPATIAL_OPS;
        lineage_*.json 走家谱段 (跳过)。坏的持久文件跳过不崩。
        兼容旧目录 (B2 前): 无 lineage_*.json 时, 每个持久算子隐式成单版本族
        (real_mae 从算子 json 的钩子字段读回)。
        """
        if not self.persist_dir or not os.path.isdir(self.persist_dir):
            return []
        loaded = []
        loaded_mae: dict[str, float | None] = {}
        for fn in sorted(os.listdir(self.persist_dir)):
            if not fn.endswith(".json") or fn.startswith("lineage_"):
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
            # B7 辅助任务: aux_ 前缀 (或 meta kind=aux) → 注入 AUX_OPS, 不走 synth 链路
            if reg_name.startswith(AUX_PREFIX) or meta.get("kind") == "aux":
                try:
                    cls = exec_operator_code(code, meta["class_name"])
                    ops_mod.AUX_OPS[reg_name] = cls
                    self._registered_aux[reg_name] = _RegisteredAux(
                        reg_name, meta["class_name"], code, plan=dict(meta.get("plan", {})))
                    loaded.append(reg_name)
                except Exception:
                    continue  # 坏的持久 aux 跳过, 不崩
                continue
            try:
                cls = exec_operator_code(code, meta["class_name"])
                ops_mod.SPATIAL_OPS[reg_name] = _make_factory(cls)
                ops_mod.OP_CATEGORY[reg_name] = meta.get("category", "spatiotemporal")
                self._registered[reg_name] = _RegisteredOp(
                    reg_name, meta["class_name"], code, meta.get("needs_adj", False),
                    composition=meta.get("composition", ""),
                    source_mechanisms=list(meta.get("source_mechanisms", [])))
                loaded.append(reg_name)
                loaded_mae[reg_name] = meta.get("real_mae")
            except Exception:
                continue  # 坏的持久算子跳过, 不崩

        # 家谱: 先读 lineage 文件 (显式记录优先)
        for fn in sorted(os.listdir(self.persist_dir)):
            if not (fn.startswith("lineage_") and fn.endswith(".json")):
                continue
            fam = fn[len("lineage_"):-5]
            try:
                with open(os.path.join(self.persist_dir, fn)) as f:
                    data = json.load(f)
                versions = [v for v in data.get("versions", [])
                            if isinstance(v, dict) and "reg_name" in v]
            except Exception:
                continue            # 坏 lineage 文件跳过, 不崩
            if versions:
                for v in versions:
                    v.setdefault("version", version_of(v["reg_name"]))
                versions.sort(key=lambda v: v["version"])
                self._lineages[fam] = versions

        # 隐式族: 已注册但不在任何 lineage 里的算子 (旧目录单版本族成立)
        for reg_name in loaded:
            if reg_name in self._registered_aux:
                continue                # B7 辅助任务不进算子家谱 (独立表, 无版本族语义)
            fam = family_of(reg_name)
            have = {v["reg_name"] for v in self._lineages.get(fam, [])}
            if reg_name not in have:
                self._lineages.setdefault(fam, []).append(
                    {"reg_name": reg_name, "version": version_of(reg_name),
                     "code_file": reg_name + ".py", "real_mae": loaded_mae.get(reg_name),
                     "round_idx": None, "created_at": None, "parent_reg_name": None})
        for versions in self._lineages.values():
            versions.sort(key=lambda v: v["version"])
        return loaded


def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


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
