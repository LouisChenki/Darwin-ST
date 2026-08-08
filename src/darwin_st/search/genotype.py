"""架构基因型 (Architecture Genotype) —— 进化式 NAS 的可变异离散编码。

genotype 是「离散、可序列化、可变异」的架构表示 (phenotype=builder 解码出的真实网络)。
进化算子 (evolution.py) 在 genotype 上做 op-swap / depth-change 等突变, builder.py
把它编译成 nn.Module。

编码结构 (见 docs/BLUEPRINT.md §4.1):
  - blocks: 若干 ST-block, 每块 = {spatial_op, temporal_op, fusion}
  - depth/hidden: 全局深度与隐藏维
  - adj_mode: 邻接归一化模式 (sym/rw/none, 仅对需要外部图的空间算子)
  - aux_op: B7 辅助任务槽位 (单槽, 注册名或 None) —— 防空间爆炸的关键设计:
    辅助任务不是每块一个, 而是全模型至多挂一个自监督辅助损失 (STD-MAE 路线)
  - protected: 创新点保护区 —— Agent 在 Tier2 写入的创新算子名列表, **进化禁止删除**

铁律 (见 docs/P2_ALGORITHM_DESIGN.md 架构铁律):
  - 【创新点保护区】protected 中的算子在任何变异下不得被移除 (可调其超参/接线, 不可删)
  - 节点维 N 由算子层保证不丢; genotype 层不引入展平
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict

from darwin_st.search.operators import SPATIAL_OPS, SPATIOTEMPORAL_OPS, TEMPORAL_OPS

__all__ = ["STBlock", "EmbeddingConfig", "Genotype", "mutate", "random_genotype"]


def _all_op_names() -> set[str]:
    """全部可用算子名 (三类内置 + 已注入 synth)。synth 注册进 SPATIAL_OPS dict, 故已含。"""
    return set(SPATIAL_OPS) | set(TEMPORAL_OPS) | set(SPATIOTEMPORAL_OPS)


def _block_to_dict(b: "STBlock") -> dict:
    """STBlock → dict, **省略 joint_op=None** (omit-None): 保旧线性 genotype 签名字节不变。"""
    d = {"spatial_op": b.spatial_op, "temporal_op": b.temporal_op, "fusion": b.fusion}
    if b.joint_op is not None:
        d["joint_op"] = b.joint_op
    return d

VALID_FUSION = {
    "sequential",      # 先空间后时序 (S→T, 现状默认)
    "sequential_ts",   # 先时序后空间 (T→S)
    "parallel",        # 空间(x) + 时序(x) 相加
    "residual",        # x + (S→T)
    "cross",           # 门控交叉交互: T(x) * sigmoid(S(x)) (时空互相调制)
    "iterative",       # 双向迭代两轮: S→T→S→T
}
VALID_ADJ_MODE = {"sym", "rw", "none"}


@dataclass
class STBlock:
    """单个时空块。两种模式:
      - 分离模式 (joint_op=None): 一个空间算子 + 一个时序算子, 按 fusion 接线。
      - 一体模式 (joint_op 非空): 单个时空一体算子 (joint) 直接建模时空, 取代 S+T 对。
        joint_op 优先 (非空时 spatial_op/temporal_op 忽略, 但保留作占位)。
    """

    spatial_op: str          # SPATIAL_OPS 中的键 (joint 模式下作占位)
    temporal_op: str         # TEMPORAL_OPS 中的键 (joint 模式下作占位)
    fusion: str = "sequential"  # 见 VALID_FUSION (分离模式接线方式)
    joint_op: str | None = None  # 非空 → 一体模式, 用此时空算子 (取代 S+T)

    def validate(self) -> None:
        if self.joint_op is not None:
            # 一体模式: 校验 joint 算子在算子库 (类别由 op_category 兜底, 签名相同任意类别都能跑)
            if self.joint_op not in _all_op_names():
                raise ValueError(f"未知时空一体算子: {self.joint_op}")
        else:
            # 分离模式: 类别兼容校验 (放宽 —— 允许 LLM 时空算子坐任一槽)
            if self.spatial_op not in _all_op_names():
                raise ValueError(f"未知空间算子: {self.spatial_op}")
            if self.temporal_op not in _all_op_names():
                raise ValueError(f"未知时序算子: {self.temporal_op}")
        if self.fusion not in VALID_FUSION:
            raise ValueError(f"未知融合方式: {self.fusion} (可选 {sorted(VALID_FUSION)})")


@dataclass
class EmbeddingConfig:
    """身份嵌入配置 (一等可变异基因)。

    研究结论: 节点嵌入比图算子更提精度 (去掉 → MAE+18%)。故默认开节点嵌入。
    time-of-day / day-of-week 需数据管道提供时间索引, 默认关 (就绪后可开)。
    """

    use_node: bool = True
    node_dim: int = 64
    use_tod: bool = True
    tod_dim: int = 64
    use_dow: bool = True
    dow_dim: int = 64

    VALID_DIMS = (32, 64, 96, 128)

    def validate(self) -> None:
        for label, dim in (("node", self.node_dim), ("tod", self.tod_dim), ("dow", self.dow_dim)):
            if dim <= 0:
                raise ValueError(f"{label}_dim 必须为正: {dim}")


@dataclass
class Genotype:
    """完整架构基因型。

    aux_op (B7): 辅助任务槽位 —— None (默认) 或 AUX_OPS 注册名 (aux_ 前缀)。
    只校验"None 或合法标识符"; 注册与否由 build_model 挂载时查 (genotype 层
    不依赖运行时注册表, 保序列化/签名纯数据语义)。
    **签名语义注意**: to_dict 对 aux_op=None 走 omit-None (同 joint_op) —— 无 aux 的
    genotype 签名字节与引入本字段前完全一致 (零换代); 一旦 aux_op 非 None, genotype
    签名即变 → 记忆/查重把"同架构带不带 aux"当不同 trial 对待 (不串)。注意
    space_version (operators.builtin_op_signature: 内置算子名+schema 标记的哈希) **不受
    本字段影响** —— aux 是训练期辅助损失而非架构搜索空间代际变更, 有无 aux 的架构留在
    同代比较 (消融对比正是研究所需), 这是有意设计而非遗漏。
    """

    blocks: list[STBlock]
    hidden: int = 64
    adj_mode: str = "sym"
    # 身份嵌入配置 (一等基因, 默认开节点嵌入)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # 创新点保护区: 这些空间/时序算子名在变异中不可被删除 (可改超参)
    protected: list[str] = field(default_factory=list)
    # B7 辅助任务 (单槽): None = 无辅助损失 (现状不变); 非 None = AUX_OPS 注册名
    aux_op: str | None = None

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def validate(self) -> None:
        if not self.blocks:
            raise ValueError("genotype 至少需要 1 个 block")
        if self.hidden <= 0:
            raise ValueError(f"hidden 必须为正: {self.hidden}")
        if self.adj_mode not in VALID_ADJ_MODE:
            raise ValueError(f"未知 adj_mode: {self.adj_mode} (可选 {sorted(VALID_ADJ_MODE)})")
        # aux_op: None 或合法标识符 (注册与否由 build_model 挂载时查, 此处不依赖注册表)
        if self.aux_op is not None and not self.aux_op.isidentifier():
            raise ValueError(f"aux_op '{self.aux_op}' 须为 None 或合法 Python 标识符")
        self.embedding.validate()
        for b in self.blocks:
            b.validate()
        # 保护区算子必须确实存在于某个 block 中
        for p in self.protected:
            if not self._uses_op(p):
                raise ValueError(f"保护区算子 '{p}' 未在任何 block 中使用")

    def _uses_op(self, op_name: str) -> bool:
        return any(
            b.spatial_op == op_name or b.temporal_op == op_name or b.joint_op == op_name
            for b in self.blocks
        )

    # -- 序列化 --
    def to_dict(self) -> dict:
        d = {
            "blocks": [_block_to_dict(b) for b in self.blocks],   # omit-None joint_op 保旧签名
            "hidden": self.hidden,
            "adj_mode": self.adj_mode,
            "embedding": asdict(self.embedding),
            "protected": list(self.protected),
        }
        # omit-None aux_op (B7): 无辅助任务的 genotype 签名字节与引入 aux_op 前一致 (零换代);
        # 非 None → genotype 签名变 (带/不带 aux 是不同 trial)。space_version
        # (builtin_op_signature) 有意不受本字段影响, 见类 docstring
        if self.aux_op is not None:
            d["aux_op"] = self.aux_op
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Genotype":
        blocks = [STBlock(**b) for b in d["blocks"]]
        emb_d = d.get("embedding", {})
        # 兼容旧 genotype (无 embedding 字段) → 用默认配置
        embedding = EmbeddingConfig(**emb_d) if emb_d else EmbeddingConfig()
        return cls(
            blocks=blocks,
            hidden=d.get("hidden", 32),
            adj_mode=d.get("adj_mode", "sym"),
            embedding=embedding,
            protected=list(d.get("protected", [])),
            aux_op=d.get("aux_op"),              # 兼容旧 dict (无 aux_op 键) → None
        )

    def signature(self) -> str:
        """稳定哈希 (查重键), 与 memory.compute_signature 口径一致 (有序 JSON)。"""
        blob = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()

    def copy(self) -> "Genotype":
        return Genotype.from_dict(copy.deepcopy(self.to_dict()))


# ---------------------------------------------------------------------------
# 变异算子 (进化式 NAS 的突变操作)
# ---------------------------------------------------------------------------


def _is_protected(geno: Genotype, block: STBlock) -> bool:
    """该 block 是否触及保护区算子 (触及则其算子不可被替换/删除)。含 joint 槽。"""
    return (block.spatial_op in geno.protected or block.temporal_op in geno.protected
            or (block.joint_op is not None and block.joint_op in geno.protected))


def mutate(geno: Genotype, op: str, rng_index: int = 0, **params) -> Genotype:
    """对 genotype 施加一种变异, 返回**新** genotype (不改原对象)。

    op:
      - "swap_spatial":  把第 i 块的空间算子换成 new_op
      - "swap_temporal": 把第 i 块的时序算子换成 new_op
      - "swap_joint":    把第 i 块 (一体模式) 的时空算子换成 new_op
      - "toggle_joint":  第 i 块在 分离↔一体 模式间切换 (进/出 joint 模式)
      - "change_fusion": 改第 i 块融合方式
      - "add_block":     在末尾加一块 (new_block)
      - "remove_block":  删第 i 块 (保护区块拒删)
      - "change_hidden": 改隐藏维
      - "change_adj_mode": 改邻接归一化模式
      - "aux_toggle":    B7 辅助任务开关: None→挂 new_op (AUX_OPS 注册名); 已挂→置 None
      - "aux_swap":      B7 换一个辅助任务 (须已挂, new_op 为新注册名)

    变异铁律: 触及保护区算子的块, 其对应算子不可被 swap/remove。
    rng_index/params 由调用方 (进化层) 提供具体选择, 本函数保持确定性。
    """
    g = geno.copy()

    if op == "swap_spatial":
        i, new_op = params["index"], params["new_op"]
        if g.blocks[i].spatial_op in g.protected:
            raise ValueError(f"block[{i}] 空间算子在保护区, 不可替换")
        g.blocks[i].spatial_op = new_op

    elif op == "swap_temporal":
        i, new_op = params["index"], params["new_op"]
        if g.blocks[i].temporal_op in g.protected:
            raise ValueError(f"block[{i}] 时序算子在保护区, 不可替换")
        g.blocks[i].temporal_op = new_op

    elif op == "swap_joint":
        i, new_op = params["index"], params["new_op"]
        if g.blocks[i].joint_op is None:
            raise ValueError(f"block[{i}] 非一体模式, 无 joint 算子可换 (先 toggle_joint)")
        if g.blocks[i].joint_op in g.protected:
            raise ValueError(f"block[{i}] 时空一体算子在保护区, 不可替换")
        g.blocks[i].joint_op = new_op

    elif op == "toggle_joint":
        # 分离↔一体 模式切换。进 joint: 设 joint_op (取代 S+T); 出 joint: 清空回分离。
        i = params["index"]
        b = g.blocks[i]
        if b.joint_op is not None:
            if b.joint_op in g.protected:
                raise ValueError(f"block[{i}] 一体算子在保护区, 不可退出一体模式")
            b.joint_op = None                    # 退回分离模式 (S+T 占位符复活)
        else:
            b.joint_op = params["new_op"]        # 进入一体模式

    elif op == "change_fusion":
        i, new_fusion = params["index"], params["new_fusion"]
        g.blocks[i].fusion = new_fusion

    elif op == "add_block":
        nb = params["new_block"]
        g.blocks.append(nb if isinstance(nb, STBlock) else STBlock(**nb))

    elif op == "remove_block":
        i = params["index"]
        if len(g.blocks) <= 1:
            raise ValueError("至少保留 1 块, 不可再删")
        if _is_protected(g, g.blocks[i]):
            raise ValueError(f"block[{i}] 触及保护区, 不可删除")
        g.blocks.pop(i)

    elif op == "change_hidden":
        g.hidden = params["new_hidden"]

    elif op == "change_adj_mode":
        g.adj_mode = params["new_adj_mode"]

    elif op == "toggle_embedding":
        # 把身份嵌入当可变异基因: 开关某类嵌入 / 改其维度
        which = params["which"]              # "node" | "tod" | "dow"
        if which not in ("node", "tod", "dow"):
            raise ValueError(f"toggle_embedding 的 which 非法: {which}")
        if "enable" in params:
            setattr(g.embedding, f"use_{which}", bool(params["enable"]))
        if "dim" in params:
            setattr(g.embedding, f"{which}_dim", int(params["dim"]))

    elif op == "aux_toggle":
        # B7 辅助任务开关 (单槽): 未挂 → 挂 new_op; 已挂 → 置 None (与 toggle_joint 同模式)
        if g.aux_op is not None:
            g.aux_op = None                          # 关闭辅助任务
        else:
            g.aux_op = params["new_op"]              # 挂载 (注册名合法性由 build_model 查)

    elif op == "aux_swap":
        # B7 换辅助任务: 须已挂 (未挂用 aux_toggle 开启)
        if g.aux_op is None:
            raise ValueError("未挂辅助任务, 无可换 (先 aux_toggle 开启)")
        g.aux_op = params["new_op"]

    else:
        raise ValueError(f"未知变异算子: {op}")

    g.validate()
    return g


def random_genotype(
    depth: int = 2, hidden: int = 32, spatial: str = "gcn", temporal: str = "tcn",
    fusion: str = "sequential", adj_mode: str = "sym", protected: list[str] | None = None,
) -> Genotype:
    """构造一个简单的初始 genotype (确定性, 非随机 —— 名字沿用领域习惯)。

    用于冷启动基线; 进化层在此基础上变异。protected 标记创新点保护区。
    """
    blocks = [STBlock(spatial_op=spatial, temporal_op=temporal, fusion=fusion) for _ in range(depth)]
    g = Genotype(blocks=blocks, hidden=hidden, adj_mode=adj_mode, protected=protected or [])
    g.validate()
    return g
