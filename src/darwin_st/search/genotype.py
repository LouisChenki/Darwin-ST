"""架构基因型 (Architecture Genotype) —— 进化式 NAS 的可变异离散编码。

genotype 是「离散、可序列化、可变异」的架构表示 (phenotype=builder 解码出的真实网络)。
进化算子 (evolution.py) 在 genotype 上做 op-swap / depth-change 等突变, builder.py
把它编译成 nn.Module。

编码结构 (见 docs/BLUEPRINT.md §4.1):
  - blocks: 若干 ST-block, 每块 = {spatial_op, temporal_op, fusion}
  - depth/hidden: 全局深度与隐藏维
  - adj_mode: 邻接归一化模式 (sym/rw/none, 仅对需要外部图的空间算子)
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

from darwin_st.search.operators import SPATIAL_OPS, TEMPORAL_OPS

__all__ = ["STBlock", "EmbeddingConfig", "Genotype", "mutate", "random_genotype"]

VALID_FUSION = {"sequential", "parallel", "residual"}
VALID_ADJ_MODE = {"sym", "rw", "none"}


@dataclass
class STBlock:
    """单个时空块: 一个空间算子 + 一个时序算子 + 融合方式。"""

    spatial_op: str          # SPATIAL_OPS 中的键
    temporal_op: str         # TEMPORAL_OPS 中的键
    fusion: str = "sequential"  # sequential(S后T) / parallel(S||T相加) / residual(+输入)

    def validate(self) -> None:
        if self.spatial_op not in SPATIAL_OPS:
            raise ValueError(f"未知空间算子: {self.spatial_op} (可选 {sorted(SPATIAL_OPS)})")
        if self.temporal_op not in TEMPORAL_OPS:
            raise ValueError(f"未知时序算子: {self.temporal_op} (可选 {sorted(TEMPORAL_OPS)})")
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
    """完整架构基因型。"""

    blocks: list[STBlock]
    hidden: int = 64
    adj_mode: str = "sym"
    # 身份嵌入配置 (一等基因, 默认开节点嵌入)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    # 创新点保护区: 这些空间/时序算子名在变异中不可被删除 (可改超参)
    protected: list[str] = field(default_factory=list)

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
        self.embedding.validate()
        for b in self.blocks:
            b.validate()
        # 保护区算子必须确实存在于某个 block 中
        for p in self.protected:
            if not self._uses_op(p):
                raise ValueError(f"保护区算子 '{p}' 未在任何 block 中使用")

    def _uses_op(self, op_name: str) -> bool:
        return any(b.spatial_op == op_name or b.temporal_op == op_name for b in self.blocks)

    # -- 序列化 --
    def to_dict(self) -> dict:
        return {
            "blocks": [asdict(b) for b in self.blocks],
            "hidden": self.hidden,
            "adj_mode": self.adj_mode,
            "embedding": asdict(self.embedding),
            "protected": list(self.protected),
        }

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
    """该 block 是否触及保护区算子 (触及则其算子不可被替换/删除)。"""
    return block.spatial_op in geno.protected or block.temporal_op in geno.protected


def mutate(geno: Genotype, op: str, rng_index: int = 0, **params) -> Genotype:
    """对 genotype 施加一种变异, 返回**新** genotype (不改原对象)。

    op:
      - "swap_spatial":  把第 i 块的空间算子换成 new_op
      - "swap_temporal": 把第 i 块的时序算子换成 new_op
      - "change_fusion": 改第 i 块融合方式
      - "add_block":     在末尾加一块 (new_block)
      - "remove_block":  删第 i 块 (保护区块拒删)
      - "change_hidden": 改隐藏维
      - "change_adj_mode": 改邻接归一化模式

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
