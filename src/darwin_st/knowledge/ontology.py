"""机制中心知识本体 (Mechanism-Centric Ontology) —— 跨域类比迁移的数据模型。

设计依据: docs/TIER2_RESEARCH_FINDINGS.md §3-4 (结构映射理论 + GoF/SAPPhIRE 三元组)。

核心思想 (Gentner 结构映射): 类比迁移靠**关系结构**(抽象功能+前提条件+因果)而非
表面属性。故 Mechanism 节点必须把"抽象功能""前提条件""因果行为"做成结构化字段,
才能被跨域检索到——"机制 M 来自域 X 满足前提 P; 我的 ST 任务也满足 P; 故值得移植"。

两层置信度 (抽取可靠性): stated(grounded, 高置信) vs inferred(需人工复核, 低置信)。

存储后端可插拔: 本数据模型是规范表示, 可序列化到 Neo4j 或纯 Python 嵌入式图。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict

__all__ = [
    "Evidence",
    "Mechanism",
    "Precondition",
    "DOMAINS",
    "PRECONDITION_VOCAB",
    "FUNCTION_VOCAB",
]

# 来源领域 (跨域过滤键) —— origin_domain 必须在此集合
DOMAINS = {
    "CV",            # 计算机视觉
    "NLP",           # 自然语言处理
    "SSM",           # 状态空间模型 / 序列建模
    "GraphLearning", # 图表示学习
    "SelfSupervised",# 自监督学习 (跨域范式)
    "TimeSeries",    # 通用时间序列
    "ST",            # 时空预测 (本域)
    "Optimization",  # 优化/训练
    "Generative",    # 生成模型
    "Control",       # 控制论/动力系统
}

# 受控前提词表 (类比钩子, 共享节点) —— 跨域迁移的关键
# 这些是"抽象的、域无关的"数据/任务性质, 让"前提匹配"成为可查询的约束
PRECONDITION_VOCAB = {
    "redundant_structure",     # 数据有冗余/低秩结构 (掩码/压缩类机制的前提)
    "label_scarcity",          # 标签稀缺 (自监督的前提)
    "long_range_dependency",   # 存在长程依赖 (SSM/注意力的前提)
    "non_euclidean_topology",  # 非欧几里得拓扑/图结构 (图卷积的前提)
    "spatial_smoothness",      # 空间平滑性 (扩散/图正则的前提)
    "temporal_periodicity",    # 时间周期性 (周期嵌入的前提)
    "multi_scale_structure",   # 多尺度结构 (金字塔/膨胀的前提)
    "distribution_shift",      # 分布漂移 (域适应/鲁棒的前提)
    "node_indistinguishability",  # 节点不可区分 (身份嵌入的前提)
    "high_dimensionality",     # 高维 (降维/瓶颈的前提)
    "sequential_order",        # 有序序列 (因果卷积/递归的前提)
    "sparse_interaction",      # 稀疏交互 (稀疏注意力的前提)
    "heterogeneity",           # 异质性 (混合专家/元学习的前提)
    "noise_corruption",        # 噪声污染 (去噪/扩散的前提)
}

# 受控功能词表 (verb+flow 元组的简化, 域无关的"做什么")
FUNCTION_VOCAB = {
    "self_supervised_representation",  # 自监督表示学习
    "long_range_mixing",               # 长程信息混合
    "spatial_aggregation",             # 空间邻域聚合
    "temporal_modeling",               # 时序建模
    "identity_injection",              # 身份/位置信息注入
    "denoising",                       # 去噪
    "multi_scale_fusion",              # 多尺度融合
    "adaptive_weighting",              # 自适应加权
    "dimensionality_reduction",        # 降维/瓶颈
    "robustness_regularization",       # 鲁棒性正则
    "generative_modeling",             # 生成建模
}


@dataclass
class Evidence:
    """机制的量化证据 (stated 层, grounded)。"""

    dataset: str             # 在什么数据集上
    metric: str              # 什么指标 (MAE/Acc/...)
    delta: str               # 改进幅度描述 (如 "MAE 18.29→21.65 去掉后" 或 "+18%")
    grade: str = "moderate"  # GRADE 证据等级: high/moderate/low
    source: str = ""         # 出处 (论文/arXiv id)


@dataclass
class Mechanism:
    """机制节点 (本体新中心)。

    字段分两层置信度:
      stated(grounded): name/math_structure/consequences/evidence/origin_domain
      inferred(需人工复核): abstract_function/preconditions/causal_behavior
    """

    name: str                              # 规范名, 如 "masked_autoencoding"

    # -- inferred 层 (类比的关键, 需人工把关) --
    abstract_function: str                 # 域无关一句话目的 (会被嵌入做 MAC 检索)
    preconditions: list[str]               # 适用前提 (须 ∈ PRECONDITION_VOCAB) —— 类比钩子
    causal_behavior: str                   # 结构→功能的因果链 (类比桥)
    function_tags: list[str] = field(default_factory=list)  # ∈ FUNCTION_VOCAB

    # -- stated 层 (grounded) --
    math_structure: str = ""               # 具体数学/算子描述 (喂 Aider 合成的素材)
    consequences: str = ""                 # 权衡 (算力/内存/不稳定 等代价)
    origin_domain: str = "ST"              # 来源领域 (须 ∈ DOMAINS) —— 跨域过滤键
    abstraction_level: str = "concept"     # 抽象层级: concept(高,跨域) / variant(低,同域)
    evidence: list[Evidence] = field(default_factory=list)
    anti_patterns: list[str] = field(default_factory=list)  # 反模式/雷区
    related: dict[str, list[str]] = field(default_factory=dict)  # {REFINES/ALTERNATIVE_TO/COMPOSES_WITH: [name]}
    provenance: str = ""                   # 出处
    inferred: bool = True                  # 该卡的 inferred 层是否已人工复核 (True=待核)

    def validate(self) -> None:
        if self.origin_domain not in DOMAINS:
            raise ValueError(f"{self.name}: origin_domain '{self.origin_domain}' 不在 DOMAINS {sorted(DOMAINS)}")
        for p in self.preconditions:
            if p not in PRECONDITION_VOCAB:
                raise ValueError(f"{self.name}: precondition '{p}' 不在受控词表; 请先在 PRECONDITION_VOCAB 登记")
        for f in self.function_tags:
            if f not in FUNCTION_VOCAB:
                raise ValueError(f"{self.name}: function_tag '{f}' 不在受控词表")
        if self.abstraction_level not in ("concept", "variant"):
            raise ValueError(f"{self.name}: abstraction_level 须为 concept/variant")
        if not self.abstract_function.strip():
            raise ValueError(f"{self.name}: abstract_function 不可为空 (类比检索依赖它)")

    def signature(self) -> str:
        return hashlib.sha1(self.name.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def embedding_text(self) -> str:
        """用于 MAC 向量检索的文本 = 抽象功能 + 前提 (绝不含表面描述/域名)。

        研究关键: 只嵌"抽象功能+前提", 不嵌表面/域, 否则检索退化为同域语义相似。
        """
        return self.abstract_function + " || preconditions: " + ", ".join(self.preconditions)


@dataclass
class Precondition:
    """前提节点 (共享, 类比钩子)。多个机制经同一前提相连 → 跨域检索的桥。"""

    name: str                    # ∈ PRECONDITION_VOCAB
    description: str = ""

    def validate(self) -> None:
        if self.name not in PRECONDITION_VOCAB:
            raise ValueError(f"未知前提 '{self.name}'")
