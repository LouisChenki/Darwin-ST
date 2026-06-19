"""种子机制卡 (Seed Mechanism Cards) —— 人工精选的跨域机制库 v1。

挑选准则: 真正发生过(或可能)跨域迁移到时空预测的经典机制, 覆盖多个 origin_domain,
让跨域类比检索有真实的"远域类比"可找。质量靠人工把关(非自动抽取)。

这是 P3-a 的核心人工资产, 供用户审核。每张卡的 inferred 层(abstract_function/
preconditions/causal_behavior)是类比的关键, 已尽量基于公认认知填写。

注: STD-MAE 把 CV 的掩码自编码搬到 ST = 本库要让 LLM 能独立重现的那类移植。
"""

from __future__ import annotations

from darwin_st.knowledge.ontology import Evidence, Mechanism

# ---------------------------------------------------------------------------
# 自监督 / 预训练范式 (跨域迁移的高价值来源)
# ---------------------------------------------------------------------------

SEED_MECHANISMS: list[Mechanism] = [
    Mechanism(
        name="masked_autoencoding",
        abstract_function="通过遮盖部分输入并重建, 学习利用数据冗余结构的自监督表示",
        preconditions=["redundant_structure", "label_scarcity"],
        causal_behavior="随机遮盖输入 → 模型被迫从可见上下文推断缺失部分 → 学到数据内在的冗余/相关结构 → 表示更鲁棒",
        function_tags=["self_supervised_representation", "robustness_regularization"],
        math_structure="encoder(mask(x)) → decoder → reconstruct(x); loss = ||x - x_hat||² 仅在遮盖位",
        consequences="需预训练阶段(额外算力); 掩码率过高→欠拟合, 过低→任务太易",
        origin_domain="CV",
        abstraction_level="concept",
        evidence=[Evidence("ImageNet", "Acc", "MAE 预训练显著提升下游", "high", "He et al. 2021 MAE arXiv:2111.06377"),
                  Evidence("PeMS04/08", "MAE", "STD-MAE 借此达 SOTA 17.80/13.44", "high", "arXiv:2312.00516")],
        anti_patterns=["掩码率不当导致预训练信号过弱或过强"],
        provenance="MAE (He et al. 2021); STD-MAE 已移植到 ST",
    ),
    Mechanism(
        name="contrastive_learning",
        abstract_function="通过拉近正样本对、推远负样本对, 学习判别性自监督表示",
        preconditions=["label_scarcity", "redundant_structure"],
        causal_behavior="对同一样本的两个增强视图 → 在表示空间拉近; 不同样本推远 → 学到对增强不变的本质特征",
        function_tags=["self_supervised_representation"],
        math_structure="InfoNCE loss: -log[ exp(sim(z_i,z_j)/τ) / Σ_k exp(sim(z_i,z_k)/τ) ]",
        consequences="对负样本数量/质量敏感; 需精心设计数据增强",
        origin_domain="SelfSupervised",
        abstraction_level="concept",
        evidence=[Evidence("ImageNet", "Acc", "SimCLR/MoCo 接近监督", "high", "Chen et al. 2020 SimCLR")],
        anti_patterns=["增强过弱→表示坍缩; 负样本不足→判别力差"],
        provenance="SimCLR/MoCo (2020)",
    ),

    # -----------------------------------------------------------------------
    # 序列建模 (长程依赖的跨域来源)
    # -----------------------------------------------------------------------
    Mechanism(
        name="state_space_model",
        abstract_function="用线性状态空间递推在序列上做长程信息混合, 兼顾长程依赖与线性复杂度",
        preconditions=["long_range_dependency", "sequential_order"],
        causal_behavior="隐状态 h_t = A h_{t-1} + B x_t 沿序列传播 → 选择性保留/遗忘历史 → 以 O(L) 捕获长程依赖",
        function_tags=["long_range_mixing", "temporal_modeling"],
        math_structure="h_t = Ā h_{t-1} + B̄ x_t; y_t = C h_t; (Mamba: A,B,C 输入相关/选择性)",
        consequences="比注意力省算力(线性 vs 二次); 但状态容量有限",
        origin_domain="SSM",
        abstraction_level="concept",
        evidence=[Evidence("Long Range Arena", "Acc", "S4/Mamba 超 Transformer", "high", "Gu et al. S4/Mamba arXiv:2312.00752")],
        anti_patterns=["状态维过小→长程信息丢失"],
        provenance="S4 (2021) / Mamba (2023)",
    ),
    Mechanism(
        name="self_attention",
        abstract_function="通过查询-键相似度动态加权聚合, 实现任意两位置间的直接信息交互",
        preconditions=["long_range_dependency", "sparse_interaction"],
        causal_behavior="每个位置生成 query, 与所有位置的 key 算相似度 → softmax 加权聚合 value → 任意距离直接交互, 无递归衰减",
        function_tags=["long_range_mixing", "adaptive_weighting"],
        math_structure="Attention(Q,K,V) = softmax(QKᵀ/√d) V",
        consequences="二次复杂度 O(L²); 长序列算力/显存爆炸",
        origin_domain="NLP",
        abstraction_level="concept",
        evidence=[Evidence("WMT", "BLEU", "Transformer 革新序列建模", "high", "Vaswani et al. 2017"),
                  Evidence("PeMS04", "MAE", "STAEformer/PDFormer 用时空注意力达 SOTA 级", "high", "arXiv:2308.10425")],
        anti_patterns=["长序列二次复杂度; 无位置信息则置换不变需额外编码"],
        provenance="Transformer (2017); 已广泛用于 ST",
    ),
    Mechanism(
        name="dilated_causal_convolution",
        abstract_function="用指数膨胀的因果卷积以对数层数覆盖大感受野, 高效捕获多尺度时序",
        preconditions=["long_range_dependency", "sequential_order", "multi_scale_structure"],
        causal_behavior="卷积核间隔随层指数增长 → 感受野指数扩大 → 少量层即覆盖长历史, 且因果填充不看未来",
        function_tags=["long_range_mixing", "multi_scale_fusion", "temporal_modeling"],
        math_structure="y_t = Σ_k w_k · x_{t - d·k}; 膨胀率 d = 2^layer",
        consequences="比 RNN 可并行; 感受野固定(非自适应)",
        origin_domain="SSM",
        abstraction_level="variant",
        evidence=[Evidence("PeMS", "MAE", "Graph WaveNet/TCN 主干", "high", "WaveNet 2016; GWNet 2019")],
        anti_patterns=["膨胀率与序列长度不匹配→感受野不足或浪费"],
        provenance="WaveNet (2016) / TCN",
    ),

    # -----------------------------------------------------------------------
    # 图学习 (空间结构的本域+邻域来源)
    # -----------------------------------------------------------------------
    Mechanism(
        name="adaptive_adjacency",
        abstract_function="从可学习节点嵌入自动学出图邻接, 无需预定义图也能建模实体间关系",
        preconditions=["non_euclidean_topology", "node_indistinguishability"],
        causal_behavior="可学习节点嵌入 E1,E2 → softmax(relu(E1 E2ᵀ)) 生成自适应邻接 → 数据驱动地发现节点间隐含关系",
        function_tags=["spatial_aggregation", "adaptive_weighting"],
        math_structure="A_adp = softmax(ReLU(E1 E2ᵀ)); 然后图卷积 A_adp X W",
        consequences="不依赖先验图; 但 N² 邻接对大图开销大",
        origin_domain="GraphLearning",
        abstraction_level="concept",
        evidence=[Evidence("METR-LA", "MAE", "Graph WaveNet 自适应图", "high", "arXiv:1906.00121")],
        anti_patterns=["大图 N² 开销; 无正则易过拟合"],
        provenance="Graph WaveNet (2019) / MTGNN",
    ),
    Mechanism(
        name="diffusion_convolution",
        abstract_function="用随机游走在图上多步扩散信息, 建模有向的、多跳的空间依赖",
        preconditions=["non_euclidean_topology", "spatial_smoothness"],
        causal_behavior="信息按转移概率矩阵 P 沿图多步扩散 → K 步覆盖 K-hop 邻域 → 捕获有向的空间传播(如交通流向)",
        function_tags=["spatial_aggregation"],
        math_structure="Σ_{k=0}^{K} P^k X W_k; P = D⁻¹A 随机游走矩阵(双向)",
        consequences="建模有向传播; K 大则过平滑",
        origin_domain="GraphLearning",
        abstraction_level="variant",
        evidence=[Evidence("METR-LA", "MAE", "DCRNN 扩散卷积", "high", "arXiv:1707.01926")],
        anti_patterns=["K 过大→过平滑, 节点表示趋同"],
        provenance="DCRNN (2017)",
    ),

    # -----------------------------------------------------------------------
    # 身份/位置注入 (近年 ST SOTA 的关键)
    # -----------------------------------------------------------------------
    Mechanism(
        name="identity_embedding",
        abstract_function="给每个实体/位置注入可学习身份向量, 打破实体间的不可区分性",
        preconditions=["node_indistinguishability", "temporal_periodicity"],
        causal_behavior="为每个节点/时刻分配可学习嵌入并拼接到特征 → 模型能区分'谁'和'何时' → 解决空间/时间不可区分瓶颈",
        function_tags=["identity_injection"],
        math_structure="Z = concat(proj(x), E_node[i], E_tod[t], E_dow[d]); E_* 可学习查表",
        consequences="几乎零额外算力, 提升巨大; 但嵌入随节点数线性增长",
        origin_domain="ST",
        abstraction_level="concept",
        evidence=[Evidence("PeMS04", "MAE", "去掉节点嵌入 MAE 18.29→21.65 (+18%)", "high", "STID arXiv:2208.05233"),
                  Evidence("PeMS04", "MAE", "STAEformer 自适应嵌入去掉 +19%", "high", "arXiv:2308.10425")],
        anti_patterns=["相加而非拼接会稀释信号"],
        provenance="STID/STAEformer (2022-23) —— 本项目已实现",
    ),
    Mechanism(
        name="rotary_position_encoding",
        abstract_function="用旋转矩阵编码相对位置, 让注意力自然感知相对距离",
        preconditions=["sequential_order", "long_range_dependency"],
        causal_behavior="对 query/key 按位置施加旋转 → 内积自动含相对位置项 → 注意力感知相对距离而非绝对",
        function_tags=["identity_injection"],
        math_structure="q_m·k_n 经旋转后 = f(q,k, m-n); 仅依赖相对位置 m-n",
        consequences="外推性好; 仅适合有序序列",
        origin_domain="NLP",
        abstraction_level="variant",
        evidence=[Evidence("语言建模", "PPL", "RoPE 改善长度外推", "moderate", "RoFormer arXiv:2104.09864")],
        anti_patterns=["用于无序集合无意义"],
        provenance="RoFormer (2021)",
    ),

    # -----------------------------------------------------------------------
    # 生成 / 去噪 (跨域范式)
    # -----------------------------------------------------------------------
    Mechanism(
        name="denoising_diffusion",
        abstract_function="学习逐步从噪声恢复数据的反向过程, 实现高质量条件生成/不确定性建模",
        preconditions=["noise_corruption", "redundant_structure"],
        causal_behavior="前向逐步加噪→纯噪声; 学习反向逐步去噪 → 从噪声采样生成数据 → 天然建模预测不确定性/多模态",
        function_tags=["generative_modeling", "denoising"],
        math_structure="前向 q(x_t|x_{t-1})=N(√(1-β)x_{t-1}, βI); 反向学 p_θ(x_{t-1}|x_t)",
        consequences="可建模不确定性/多模态; 采样慢(多步)",
        origin_domain="Generative",
        abstraction_level="concept",
        evidence=[Evidence("图像生成", "FID", "DDPM 革新生成", "high", "Ho et al. 2020"),
                  Evidence("交通预测", "CRPS", "DiffSTG 等扩散时空预测", "moderate", "扩散用于概率时空预测")],
        anti_patterns=["多步采样慢; 点预测任务可能过度复杂"],
        provenance="DDPM (2020); 已有扩散时空预测工作",
    ),

    # -----------------------------------------------------------------------
    # 多尺度 / 频域
    # -----------------------------------------------------------------------
    Mechanism(
        name="frequency_decomposition",
        abstract_function="在频域分解信号, 分别建模不同频率成分(趋势/周期/突变)",
        preconditions=["temporal_periodicity", "multi_scale_structure"],
        causal_behavior="傅里叶/小波变换 → 信号分解为不同频率 → 低频建模趋势、高频建模波动 → 周期性显式可学",
        function_tags=["multi_scale_fusion", "temporal_modeling"],
        math_structure="X_freq = FFT(x); 在频域滤波/学习; x' = iFFT(...)",
        consequences="显式建模周期; 对非平稳信号需配合分解",
        origin_domain="TimeSeries",
        abstraction_level="concept",
        evidence=[Evidence("长期预测", "MSE", "FEDformer/Autoformer 频域分解", "high", "FEDformer arXiv:2201.12740")],
        anti_patterns=["非平稳信号直接 FFT 失真"],
        provenance="Autoformer/FEDformer (2021-22)",
    ),
    Mechanism(
        name="series_decomposition",
        abstract_function="把序列分解为趋势项与季节项分别建模, 降低非平稳性",
        preconditions=["temporal_periodicity", "distribution_shift"],
        causal_behavior="移动平均提取趋势 → 残差为季节项 → 趋势与季节分开建模 → 缓解非平稳带来的预测难度",
        function_tags=["temporal_modeling", "robustness_regularization"],
        math_structure="trend = AvgPool(x); seasonal = x - trend; 分别预测后相加",
        consequences="简单有效缓解非平稳; 分解窗口需调",
        origin_domain="TimeSeries",
        abstraction_level="variant",
        evidence=[Evidence("长期预测", "MSE", "Autoformer/DLinear 分解", "high", "DLinear arXiv:2205.13504")],
        anti_patterns=["窗口不当→趋势/季节分离不净"],
        provenance="Autoformer (2021) / DLinear",
    ),

    # -----------------------------------------------------------------------
    # 训练范式 / 正则 (C5 训练过程优化的素材)
    # -----------------------------------------------------------------------
    Mechanism(
        name="curriculum_learning",
        abstract_function="由易到难安排训练样本/任务顺序, 提升收敛与泛化",
        preconditions=["distribution_shift", "heterogeneity"],
        causal_behavior="先训练简单样本 → 模型建立基础表示 → 逐步引入难样本 → 避免早期被难样本误导, 收敛更稳",
        function_tags=["robustness_regularization"],
        math_structure="按难度排序样本; 训练中逐步放开难度阈值",
        consequences="收敛更稳; 需定义'难度'度量",
        origin_domain="Optimization",
        abstraction_level="concept",
        evidence=[Evidence("多任务", "—", "Bengio 等证明有益", "moderate", "Bengio et al. 2009")],
        anti_patterns=["难度度量不当→课程无效"],
        provenance="Curriculum Learning (Bengio 2009)",
    ),
    Mechanism(
        name="mixture_of_experts",
        abstract_function="用门控网络为不同输入路由到不同专家子网络, 条件计算提升容量",
        preconditions=["heterogeneity", "high_dimensionality"],
        causal_behavior="门控为每个输入选 top-k 专家 → 不同模式由专门子网处理 → 大容量但激活稀疏, 算力可控",
        function_tags=["adaptive_weighting"],
        math_structure="y = Σ_i g_i(x) · expert_i(x); g = top-k softmax 门控",
        consequences="容量大且条件计算省算力; 负载均衡难调",
        origin_domain="NLP",
        abstraction_level="concept",
        evidence=[Evidence("语言建模", "PPL", "MoE 大模型主流", "high", "Switch Transformer 2021"),
                  Evidence("ST", "MAE", "异质节点/时段可用 MoE", "low", "潜在移植")],
        anti_patterns=["门控坍缩→只用少数专家; 负载不均"],
        provenance="MoE / Switch Transformer (2021)",
    ),
    Mechanism(
        name="meta_node_parameter",
        abstract_function="为不同节点/时段生成专属参数, 建模时空异质性",
        preconditions=["heterogeneity", "node_indistinguishability"],
        causal_behavior="从节点/时间嵌入生成该节点专属的卷积核/权重 → 每个节点有个性化变换 → 显式建模'不同路口行为不同'",
        function_tags=["identity_injection", "adaptive_weighting"],
        math_structure="W_i = hypernet(E_node[i], E_time[t]); 然后用 W_i 做该节点变换",
        consequences="建模异质性强; 参数生成增加复杂度",
        origin_domain="ST",
        abstraction_level="concept",
        evidence=[Evidence("METR-LA", "MAE", "HimNet 元参数学习达 2.92", "high", "arXiv:2405.10800")],
        anti_patterns=["参数生成网络过大→过拟合"],
        provenance="HimNet (2024) —— 异质性元学习",
    ),
    Mechanism(
        name="residual_connection",
        abstract_function="用跳跃连接让信息绕过变换直达, 缓解深层退化与过平滑",
        preconditions=["spatial_smoothness", "multi_scale_structure"],
        causal_behavior="输出 = 输入 + F(输入) → 梯度可直接回传, 深层不退化 → 图网络中抵抗过平滑(保留原始节点信息)",
        function_tags=["robustness_regularization"],
        math_structure="y = x + F(x); F 为残差变换",
        consequences="使极深网络可训; 几乎零代价",
        origin_domain="CV",
        abstraction_level="concept",
        evidence=[Evidence("ImageNet", "Acc", "ResNet 革新深度", "high", "He et al. 2015"),
                  Evidence("ST-GNN", "MAE", "抵抗深层图卷积过平滑", "high", "广泛使用")],
        anti_patterns=["维度不匹配需投影"],
        provenance="ResNet (2015) —— 本项目 builder 已用",
    ),
]


def all_seed_mechanisms() -> list[Mechanism]:
    """返回全部种子机制卡 (已校验)。"""
    for m in SEED_MECHANISMS:
        m.validate()
    return SEED_MECHANISMS
