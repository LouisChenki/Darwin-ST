# 🧠 AutoResearch 贝叶斯实验记忆库 (Bayesian Hippocampus)

> **智能体使用指南 (Agent Directives):**
> 本文件是你跨越数百次实验的长期“信念（Beliefs）”分布锚点。你的目标是以科学家的严谨态度，动态调整对各模块架构与参数空间的贝叶斯信心。每次突变前必须对比先验，每次试验后必须更新后验概率。

---

## 🎯 领域先验 (Domain Priors)
*(Phase 1 初始化时，必须通过检索 `baseline_registry.py`，在此写入该目标数据集所有公认前沿基线的标尺均值)*

- **[领域硬约束警告]**: 任何测试完毕、未能降至此先验误差射程内（如远大于最差 Baseline 的 MAE）的模型架构变动，均被严厉判定为“未达标的实验性信念 (Unqualified Experimental Belief)”，仅存入禁区，**绝对严禁**混入下方的高置信度核心基因区！

## 🧬 核心基因 (High-Confidence Components)
*(记录在多次迭代以及极值似然性评估中，被证明具有绝对高置信度 (High Confidence) 及稳定性能贡献幅度的核心架构路线)*

- **当前主流架构的信念分布 (Current SOTA Architecture Belief)**: [初始态：TemporalEmbedding + DCRNN/LNN占位]
- **空间感知机制置信度 (Spatial Conv Confidence)**: [例如：两层基于 adj.npy 的 GraphConvolution (初始先验极高置信度)]
- **时间频率机制置信度 (Temporal Prior Confidence)**: [例如：TOD / DOW 双频信号加和嵌入 (初始先验高置信度)]
- **最优后验概率落点 (Best MAE Posterior Drop)**: [待更新后验数值...]

## 🎛️ 参数先验区间 (Parameter Ranges)
*(记录关键超参数的贝叶斯分布演进，标注哪些数值区间已被确认为高概率良性空间，哪些区间已被彻底证伪)*

- **内部表征隐藏维度 (d_model/Hidden Dim)**: [待收敛分布先验...]
- **模型学习能力调度 (LR / AMP Engine)**: [待收敛分布先验...]
- **架构专属算子参数 (如：图传递深度、LNN时间常数流速)**: [待收敛分布先验...]

## ☠️ 实验禁区 (The Graveyard)
*(记录似然性极差、遭遇梯度爆炸、导致灾难性崩塌或 Shape Mismatch 的那些被证伪假设)*

1. **[被证伪假设 1 - 架构类]**: 
   - **错误推断 (Failed Hypothesis)**: [例如：意图粗暴通过 `[B, T_in, N, C] -> [B, N*C]` 抹杀图属性]
   - **崩溃结症 (Likelihood Collapse)**: [严重违背 GeoAI 拓扑隔离理论，极大偏离时空先验分布边界]
   - **后验纠正 (Posterior Correction)**: [一切实验绝不能丢弃 N 维独立性！绝对禁止无脑 MLP 平铺！]

2. **[被证伪假设 2 - 待捕获]**: ...
