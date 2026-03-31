# 🧠 AutoResearch 长期实验记忆库 (Hippocampus)

> **智能体使用指南 (Agent Directives):**
> 本文件是你跨越数百次时空预测实验的唯一长期记忆锚点。在每次突变 `train.py` 前，**必须**先行读取此文件汲取前车之鉴；在每次完成验证决定 KEEP 或 DISCARD 后，**必须**基于实验结晶动态更新此文件。依赖“增量反思”，避免原点踏步。

---

## 🧬 核心基因序列 (SOTA DNA)
*(请记录当前达到 Best-ever MAE 的核心架构流派与组装联结逻辑，方便下一局复刻与强化)*

- **当前最优架构 (Current SOTA Architecture)**: [初始态：TemporalEmbedding + DCRNN/LNN占位]
- **物理空间先验 (Spatial Information Flow)**: [初始态：直接采用 adj.npy 进行 GraphConvolution]
- **周期频率先验 (Temporal Periodicity)**: [初始态：接入确定性提取的 TOD / DOW 双频信号]
- **防御水位 (Best MAE to Beat)**: [待点亮...]

## 🎛️ 弹性调参笔记 (Elastic Parameter Insights)
*(请记录已经被证明对验证精度有明显影响、或是极度敏感的超参数稳定区间)*

- **内部隐藏表征维 (d_model/Hidden Dim)**: [待探索...]
- **学习率/混合精度调度 (LR / AMP Engine)**: [待探索...]
- **特殊拓扑算子参数 (e.g. Diffusion steps)**: [待探索...]

## ☠️ 实验禁区 (The Graveyard)
*(极其重要！记录那些直接导致 Loss=NaN、显存溢出 (OOM)、维度坍缩崩溃及精度大掉血的血泪尝试。)*

1. **[血泪教训 1 - 架构类]**: 
   - **错误推断**: [例如：意图粗暴通过 `[B, T_in, N, C] -> [B, N*C]` 送入全连接核]
   - **崩溃结症**: [严重违背 GeoAI 拓扑隔离常理，导致参数量爆表并使 MAE 退化至原始形态]
   - **纪律锚定**: [严守 Mandatory Spatial Dependency 纪律！维护 N 维度存续！]

2. **[血泪教训 2 - 待捕获]**: ...
