# AutoResearch 海马体 (Bayesian Hippocampus)

## 🎯 领域先验 (Domain Priors)
- **目标数据集**: PeMS04
- **DCRNN 基线 MAE**: 24.1

## 🧬 核心基因 (High-Confidence Components)
- **Residual Liquid Cell**: [High Confidence] 核心稳定性。
- **Spatial Self-Attention**: [High Confidence] 优于所有尝试过的 GCN 变体。
- **Static Node Embedding**: [High Confidence] 简单即美。

## 🪦 墓地 (The Graveyard)
- **Global Learnable GCN**: [Failure] 优化极难。
- **Sparse Top-K GCN (Iteration 7)**: [Failure] 导致训练极度不稳定 (MAE 22.75)。
- **Stacked LNN**: [Weakness] 震荡严重。

## 📊 参数先验区间 (Parameter Ranges)
- **hidden_dim**: 64-96

## 🧠 演化日志 (Evolution Logic)
...
4. **Spatial Attn LNN**: 20.01 MAE. (Best so far)
7. **Sparse GCN LNN**: 22.75 MAE. (Discarded) - 确认了自学习拓扑的有害性。
