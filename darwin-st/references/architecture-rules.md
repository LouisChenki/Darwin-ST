# 🧬 时空架构演化法则 (ST-Architecture Mental Models & Evolution Rules)

这是你在时空图神经网络 (Spatio-Temporal GNN) 中进行网络突变 (Mutation) 与架构修改时**必须**遵循的核心领域直觉和纪律。在任何调整 `train.py` 之前，请深刻理解并执行以下守则。

## 1. 空间关系约束 (Spatial Topology Prior)
You are highly discouraged from violently flattening spatial nodes (e.g., `[B, T_in, N, C] -> [B, T_in, N*C]`). This completely obliterates the fundamental physical spatial topology of traffic nodes.

## 2. 强制进化方向 (Mandatory Evolution Direction)
In the iterative evolution loops, you MUST attempt to **preserve the node dimension `N`**. You should actively explore introducing Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Spatio-Temporal Synchronous Convolutions, or independent Per-Node Processing fusion mechanisms to enable the model to naturally comprehend spatial adjacency relationships!

## 3. 奥卡姆剃刀原则与计算折中 (Occam's Razor & Compute Trade-offs)
- 在验证误差 (`val_mae`) 相似或差异极小的情况下，你**必须**优先保留最简单、对计算资源最友好的代码分支。
- 如果你的训练触碰到了 15 分钟的硬性时间预算墙，**这绝不是停止 24/7 循环或中止任务的借口！** 相反，请执行优雅的算力折中。主动降低隐藏维度 (`d_model`)、减少注意力头数 (attention heads)、或者大胆削减 batch size，以确保重型空间操作（Heavy Spatial Operators）能在严苛的时间限制内运行完毕。

## 4. 严格张量追踪与防御 (Mandatory Dimension Tracking & Defense)
每个张量变换操作 (`view`, `reshape`, `permute`, `einsum`) **必须**在行尾带有严格的中文注释，动态展示维度的变化过程。
例如: `# 张量变换: [Batch, Nodes, Time, Features] -> [B, T, N, C]`。
这能有效避免 Shape 幻觉。在进行复杂的维度操作之前，**必须**积极利用 `assert` 语句来进行防御性编程，提前规避 Shape 不匹配的灾难。

## 5. 专家级微调韧性 (Hierarchical Expert Tuning)
- 拒绝轻易全盘否定 (Resilience over Replacement): 如果一个逻辑上合理的 GeoAI 机制（比如深层 GCN 或时空注意力）最初导致了 MAE 的退化，**不要瞬间抛弃它并完全更换全新的架构。** 请像资深专家一样思考，执行分层调试 (Hierarchical Debugging)：
  - 原始梯度是否消失或爆炸了？
  - 是否需要注入 `LayerNorm`？
  - 是否应该添加残差连接 (Residual Connections) 来抵御过平滑现象 (Over-smoothing)？
  - 学习率是否过大导致特征发散？
- 在完全摧毁一艘战舰前，先试着去打磨和淬炼它的内部！
