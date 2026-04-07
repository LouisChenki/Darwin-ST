# 🧬 时空架构演化法则 (ST-Architecture Mental Models & Evolution Rules)

这是你在时空图神经网络 (Spatio-Temporal GNN) 中进行网络突变 (Mutation) 与架构修改时**必须**遵循的核心领域直觉和纪律。在任何调整 `train.py` 之前，请深刻理解并执行以下守则。

## 1. 空间关系约束 (Spatial Topology Prior)
You are highly discouraged from violently flattening spatial nodes (e.g., `[B, T_in, N, C] -> [B, T_in, N*C]`). This completely obliterates the fundamental physical spatial topology of traffic nodes.

## 2. 强制进化方向 (Mandatory Evolution Direction)
In the iterative evolution loops, you MUST attempt to **preserve the node dimension `N`**. You should actively explore introducing Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Spatio-Temporal Synchronous Convolutions, or independent Per-Node Processing fusion mechanisms to enable the model to naturally comprehend spatial adjacency relationships!

## 3. 奥卡姆剃刀原则与计算折中 (Occam's Razor & Compute Trade-offs)
- 在验证误差 (`val_mae`) 相似或差异极小的情况下，你**必须**优先保留最简单、对计算资源最友好的代码分支。
- 如果你的训练触碰到了 20 分钟的硬性时间预算墙，**这绝不是停止 24/7 循环或中止任务的借口！** 相反，请执行优雅的算力折中。主动降低隐藏维度 (`d_model`)、减少注意力头数 (attention heads)、或者大胆削减 batch size，以确保重型空间操作（Heavy Spatial Operators）能在严苛的时间限制内运行完毕。

## 4. 严格张量追踪与防御 (Mandatory Dimension Tracking & Defense) - 【防挂死必读】
每个张量变换操作 (`view`, `reshape`, `permute`, `einsum`) **必须**在行尾带有严格的中文注释，动态展示维度的变化过程。
例如: `# 张量变换: [Batch, Nodes, Time, Features] -> [B, T, N, C]`。
这能有效避免 Shape 幻觉。在进行复杂的维度操作之前，**必须**积极利用 `assert` 语句来进行防御性编程，提前规避 Shape 不匹配的灾难。
- **NaN 熔断代码防御**: 神经网络前向/反向传播中极为容易因时空图卷积造成数值崩溃。你**必须**在 `train.py` 的主训练循环 (Training Loop) 中植入自动 `NaN` 监测机制（例如：`if torch.isnan(loss) or torch.isinf(loss):`）。
  一旦捕获到无穷大或 NaN，应当直接打印 `Epoch X: Loss=nan, Val MAE=inf` 的崩溃日志，并强制调用 `sys.exit(1)` 脱出脚本，**绝对不要让空跑毫无意义的 Epoch 占用时间**。系统外围外挂进程才能最快识别并开启下一次演化。

## 5. 专家级微调韧性 (Hierarchical Expert Tuning)
- 拒绝轻易全盘否定 (Resilience over Replacement): 如果一个逻辑上合理的 GeoAI 机制（比如深层 GCN 或时空注意力）最初导致了 MAE 的退化，**不要瞬间抛弃它并完全更换全新的架构。** 请像资深专家一样思考，执行分层调试 (Hierarchical Debugging)：
  - 原始梯度是否消失或爆炸了？
  - 是否需要注入 `LayerNorm`？
  - 是否应该添加残差连接 (Residual Connections) 来抵御过平滑现象 (Over-smoothing)？
  - 学习率是否过大导致特征发散？
- 在完全摧毁一艘战舰前，先试着去打磨和淬炼它的内部！

## 6. 记忆与知识的双轨制 (Memory vs Knowledge)
这是你作为超级 Agent 认知世界的两大思维基石，绝不允许混淆其物理边界：
- **情景记忆 (Memory - `memory.md`) = 本次实验个体的日记**：只记录**此时此刻**在当前数据集、当前这个特定代码分支上，哪些改动取得了微茫的进展，哪些改动触发了数值爆炸（The Graveyard）。这是真理的最后防线，你必须绝对捍卫 Memory 中的局部最优基因，不能因为外部知识说这不行就盲目全盘否决当前正在运行的有效代码。
- **语义知识 (Knowledge - `GraphRAG`) = 集体顶会与跨越周期的智慧**：当自身演化彻底失去灵感（如长时依赖迟迟无法捕获），或者你想引入一个极其庞大复杂的新数学理论时，使用命令行呼叫 `graph_rag.py` 索要前瞻性处方。
- **💥【创新铁律】跨域涌现 (Emergent Fusion)**：当 RAG 为你提供了一个所谓的 SOTA 代码公式或结构模版时，**严禁将其当作“圣旨”强行生硬拼接到 `train.py` 中并抛弃现有心血！** 
你必须发挥你的大模型创造力：只能提取 RAG 中的数学机制、特别是其预警的 `Anti-pattern` (防雷教训) 作为**调味灵感 (Ingredients)**。然后，将这股灵感与你 `memory.md` 中证明有效的基础模块，进行有机的、带有创造性的**化学融合 (Chemical Fusion)**。保持代码独特，拒绝教条化！
