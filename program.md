# AutoResearch: Spatio-Temporal Edition

This is an experiment to have the AI agent autonomously conduct Deep Learning research on Spatio-Temporal Neural Networks (like Traffic Flow Prediction).
The agent will iteratively modify `train.py`, run experiments for a specified time budget, keep improvements, and discard failures.

## 全局规范 (Global Constraints - MUST READ)

### 📐 编码纪律 (Coding Directives)
1. **中文双语注释 (Bilingual Comments)**: The agent must write ALL code comments primarily in Simplified Chinese. Professional deep learning terms should use Chinese accompanied by their English names (e.g. 损失函数 (Loss Function), 时空图卷积 (Spatio-Temporal Graph Convolution)).
2. **严格张量追踪与防御 (Mandatory Dimension Tracking & Defense)**: Every tensor manipulation (`view`, `reshape`, `permute`, `einsum`) MUST be strictly annotated with a Chinese comment showing the exact dimension changes dynamically, e.g., `# 张量变换: [Batch, Nodes, Time, Features] -> [B, T, N, C]` to avoid shape hallucinations. Furthermore, you must aggressively utilize `assert` statements BEFORE complex operations to preemptively defend against Shape Mismatches.
3. **奥卡姆剃刀原则 (Occam's Razor)**: In cases where the validation error (`val_mae`) is similar or only negligibly different, you MUST prioritize maintaining the simplest, most computationally efficient code branch.

### 🧬 时空架构演化法则 (ST-Architecture Evolution Rules)
这是你进行网络突变 (Mutation) 时必须遵循的领域直觉：
- **空间关系约束 (Spatial Topology Prior)**: You are highly discouraged from violently flattening spatial nodes (e.g., `[B, T_in, N, C] -> [B, T_in, N*C]`). This completely obliterates the fundamental physical spatial topology of traffic nodes.
- **强制进化方向 (Mandatory Evolution Direction)**: In the iterative evolution loops, you MUST attempt to **preserve the node dimension `N`**. You should actively explore introducing Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), Spatio-Temporal Synchronous Convolutions, or independent Per-Node Processing fusion mechanisms to enable the model to naturally comprehend spatial adjacency relationships!

## Phase 1: Setup & Initialization (交互式预启动)

To set up a new experiment, DO NOT start coding independently. You **MUST** first ask the user to fill out the following configuration template exactly as shown below:

```markdown
**【AutoResearch 时空研究任务配置单】**
请您提供以下信息以初始化实验：
- **核心创新点 (Innovation)**: [请描述你想引入的网络机制，例如：液态神经网络 / 因果时空注意力等]
- **目标数据集 (Dataset)**: [请选择一项：PeMS04 / PeMS08 / METR-LA / PEMS-BAY]
- **对比基线 (Baselines)**: [请选择一项或多项：ARIMA / DCRNN / ST-Transformer / UniST，或填“无”]
- **实验分支名 (Git Branch Name)**: [请提供一个简短的英文分支名，例如：exp/lnn-gcn-base，或填“自动生成”]
```

Wait carefully for the user's response. Once received, do the following:

1. **配置环境与锁定基线**: Read the dataset and baselines. Note them down in `train.py` structure so they automatically plot the corresponding boundaries from `baseline_registry.py`.
2. **构建基座代码 & 网络复杂度匹配 (Construct Initial `train.py` & Complexity Matching)**: Based on the innovation, write the initial `train.py` from scratch.
   - **🥶 冷启动隔离 (Cold Start Isolation)**: You are **ABSOLUTELY FORBIDDEN** from reading or referencing `memory.md` during this initialization phase! Your initial codebase must act as a sterile control group ("Tabula Rasa"), built 100% strictly relying on the user's provided core innovation and baseline. Do not pollute the initial model with historical architectural memories!
   - **CRITICAL CONSTRAINT**: You MUST ensure that the parameter count (Model Complexity) of the architecture you construct is roughly in the same order of magnitude as the user's selected Baseline! (Do not build a 50-million parameter monster when comparing to a lightweight DCRNN framework). Ensure fair architectural groundings.
   - You MUST wrap the user's core innovation inside the "【创新点保护区 (Innovation Protected Zone)】". During Phase 2, you are FORBIDDEN from deleting this protected module.
3. **环境数据初始化**: Ensure to tell the user that the background data downloading logic (`prepare.py`) will automatically take over based on their dataset choice by parsing an environment variable, e.g. `DATASET=PeMS08 python3 prepare.py` (assuming you've edited prepare to support it).
4. **强制物理隔离与创建版本库 (Enforce Git Isolation)**: As your VERY FIRST terminal action, execute `git checkout -b <branch_name>`.
   - If the user wrote "自动生成" (auto-generate), engineer a highly readable English branch name based on the core innovation.
   - You MUST ensure the terminal successfully switches to this new branch. ONLY after confirming Git branch creation/switching is successful, initialize `results.tsv`.
5. **构建先验定锚 (Establish Domain Priors)**: Before transitioning, you MUST execute a quick Python script to query `baseline_registry.py` for all baseline MAEs of the target dataset. Write these baseline metrics at the very top of `memory.md` under **"🎯 领域先验 (Domain Priors)"**. Explicitly declare that any architecture producing an MAE disastrously outside this academic range is merely an "未达标的实验性信念 (Unqualified Experimental Belief)" and is absolutely forbidden from entering the SOTA DNA.
6. **🚀 无缝衔接全自动炼丹 (Seamless Autonomous Transition)**: Once the initial code, environment, and Bayesian memories are set up, **DO NOT STOP to ask for human permission!** You must IMMEDIATELY and AUTONOMOUSLY transition into Phase 2's experimentation loop. The human has completely handed over the control of the research lab to you. Operate 24/7 unattended!

## Phase 2: The Autonomous Experimentation Loop

**🧠 贝叶斯核心工作流：读写海马体 (Bayesian Read-Write Hippocampus)**
1. **先验检索 (Read Before Mutation)**: **Starting ONLY from the first evolutionary mutation of Phase 2**, before EVERY single subsequent modification on `train.py`, you MUST read `memory.md`. Comprehend the `High-Confidence Components` (SOTA DNA) and entirely dodge the fatal hypotheses located in `The Graveyard`. (Reminder: Do NOT read `memory.md` during Phase 1 Initialization).
   - **连续演化论 (Continuous Evolution Comments)**: When modifying `train.py`, your code comments MUST explicitly state the Bayesian belief update directing your hypothesis. (e.g., `# 假设更新: 基于先验空间聚集性，增加对 GCN 核的信心 (Belief Update: Increased confidence in GCN kernel based on spatial prior)`).
2. **执行实验 (Execute Experiment)**: Launch the script via `python3 train.py`. The training script MUST run and cleanly terminate within a **fixed time budget of 15 minutes** (wall clock training time).
3. **贝叶斯后验反思 (Bayesian Posterior Update)**:
   - **似然性评估 (Likelihood Check)**: Before blindly keeping a model, contrast its `val_mae` against the "🎯 Domain Priors" in `memory.md`. If the error is vastly inferior to the worst baseline, it is a failed hypotheses. Record the failure etiology in `The Graveyard` and DO NOT pollute the core DNA.
   - **后验更新 (Posterior Update - KEEP)**: If `val_mae` demonstrates a statistically significant improvement and securely hits the domain prior benchmarks, you successfully reinforced a scientific belief! Append the structured reflection to `evolution_log.jsonl` AND proactively edit `memory.md` -> updating `核心基因 (High-Confidence Components)` and `参数先验区间 (Parameter Ranges)` with your conquering mechanisms.
   - **拒绝硬锁死 (Soft Preservation)**: You possess the utmost freedom to modify ANY component in `train.py`. However, if you rip out a module currently anchored as a High-Confidence constituent, you must physically and mathematically justify this "belief shift" in your reflection logs.

**The Goal: Penetrate the Baselines (超越靶场基线) !**
Your prime objective is not just to minimize `val_mae`. It is to push `val_mae` below the horizontal dashed line of the toughest User-Selected Baseline. You must evaluate **"Relative Improvement"** over the baseline.

## Output Format

Once the `train.py` script finishes, it must print a summary:
```
---
val_mae:          15.4200
val_rmse:         22.1400
training_seconds: 900.1
total_seconds:    905.9
peak_vram_mb:     4506.2
num_steps:        1500
num_params_M:     0.95
```

## Logging and Plotting

After every single run, log it to `results.tsv` (tab-separated):
```
commit	val_mae	val_rmse	peak_vram_mb	status	description
```

**MANDATORY OUTPUT**: You MUST write code in `train.py` or a helper script that, at the end of every successful loop, generates a data visualization plot `progress.png`. 
**Horizontal Comparators**: This plot MUST read from `baseline_registry.py` and draw perfectly horizontal dashed lines representing the `val_mae` limits of the user-selected Baselines on the plot background. Watch your actual model curve trace down and aggressively intersect with these fixed literature-bound targets!
