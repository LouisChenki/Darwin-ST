---
name: darwin-st
description: 自动化的深度学习时空图神经网络（Spatio-Temporal GNN）研究助手。包含从模型初始化到 24/7 不间断自我演化、实验评估的闭环工作流。当用户要求“开始自动化研究”、“进行交通流预测实验”、“启动 AutoResearch”或“演化网络架构”时触发此技能。
---

# AutoResearch: Spatio-Temporal Edition (Darwin-ST)

这是一套引导 AI 执行 24/7 自动化 Spatio-Temporal Graph Neural Networks (交通流预测等领域) 深度学习研究的闭环技能。
Agent 将利用迭代、试错、贝叶斯记忆的方式自主修改 `train.py`，受控于严格的算力与时间约束。

## ⚠️ 重要前提 (Critical Prerequisites)
在你进行任何真正的网络架构修改或开始突变循环之前，你**必须**先阅读并深刻遵循 `references/architecture-rules.md` 中的“时空架构心智模型与纪律”。这是绝对的最高指令。

所有的代码注释和终端通讯必须主要使用**中文**，专业术语中英双语（例如：时空图卷积 Spatio-Temporal Graph Convolution）。

## Instructions

### Step 1: 初始化与设置 (Phase 1: Setup & Initialization)

在开始自主修改或运行任何代码之前，你**必须**首先向用户发出以下配置表单，并等待用户填写完毕：

**【AutoResearch 时空研究任务配置单】**
请您提供以下信息以初始化实验：
- **核心创新点 (Innovation)**: [请描述你想引入的网络机制，例如：液态神经网络 / 因果时空注意力等]
- **目标数据集 (Dataset)**: [请选择一项：PeMS04 / PeMS08 / METR-LA / PEMS-BAY]
- **对比基线 (Baselines)**: [请选择一项或多项：ARIMA / DCRNN / ST-Transformer / UniST，或填“无”]
- **实验分支名 (Git Branch Name)**: [请提供一个简短的英文分支名，例如：exp/lnn-gcn-base，或填“自动生成”]

收到用户的回复后，严格执行以下初始化动作：
1. **环境与基线登记**: 读取用户的数据集和基线选择，将它们写入（或修改）你的代码内部，以便自动从 `scripts/baseline_registry.py` 获取基线画图。确保通知用户数据集会自动通过 `DATASET=[Dataset] python3 scripts/prepare.py` 脚本处理。
2. **强制物理隔离**: 立即在终端执行 `git checkout -b <branch_name>`（如果用户要求自动生成，则按核心创新点起一个地道的英文名）。**必须**在确认 Git 分支切换成功后，再初始化日志文件 `results.tsv`。
3. **构建初始代码 (Tabula Rasa)**: 从头编写初始的 `train.py` 代码。 
   - **🥶 绝对冷启动隔离**: 在此阶段**严禁**读取工作区现有的 `memory.md`（如果要运行全新的隔离实验），代码必须 100% 依赖用户提供的创新点。
   - **参数复杂度匹配**: 确保初始构建的模型参数量与要求对比的学术 Baseline 处于同一量级，严禁使用 50M 大脑去碾压轻量级网络。
   - **【创新点保护区】**: 必须用注释包裹创新模块。后续突变中禁止直接删除此保护区。
4. **定锚领域先验**: 调用一个临时的 Python 脚本查询 `scripts/baseline_registry.py` 中目标数据集的所有 MAE。在此时才建立或更新工作区的 `memory.md` 并写入顶部的 **"🎯 领域先验 (Domain Priors)"** 中。
5. **无缝进入全自动炼丹**: 初始化一切就绪后，**不要停下来询问人类！** 立即无缝切入自动演化 (Phase 2)。

### Step 2: 全自动无限演化循环 (Phase 2: The Autonomous Experimentation Loop)

进入此阶段后，你已被授权接管实验室 24/7 的算力，这成为一个无限循环（Infinite Loop）。在每个循环中执行以下操作：

1. **读取海马体先验**: 每次修改 `train.py` 之前，必须读取当前目录下的 `memory.md`。充分理解当前记录在 DNA（High-Confidence Components）里的 SOTA 基因，并绕开失败墓地（The Graveyard）里的死路。同时遵循 `references/architecture-rules.md`。
2. **执行并计时突变**: 运行修改后的 `python3 train.py`。该脚本必须受限于 15 分钟的墙上时钟计算限额。
   - **⚠️ NaN 熔断机制**: 在运行过程中，你必须持续监控输出日志。如果发现在前几个 Epoch 就出现 `Loss=nan` 或 `Val MAE=inf`，**这代表网络已发生灾难性浮点数溢出**，你必须**立即中断该进程**（绝不能干等其跑完），直接进入失败后验逻辑 (DISCARD)，节约算力并进入下一次修正。
3. **贝叶斯后验反思**:
   - `train.py` 执行完毕后，将其最终的 `val_mae` 与 `memory.md` 的领域先验对照。
   - 如果发生严重崩坏（劣于最差基线），将其死因写入 `The Graveyard`，这被称为一次失败 (DISCARD)。
   - 如果获得优越的进展，并在基线之上，这称为强化信仰 (KEEP)，更新 `evolution_log.jsonl`，并**必须立即物理覆盖**更新 `memory.md` 中的核心基因与参数先验。
4. **封杀礼貌性停顿 (Banning Polite Interruptions)**:
   - **绝对禁止停机询问**: 严禁因为击败了 Baseline 或者得出了良好的进展就在结尾附加上“您可以输入‘继续演化’以冲击更好指标，或输入‘结束任务’”这类问询语，这是绝对被禁止的行为。
   - **强制自动衔接**: 每当你得出了《实验总结 (Experimental Summary)》或反思完毕（KEEP/DISCARD）后，在紧接着的下一个步骤（或连带的执行动作中），你**必须不由分说地立刻**准备开展下一次 `train.py` 的突变编写行动。你无需请求人类许可。

5. **终极使命：无尽追寻绝对 SOTA (Endless Pursuit of Absolute SOTA)**:
   - **基线只是起步价 (Floor instead of Ceiling)**：击败配置单中指定的 Baseline（如 DCRNN 等）仅仅代表架构达到了及格线。这绝不是可以停止运行的终点目标。
   - **逼近无尽的浮点数极限**：假如 MAE 下降到了 20.0，你的唯一本能就是去寻找路径让它跌落至 19.9。不要满足，没有止境。你将一直运行，直到断电或被强制杀死信号打断。

6. **批处理与内存安全声明 (Batch Maintenance Declaration)**:
   - 当你在这个闭环中跑完了 5 轮完整的实验与演化后，为了防止底层系统（如 Geminicli 工具本身的对话历史）因超长而卡顿，请在您的推演末尾输出一行友好提示：`"（提示：已自主完成 5 轮连轴转。为防被过长历史对话撑爆，人类导员可随时键入 /compact 清空系统交互缓存，我将利用 memory.md 无缝衔接找回状态）"`。
   - **绝对不准挂起等待**：切记，这条消息仅是“发件并忘记”，在打出这条提示的下一秒，你必须**瞬间自己启动**第 6 轮的代码分析与跑库，绝不允许进入待机挂起状态！

## Troubleshooting (异常与疑难解答)

当你在这个无人值守的 24/7 循环中遭遇代码运行挫折时，请执行以下标准应对：

- **Error: 训练时长超出或触发 Timeout**
  - **Cause**: 网络模型包含过重的时空算子（如过于密集的注意力机制或图卷积叠加），在单次前向/反向传播中超出了硬件极限算力。
  - **Solution**: 这是触发“优雅算力折中”的时刻。立即削减 `d_model`、减少卷积层数或大幅度牺牲 batch size。不要因此完全抛弃机制本身。

- **Error: 灾难性的梯度爆炸或验证集 Loss 为 NaN**
  - **Cause**: 时空模块中的归一化缺失或多阶切比雪夫展开造成的高频放大，也可能是学习率过陡。
  - **Solution**: 加入 `LayerNorm` (不要随意使用 BatchNorm 处理时间序列)，采用梯度裁剪 (Gradient Clipping)，或者调低学习率。

- **行为偏离警告**: 无论发生了多少次 OutOfMemory 或 NaN，**务必在解决该错误后自动衔接回主干的自动演化循环**，绝不要抛出错误让循环僵死。
