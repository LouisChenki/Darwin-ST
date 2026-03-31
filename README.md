# AutoResearch: Spatio-Temporal Edition


这是一个旨在让 AI 智能体自主进行时空神经网络（Spatio-Temporal Neural Networks，例如交通流预测）自动深度学习研究的实验性项目。

核心思想：给予 AI 智能体一个针对时空序列预测的训练框架，让它彻夜自主实验。智能体会自动修改 `train.py` 中的网络架构、引入新的机制（如液态神经网络 LNN、因果时空注意力等），在固定的 **15分钟时间预算** 内进行模型训练，对比验证集表现，保留提升的实验并丢弃失败的尝试，不断循环。这个过程将使得时空神经网络自动进化，自动突破现有的经典基线（Baseline）模型。

## 核心特性 (Key Features)

- **时空交通流基准对齐**: 内置对标四大主流数据集（PeMS04, PeMS08, METR-LA, PEMS-BAY）和四种经典模型基准线 (ARIMA, DCRNN, ST-Transformer, UniST)。以突破靶场基线作为最高目标。
- **15分钟严格训练预算**: 每次架构迭代必须在 15 分钟内完成，确保所有创新的迭代都能在这种极不平衡的快速节奏中接受公平考验，并在最后输出绝对客观的 `val_mae` 指标反馈。
- **硬件级加速 (Apple M4 Max)**: 脚本级深度优化统一内存操作（Pin Memory）与底层显存调用，支持 Apple MPS 平台及常见 GPU 架构。在推理级采用混合精度预设方案预防梯度饥饿。
- **创新点保护区 (Innovation Protected Zone)**: 强制保护人类指定的最新架构机制，在此基础上扩展特征融合探索。随后的架构变异不改变基座逻辑，避免发生灾难性遗忘。

## 文件结构 (Project Structure)

项目主要由三个核心文件组成：

- **`prepare.py`** — 数据预处理及评估工具。负责下载时空交通流数据集 (npz 格式)、构建基于滑动窗口 (`seq_len=12`) 的训练数据，并提供客观唯一的 MAE/RMSE 官方度量手段。(不被 AI 智能体修改)
- **`train.py`** — 包含神经网络的主体架构、优化器定义和训练循环，智能体将主要在这个文件中不断迭代与魔改，这是唯一一个经过智能体无数自我变异打磨的竞技场。(由 AI 智能体修改)
- **`program.md`** — 智能体运行的全局系统提示词与进化准则 (System Prompt & Rulebook)。定义了严格的张量维度追踪与防御性编程纪律。(由人类研究员修改和配置)

辅助支持组件：
- **`baseline_registry.py`** — 基线指标测算库，记录 SOTA 模型的官方指标数据标尺。
- **`results.tsv`** & **`progress.png`** — 自动演化日志记录以及动态可视化的指标衰减进展图。

## 智能体演化能力展示 (Agent Evolution Capability Demonstration)

随着自动实验循环的推进，智能体会自主记录并绘制每一个被 `keep` 的变异架构的性能轨迹。如下方的 `progress.png` 所示，图表不仅直观展现了随着实验世代的收敛，预测绝对误差逐渐逼近最优的过程；同时它也将客观比对其与用户选定的顶级学术基准线（如 ARIMA, DCRNN 等），为您的自动科研提供一目了然的标尺：

![Agent Validation Progress](progress.png)

## 快速开始 (Quick Start)

**环境与硬件要求 (Requirements):** Python 3.10+, 推荐使用 [uv](https://docs.astral.sh/uv/)。测试环境最佳支持 Apple MPS 硬件加速架构，亦兼容普通独立 GPU。

```bash
# 1. 初始化并安装依赖库 (Install dependencies)
uv sync

# 2. 选择数据集，准备并在本地预处理数据 (Download & preprocess data)
# 数据集支持: PeMS04, PeMS08, METR-LA, PEMS-BAY
DATASET=PeMS04 uv run prepare.py

# 3. 手动进行一次基座训练验证 (~15分钟预算)
DATASET=PeMS04 uv run train.py
```

## 开启自主研究模式 (Autonomous Experimentation Mode)

### 工作流程图 (Agent Workflow)

以下展现了 AutoResearch 智能体从接受初始化指令到进入自主突变循环的完整工作流：

```mermaid
graph TD
    A[人类输入: 核心创新点、数据集、目标基线] -->|系统唤醒| B(Phase 1: 指令解析与环境对齐)
    B --> C[构建包含 '创新点保护区' 的初始 train.py]
    C --> D{人类确认识别与允许?}
    D -- 否 (修改提示词) --> A
    D -- 是 (开始放权) --> E[Phase 2: 进入自动化研究循环]

    subgraph 15分钟快速进化周期 (15-Min Evolution Loop)
        E --> F[智能体变异: 修改网络架构、通道数、融合机制]
        F --> G[基于 M4 Max / GPU 执行 15 分钟预算内训练]
        G --> H{客观验证: 验证集 MAE 是否下降?}
        H -- 成功 (Keep) --> I[保留代码, 记录 results.tsv, 绘制 progress.png]
        H -- 失败 (Discard) --> J[丢弃本次修改, 代码回滚至上一极值态]
        I --> F
        J --> F
    end
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style E fill:#bbf,stroke:#333,stroke-width:2px
    style I fill:#dfd,stroke:#333,stroke-width:2px
    style J fill:#fdd,stroke:#333,stroke-width:2px
```

加载您最习惯的代码 Agent 插件，如 Claude 智能体、Cursor，或任意支持本地文件读写的 AutoResearch 终端，在当前根目录下发起系统唤醒交互：

> "请阅读 `program.md` 文件中的指令模板，我想引入『液态时空神经网络』机制到 `PeMS04` 数据集上，对比基线设为 `DCRNN`，开始 Phase 1 的初始化！"

接下来智能体会自动执行：
1. 分析核心需求并基于 `train.py` 搭建出处于 "**创新点保护区**" 的初始探索基础架构。
2. 约束参数体量，确保其与目标基线（如 DCRNN）的网络复杂度对比处在同一合理的量级内。
3. 获取您的人为确认许可后，进入无限循环的超高速自动演进（Phase 2），记录实验 `commit` 日志，绘制验证损失对抗基线的 `progress.png` 轨迹。您只需在一夜之后验收架构的突变结果。

## 许可证 (License)

MIT License. 基于原始 `nanochat-autoresearch` 重构用于特定的重度专业向时空网络预测场景。
