# AutoResearch: Darwin-ST

🤖 **Darwin-ST** 是一套专为大型语言模型 Agent（如 Geminicli、Claude 等）打造的“时空图神经网络 (Spatio-Temporal GNN) 自动化研究技能”。

基于本仓库提供的标准 Agent Skill，你的 AI 助手可以全天候 24/7 接管算力，自主完成网络的搭建、训练、验证，并通过贝叶斯后验机制不断进行反思与架构突变，以不断突破时空预测基线。

---

## 📂 仓库结构 (Repository Structure)

本仓库采用了 [Agent Skills 标准结构](https://github.com/anthropics/skills) 进行构建：

```bash
/
├── README.md               # 也就是您现在看到的这份人类阅读文档
├── darwin-st/              # ⬅️ 核心 Agent Skill 文件夹（纯净版）
│   ├── SKILL.md            # Agent 所阅读的核心系统指令与工作流控制
│   ├── scripts/            # 数据预处理与环境注册等 Python 挂载脚本
│   └── references/         # Agent 执行突变时的时空架构“心智模式”先验指导
├── pyproject.toml / uv.lock # Python 运行环境的依赖控制文件
└── reference/              # 私有研究参考资料（不参与 Skill 发布层）
```

---

## ⚡ 安装说明 (Installation)

**注意**：`darwin-st` 是一个 Agent 技能插件，而不是独立运行的普通 Python 包。你需要一个支持读取 Agent Skill 的运行环境（比如 [Geminicli](https://github.com/supratikpm/gemini-autoresearch)）。

### 安装技能到 Agent
您可以选择将本技能安装在特定工作区，或安装为您系统的全局技能。

1. **工作区本地安装（推荐）**
   克隆本项目后，直接将库中的 `darwin-st` 文件夹拖入您目标研究项目的 `.agents/skills/` 目录下即可：
   ```bash
   mkdir -p .agents/skills/
   cp -r darwin-st/ .agents/skills/
   ```

2. **全局安装**
   将其放入全局插件目录下，之后您在任何终端都能召唤它（以 Geminicli 为例）：
   ```bash
   cp -r darwin-st/ ~/.gemini/antigravity/skills/
   ```

### 准备 Python 环境
本技能中的 Agent 循环在运行时会挂载 `scripts/` 下的 Python 脚本。请在使用前确保环境配齐：
```bash
# 推荐使用 uv 同步依赖
uv sync
```

---

## 🚀 使用指南 (How to Use)

技能挂载成功后，直接通过自然语言与您的 Agent 对话来触发工作流。在聊天框中发送以下**触发词**：

> *"帮我启动 AutoResearch"*
> *"我要开始自动化研究，进行交通流预测实验"*
> *"帮我演化一个时空网络架构"*

### 自动化工作流展示
触发后，Agent 会严格接管下述流程：

1. **需求问答**: Agent 会向您发送一张《AutoResearch 时空研究任务配置单》，向您索要创新点描述、目标数据集 (PeMS04/08等)、对比基线等参数。
2. **物理隔离**: 自动新建 Git 分支剥离环境。
3. **闭环演化**: 根据 `references/architecture-rules.md` 中的理论指导构造 `train.py`。
4. **24/7 自主迭代**: 不断地通过 15 分钟级的训练来验证假设，好的代码会被保留 (KEEP) 到 `memory.md` 作为强基因，坏的代码会被记录并舍弃。直到您强行停止。
