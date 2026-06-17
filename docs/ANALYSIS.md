# Darwin-ST 项目深度审查与重构需求分析

> 版本: 2026-06-16 · 状态: 需求澄清阶段（待用户确认后转为实施蓝图）
> 本文档是对 Darwin-ST 当前实现的代码级审查 + 结合时空预测/NAS/HPO/GraphRAG 研究的需求梳理。

---

## 0. 一句话结论

Darwin-ST 的**愿景是对的**（让 Agent 24/7 自动优化时空预测模型直到超越 SOTA），但**当前实现是一个从 LLM 预训练项目（karpathy/autoresearch）机械改写的半成品**：保留了 LLM 项目的骨架（单文件 `train.py` + 固定时间预算 + keep/discard 随机爬山），却没有补上时空预测领域真正需要的基础设施（图结构、masked 指标、标准协议、结构化记忆、可用的知识库）。要达到"国际领先水平"，需要从**评测协议、优化算法、记忆系统、知识系统**四条线系统性重建。

---

## 1. 项目谱系：继承了什么，遗漏了什么

Darwin-ST 原型 = `karpathy/autoresearch`（2026-03，让 Agent 通宵自动优化一个 nanochat 量级的 LLM）。

| 维度 | 原始 AutoResearch (LLM) | Darwin-ST 当前 (时空) | 问题 |
|---|---|---|---|
| 优化对象 | `train.py` 单文件（GPT+Muon 优化器） | `train.py` 单文件（agent 从零写） | 继承单文件范式 ✅ |
| 评测指标 | `val_bpb`（bits-per-byte，vocab 无关） | MAE/RMSE | 指标换了，但**协议没跟上**（见 §2） |
| 时间预算 | 固定 5 分钟 | 固定 20 分钟 | 合理 ✅ |
| 优化策略 | keep/discard 随机爬山 | keep/discard 随机爬山 | **直接继承，未升级**（见 §3） |
| 记忆 | `results.tsv`（5 列）+ 分支推进 | `memory.md` + `results.tsv` + `evolution_log.jsonl` | **只在 SKILL.md 里描述，无模板/无代码**（见 §4） |
| 知识库 | 无 | Neo4j GraphRAG | **空壳，无建图数据**（见 §5） |
| 数据 | 自动下载+BPE | PeMS 下载 | **不处理图结构**（见 §2） |
| `analysis.ipynb` | 分析 val_bpb | **原封不动分析 val_bpb** | **LLM 残留，与时空无关** |

**核心判断**：单文件 + 固定预算 + 自主循环这套"骨架"值得保留；但 LLM 项目的"评测即一个标量、优化即随机爬山、记忆即一个 tsv"这套"内脏"完全不适配时空预测这种**有图结构、有标准协议、解空间高度结构化**的任务。

---

## 2. 致命问题 A：评测协议不符合学术标准 → 数字不可信、不可比

这是最优先要修的问题。如果评测本身错了，后面所有"超越 SOTA"的判断都是空中楼阁。研究核实（BasicTS / STAEformer / DCRNN 协议）后确认当前 `prepare.py` 有以下硬伤：

1. **没有 masked MAE**（`prepare.py:205`）。学术界标准：缺失/故障传感器读数编码为 0，计算 MAE/RMSE 时必须 `mask = (y != 0)` 并做 `mask /= mask.mean()` 重加权。当前实现把 0 当真值算进误差 → 数字与任何论文都不可比。
2. **split 比例错误且无测试集**（`prepare.py:40-41`）。当前 `TRAIN=0.6 / VAL=0.2`，剩下 0.2 既不切也不加载 → **没有 held-out 测试集**，且 PeMS04/08 学术标准是 6/2/2、METR-LA/PEMS-BAY 是 7/1/2，需按数据集分流。当前用验证集当测试集，有信息泄漏风险。
3. **不处理图结构**（整个 `prepare.py` 无 adjacency）。PeMS04/08 自带 `distance.csv`、METR-LA/PEMS-BAY 自带 `adj_mx.pkl`，绝大多数 ST-GNN（DCRNN/STGCN/ASTGCN/GraphWaveNet）**必须有邻接矩阵才能跑**。当前管线让 GNN 类创新点根本无法实现 → 与 SKILL.md "强制保留节点维度 N、引入 GCN/GAT" 的架构纪律直接矛盾。
4. **数据集 URL 一半是占位符**（`prepare.py:30-31`）。METR-LA/PEMS-BAY 是 `chenkaiqi/..._placeholder_url` 假地址，且这俩是 `.h5` 速度数据，与现有 `.npz` 流量加载逻辑不兼容。
5. **归一化口径需明确**。学术标准是**单一全局标量** z-score（仅用训练集统计量、仅对 flow/speed 通道），当前实现方向对，但缺少 per-node 选项与通道处理的显式说明。
6. **baseline_registry 数字与真实文献偏差大**。核实：registry 写 PeMS04 DCRNN MAE=24.1，真实 ≈19.6；registry UniST=21.5，真实 SOTA ≈17.8–18.2。**用错误的基线当"及格线"会导致 Agent 误判自己已超越 SOTA**。

> 研究要点：应直接对齐 **BasicTS**（github.com/GestaltCogTeam/BasicTS）的协议——它是社区公认的标准化基准，固定了 split、masked 指标、train-only scaler、inverse-transform-before-scoring。当前 SOTA 集群：PeMS04 ≈17.8–18.2、PeMS08 ≈13.4–13.5、METR-LA ≈2.9、PEMS-BAY ≈1.50。提升空间已 <1%、落在噪声内，**必须多 seed 平均 + 显著性检验**才能避免追逐噪声。

---

## 3. 致命问题 B：优化策略是"随机游走"，不是真正的 AutoML

用户明确要求："不能是简单的随机游走调优，而是应该结合 NAS、HPO 等领域的算法"。当前 SKILL.md 的 Phase 2 循环本质上是**(1+1) 随机变异爬山**：变异一次 → 跑一次 → 更好就 keep、更差就 discard。缺四样东西：代理模型(surrogate)、多保真早停(multi-fidelity)、结构化档案(archive)、多样性保持(diversity)。

研究结论给出的**分层方案**（关键洞察：Agent 的"编辑代码→跑一次→看指标→keep/discard"在数学上就是一个**离散黑盒进化搜索步**，不是梯度步、不是超网）：

```
外层循环（Agent 拥有 —— 架构级 NAS）
  ├─ 档案 Archive: MAP-Elites 风格网格 + 岛屿多样性
  │      行为描述子 = {图卷积类型, 时序模块类型, 深度, 参数量}
  ├─ 变异算子: Agent 作为"有语义的智能变异算子"
  │      (op-swap / edge-rewire / fusion 改动), 用 OPRO 风格把
  │      历史 (genotype, score) 升序排列喂进上下文
  ├─ 预筛 predictor: 用自己的试验档案训练一个廉价排序预测器,
  │      跳过预测会变差的变异(BRP-NAS 风格, 排序比回归省样本)
  └─ 对每个存活候选基因型:
        genotype → phenotype (编译成 train.py)
        │
        内层循环（调库 —— HPO）
          └─ Optuna TPE (+ Hyperband/ASHA pruner 早停)
             调 LR / hidden dims / dropout / batch
        │
        多保真门: 5/10/20 分钟检查点早杀明显的输家
        │
        观测最终指标 → 放回档案(打败 cell 内现任才留;
             aging 淘汰 / 岛屿重置保多样性) → 回灌 trajectory + predictor
```

**落地优先级**（投入产出比排序）：
- **最高 ROI**：(a) `train.py` 支持检查点 + 上报中间指标 → 解锁所有多保真早停；(b) 用 **MAP-Elites 档案 + aging evolution** 替换单一现任的 keep/discard。
- **分工铁律**：架构级搜索 Agent 自己做（它擅长提出"有语义"的变异）；架构内的 HPO **直接调 Optuna TPE**，别让 Agent 手推超参（库做得更好）。
- **不要碰**：DARTS（需要可微超网，与离散试验范式相反）、真 PBT（需要并行 + 权重检查点）。如需 DARTS 速度，把 AutoCTS/AutoSTG 当一次性工具调用，返回一个 genotype 再由 Agent 精修。

参考系统：OPRO（LLM 作优化器）、FunSearch/AlphaEvolve（LLM 进化代码 + 岛屿档案）、AIDE（解空间树搜索，MLE-bench 最强 scaffold）、Regularized Evolution（aging 进化）、AutoCTS++/AutoSTF（时空专用 NAS）。

---

## 4. 致命问题 C：Memory 模块只存在于"口头描述"

用户要求："构建一个 memory 模块，结构化记录过往优化的成功与失败经验，防止无效优化"。

现状：SKILL.md 反复引用 `memory.md`、`results.tsv`、`evolution_log.jsonl`、"The Graveyard"，但**仓库里没有任何模板、没有 schema、没有读写代码** —— 完全靠 Agent 每次即兴发挥。这导致：格式漂移、字段不一致、无法被程序消费、更无法做 §3 里的"预筛 predictor / 档案"。

研究给出的记忆模式（直接可借）：
- **Reflexion**：把失败的语言化反思存进情景记忆 → "我是不是已经试过这个了？"
- **ExpeL**：把跨试验经验蒸馏成"条件式自然语言洞察"，决策时检索最近邻历史轨迹（**最贴合本场景的模板**）。
- **Generative Agents**：记忆排序 = recency × importance × relevance（不是只看余弦相似度）。
- **MAP-Elites 档案**：行为网格里每格只留最优 → 天然就是"什么有效"的结构化记忆。

**需求**：把 memory 从"一个自由格式的 md"升级为**结构化的、程序可读写的试验数据库**（建议 JSONL/SQLite + 一个生成/查询脚本），字段至少含：基因型、超参、masked 指标(多 seed)、状态(keep/discard/crash)、失败死因、行为描述子、时间戳、所基于的父代。失败经验（Graveyard）要能被**硬性查询**以阻止重复提议已知坏配置。

---

## 5. 致命问题 D：知识图谱是"空壳"且检索方式落后

用户反馈："knowledge 这一块当前效果不佳"。审查确认问题比想象严重：

1. **没有任何建图/注入脚本**（`grep CREATE|MERGE|ingest` 全仓库无命中）。`graph_rag.py` 只有**查询**接口，没有**任何把文献写进 Neo4j 的代码**。也就是说图谱**要么是空的、要么是手动塞的少量数据** → 检索自然"效果不佳"。这是第一性问题。
2. **检索是子串匹配，不是语义检索**（`graph_rag.py:45,50` 的 `toLower(...) CONTAINS $kw`）。"over-smoothing" 匹配不到 "oversmoothing" 或 "节点表示坍缩"。代码里那句"句子过长无法定位概念节点"的报错就是这个问题在咬人。
3. **schema 丢掉了让建议"可执行"的关键字段**。`(Dataset)-(Symptom)-(Action)-(Concept)` 骨架方向对，但缺：Condition（在什么条件/数据集/horizon 下适用）、量化证据（Δmetric + 基线 + 数据集，而非自由文本 effectiveness）、provenance（出处/DOI）、defeasibility（"在条件 C 下失败"的负知识结构化）。
4. **`inspire` 是纯 `rand() LIMIT 10`**。无相关性门控 → 最大化"近似但不相关的干扰项"（Power of Noise 证明这种 top-ranked-but-irrelevant 干扰对准确率伤害最大），还会触发"lost in the middle"。
5. **文献知识与 Agent 自身经验完全割裂**。`memory.md`/Graveyard 在图谱之外，检索时不会一起召回。

研究给出的重构方向（**保留 Neo4j，不要推倒重来**——fix/evidence 查询确实是多跳的，图有价值）：
- **(a) 换掉子串前门 → 混合向量+图检索**：用 Neo4j 原生向量索引对 Symptom/Concept/Action 描述做 embedding，先向量相似找入口节点，再图遍历取"修复+证据+反模式"子图。这是最急、最便宜、收益最大的单点修复。
- **(b) 只 reify 一个节点 + 加条件**：把 `RESOLVES` 边升级成 `Resolution/Claim` 节点，带 condition / effect(结构化Δ) / evidence_grade / provenance / polarity（能表达失败）。别上 ORKG/nanopub 全套。
- **(c) 融合自身经验**：把 `results.tsv`/Graveyard 作为一等 `Experiment` 节点纳入检索，冲突时**自身结果压过文献**，已知失败配置**硬阻断**。
- 还需补：**建图 pipeline**（schema 约束 LLM 抽取 → 出处接地 → NLI 验证防幻觉边 → 去重规范化），否则图谱永远是空壳。

---

## 6. 致命问题 E：Agent 会"礼貌性暂停"问用户

用户要求："如果精度没有超过 SOTA，则需要一直优化下去，不能间断……之前的版本经常会暂停询问用户。"

SKILL.md 其实已有大量"封杀礼貌性停顿"的指令（§6.61-63、§7、§8），方向对。但**纯靠 prompt 纪律约束不可靠**——这是 LLM Agent 的已知通病（MLAgentBench 的 "Fact Check"、AutoResearch 的 "NEVER STOP" 都在和这个斗争）。需要从**机制层**加固：
- 用 harness 层的循环驱动（如本仓库 launcher 生成的 CLAUDE.md + 外部 `/loop`）而非只靠模型自觉；
- 把"是否达到停止条件(超越 SOTA)"做成**程序判定**（读 memory DB + baseline registry 比较），而非模型口头判断；
- 每轮结束**自动衔接**下一轮的动作要由脚本/流程兜底。

---

## 7. 其它代码级问题（较小但需清理）

- `analysis.ipynb`：整本是 LLM 项目残留（分析 `val_bpb`/`memory_gb`/5 列 tsv），与时空指标无关 → 重写或删除。
- `pyproject.toml`：(a) `graph_rag.py` 依赖 `neo4j` 但**未声明**；(b) 残留 LLM 依赖 `kernels`/`rustbpe`/`tiktoken`；(c) 描述仍是 "Autonomous pretraining research swarm"；(d) torch 锁 `pytorch-cu128`(CUDA) 但本机是 Mac/MPS，`uv sync` 装不了。
- 本机 `python3` 是 3.9.6，项目要求 ≥3.10（uv 管理的 venv 可解决）。

---

## 8. 重构路线图（建议分期，待用户确认范围）

> 下列为**建议**，具体做哪些、先后顺序需用户拍板。

**P0 — 可信评测地基（没有这个，一切优化都是假的）**
- 重写 `prepare.py`：masked MAE/RMSE、按数据集分流的 6/2/2 与 7/1/2、真正的 test 集、邻接矩阵加载（distance.csv → Gaussian kernel；adj_mx.pkl）、修复数据集 URL、多 seed。
- 校正 `baseline_registry.py` 为真实文献数字，并标注协议(masked/horizon)。

**P1 — 结构化 Memory 模块**
- 设计试验数据库 schema（JSONL/SQLite）+ 读写/查询脚本；落地 Graveyard 硬阻断与"最近邻经验检索"。

**P2 — 真正的 AutoML 优化循环**
- `train.py` 检查点 + 中间指标上报；接入 Optuna TPE + Hyperband pruner 做内层 HPO；外层用 genotype + MAP-Elites + aging evolution 替换随机爬山。

**P3 — 知识系统重构**
- 建图 pipeline（抽取→接地→验证→去重）；Neo4j 向量索引混合检索；reify Resolution 节点；融合自身经验；修复 `inspire` 相关性门控。

**P4 — 自主性机制加固 + 清理**
- 程序化停止判定与自动衔接；清理 LLM 残留（ipynb/依赖/描述/torch 后端）。

---

## 9. 待用户澄清的关键决策

见对话中的提问。核心几条：
1. 运行硬件（本机 Mac/MPS？还是有 NVIDIA GPU / 远程集群？）—— 决定 torch 后端、时间预算、能否并行(影响 ASHA/PBT)。
2. 重构范围与节奏（P0 先行，还是端到端蓝图一次性给全？）。
3. 知识图谱：是否有现成文献语料/已建好的图数据？还是需要从零搭建建图 pipeline？
4. 评测协议是否锁定对齐 BasicTS（强烈建议是）。
5. memory 落地形态偏好（JSONL 轻量 / SQLite 可查询）。
