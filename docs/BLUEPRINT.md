# Darwin-ST 端到端重构蓝图 (BLUEPRINT)

> 版本: 2026-06-16 · 状态: 待用户审阅 → 审定后转为实施
> 前置文档: `docs/ANALYSIS.md`（问题诊断）
> 关键决策（已确认）: **多卡/集群可并行** · **端到端蓝图先行** · **知识图谱从零搭建** · **升级为 `src/darwin_st/` 包结构** · **分布式调度用 Ray Tune** · **数据集+创新点为每次运行的可配置输入（数据集给常用选项，创新点自由文本）**

---

## 0. 设计哲学：双层自治 (Two-Tier Autonomy)

这是整个重构的核心思想，用来同时满足你的两个看似矛盾的要求——"不要随机游走、要用 NAS/HPO 算法" vs "最大化发挥 Agent 能力"。

```
┌─────────────────────────────────────────────────────────────────┐
│  Tier 2 · 创造层 (Agent / LLM 主导, 低频, 突破性)                 │
│  ─ 把"创新点"实现为一等算子 (如 液态神经元 Liquid Cell)          │
│  ─ 查知识图谱 + 读记忆, 当搜索停滞时注入全新机制/重定向搜索空间   │
│  ─ 诊断系统性失败 (过平滑/梯度爆炸), 写新的算子代码               │
│       │ 注入算子 / 扩展搜索空间 / 重定向                          │
│       ▼                                                           │
│  Tier 1 · 系统层 (脚本/算法主导, 高频, 7×24 不停)                │
│  ─ 进化式 NAS: 在 genotype 搜索空间上做 aging evolution + 锦标赛 │
│  ─ Optuna TPE 做架构内 HPO + ASHA 多保真早停                     │
│  ─ 跨集群并行派发 trial, MAP-Elites 档案保多样性                 │
│  ─ 程序化判定"是否超越 SOTA" → 决定是否停止                       │
└─────────────────────────────────────────────────────────────────┘
```

**为什么这样分**（研究结论支撑）：
- Agent 的"编辑代码→跑一次→看指标→keep/discard"在数学上**就是一个离散黑盒进化搜索步**。让 Agent 用自然语言去手推超参/盲目变异，正是"随机游走"的根源。
- 真正高效的做法：**系统化的搜索交给成熟算法**（Optuna/aging evolution/ASHA，它们有 surrogate、多保真、档案、多样性），**Agent 只做它独有优势的事**——理解"创新点"的语义、写出新颖算子代码、在搜索停滞时做有理论依据的跨界重定向。
- "创新点保护区"（原 SKILL.md 的正确直觉）在新架构里升级为 **genotype 里的一个受保护算子节点**：NAS 围绕它探索最佳接线，HPO 调它的超参，但进化算子禁止删除它。

---

## 1. 目标系统架构（文件树）

```
Darwin-ST/
├── darwin-st/                       # Agent Skill 发布层
│   ├── SKILL.md                     # 重写: 双层自治工作流 + 程序化停止
│   └── references/
│       ├── architecture-rules.md    # 保留+扩充: 时空架构纪律
│       ├── search-space.md          # 新: genotype 算子词典(给 Agent 读)
│       └── optimization-playbook.md # 新: NAS/HPO 策略手册(给 Agent 读)
│
├── src/darwin_st/                   # 核心 Python 包(可被 import/测试)
│   ├── data/
│   │   ├── prepare.py               # 重写: 标准协议数据管道
│   │   ├── adjacency.py             # 新: 邻接矩阵构建(Gaussian/adj_mx)
│   │   ├── protocol.py              # 新: 按数据集分流的 split/指标 profile
│   │   └── metrics.py               # 新: masked MAE/RMSE/MAPE(唯一标尺)
│   │
│   ├── search/
│   │   ├── genotype.py              # 新: 架构基因型 (DAG of S-ops/T-ops)
│   │   ├── operators.py             # 新: 算子库(GCN/GAT/TCN/Attn/...+创新点)
│   │   ├── builder.py               # 新: genotype → nn.Module 编译器
│   │   └── space.py                 # 新: 搜索空间定义 + 变异算子
│   │
│   ├── optim/
│   │   ├── orchestrator.py          # 新: 外层进化 NAS + 跨集群派发
│   │   ├── evolution.py             # 新: aging evolution + 锦标赛选择
│   │   ├── archive.py               # 新: MAP-Elites 档案 + 岛屿多样性
│   │   ├── hpo.py                   # 新: Optuna TPE 内层 HPO
│   │   ├── scheduler.py             # 新: ASHA 多保真早停 + 分布式调度
│   │   └── predictor.py             # 新: genotype 排序预测器(预筛)
│   │
│   ├── train/
│   │   ├── train.py                 # 重写: 参数化(吃 genotype+HP), 出检查点+中间指标
│   │   └── trial.py                 # 新: 单次 trial 的隔离执行单元
│   │
│   ├── memory/
│   │   ├── store.py                 # 新: 试验数据库读写(SQLite)
│   │   ├── schema.sql               # 新: 试验/谱系/Graveyard 表结构
│   │   └── reflect.py               # 新: ExpeL 式经验蒸馏 + 最近邻检索
│   │
│   └── knowledge/
│       ├── ingest/
│       │   ├── fetch.py             # 新: 文献获取(arXiv/PwC/Semantic Scholar)
│       │   ├── extract.py           # 新: schema 约束 LLM 抽取三元组
│       │   ├── ground.py            # 新: 出处接地 + NLI 防幻觉边
│       │   └── canonicalize.py      # 新: 概念去重规范化(对齐 CSO)
│       ├── schema.cypher            # 新: Neo4j 图 schema(reified Claim)
│       ├── build_graph.py           # 新: 注入图谱(当前完全缺失!)
│       ├── embed.py                 # 新: 节点 embedding + 向量索引
│       └── retrieve.py              # 重写: 混合向量+图检索(替换子串匹配)
│
├── configs/
│   ├── datasets/{pems04,pems08,metr-la,pems-bay}.yaml  # 协议 profile(常用选项)
│   └── experiment.yaml              # 运行时输入: 创新点(自由文本)/数据集(选项)/SOTA目标/分支/预算
│
├── tests/                           # 新: 协议正确性的回归测试(关键!)
│   ├── test_metrics.py              # masked 指标对拍已知实现
│   ├── test_protocol.py            # split/scaler/无泄漏 断言
│   └── test_builder.py             # genotype→model 形状/可微 断言
│
├── docs/
│   ├── ANALYSIS.md                  # 已有: 问题诊断
│   └── BLUEPRINT.md                 # 本文档
│
├── pyproject.toml                   # 修: 加 neo4j/optuna/..., 清 LLM 残留
└── README.md                        # 修: 对齐新架构
```

> 注：从"单文件 `train.py` 人改"升级为"包化 + Agent 在 Tier 2 改算子/搜索空间"。原型的"单文件"哲学保留在精神上（diff 可审、scope 可控），但落到 **算子级**——Agent 编辑 `operators.py` 里的一个新算子，而非每轮重写整个 train.py。

---

## 2. P0 — 可信评测地基

> 没有这个，"超越 SOTA" 无从谈起。这是**第一优先**，且要有测试护栏。

### 2.1 `data/metrics.py` — 唯一标尺
```python
def masked_mae(preds, labels, null_val=0.0): ...   # mask=(y!=null); mask/=mask.mean()
def masked_rmse(preds, labels, null_val=0.0): ...
def masked_mape(preds, labels, null_val=0.0): ...   # 二次掩码防除零
```
对拍 Graph WaveNet / DCRNN / BasicTS 的 `util.py` 实现，写进 `tests/test_metrics.py`。

### 2.2 `data/protocol.py` — 按数据集分流
| 数据集 | 节点 | split | 报告口径 | 图来源 |
|---|---|---|---|---|
| PeMS04 | 307 | **6/2/2** | 12 步平均 MAE | distance.csv→Gaussian |
| PeMS08 | 170 | **6/2/2** | 12 步平均 MAE | distance.csv→Gaussian |
| METR-LA | 207 | **7/1/2** | 逐 horizon @3/6/12 | adj_mx.pkl |
| PEMS-BAY | 325 | **7/1/2** | 逐 horizon @3/6/12 | adj_mx.pkl |

- **真测试集**：三切分都生成并加载（当前缺 test）。
- **scaler 仅训练集统计量**、仅 flow/speed 通道、**评测前 inverse-transform**。
- **多 seed**（≥3）平均 + 标准差，避免追噪声（SOTA 提升已 <1%）。

### 2.3 `data/adjacency.py` — 图结构（当前完全缺失）
- PeMS04/08: 读 `distance.csv` → 阈值高斯核 `W_ij=exp(-d²/σ²)`, `σ=距离标准差`, 阈值 κ=0.1。
- METR-LA/PEMS-BAY: 直接加载 `adj_mx.pkl` 的稠密 N×N。
- 支持对称归一化 `D^{-1/2}ÃD^{-1/2}` 与随机游走 `D^{-1}A` 两种（作为搜索维度）。
- 修复占位符 URL：用 DCRNN 官方源（liyaguang/DCRNN）。

### 2.4 `data/baseline_registry` 校正
用真实文献数字替换（PeMS04 SOTA≈17.8–18.2、PeMS08≈13.4–13.5、METR-LA≈2.9、PEMS-BAY≈1.50），每条标注协议（masked? 平均/逐horizon）。**对齐 BasicTS**。

**验收**：`pytest tests/` 全绿；能复现 STID 量级的 sanity 数字（PeMS04 ~18.x）。

---

## 3. P1 — 结构化记忆模块

> 把 memory 从"自由格式 md"升级为"程序可读写的试验数据库"。借鉴 ExpeL/Reflexion/Generative Agents。

### 3.1 `memory/schema.sql`（SQLite，可查询）
```
experiments(
  id, parent_id, branch, genotype_json, hp_json,
  val_mae_mean, val_mae_std, test_mae, rmse, mape,   -- 多 seed
  status,              -- KEEP / DISCARD / CRASH / PRUNED
  behavior_descriptor, -- {graph_conv, temporal, depth, params} → MAP-Elites cell
  fail_reason,         -- Graveyard: nan/oom/timeout/regression
  wall_seconds, num_params, peak_mem_gb,
  created_at, commit_hash
)
lineage(child_id, parent_id, mutation_op, mutation_detail)
insights(id, condition, insight_text, evidence_exp_ids, confidence)  -- ExpeL 蒸馏
```

### 3.2 `memory/store.py` 接口
```python
record_trial(trial) -> id
best_so_far(metric, dataset) -> Experiment        # 程序化 SOTA 判定用
query_graveyard(genotype_signature) -> bool       # 硬阻断已知坏配置
nearest_experiments(genotype, k) -> [Experiment]   # 最近邻经验检索
distill_insights() -> [Insight]                    # ExpeL: 跨试验蒸馏条件式洞察
```

**关键功能**：
- **Graveyard 硬阻断**：变异前查签名，命中已知失败 → 跳过，不浪费 20 分钟。
- **最近邻检索**：决策时召回相似历史试验，排序 = recency × importance × similarity。
- 失败/成功都结构化，供 P2 的 predictor 和 archive 消费。

---

## 4. P2 — 真正的 AutoML 优化循环（适配并行集群）

> 核心：用 **进化式 NAS（外层）+ Optuna TPE（内层）+ ASHA（多保真）+ MAP-Elites（档案）** 替换随机爬山。多卡 → 并行种群 + 异步调度。

### 4.1 `search/genotype.py` — 架构基因型
离散、可变异、可编译的编码（phenotype=真实网络）：
```python
Genotype = {
  "nodes": [...],                  # DAG 潜在特征节点
  "edges": [(src,dst,op_code)],    # 每条边一个算子
  "s_ops": [...],                  # 空间算子序列(GCN/GAT/Diffusion/Adaptive/...)
  "t_ops": [...],                  # 时序算子序列(TCN/GRU/Attn/...)
  "fusion": "factorized|joint|attn",
  "depth": int, "hidden": int, "adj_mode": "sym|rw|adaptive",
  "protected": ["liquid_cell"],    # 创新点保护区, 变异禁删
}
```

### 4.2 `search/operators.py` — 算子库
内置标准算子（ChebGCN, GAT, DiffusionConv, AdaptiveAdj, DilatedTCN, GRU, TemporalAttn, STID-embedding…）+ **Agent 在 Tier 2 写入的创新算子**（如 `liquid_cell`）。每个算子是一个 `nn.Module` 工厂。

### 4.3 `optim/evolution.py` — aging evolution
- 维护种群 P（带 accuracy + age）。
- 每代：锦标赛选择父代 → **Agent 或规则变异算子**（op-swap/edge-rewire/depth）→ 训练评估子代 → **淘汰最老**（不是最差，aging 抗噪声）。
- 变异算子两路：规则变异（系统层，高频）+ Agent 语义变异（创造层，停滞时）。

### 4.4 `optim/archive.py` — MAP-Elites + 岛屿
- 行为网格 = {图卷积族 × 时序族 × 深度档 × 参数量档}，每格留最优。
- 保多样性：避免坍缩到单一局部最优；停滞时岛屿重置/重播种（FunSearch 式）。
- archive 即"什么有效"的结构化记忆，与 P1 memory 打通。

### 4.5 `optim/hpo.py` — Optuna 内层
- 对每个 genotype，`train.py` 内调 Optuna **TPE sampler** 调 LR/hidden/dropout/batch/dilation。
- 挂 **HyperbandPruner / SuccessiveHalvingPruner**（=ASHA）早杀弱配置。

### 4.6 `optim/scheduler.py` — ASHA 多保真 + 分布式
- 多卡解锁 **ASHA 异步多保真**：5/10/20 分钟检查点处按 rung 晋级/淘汰，promote-on-demand 无同步障碍，近线性扩展。
- 跨 worker 派发 trial；archive/memory 作为共享状态（DB 加锁或中心 orchestrator）。

### 4.7 `optim/predictor.py` — 排序预测器(预筛)
- 用自身试验档案训练廉价 **排序预测器**（BRP-NAS 式，排序比绝对回归省样本）。
- 变异候选先过预测器，跳过预测变差的，省 20 分钟/个。

**验收**：在固定创新点上，优化曲线显著优于纯随机爬山 baseline（多 seed 对照）。

---

## 5. P3 — 知识系统重构（从零搭建）

> 现状：**无任何建图数据、无注入脚本、检索是子串匹配**。需从语料获取做起。保留 Neo4j（fix/evidence 查询确属多跳，图有价值），但重建检索与 schema。

### 5.1 建图 pipeline（`knowledge/ingest/`）—— 当前完全缺失
```
fetch.py        → 拉文献: arXiv API + Papers-with-Code + Semantic Scholar
                  (时空预测/ST-GNN 主题, 含摘要/方法/结果/消融)
extract.py      → schema 约束 LLM 抽取: (Method, Symptom, Action, Condition,
                  Claim{effect,dataset,metric}, Concept) 三元组
ground.py       → 每条边接地到 source span; NLI 蕴含门控丢弃幻觉边
                  (RefChecker/GraphEval 式)
canonicalize.py → 概念去重(CESI/EDC), 对齐 CSO 计算机科学本体
build_graph.py  → 写入 Neo4j
embed.py        → 节点描述 embedding + Neo4j 原生向量索引
```

### 5.2 `knowledge/schema.cypher` — reified Claim
把 `RESOLVES` 边升级为节点，补齐让建议"可执行"的字段：
```
(Symptom)-[:ADDRESSED_BY]->(Resolution:Claim {
   condition,            # 何时适用: dataset_type/horizon/regime
   effect,               # 结构化 Δmetric + baseline + dataset (非自由文本)
   evidence_grade,       # GRADE: High/Moderate/Low
   polarity,             # +resolves / -fails_under (负知识结构化)
   provenance            # → Paper{doi/arxiv_id}
})-[:BASED_ON]->(Concept)
(Resolution)-[:HAS_RECIPE]->(CodeSnippet)   # 可运行 config/代码 diff
```

### 5.3 `knowledge/retrieve.py` — 混合检索（替换子串匹配）
- **向量入口**：query embedding → 向量相似找入口节点（解决"oversmoothing≠over-smoothing"）。
- **图遍历**：从入口取"修复+证据+反模式"连通子图。
- **applicability 门控**（Self-RAG 式）：每条结果先判"适配当前 run 的 regime 吗"，丢弃近似干扰项（Power of Noise）。
- **融合自身经验**：同时召回 P1 memory 的最近邻试验；**冲突时自身结果压过文献**；Graveyard 已知失败**硬阻断**。
- `inspire` 修复：从纯 `rand()` 改为**按 Concept 族分层 + 相关性门控**，≤3 条，最相关放上下文边缘（避免 lost-in-the-middle）。

**验收**：给定症状（如"长时预测时序依赖丢失"）能召回带量化证据+可运行 recipe 的处方，且能正确屏蔽不适配/已试错项。

---

## 6. P4 — 自主性机制加固 + 清理

### 6.1 程序化"永不停"（替代纯 prompt 纪律）
- 停止判定做成**程序**：`best_so_far()` vs `baseline_registry[SOTA]`，超越才停；否则 orchestrator 自动派发下一轮。
- 循环驱动放到 **harness/编排层**（orchestrator 进程 + 可选 `/loop`），不依赖模型"自觉不暂停"。
- Agent 侧 SKILL.md 仍保留"封杀礼貌性停顿"，但只作为兜底。

### 6.2 清理 LLM 残留
- `analysis.ipynb` 重写为时空分析（读 memory DB：MAE 收敛曲线 / KEEP 率 / archive 热力图）。
- `pyproject.toml`：加 `neo4j optuna ray[tune] h5py pyyaml`；清 `kernels/rustbpe/tiktoken`；改描述；torch 后端 **CUDA(cu128)**（集群）。
- README/SKILL.md 对齐新架构。

---

## 7. 端到端时序（一次实验的生命周期）

```
用户填配置单(创新点/数据集/SOTA目标/分支/预算)
        │
[Setup] git 分支隔离 → prepare.py 按协议建数据+邻接 → 校正 baseline
        │
[Tier2 创造] Agent 读创新点 → 查知识图谱 → 把"液态神经元"实现为 operators.py 算子
             → 定义初始 genotype(含保护区) → 写初始 insight 到 memory
        │
[Tier1 系统] orchestrator 启动并行进化:
   ┌──────────────────────────────────────────────┐
   │ for 每代 (跨集群并行):                         │
   │   选父代(锦标赛) → 变异(规则/预测器预筛)        │
   │   → 编译 genotype → train.py(+Optuna HPO)      │
   │   → ASHA 5/10/20min 早停 → masked 多seed 评测   │
   │   → 记 memory + 放 MAP-Elites 档案(aging淘汰)   │
   │   → 程序判定: 超越 SOTA?                        │
   │       否 → 继续(永不暂停问人)                   │
   │       是 → 记录里程碑, 继续逼近浮点极限          │
   └──────────────────────────────────────────────┘
        │  ←── 搜索停滞时回到 Tier2: Agent 诊断+注入新机制/重定向
        ▼
   持续运行直到被手动中断
```

---

## 8. 里程碑与验收

| 阶段 | 交付 | 验收标准 |
|---|---|---|
| **M0** 评测地基 | P0 全部 + tests | `pytest` 绿；复现 STID 量级 sanity 数 |
| **M1** 记忆 | P1 SQLite + 接口 | Graveyard 阻断/最近邻检索可用 |
| **M2** 优化循环 | P2 orchestrator | 优化曲线显著优于随机爬山(多seed对照) |
| **M3** 知识系统 | P3 pipeline+检索 | 召回带证据+recipe 的处方，屏蔽干扰项 |
| **M4** 自主+清理 | P4 | 程序化不间断；残留清空 |
| **M5** 集成 | SKILL.md 重写 | 端到端跑通一个创新点直到逼近 SOTA |

---

## 9. 关键技术选型

| 关注点 | 选型 | 理由 |
|---|---|---|
| 评测标准 | 对齐 **BasicTS** | 社区公认标准化基准，跨模型可比 |
| 内层 HPO | **Optuna TPE + Hyperband/ASHA pruner** | 条件空间友好、多保真、成熟 |
| 外层 NAS | **aging evolution + 锦标赛** | 离散黑盒、抗噪声、与 Agent 变异天然契合 |
| 档案 | **MAP-Elites + 岛屿** | 多样性、防坍缩、即"什么有效"记忆 |
| 并行调度 | **ASHA (Ray Tune)** | 多卡近线性扩展、异步无障碍 |
| 预筛 | **排序预测器 (BRP-NAS 式)** | 排序省样本，跳过坏变异 |
| 记忆 | **SQLite + ExpeL 蒸馏** | 可查询、可阻断、可检索最近邻 |
| 知识库 | **Neo4j + 向量索引混合检索** | 多跳查询保留，语义入口修复 |
| 抽取防幻觉 | **schema 约束 + NLI 门控** | 出处接地、可信三元组 |

---

## 10. 实施前决策（已确认 / 待定）

**已确认：**
1. ✅ **包结构**：升级为 `src/darwin_st/`（NAS/HPO/并行的承载基础）。
2. ✅ **分布式栈**：**Ray Tune**（最主流，与 Optuna 原生集成，ASHA `ASHAScheduler` 开箱即用，多卡近线性扩展）。
3. ✅ **运行时输入**：每次实验自由指定**创新点（自由文本）**与**数据集（PeMS04/08、METR-LA、PEMS-BAY 常用选项，亦可扩展）**，经配置单 → `configs/experiment.yaml` 驱动。系统对数据集与创新点都不做硬编码。

**待定（可在实施中逐步定，不阻塞 M0）：**
4. **知识图谱语料来源**：建议 arXiv API + Papers-with-Code 自动拉取，首批 200–500 篇 ST-GNN 论文。是否认可？
5. **LLM 调用**：建图抽取与混合检索需调用 LLM，默认走 **Claude API**（涉及密钥与成本）。是否认可？
6. **M0 首做数据集**：因数据集是运行时输入，M0 仅需先打通**一个**做协议验证；建议 **PeMS04**（最具区分度、图来源清晰），其余 profile 随后补齐。

> 设计原则：数据集与创新点既然是运行时自由输入，**P0 的协议层（`data/protocol.py`）以"按数据集 profile 分流"为一等设计**——新增数据集 = 加一个 yaml + 一个下载/邻接适配器，不改核心逻辑；**创新点经 Tier 2 由 Agent 实现为 `operators.py` 的算子并纳入 genotype 保护区**，对系统层透明。
