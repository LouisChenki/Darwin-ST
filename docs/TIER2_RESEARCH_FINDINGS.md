# Tier-2 实现级研究结论 (RESEARCH-BACKED)

> 2026-06-18 · 4 路深度调研(均源码/论文核实, 子 agent 隔离)的蒸馏结论。指导 P3 + Tier-2 实现。
> 前置: docs/TIER2_DESIGN_DECISIONS.md(决策). 本文档是其落地依据。

## 0. 贯穿一切的铁律(4 路调研一致结论)

**LLM 是快速但会幻觉的"提议者"; 正确性与进展完全来自确定性评测器 + 进化记忆。**
我们的 P0 评测地基 + builder 测试 + archive/memory 正是这个"壳"。LLM 永远只提议, 真实 MAE(20min GPU)永远是唯一裁判。这条铁律决定了下面所有"保留/砍掉"的判断。

## 1. LLM 算子合成 (C1/C5) — FunSearch/AlphaEvolve/ADAS 实现机制

**固定骨架 + 只变异一个函数/类(FunSearch 第一课)**: 人写骨架(driver+evaluator, LLM 不可见/不可改), LLM 只填一个固定签名的函数体。对我们 = 固定 nn.Module 契约 `forward(x,adj)->[B,T,N,C]`, LLM 只填内部计算。骨架越保护, LLM 算力越聚焦在值得变异的部分。

**评测器是唯一幻觉防线**: FunSearch 只接受"能跑+输出数值分+不调祖先函数"的程序, 其余静默丢弃; 无人读代码, 错代码自然不繁殖。AlphaEvolve 用 **EVOLVE-BLOCK 标记可改区 + SEARCH/REPLACE diff + 评测级联(cheap→expensive)**。

**ADAS(最相关)**: LLM 把"agent 写成代码"(<100 行框架 + 自由 forward 体), 条件于整个 archive 生成新设计 + 自反思 debug(≤5次)+ 入档。关键发现: **"踏脚石"——低分但新颖的设计是后续突破的原料, 不要纯按 fitness 剪枝算子库**。开放式来自"代码是图灵完备"而非固定菜单。

**防 reward hacking(非协商项)**: ①LLM 绝不见/不改测试+评测+held-out(物理隔离, OS 只读) ②held-out 独立 ③沙箱子进程(限内存/时间/无网络) ④硬门: 常数输出检测/标签泄漏检测/参数量FLOP上限/NaN门/gradcheck ⑤对照 trivial baseline ⑥入档候选换 seed 重评。环境硬化实测降 reward hacking 87.7%。

**Aider 作为子例程(参考 AI Scientist)**: SEARCH/REPLACE diff 直接编辑+自动跑测试+读报错自我 debug(不靠"抽取"靠"约束生成+验证")。AI-Scientist 参数: MAX_RUNS=5, MAX_ITERS=4, stderr 截断 1500 字符, 每 idea 在模板副本上跑防污染。调用: CLI `aider --message-file --yes --no-auto-commits --test-cmd "pytest ..." --auto-test operators.py`; editable=仅 operators.py, read_only=spec+测试文件。坑: 钉版本/CLI比API稳/截断stderr/硬限迭代/限文件/限预算。

## 2. 轻量 idea 面板 (承载 C1-C5) — AI-Scientist/co-scientist 适配

**多智能体真相(关键, 省钱)**: 等算力下 multi-agent debate **打不过** CoT+self-consistency; 增益主要来自集成而非"辩论"; 单基模型多 agent = 模式坍缩。**唯一有效的多样性杠杆 = 不同 persona/system prompt + 温度, 不是加更多同质 agent**。LLM 自评不可靠(intrinsic self-correction 会降分); 必须用外部裁判(我们有真实 MAE)。

**面板 4 角色(每个=独立 system prompt 的 LLM 调用, 编排在 orchestrator)**:
- **Generator**(温度~0.9, N 个 persona 并行如"感受野最大化者/参数吝啬者/谱图专家"): 输入 task+算子库+genotype schema+**带 MAE 的 scored archive**+meta-review 反馈+当前最优; 输出 K=8-12 个候选 genotype 变异+理由。
- **Critic**(对抗, "不要改进, 预测失败"): 输出每候选的结构化失败列表 {failure_mode/location/severity/falsifiable_prediction/fix/confidence}; fatal(可证 OOM/shape 非法)在烧 GPU 前丢弃。**ST 重定向: 不查文献新颖性**, 查失败预测/形状可微/资源/对症/自身查重。
- **Ranker**(pairwise 双向消偏, **不是** pointwise): PRP-Sliding-K(O(N), 求 top-K)→ Bradley-Terry/Elo 聚合。这是 co-scientist Elo 锦标赛去掉昂贵 debate 的版本。
- **Meta-review**(每代一次): 复发失败模式→字符串附加到下代 Generator/Critic prompt(无需微调); **算 Spearman ρ/Kendall τ(预测排名 vs 真实 MAE)做校准门, 差则退回随机/历史选择**。

**编排循环**: Generator(N persona 并行)→Critic(丢 fatal, 廉价无 GPU)→surrogate 预测→Ranker(pairwise)→选 top-4(=4卡)→真实训练 20min(唯一 ground truth)→更新 archive+重训 surrogate→Meta-review 校准门→aging/MAP-Elites→循环。**经济本质: 廉价 LLM 面板做 acquisition function, 昂贵 GPU 只烧 top-4**。

**砍掉(开放式发现的产物, 对降 MAE 有害/浪费)**: Semantic Scholar 文献新颖性检查; 自评 Interestingness/Novelty; 多轮开放式"科学辩论"散文; 无 grounding 的多轮自纠; 有偏 NeurIPS reviewer+论文写作; 为多而多的 agent 扩张。

## 3. 跨域知识本体 (C3, P3 核心) — 结构映射理论

**为何当前 schema 做不了跨域**: `Dataset←Symptom←Action→Concept` 是诊断中心, 按 symptom/dataset(表面事实)索引→检索只返回同域处方。

**结构映射理论(Gentner, 载重理论)**: 类比映射**关系结构非表面属性**。系统性原理: CAUSE/ENABLES 绑定的关系系统优先迁移。**MAC/FAC 两阶段检索 = 向量搜索(MAC 廉价内容向量)+图结构重排(FAC 昂贵对齐)** —— 直接映射到向量+图栈。人类检索被表面相似主导, 但有用性被关系相似主导——这正是朴素 embedding 检索做不了类比的原因。

**工程设计早已解决跨域类比(偷 schema)**: GoF 设计模式(Intent+**Applicability**+Consequences)/TRIZ(抽象阶梯)/SBF-SAPPhIRE(Function+conditions+Structure+**Behavior 因果链**)。共同三元组: **抽象功能(做什么)+前提/适用条件(何时)+机制(怎么做)+因果解释+权衡**。

**LLM 能类比但只作提议者非验证者**: Webb 2023 证明能力; 但反事实/置换变体上脆弱→**验证映射别信任**。Analogical Prompting 在参数近邻最强、远距类比最弱→**这正是需要外部检索的理由**。GraphRAG 对"连点"全局查询胜过向量 RAG(机制=surfacing 语义近邻外的连接=跨域迁移)。

**抽取**: 两层置信度。**Stated 层**(高置信, grounded): method/task/dataset/metric/math_structure/evidence, 每个带 provenance{section,span}。**Inferred 层**(inferred=true, 低置信): abstract_function/preconditions/causal_behavior, 需 justification+supporting_span, **span 不存在就拒**。Inferred 层路由人工复核(无验证基准, 当假设非真理)。

## 4. P3 本体 v1 schema(机制中心)

`:Mechanism` 为新中心(取代 :Action 为主):
- `abstract_function`*(inferred, 嵌入)*: 域无关一句话目的, 如"利用输入冗余的自监督表示学习"
- `function_signature`: 受控 verb+flow 元组如 `(reconstruct, masked-signal)`
- `math_structure`*(stated, grounded)*: 具体算子, 喂 Aider
- `causal_behavior`*(inferred)*: **类比桥**, 结构→功能因果链如"掩码迫使从上下文推断缺失→学到冗余结构"
- `consequences`: 权衡(算力/内存/不稳定)
- `abstraction_level`: SAPPhIRE/TRIZ 阶梯, 调检索粒度
- `origin_domain`: **跨域过滤键**(CV/NLP/SSM/ST/control)
- `evidence`{dataset,Δ,grade} · `provenance`(必需) · `function_embedding`(MAC, 只嵌 function+preconditions)

新一等节点: `:Precondition`(共享, 类比钩子, 受控~30-50 如"数据有冗余/低秩结构""标签稀缺""长程依赖""非欧拓扑") · `:Function`(受控功能词表)。保留 `:Concept/:Dataset/:Symptom/:Anti_Pattern` 向后兼容但降级为元数据非类比索引。
边: HAS_PRECONDITION · ACHIEVES · CAUSES{ENABLES}(高阶, 系统性) · REFINES/ALTERNATIVE_TO/COMPOSES_WITH · EVIDENCED_BY · HAS_ANTI_PATTERN。

**跨域检索流(graph_rag.py 加 find_cross_domain_analogy)**: Step-Back 抽象 ST 瓶颈→{abstract_function,preconditions}域剥离 → MAC 向量搜 function_embedding(**绝不嵌表面描述**)→ 2-hop precondition 扩展 → **origin_domain<>target 硬过滤** → FAC 按共享 precondition 数重排(非余弦)→ 返回机制卡+角色映射+前提满足检查 prompt + 对抗 paraphrase 检查。

**v1 验收门**: "能否提出一个成功跨域移植"(如独立重发现掩码预训练→ST 即 STD-MAE)。

## 5. 诚实的 solid vs 研究开放

**Solid**: 结构映射理论+字段集(40年); MAC/FAC=向量+图; 功能分离检索胜表面; 稠密检索表面偏置; FunSearch/ADAS 代码进化机制; 防 reward-hack 栈; pairwise>pointwise; Aider 工程范式; stated 层抽取(~80%F1)。
**研究开放(需人工兜底)**: preconditions/abstract_function 清洁抽取无基准(外在幻觉风险); precondition 节点规范化(过/欠连接); SME 结构重排规模化; 类比距离控制; KG 是否绝对胜参数知识(远距迁移真, 近域未证)。**贯穿缓解: KG 作提议+grounding, P0 评测器作最终裁判——与 C-hybrid 赌注一致**。

## 6. 落地顺序(P3 → Tier-2)
P3: ①机制中心 schema(Neo4j)②建图 pipeline(语料扩 CV/序列/自监督/图学习, schema 引导抽取+两层置信+人工门)③向量+图混合 + find_cross_domain_analogy ④v1 验收: 一次成功跨域移植。
Tier-2(P2.5, 用 P3 知识): ①Aider 算子合成(沙箱 worktree+测试验收)②idea 面板(Generator/Critic/Ranker/Meta-review)③接 orchestrator(面板=提议+预排, GPU 只烧 top-4)④surrogate predictor + 校准门。
