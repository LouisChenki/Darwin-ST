# P2 AutoML 算法设计 (实现级研究结论)

> 2026-06-17 · 基于 4 路深度前沿研究 (ST-GNN NAS / Optuna+ASHA / MAP-Elites / LLM-as-optimizer)
> 本文档把研究结论固化为 P2 各模块的实现级设计依据。前置: docs/BLUEPRINT.md §4。

## 0. 最重要的发现 (改变设计)

**节点/时间身份嵌入 (identity embeddings) 比图算子的选择更能提升精度。**
- 证据: 去掉 STID 的空间节点嵌入 → MAE +18% (18.29→21.65);去掉 STAEformer 的自适应嵌入 → +19% (18.22→21.63)。
- 现状: 我们的 genotype/operators **完全没有这些嵌入**。
- 结论: **不补嵌入,搜出来的架构无论图算子多好都到不了 SOTA 档(只能到 GraphWaveNet 的 ~19 档,够不到 STAEformer 的 ~18 档)。**
- 行动: 把节点嵌入 + time-of-day + day-of-week 提升为**一等 genotype 元素**(可搜索开关 + 维度),注入方式=**拼接**(concat,非相加),接在输入投影处。

**SOTA 现实校准**: 所有 NAS 系统(AutoCTS/+/++、AutoSTF)都比手工 SOTA(STAEformer 18.22 / STD-MAE 17.80)低**一档**。我们若能搜到 STAEformer 档(PeMS04 ~18.2)就已是可发表水平。AutoSTF 18.38 是 NAS 线最好的。

## 1. operators.py 补强 (P2-a 已建, 需扩展)

研究指明**必备算子集**(否则进不了 AutoCTS/AutoSTF 联盟):
- 时序: **GDCC**(门控膨胀因果卷积,胜过普通 1D conv)、**Informer 式高效时序注意力**。我们已有 tcn/gru/attn,需把 tcn 升级带门控、attn 确认高效。
- 空间: **不要押注单一邻接**——做成可搜索选择(AutoSTF 式):`adaptive`(可学习 softmax(relu(E₁E₂ᵀ)),最重要)、`fixed`(预定义距离图)、`att`/Informer-S(无图空间注意力)。我们已有 gcn/cheb/gat/diffusion/adaptive,基本够,需确认 adaptive 是一等。
- **新增 embeddings 模块**(最高优先级):STID 式 `E∈ℝ^{N×D}` + ToD`[288×D]` + DoW`[7×D]`,拼接注入。

## 2. genotype.py 补强 (P2-b 已建, 需加嵌入基因)

加入嵌入相关基因(可变异):
- `use_node_emb`(bool)+ `node_emb_dim ∈ {16,32,64}`
- `use_tod`(bool)、`use_dow`(bool)
- 可选 STAEformer 式共享自适应 `[T,N,d_a]`(bool + d_a)
- 新增变异算子 `toggle_embedding`(开关嵌入/改维度)——把嵌入当可变异基因。
- 解码器固定为**直接多步线性/MLP 头**(一次出 12 步,别搜迭代解码——严格更差)。

## 3. evolution.py — Aging/Regularized Evolution (Real et al. 2019)

- 算法: 种群队列大小 P;每轮采样 S 个、取最优为父代(锦标赛)、变异、评估、入队右端、**移除最老(FIFO)而非最差**。
- 为何移除最老: 20 分钟评估有噪声,移除最差会让"侥幸高分"永久霸榜;aging 给每个个体有限寿命,血脉只能靠后代**重新训练到高分**存活 → 抗噪声正则化(关键,因我们付不起重复评估)。
- 参数: **P≈20, S≈5**(保持 S/P=0.25);只调这两个 + 网格分辨率。
- 变异: op-swap、S/T 算子换、**embedding-toggle**;连接重连须保持 DAG 合法(非法子代拒绝重采);**无 crossover**(破坏图基因合法性)。

## 4. hpo.py + scheduler.py — Optuna TPE + ASHA (源码级已核实)

**关键修正**: Optuna 的 `SuccessiveHalvingPruner` **本身就是 ASHA**(异步,无同步障碍)。没有单独的 ASHAPruner。4 卡场景**用它,不用 HyperbandPruner**(单 bracket → TPE 约 8 个完成 trial 就启动,而非 Hyperband 的 ~40)。

- Sampler: `TPESampler(multivariate=True, group=True, constant_liar=True, n_startup_trials=8)`
  - `multivariate+group`: 正确处理条件超参(如 num_heads 仅注意力时存在)
  - `constant_liar=True`: 4 并行 worker 防扎堆(**必须**)
- Pruner: `SuccessiveHalvingPruner(min_resource=3, reduction_factor=3)`
  - fidelity=**训练 epoch**;`min_resource` 设在 warmup 之后(避开早期噪声)
  - reduction_factor=3 比默认 4 温和,适合噪声 MAE 曲线
- 接口: **ask/tell**(自定义训练循环);`trial.report(mae,epoch)` + `should_prune()`。
- 并行: **每 GPU 一个进程**,共享 **JournalStorage**(Optuna 4.0 稳定,文件式,免 DB server;**绝不用 SQLite 并发写**)。
- 分工铁律: **架构不是 Optuna 参数**;每个架构开**独立 study** 调其超参;`study.enqueue_trial(parent_best_hps)` 暖启动;**绝不让 ASHA 跨架构比 MAE**(不同架构 MAE 尺度不可比)。
- TPE **不学被剪枝的 trial**(只用 COMPLETE);若 TPE 饿了,剪枝时 return 平滑 MAE 估计而非 raise。
- 僵尸防护: try/except + `tell(state=FAIL)`。

## 5. archive.py — MAP-Elites + Aging 混合 (小预算特化)

**现实**: 纯 MAP-Elites 极度样本饥渴(SAIL 报告需 10⁵-10⁷ 评估)。我们只有 ~100-300 评估,必须特化:
- **用普通网格**(BD ≤4 维、矩形/类别 → O(1),不用 CVT)。
- **BD 轴 + bin(目标 ~36 cell ≤ 预算)**:
  | 轴 | 类型 | bin | 定义 |
  |---|---|---|---|
  | 主导空间族 | 类别 | 4 | {none, local-conv[gcn,cheb], attn[gat], diffusion/adaptive} |
  | 主导时序族 | 类别 | 3 | {conv[tcn], recurrent[gru], attn[attn]} |
  | 参数量档 | 连续log | 3 | <50k / 50k-200k / >200k |
  → 4×3×3=**36 cell**。预算到 300 再加深度轴(2 bin)→72。
- **BD 从 genotype 直接算,无需训练**(参数量靠 builder 干实例化)。
- 插入: 每格精英,**严格改进**才替换(MAE 更低)。
- 选择: **混合**——40% 概率从填充格均匀采(QD 多样性);60% 概率对最近插入精英做 S≈4-5 锦标赛(aging 抗噪)。archive 本身即种群。
- **BD 不能与 fitness 相关**(否则 QD 退化成纯优化)——须验证无单一算子族碾压。
- 停滞 → **岛屿重置**(FunSearch 式):best-MAE 连 N 轮不升 → 清差的一半 cell,从最优精英重播种,把清空区交给 LLM 当目标。

## 6. orchestrator.py + Agent 交互 — 两层自治协议

研究强烈印证 BLUEPRINT §0 的两层分工。关键铁律:

| 确定性 harness (Tier1, 高频 24/7) | LLM 判断 (Tier2, 低频, 停滞/创新时) |
|---|---|
| 选择/插入/岛屿重置、训练、ASHA 剪枝、NaN/OOM 熔断 | *哪个*语义变异;*是否*注入新算子 |
| Graveyard 签名查重**硬否决** | 诊断系统性失败(过平滑→残差) |
| 打分、多 seed、**SOTA 差距、停止判定** | 停滞时重定向搜索空间;写新算子代码 |
| **循环驱动**(下一轮派发) | 每次请求只返回一个结构化提议 |

**防"礼貌性暂停"= 结构性, 非行为性**(最可靠):
- 循环放 orchestrator 进程,续跑不依赖模型"记得别停"。
- 每次 LLM 调用只干一件事(产一个提议),返回结构化对象,**无对话式收尾**。
- 提议 schema **无 ask_user 字段**;不确定也须给 best-effort 变异 + confidence。
- 停止判定是**代码**(`best_so_far()` vs `baseline_registry[SOTA]`),绝非 LLM 口头。

**LLM 提议协议**(OPRO + FunSearch + DS-Agent 合成):
- harness 组装状态包(确定性,从 DB): {objective+SOTA gap, **OPRO 式轨迹(按 MAE 升序、最优在末**), graveyard 摘要, insights, nearest_experiments, ≤3 relevance-gated KG 提示, protected 算子, parent genotype, regime}。
- LLM 返回 **schema 约束**的 MutationSpec(diff 式优先,AlphaEvolve)或 new_genotype(仅 Draft)+ rationale + confidence。
- harness: 校验 → 应用 → 签名 → graveyard 否决(命中静默重采)→ builder 形状/可微检查 → ASHA 派发 → 记录 → 更新 archive → 重算 SOTA 差距 → **无条件派发下一请求**。
- 多保真级联: 5 分钟廉价 rung 先筛,再升 20 分钟(LLM 坏提议快速便宜地死)。
- 每次 DISCARD 闭合 **Reflexion** 回路: 写一行 fail_reason,周期性蒸馏 insights。

## 7. 落地顺序 (P2 剩余)

1. **operators.py 补 embeddings 模块 + GDCC 门控**(最高优先级,决定能否到 SOTA 档)
2. **genotype.py 补嵌入基因 + toggle_embedding 变异**
3. **builder.py**(P2-c): genotype→nn.Module,含嵌入拼接 + 直接多步头;PeMS04 跑通基线(目标先到 ~20 MAE 验证管道,再调到 ~18)
4. evolution.py(aging,P≈20/S≈5)
5. hpo.py + scheduler.py(Optuna TPE + ASHA + JournalStorage + 4 卡并行)
6. archive.py(36-cell 网格 + 混合选择 + 岛屿重置)
7. orchestrator.py(两层协议 + 程序化停止 + graveyard 硬否决)
8. predictor.py(≥50 评估后,SAIL/BRP-NAS 式排序预筛)

## 架构铁律(节点维约束)

时空算子(无论进化变异还是 Tier-2 合成)必须遵守的硬约束。validation harness
(creation/validation.py)对合成算子强制检查其中的形状契约;进化/手写算子同样适用。

1. **节点维 N 神圣,张量全程 `[B, T, N, C]`**。严禁把空间节点暴力压平
   (如 `[B, T, N, C] → [B, T, N*C]`)——这会彻底摧毁交通节点的物理空间拓扑。
   算子的 `forward([B,T,N,C]) → [B,T,N,C]`,绝不 flatten/mean 掉 N。这是全库统一约定,
   也是合成算子能否通过验证门的第一关。

2. **强制保留并利用空间结构**。进化方向应主动引入 GCN / GAT / 时空同步卷积 /
   逐节点处理融合等机制,让模型理解空间邻接关系,而非把 N 当 batch 维抹平。

3. **奥卡姆剃刀与算力折中**。`val_mae` 相近时优先保留最简单、对算力最友好的分支。
   触碰时间/显存预算墙时执行优雅折中(降 `d_model`、减注意力头、削 batch),
   而非抛弃机制本身。

4. **维度防御 + NaN 熔断**。复杂张量变换(`view`/`reshape`/`permute`/`einsum`)前后
   用 `assert` 防御性检查形状,规避 Shape 幻觉。训练循环植入 `torch.isnan/isinf`
   检测,捕获即判 DISCARD,不空跑无意义的 epoch。

5. **分层调试优于全盘替换**。一个合理的时空机制(深层 GCN / 时空注意力)初期导致
   MAE 退化时,不要立即抛弃换新架构。先排查:梯度消失/爆炸?需注入 LayerNorm?
   加残差连接抵御过平滑(over-smoothing)?学习率过大致发散?先打磨再否定。

## 关键来源
STID 2208.05233 · STAEformer 2308.10425 · STD-MAE 2312.00516 · AutoCTS 2112.11174 · AutoCTS+ 2211.16126 · AutoSTF 2409.16586 · Regularized Evolution 1802.01548 · MAP-Elites 1504.04909 · CVT-MAP-Elites 1610.05729 · SAIL 1702.03713 · QD-NAS 2208.00204 · LLMatic 2306.01102 · ELM 2206.08896 · FunSearch 10.1038/s41586-023-06924-6 · OPRO 2309.03409 · AIDE 2502.13138 · ExpeL 2308.10144 · Optuna ASHA/TPE 官方文档 + 源码
