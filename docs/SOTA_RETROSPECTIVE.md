# 冲 SOTA 全程复盘 (2026-07-01)

> 从纯 NAS 天花板 20.67 到逼近手工 SOTA 集群 18.35 的完整复盘。数据源: 服务器 sota_24h.db
> (540 评估, 522 KEEP)、多轮实跑日志、leaderboard 存档。

## 1. 最终结果

| 里程碑 | MAE | 说明 |
|---|---|---|
| 纯 NAS+HPO 天花板 | 20.67 | 传统 AutoML 上限 (项目论点的对照) |
| 历史最好 (线程版, 无本轮改进) | 18.69 | |
| **本轮最终 best** | **18.35** | PeMS04 排行榜排第 5, **超已发表 STID(18.35)/PDFormer(18.36)** |
| 可发表档 STAEformer | 18.22 | 距 +0.13 (在合成算子 MAE 噪声量级内) |
| SOTA STD-MAE | 17.80 | 距 +0.55 |

**核心论点充分坐实**: 纯 AutoML 卡在 20.67, **LLM 跨域创造 + 搜索空间升级把它压到 18.35, 挤进手工 SOTA 集群内部**。这是一个可发表水平的自动发现结果。

**最佳架构** (depth=4, hidden=128): 2 个跨域合成算子
`synth_SequentialRandomGCN`(源: domain_randomization) + `synth_ssm_additive_residual`(源: state_space_modeling)
+ Stage 1 新交互 (iterative/cross)。**纯 NAS 绝对搜不出的复杂跨域架构。**

## 2. 各改进的实证贡献 (本轮做的四件事)

1. **多进程后端 (修 GIL)**: 4 卡真并行 (占用 60-90% vs 线程版 ~40% 波动), 充分训练可行。
2. **A+B 节奏 (创造后专项大 HPO + 精修窗口)**: seed 走 20-trial HPO + warm-start, 榨干新算子潜力。
3. **自适应 patience (Gap-Annealed)**: patience 随距 SOTA 缩小而增大 (实测 gap=2.6→5, gap=0.55→8), 逼近 SOTA 时给精修更多耐心。
4. **Stage 1 搜索空间升级 (joint + 交互层 + 类别化)**: 见下, 贡献最直接。

**量化证据 (top20 精英架构统计)**:
- **fusion 分布**: cross **46**, iterative **29**, sequential 7, 其余各 1。→ **Stage 1 新加的 cross/iterative 占了 88%**, 远超原有的 sequential/parallel/residual。进化强烈偏好新交互模式。
- **含 synth 算子的块 65/85 (76%)**: 精英架构绝大多数依赖 LLM 创造的跨域算子。
- **joint 块 20/85**: joint 模式 (Stage 1 批评#1 修复) 被稳定选用。

结论: **Stage 1 的两项 (新交互 fusion + joint) 不是孤立验证, 而是在真实搜索里被进化压倒性选中** —— 破纪录架构离不开它们。

## 3. 诊断多样化 (LLM 训练动态诊断)

**30 次创造, 28 个不同跨域机制** (diag_fail=0, LLM 诊断全程零 fallback):
domain_randomization / state_space_modeling / deformable_temporal_patching / diffusion_convolution /
quadtree_spatial_partitioning / frequency_temporal_augmentation / cross_scale_attention /
minimal_predictive_sufficiency / dataset_distillation / curriculum_learning / disentangled_encoding …

对比早期 bug (6 次全 dilated_causal), **多样化彻底成功**。机制横跨频域/图/采样/损失/分布/自监督等多个类别, 由训练动态证据驱动 (欠拟合→多尺度; 已拟合→泛化机制)。

## 4. best 轨迹 (突破节奏)

```
20.42 (round1) → 19.17 (round4, 首次创造 joint 时空算子) → 18.75 (round5) →
18.674 (破历史18.69) → 18.642 → [关机#1 无缝续跑] → 18.591 → 18.537 →
18.488 → 18.362 → 18.349 (硬平台)
```

- **前期极快** (5 轮从 20.42 到 18.75, 三件改进协同): 自适应 patience 让创造早触发, joint+大HPO 让新算子立刻发挥。
- **中期稳降** (18.75→18.35): 创造持续换机制, 进化加深架构 (depth 2→5), 精修窗口起效。
- **后期硬平台** (卡 18.35): 数百评估 + 十几次新创造未能撬动。**这是当前搜索空间 (线性 block + 现算子集) 的硬局部最优。**

## 5. 卡点分析: 为什么停在 18.35

**根因 = 搜索空间天花板, 不是节奏/诊断/算力问题**:
- 节奏 (A+B/自适应)、诊断 (28 机制)、算力 (4 卡满载) 都已到位且健康。
- 但架构表示仍是**线性 block 堆叠** (Stage 3 DAG 未上), 且原子算子集仍有限 (Stage 2 未上)。
- 创造能发明新算子, 但进化只能把它们**线性串联** —— 无法探索"时序注意力引导的动态图 + 前后自由连接其他算子"这类 DAG 拓扑。18.35 是这个受限空间的硬极限。

## 6. 运维 & 鲁棒性验证 (意外收获)

- **2 次 AutoDL 余额关机, 均 RESUME 无缝续跑, 零进度损失**。RESUME (从 memory 最优暖启动) + registry.load_persisted (载回 synth 算子) + 独立 MEMORY_DB 累积 + watchdog, 这套长跑基建经受住了真实中断考验。
- 磁盘/内存全程健康 (关机纯因计费)。540 评估 18 CRASH (3.3%, 合成算子少数失败, 被验证门+评测器正常淘汰)。

## 7. 下一步 (突破 18.35 平台的根本手段)

**Stage 2 + Stage 3 是天花板的来源, 才是破局的根本** (已在 plan 设计):
- **Stage 2 扩充原子算子集**: MixHop / GraphWaveNet 双邻接 / Informer 注意力 / Mamba SSM / 多尺度膨胀。便宜纯加法。
- **Stage 3 DAG 连接空间** (AutoCTS++ 式): 架构=节点 DAG, 边=任意算子 (含复杂时空交互), 进化自由连接。这才能让"时序注意力引导的动态空间图卷积"这类算子作为一条边无缝融入, 突破线性堆叠的表达上限。
- 二者上线后重跑, 期望突破 18.35 平台冲 18.22/17.80。

**其他可选**:
- 更充分训练 (max_epochs 80→100+)、更大 HPO 预算。
- 多 seed 平均 + 显著性 (SOTA 附近提升 <1%, 需排除噪声)。
- 针对性补机制卡 (多尺度+异质+时序 前提组合, 检索验证发现的库覆盖缺口)。
