# Darwin-ST IJGIS 实验台账 (Experiment Log)

> 用途：论文撰写的唯一取数入口。每个实验一行主记录：日期 / 配置 / 关键数字 / 产物位置 / 状态。
> 规程：台账入 git（本文件）；产物留本地 `paper-results/`（gitignore）与 `frozen-runs/`（冻结规程，SHA-256 manifest）。
> 更新规则：实验收关即补录；关键数字一律以台账为准（与论文表格同源）。

## 主结果候选（test 口径, PeMS04, masked MAE, 12 horizons 平均）

| 对象 | val MAE | test MAE | n_seeds | 来源 | 状态 |
|---|---|---|---|---|---|
| **v3 (tier2v1 模型)** | 18.356 | **18.356** | 1 (原存档 ckpt 直评) | paper-results/e1/best_v3_ckpt.json | ✅ 定稿 |
| **v2 (24h 长跑架构)** | 18.515 (6seed 均值) | **18.464 ± 0.036** | 6 | paper-results/e1/best_v2_s012+s34+s42.json | ✅ 定稿 |

- val→test 偏移: −0.05 ± 0.03（test 系统性略优于 val, 无选择集过拟合, R1 风险排除）
- 文献 test MAE 参照: STD-MAE 17.80 / STGformer 17.89 / HimNet 18.14 / STAEformer 18.22 / PDFormer 18.32 / STID 18.35

## 实验记录

| 实验 | 日期 | 内容 | 关键结果 | 产物 | 状态 |
|---|---|---|---|---|---|
| E1 口径对齐 | 2026-08-29 | best 模型 val→test 重评 + 榜单双口径改造 | v3 test=18.356; v2 test=18.464±0.036 (6seed) | paper-results/e1/; leaderboard/README.md | ✅ |
| E2 多 seed | 2026-08-30 | v2 架构 6 seed 重训统计 | test MAE 18.464±0.036, 偏移 −0.05±0.03 | paper-results/e1/ | ✅ |
| E13 效率曲线 | 2026-08-31 | best-so-far vs 评估数/GPU·h 双口径 | 纯进化平台 ~18.86 封顶; 含创造持续下探至 18.35 | paper-results/e13/efficiency_curves.png, summary.md | ✅ |
| E11 守卫统计 | 2026-08-31 | 验证门拦截率/CRASH 构成/典型案例 | 拦截率 0–3.1%; CRASH 率 2.4–6.2%; 典型案例 ScaleGatedGCNExpertMixer_v29 forward 门 | paper-results/e11/gate_stats.md | ✅ |
| E5-B 统一预算对照 | 2026-08-31 | 四 baseline 统一配方 (lr=1e-3, 100ep) 同炉 | DCRNN 25.43 / GWNet 22.83 / AGCRN 102.29(发散) / STID 18.95±0.042 | paper-results/e5/*_s42.json, stid_s012.json | ✅ (AGCRN 待诊断) |
| E5-A 文献配方复现 | 2026-09-01 | 各 baseline 按官方配方复现 | 进行中 | paper-results/e5/recipe/ | 🔄 |
| AGCRN 诊断 | 2026-09-01 | 102 发散的数值诊断 | 进行中 | — | 🔄 |
| 统计三件套 (HA/AR/FC-LSTM) | 2026-09-01 | 统计基线同协议评测 | HA test=42.35 / AR(12) test=32.11 / FC-LSTM test=26.65 (均与文献档吻合) | paper-results/e5/stat/ | ✅ |
| STAEformer 移植 | 2026-09-01 | 集群领头羊 baseline (18.22) | 进行中 | src/darwin_st/baselines/ | 🔄 |
| D2STGNN / ASTGNN 移植 | 2026-09-01 | 中段强模型 baseline | 进行中 | src/darwin_st/baselines/ | 🔄 |
| E12 成本核算 | 2026-08-31 | GPU·h + LLM 调用统计 | B7 v1: 741.9 GPU·h (≈31 卡·天), 550 评估 | scripts/cost_report.py | 🔄 初版 |

## 统一预算对照表 (E5-B, 2026-08-31 实测, test MAE)

| 模型 | test MAE | 备注 |
|---|---|---|
| **Darwin-ST (v2 best 架构)** | **18.464 ± 0.036** | 6 seed |
| STID | 18.945 ± 0.042 | 3 seed |
| GWNet | 22.83 | 未收敛到文献档 |
| DCRNN | 25.43 | 未收敛到文献档 |
| AGCRN | 102.29 | 发散 (诊断中) |

> 结论候选: 统一预算下 Darwin-ST 显著优于全部对比模型 (唯一进 18.5 档)。

## 文献配方复现表 (E5-A, 待填)

| 模型 | 文献报告值 (registry 一手数) | 复现值 | 判定 (≤0.2 合格) |
|---|---|---|---|
| DCRNN | 19.63 | 🔄 | — |
| GWNet (GraphWaveNet) | 18.53 | 🔄 | — |
| STID | 18.35 | 🔄 | — |
| AGCRN | ~18.8 | 🔄 (先诊断) | — |
| STAEformer | 18.22 | ⏳ | — |
| D2STGNN | ~18.3 | ⏳ | — |
| ASTGNN | ~18.6 | ⏳ | — |
