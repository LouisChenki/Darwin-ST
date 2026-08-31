# Darwin-ST 排行榜

自治时空预测研究系统 (进化 NAS + LLM 跨域创造) 的方法版本榜: 每个大版本只上榜其最佳模型, 展示方法一版版变强的演进主线。

## 方法版本榜 (PeMS04)

参照 SOTA: **STD-MAE** (MAE 17.80); MAE 越低越好。

| 版本 | 日期 | Best val MAE | Test MAE (复评) | vs SOTA | 榜单位次 | Tag | 复盘 |
|---|---|---:|---:|---:|---:|---|---|
| **v0**<br>纯 NAS + HPO 进化搜索（无 Tier-2 创造），传统 AutoML 天花板 | 2026-06-25 | **20.67** | — | +2.87 | **11** / 12<br>(val 口径) | — | [复盘](../docs/SERVER_VALIDATION.md) |
| **v1**<br>Tier-2 创造闭环首通：LLM 跨域合成算子注入进化（线程版） | 2026-06-25 | **18.69** | — | +0.89 | **8** / 12<br>(val 口径) | — | [复盘](../docs/SERVER_VALIDATION.md) |
| **v2**<br>Stage1 搜索空间升级（joint/cross/iterative fusion）+ 24h 长跑 | 2026-07-01 | **18.349** | **18.464** ± 0.036 (n=6) | +0.66 | **7** / 12 | `v-sota-pems04-18.35` | [复盘](../docs/SOTA_RETROSPECTIVE.md) |
| **v3**<br>Tier-2 v1 方法升级（履历/家谱精炼/proxy/信用分/独立采样 + 631 卡机制库） | 2026-08-07 | **18.356** | **18.356** (n=1) | +0.56 | **7** / 12 | `acceptable+18.356` | [复盘](../docs/TIER2_V1_RETROSPECTIVE.md) |
| **v4**<br>B7 训练方式创造通道上线（aux 契约/七门验证/训练链路）— 收关定性：aux 路由失效(activation=0)+反思未触发，非机制负结果 | 2026-08-20 | **18.476** | — | +0.68 | **7** / 12<br>(val 口径) | `b7-v1-routing-failure-18.476` | [复盘](../docs/B7_V1_CLOSEOUT.md) |

> 位次口径: 该版本最佳单独与已发表 baseline 混排 (不同版本互不占位); 完整模型存档见 [PeMS04.md](PeMS04.md)。

> **指标口径说明**: 「Best val MAE」为验证集历史成绩 (split 6/2/2, masked metric, 12 horizons 平均); 「Test MAE (复评)」为 E1 口径对齐后 test 集多 seed 重评值 ——与文献报告的 test MAE 同口径可比。无复评值的版本行位次仍以 val 标注; 有复评值的版本行位次按 test 计算。

## 成绩随版本演进

![PeMS04 方法版本成绩演进](assets/pems04_version_trend.png)

## 当前最佳

- **v3** (2026-08-07): Best val MAE **18.356**, Test MAE **18.356** (1 seeds), 榜单位次 **7** / 12 (test 口径), 距 SOTA (STD-MAE 17.80) +0.56
- 模型卡: [models/PeMS04/07_mae18.36_6bcc97b9](models/PeMS04/07_mae18.36_6bcc97b9/)
- git tag: `acceptable+18.356`

## 完整存档与设计文档

- 完整模型存档榜 (全部入榜模型 + baseline 混排): [PeMS04.md](PeMS04.md)
- 各版本复盘与设计文档: [docs/](../docs/)
