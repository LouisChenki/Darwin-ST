# 模型卡: darwin-st/a95d0fe4

- **数据集**: PeMS04
- **排名**: 第 507 名 / 库内 KEEP 实验
- **val MAE**: 22.61
- **val RMSE**: 35.82
- **参数量**: 627.5K
- **实验 id**: 14   **签名**: `a95d0fe41b3a9b6390df279b1a62ff906c8f2911`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-24T20:21:16.992387+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +4.81
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 128   邻接: sym
- 时空块:
  0. spatial=`stjoint_conv` temporal=`tcn` fusion=`sequential`
  1. spatial=`stjoint_conv` temporal=`tcn` fusion=`sequential`
  2. spatial=`stjoint_conv` temporal=`tcn` fusion=`sequential`
  3. spatial=`stjoint_conv` temporal=`tcn` fusion=`sequential`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 算子

纯 Tier-1 进化架构 (未用 LLM 合成算子)。

## 超参 (HPO 择优)

```json
{
  "lr": 0.0003143455704703884,
  "weight_decay": 1.3032786543170152e-05,
  "dropout": 0.42897823042355687,
  "batch_size": 32,
  "lr_schedule": "plateau"
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
