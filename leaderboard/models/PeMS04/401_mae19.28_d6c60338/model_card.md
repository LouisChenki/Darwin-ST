# 模型卡: darwin-st/d6c60338

- **数据集**: PeMS04
- **排名**: 第 401 名 / 库内 KEEP 实验
- **val MAE**: 19.28
- **val RMSE**: 31.46
- **参数量**: 561.9K
- **实验 id**: 12   **签名**: `d6c603380fabf7a785ddf0d27dcdc9165b7d2a77`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-24T18:53:13.983186+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.48
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 128   邻接: none
- 时空块:
  0. spatial=`gcn` temporal=`gru` fusion=`iterative`
  1. spatial=`gcn` temporal=`gru` fusion=`iterative`
  2. spatial=`gcn` temporal=`gru` fusion=`iterative`
  3. spatial=`gcn` temporal=`gru` fusion=`iterative`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 算子

纯 Tier-1 进化架构 (未用 LLM 合成算子)。

## 超参 (HPO 择优)

```json
{
  "lr": 0.007674062091126046,
  "weight_decay": 2.8109187663363917e-06,
  "dropout": 0.25073386081454896,
  "batch_size": 64,
  "lr_schedule": "plateau"
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
