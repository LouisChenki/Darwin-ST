# 模型卡: darwin-st/8ce5a86c

- **数据集**: PeMS04
- **排名**: 第 500 名 / 库内 KEEP 实验
- **val MAE**: 22.53
- **val RMSE**: 35.81
- **参数量**: 561.9K
- **实验 id**: 24   **签名**: `8ce5a86c910669ce14162efb7d1737be4f4d237d`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-24T23:53:56.622255+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +4.73
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 128   邻接: rw
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
  "lr": 0.0012863770302477616,
  "weight_decay": 7.882557489139826e-05,
  "dropout": 0.17747813890587916,
  "batch_size": 32,
  "lr_schedule": "plateau"
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
