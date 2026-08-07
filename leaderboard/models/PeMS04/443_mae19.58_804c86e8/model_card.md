# 模型卡: darwin-st/804c86e8

- **数据集**: PeMS04
- **排名**: 第 443 名 / 库内 KEEP 实验
- **val MAE**: 19.58
- **val RMSE**: 31.74
- **参数量**: 807.9K
- **实验 id**: 15   **签名**: `804c86e8289045d2e57e9c719b3a71faf3da99e3`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-24T20:29:10.308389+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.78
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 2   hidden: 192   邻接: rw
- 时空块:
  0. spatial=`mixhop` temporal=`gru` fusion=`residual`
  1. spatial=`mixhop` temporal=`gru` fusion=`residual`
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
