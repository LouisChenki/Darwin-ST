# 模型卡: darwin-st/a57f518d

- **数据集**: PeMS04
- **排名**: 第 433 名 / 库内 KEEP 实验
- **val MAE**: 19.50
- **val RMSE**: 31.70
- **参数量**: 132.4K
- **实验 id**: 5   **签名**: `a57f518d7e471668f4fbf9fc2de93c618cd9d745`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-24T15:52:08.333848+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.70
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: none
- 时空块:
  0. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  1. spatial=`gat` temporal=`identity` fusion=`residual`
  2. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  3. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 算子

纯 Tier-1 进化架构 (未用 LLM 合成算子)。

## 超参 (HPO 择优)

```json
{
  "lr": 0.007098936257405904,
  "weight_decay": 1.6334587611069488e-06,
  "dropout": 0.043564649850770354,
  "batch_size": 32,
  "lr_schedule": "cosine",
  "num_heads": 2
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
