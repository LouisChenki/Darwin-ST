# 模型卡: darwin-st/4dc1bfb5

- **数据集**: PeMS04
- **排名**: 第 445 名 / 库内 KEEP 实验
- **val MAE**: 19.59
- **val RMSE**: 31.76
- **参数量**: 128.0K
- **实验 id**: 35   **签名**: `4dc1bfb59b2e6f6abc6e7ae6827d0200b3168f84`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-25T03:07:20.008115+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.79
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 3   hidden: 64   邻接: sym
- 时空块:
  0. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  1. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  2. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
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
