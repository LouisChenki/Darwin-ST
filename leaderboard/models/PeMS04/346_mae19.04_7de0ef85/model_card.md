# 模型卡: darwin-st/7de0ef85

- **数据集**: PeMS04
- **排名**: 第 346 名 / 库内 KEEP 实验
- **val MAE**: 19.04
- **val RMSE**: 31.18
- **参数量**: 212.8K
- **实验 id**: 87   **签名**: `7de0ef85a97523822ba91dfd6940d3f75b9e91ef`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-26T02:25:47.972437+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.24
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  2. spatial=`diffusion` temporal=`stjoint_conv` fusion=`parallel`
  3. spatial=`dynamic_gat` temporal=`series_decomp_attn` fusion=`residual`
- 身份嵌入: node=True(64) tod=True(64) dow=True(32)

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
