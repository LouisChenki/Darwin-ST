# 模型卡: darwin-st/7835fca0

- **数据集**: PeMS04
- **排名**: 第 432 名 / 库内 KEEP 实验
- **val MAE**: 19.48
- **val RMSE**: 31.59
- **参数量**: 201.1K
- **实验 id**: 106   **签名**: `7835fca062b55c62f0a7096bb3851860f842c7cc`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-26T14:16:32.126717+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.68
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  3. spatial=`diffusion` temporal=`stjoint_conv` fusion=`parallel`
  4. spatial=`dynamic_gat` temporal=`identity` fusion=`residual`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 算子

纯 Tier-1 进化架构 (未用 LLM 合成算子)。

## 超参 (HPO 择优)

```json
{
  "lr": 0.005127791352880562,
  "weight_decay": 2.604248616872354e-06,
  "dropout": 0.47931035030916913,
  "batch_size": 64,
  "lr_schedule": "plateau",
  "num_heads": 2
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
