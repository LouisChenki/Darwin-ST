# 模型卡: darwin-st/d5883672

- **数据集**: PeMS04
- **排名**: 第 430 名 / 库内 KEEP 实验
- **val MAE**: 19.47
- **val RMSE**: 31.60
- **参数量**: 202.7K
- **实验 id**: 57   **签名**: `d588367231235435e22cce4cc1ad81ab3e2cf8de`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-25T10:37:59.772350+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.67
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: none
- 时空块:
  0. spatial=`gat` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  3. spatial=`mixhop` temporal=`stjoint_conv` fusion=`parallel`
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
