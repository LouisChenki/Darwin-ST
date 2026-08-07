# 模型卡: darwin-st/e5e8c993

- **数据集**: PeMS04
- **排名**: 第 372 名 / 库内 KEEP 实验
- **val MAE**: 19.14
- **val RMSE**: 31.22
- **参数量**: 204.7K
- **实验 id**: 71   **签名**: `e5e8c99338554e5f9cfcce44c8c56803439e1715`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-25T17:33:20.750407+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.34
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
  "lr": 0.0012229259598449068,
  "weight_decay": 1.7055210796748734e-06,
  "dropout": 0.37464990198758563,
  "batch_size": 16,
  "lr_schedule": "plateau",
  "num_heads": 1
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
