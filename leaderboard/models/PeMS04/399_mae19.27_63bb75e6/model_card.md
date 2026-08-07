# 模型卡: darwin-st/63bb75e6

- **数据集**: PeMS04
- **排名**: 第 399 名 / 库内 KEEP 实验
- **val MAE**: 19.27
- **val RMSE**: 31.41
- **参数量**: 215.1K
- **实验 id**: 91   **签名**: `63bb75e6d4237a9d21fc505181fd2c1b6c162ce0`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-26T05:02:23.970610+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.47
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: sym
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  2. spatial=`diffusion` temporal=`stjoint_conv` fusion=`parallel`
  3. spatial=`dynamic_gat` temporal=`series_decomp_attn` fusion=`residual`
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
