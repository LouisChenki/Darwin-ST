# 模型卡: darwin-st/653dc51b

- **数据集**: PeMS04
- **排名**: 第 256 名 / 库内 KEEP 实验
- **val MAE**: 18.81
- **val RMSE**: 30.95
- **参数量**: 202.5K
- **实验 id**: 165   **签名**: `653dc51b2a3c132a84cd2fdbb85e11dc08a850d0`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-27T17:12:36.819746+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.01
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  3. spatial=`synth_st_graph_attn_with_vib_compression` temporal=`identity` fusion=`residual` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_st_graph_attn_with_vib_compression`
- **组合方式**: sequential
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 50.89
- **算子代码**: [`operators/synth_st_graph_attn_with_vib_compression.py`](operators/synth_st_graph_attn_with_vib_compression.py)

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
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
