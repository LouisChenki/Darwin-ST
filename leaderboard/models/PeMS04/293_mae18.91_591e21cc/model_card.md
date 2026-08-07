# 模型卡: darwin-st/591e21cc

- **数据集**: PeMS04
- **排名**: 第 293 名 / 库内 KEEP 实验
- **val MAE**: 18.91
- **val RMSE**: 31.00
- **参数量**: 174.9K
- **实验 id**: 233   **签名**: `591e21ccc881a2c1f5f2df7835e992e575fe32fe`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-28T19:13:39.574139+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.11
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`st_separable` temporal=`synth_StGraphAttnWithMpsGated_v5` fusion=`cross` 🔬
- 身份嵌入: node=True(64) tod=True(32) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_StGraphAttnWithMpsGated_v5`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.46
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v5.py`](operators/synth_StGraphAttnWithMpsGated_v5.py)

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
