# 模型卡: darwin-st/1efae164

- **数据集**: PeMS04
- **排名**: 第 497 名 / 库内 KEEP 实验
- **val MAE**: 22.42
- **val RMSE**: 35.56
- **参数量**: 284.7K
- **实验 id**: 598   **签名**: `1efae164d88f9a5a7ff7d8fdf1fef81828da084b`
- **run_tag**: exp/tier2-v1   **时间**: 2026-08-05T04:18:45.776473+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +4.62
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 8   hidden: 64   邻接: rw
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`dynamic_gat` temporal=`identity` fusion=`cross`
  4. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`cross` 🔬
  5. spatial=`synth_StGraphAttnWithMpsGated_v4` temporal=`synth_InfoBootstrapResidual` fusion=`parallel` 🔬
  6. spatial=`synth_StGraphAttnWithMpsGated_v25` temporal=`synth_StGraphAttnWithMpsGated_v25` fusion=`iterative` 🔬
  7. spatial=`synth_StGraphAttnWithMpsGated_v21` temporal=`synth_StGraphAttnWithMpsGated_v6` fusion=`residual` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_InfoBootstrapResidual`
- **组合方式**: additive_residual
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.14
- **算子代码**: [`operators/synth_InfoBootstrapResidual.py`](operators/synth_InfoBootstrapResidual.py)

### `synth_StGraphAttnWithMpsGated_v4`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.94
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v4.py`](operators/synth_StGraphAttnWithMpsGated_v4.py)

### `synth_StGraphAttnWithMpsGated_v25`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.60
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v25.py`](operators/synth_StGraphAttnWithMpsGated_v25.py)

### `synth_StGraphAttnWithMpsGated_v21`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.83
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v21.py`](operators/synth_StGraphAttnWithMpsGated_v21.py)

### `synth_StGraphAttnWithMpsGated_v6`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.00
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v6.py`](operators/synth_StGraphAttnWithMpsGated_v6.py)

## 超参 (HPO 择优)

```json
{
  "lr": 0.0012520653814999472,
  "weight_decay": 0.0001398196140899405,
  "dropout": 0.30138168803582194,
  "batch_size": 64,
  "lr_schedule": "plateau",
  "num_heads": 2
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
