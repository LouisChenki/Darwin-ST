# 模型卡: darwin-st/b3c239e6

- **数据集**: PeMS04
- **排名**: 第 168 名 / 库内 KEEP 实验
- **val MAE**: 18.62
- **val RMSE**: 30.69
- **参数量**: 743.8K
- **实验 id**: 649   **签名**: `b3c239e69c78b2ca3e619f746a2bb0b09a9996b7`
- **run_tag**: exp/tier2-v1   **时间**: 2026-08-06T08:07:51.548644+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.82
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 6   hidden: 128   邻接: rw
- 时空块:
  0. spatial=`synth_StGraphAttnWithMpsGated_v7` temporal=`synth_StGraphAttnWithMpsGated_v14` fusion=`cross` 🔬
  1. spatial=`st_separable` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`dynamic_gat` temporal=`identity` fusion=`cross`
  4. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`cross` 🔬
  5. spatial=`synth_StGraphAttnWithMpsGated_v4` temporal=`synth_StGraphAttnWithMpsGated_v24` fusion=`residual` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_StGraphAttnWithMpsGated_v7`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 46.55
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v7.py`](operators/synth_StGraphAttnWithMpsGated_v7.py)

### `synth_StGraphAttnWithMpsGated_v14`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.82
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v14.py`](operators/synth_StGraphAttnWithMpsGated_v14.py)

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

### `synth_StGraphAttnWithMpsGated_v24`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.96
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v24.py`](operators/synth_StGraphAttnWithMpsGated_v24.py)

## 超参 (HPO 择优)

```json
{
  "lr": 0.0011535473443601459,
  "weight_decay": 1.2674729295260836e-06,
  "dropout": 0.2935490557565828,
  "batch_size": 16,
  "lr_schedule": "plateau",
  "num_heads": 1
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
