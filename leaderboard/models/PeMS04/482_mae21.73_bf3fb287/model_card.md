# 模型卡: darwin-st/bf3fb287

- **数据集**: PeMS04
- **排名**: 第 482 名 / 库内 KEEP 实验
- **val MAE**: 21.73
- **val RMSE**: 34.56
- **参数量**: 217.8K
- **实验 id**: 324   **签名**: `bf3fb287631df8599fc424121e5abd7732bbca16`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-30T17:40:02.564844+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +3.93
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 7   hidden: 64   邻接: rw
- 时空块:
  0. spatial=`synth_StGraphAttnWithMpsGated_v7` temporal=`identity` fusion=`cross` 🔬
  1. spatial=`st_separable` temporal=`st_separable` fusion=`cross`
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`dynamic_gat` temporal=`identity` fusion=`cross`
  4. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`cross` 🔬
  5. spatial=`synth_StGraphAttnWithMpsGated_v4` temporal=`synth_InfoBootstrapResidual` fusion=`residual` 🔬
  6. spatial=`synth_StGraphAttnWithMpsGated_v7` temporal=`synth_StGraphAttnWithMpsGated_v5` fusion=`sequential_ts` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_StGraphAttnWithMpsGated_v7`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 46.55
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v7.py`](operators/synth_StGraphAttnWithMpsGated_v7.py)

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

### `synth_StGraphAttnWithMpsGated_v5`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.46
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v5.py`](operators/synth_StGraphAttnWithMpsGated_v5.py)

## 超参 (HPO 择优)

```json
{
  "lr": 0.0021203661896609894,
  "weight_decay": 5.750084934481088e-05,
  "dropout": 0.20180992846474835,
  "batch_size": 64,
  "lr_schedule": "plateau"
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 权重存档: [`model.pt`](model.pt) (训练最优 val-MAE 步的 state_dict, 附 sidecar 元数据 `.meta.json`)
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
