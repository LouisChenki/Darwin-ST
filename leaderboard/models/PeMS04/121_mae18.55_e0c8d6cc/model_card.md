# 模型卡: darwin-st/e0c8d6cc

- **数据集**: PeMS04
- **排名**: 第 121 名 / 库内 KEEP 实验
- **val MAE**: 18.55
- **val RMSE**: 30.63
- **参数量**: 204.8K
- **实验 id**: 547   **签名**: `e0c8d6ccd5878a5df321e0cf3c04240ed96f951e`
- **run_tag**: exp/tier2-v1   **时间**: 2026-08-04T03:21:52.666330+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.75
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 6   hidden: 64   邻接: sym
- 时空块:
  0. spatial=`synth_StGraphAttnWithMpsGated_v15` temporal=`synth_StGraphAttnWithMpsGated_v2` fusion=`cross` 🔬
  1. spatial=`synth_StGraphAttnWithMpsGated_v13` temporal=`st_graph_attn` fusion=`cross` 🔬
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`synth_StGraphAttnWithMpsGated_v13` temporal=`synth_StGraphAttnWithMpsGated_v13` fusion=`cross` 🔬
  4. spatial=`synth_InfoBootstrapResidual` temporal=`ssm` fusion=`cross` 🔬
  5. spatial=`synth_StGraphAttnWithMpsGated_v4` temporal=`synth_InfoBootstrapResidual` fusion=`residual` 🔬
- 身份嵌入: node=True(32) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_StGraphAttnWithMpsGated_v15`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.96
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v15.py`](operators/synth_StGraphAttnWithMpsGated_v15.py)

### `synth_StGraphAttnWithMpsGated_v2`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.32
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v2.py`](operators/synth_StGraphAttnWithMpsGated_v2.py)

### `synth_StGraphAttnWithMpsGated_v13`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.87
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v13.py`](operators/synth_StGraphAttnWithMpsGated_v13.py)

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
