# 模型卡: darwin-st/f1eadbb4

- **数据集**: PeMS04
- **排名**: 第 362 名 / 库内 KEEP 实验
- **val MAE**: 19.09
- **val RMSE**: 31.08
- **参数量**: 188.2K
- **实验 id**: 223   **签名**: `f1eadbb448966c430bfeab08eb6af45555283979`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-28T15:26:29.218808+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.29
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`synth_StGraphAttnWithMpsGated_v2` fusion=`cross` 🔬
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`dynamic_gat` temporal=`identity` fusion=`cross`
  4. spatial=`synth_InfoBootstrapResidual` temporal=`synth_StGraphAttnWithMpsGated_v6` fusion=`cross` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_StGraphAttnWithMpsGated_v2`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.32
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v2.py`](operators/synth_StGraphAttnWithMpsGated_v2.py)

### `synth_InfoBootstrapResidual`
- **组合方式**: additive_residual
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.14
- **算子代码**: [`operators/synth_InfoBootstrapResidual.py`](operators/synth_InfoBootstrapResidual.py)

### `synth_StGraphAttnWithMpsGated_v6`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.00
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v6.py`](operators/synth_StGraphAttnWithMpsGated_v6.py)

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
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
