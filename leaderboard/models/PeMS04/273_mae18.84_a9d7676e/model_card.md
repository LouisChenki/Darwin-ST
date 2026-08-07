# 模型卡: darwin-st/a9d7676e

- **数据集**: PeMS04
- **排名**: 第 273 名 / 库内 KEEP 实验
- **val MAE**: 18.84
- **val RMSE**: 30.94
- **参数量**: 669.3K
- **实验 id**: 245   **签名**: `a9d7676e72330e3460b9c1d75da81a5501327a33`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-29T07:20:46.468958+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.04
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 6   hidden: 128   邻接: rw
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`adaptive` temporal=`st_separable` fusion=`cross`
  3. spatial=`dynamic_gat` temporal=`identity` fusion=`cross`
  4. spatial=`synth_InfoBootstrapResidual` temporal=`synth_StGraphAttnWithMpsGated_v6` fusion=`cross` 🔬
  5. spatial=`synth_StGraphAttnWithMpsGated_v4` temporal=`synth_InfoBootstrapResidual` fusion=`residual` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

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

### `synth_StGraphAttnWithMpsGated_v4`
- **组合方式**: gated_routed
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 18.94
- **算子代码**: [`operators/synth_StGraphAttnWithMpsGated_v4.py`](operators/synth_StGraphAttnWithMpsGated_v4.py)

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
