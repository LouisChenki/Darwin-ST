# 模型卡: darwin-st/7d3e06c1

- **数据集**: PeMS04
- **排名**: 第 350 名 / 库内 KEEP 实验
- **val MAE**: 19.06
- **val RMSE**: 31.19
- **参数量**: 228.3K
- **实验 id**: 196   **签名**: `7d3e06c15a1ae591a14cfc59f6f3829992a7a209`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-28T06:48:46.557397+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.26
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 6   hidden: 64   邻接: none
- 时空块:
  0. spatial=`st_graph_attn` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`st_separable` temporal=`ssm` fusion=`cross`
  2. spatial=`st_graph_attn` temporal=`st_separable` fusion=`cross`
  3. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`parallel` 🔬
  4. spatial=`dynamic_gat` temporal=`identity` fusion=`residual`
  5. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`cross` 🔬
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_InfoBootstrapResidual`
- **组合方式**: additive_residual
- **源机制 (跨域)**: minimal_predictive_sufficiency
- **该算子实测 MAE**: 19.14
- **算子代码**: [`operators/synth_InfoBootstrapResidual.py`](operators/synth_InfoBootstrapResidual.py)

## 超参 (HPO 择优)

```json
{
  "lr": 0.00203566698341695,
  "weight_decay": 3.2394974086641416e-06,
  "dropout": 0.12822711564090358,
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
