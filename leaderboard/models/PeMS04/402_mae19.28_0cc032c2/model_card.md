# 模型卡: darwin-st/0cc032c2

- **数据集**: PeMS04
- **排名**: 第 402 名 / 库内 KEEP 实验
- **val MAE**: 19.28
- **val RMSE**: 31.37
- **参数量**: 175.5K
- **实验 id**: 180   **签名**: `0cc032c2e277e51c4249725ad1e2f0c16d6a9136`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-27T23:04:52.314590+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.48
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 64   邻接: none
- 时空块:
  0. spatial=`gat` temporal=`st_graph_attn` fusion=`cross`
  1. spatial=`adaptive` temporal=`st_graph_attn` fusion=`cross`
  2. spatial=`gwnet_adp` temporal=`st_separable` fusion=`cross`
  3. spatial=`synth_InfoBootstrapResidual` temporal=`stjoint_conv` fusion=`parallel` 🔬
  4. spatial=`dynamic_gat` temporal=`identity` fusion=`residual`
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
