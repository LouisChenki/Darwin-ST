# 模型卡: darwin-st/e35741c6

- **数据集**: PeMS04
- **排名**: 第 360 名 / 库内 KEEP 实验
- **val MAE**: 19.08
- **val RMSE**: 31.17
- **参数量**: 144.9K
- **实验 id**: 163   **签名**: `e35741c66b63473947054243066d16d9d2cb50df`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-27T16:13:42.134285+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +1.28
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: sym
- 时空块:
  0. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  1. spatial=`gat` temporal=`synth_StGraphAttnWithMpsGated_v4` fusion=`residual` 🔬
  2. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  3. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

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
