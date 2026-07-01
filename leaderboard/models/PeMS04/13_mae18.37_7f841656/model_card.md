# 模型卡: darwin-st/7f841656

- **数据集**: PeMS04
- **排名**: 第 13 名 / 库内 KEEP 实验
- **val MAE**: 18.37
- **val RMSE**: —
- **参数量**: 556.8K
- **实验 id**: 502   **签名**: `7f841656e93c09e52e55a373acabcc978fae56bd`
- **run_tag**: exp/sota-24h   **时间**: 2026-06-30T13:36:11.825266+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.57
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 128   邻接: none
- 时空块:
  0. spatial=`identity` temporal=`identity` fusion=`iterative`
  1. spatial=`adaptive` temporal=`gru` fusion=`cross`
  2. spatial=`synth_SequentialRandomGCN` temporal=`identity` fusion=`iterative` 🔬
  3. spatial=`synth_ssm_additive_residual` temporal=`attn` fusion=`cross` 🔬
  4. spatial=`synth_gated_routed_adaptive` temporal=`gru` fusion=`iterative` 🔬
- 身份嵌入: node=True(64) tod=True(96) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_SequentialRandomGCN`
- **组合方式**: sequential
- **源机制 (跨域)**: domain_randomization
- **算子代码**: [`operators/synth_SequentialRandomGCN.py`](operators/synth_SequentialRandomGCN.py)

### `synth_ssm_additive_residual`
- **组合方式**: additive_residual
- **源机制 (跨域)**: state_space_modeling
- **算子代码**: [`operators/synth_ssm_additive_residual.py`](operators/synth_ssm_additive_residual.py)

### `synth_gated_routed_adaptive`
- **组合方式**: gated_routed
- **源机制 (跨域)**: relational_spatial_encoding, masked_multi_modal_conditioning
- **算子代码**: [`operators/synth_gated_routed_adaptive.py`](operators/synth_gated_routed_adaptive.py)

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
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
