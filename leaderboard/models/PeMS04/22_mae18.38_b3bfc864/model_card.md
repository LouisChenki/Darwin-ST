# 模型卡: darwin-st/b3bfc864

- **数据集**: PeMS04
- **排名**: 第 22 名 / 库内 KEEP 实验
- **val MAE**: 18.38
- **val RMSE**: —
- **参数量**: 390.8K
- **实验 id**: 494   **签名**: `b3bfc86411d8c6f10223b54394b81f0f5b21d7bd`
- **run_tag**: exp/sota-24h   **时间**: 2026-06-30T11:34:22.437584+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.58
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 128   邻接: none
- 时空块:
  0. spatial=`identity` temporal=`identity` fusion=`iterative`
  1. spatial=`adaptive` temporal=`gru` fusion=`cross`
  2. spatial=`synth_SequentialRandomGCN` temporal=`identity` fusion=`iterative` 🔬
  3. spatial=`synth_ssm_additive_residual` temporal=`attn` fusion=`cross` 🔬
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

## 超参 (HPO 择优)

```json
{
  "lr": 0.002586308826954624,
  "weight_decay": 1.9204274630632027e-06,
  "dropout": 0.08716917319095449,
  "batch_size": 16,
  "lr_schedule": "cosine",
  "num_heads": 1
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
