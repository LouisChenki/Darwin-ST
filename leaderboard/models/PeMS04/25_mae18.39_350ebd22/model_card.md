# 模型卡: darwin-st/350ebd22

- **数据集**: PeMS04
- **排名**: 第 25 名 / 库内 KEEP 实验
- **val MAE**: 18.39
- **val RMSE**: —
- **参数量**: 752.2K
- **实验 id**: 460   **签名**: `350ebd2297c198e782e4888a741efc17ba507b6c`
- **run_tag**: exp/sota-24h   **时间**: 2026-06-30T02:07:30.609356+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.59
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 128   邻接: none
- 时空块:
  0. spatial=`synth_DeformTemporalPatchParallelFusion` temporal=`gru` fusion=`sequential` 🔬
  1. spatial=`adaptive` temporal=`gru` fusion=`cross`
  2. spatial=`synth_SequentialRandomGCN` temporal=`identity` fusion=`iterative` 🔬
  3. spatial=`synth_DataDistillParallel` temporal=`attn` fusion=`cross` 🔬
  4. spatial=`synth_ParResDec` temporal=`identity` fusion=`cross` 🔬
- 身份嵌入: node=True(64) tod=True(96) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_DeformTemporalPatchParallelFusion`
- **组合方式**: parallel
- **源机制 (跨域)**: deformable_temporal_patching
- **算子代码**: [`operators/synth_DeformTemporalPatchParallelFusion.py`](operators/synth_DeformTemporalPatchParallelFusion.py)

### `synth_SequentialRandomGCN`
- **组合方式**: sequential
- **源机制 (跨域)**: domain_randomization
- **算子代码**: [`operators/synth_SequentialRandomGCN.py`](operators/synth_SequentialRandomGCN.py)

### `synth_DataDistillParallel`
- **组合方式**: parallel
- **源机制 (跨域)**: dataset_distillation
- **算子代码**: [`operators/synth_DataDistillParallel.py`](operators/synth_DataDistillParallel.py)

### `synth_ParResDec`
- **组合方式**: parallel
- **源机制 (跨域)**: residual_decomposition
- **算子代码**: [`operators/synth_ParResDec.py`](operators/synth_ParResDec.py)

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
