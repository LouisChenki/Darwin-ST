# 模型卡: darwin-st/a029d7c7

- **数据集**: PeMS04
- **排名**: 第 10 名 / 库内 KEEP 实验
- **val MAE**: 18.36
- **val RMSE**: —
- **参数量**: 686.0K
- **实验 id**: 463   **签名**: `a029d7c7c24189995bcfedb2994614365849e35c`
- **run_tag**: exp/sota-24h   **时间**: 2026-06-30T03:32:56.705093+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.56
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 128   邻接: rw
- 时空块:
  0. spatial=`identity` temporal=`gru` fusion=`sequential`
  1. spatial=`adaptive` temporal=`gru` fusion=`cross`
  2. spatial=`synth_SequentialRandomGCN` temporal=`identity` fusion=`iterative` 🔬
  3. spatial=`synth_DataDistillParallel` temporal=`attn` fusion=`cross` 🔬
- 身份嵌入: node=True(64) tod=True(96) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_SequentialRandomGCN`
- **组合方式**: sequential
- **源机制 (跨域)**: domain_randomization
- **算子代码**: [`operators/synth_SequentialRandomGCN.py`](operators/synth_SequentialRandomGCN.py)

### `synth_DataDistillParallel`
- **组合方式**: parallel
- **源机制 (跨域)**: dataset_distillation
- **算子代码**: [`operators/synth_DataDistillParallel.py`](operators/synth_DataDistillParallel.py)

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
