# 模型卡: darwin-st/07f2a42f

- **数据集**: PeMS04
- **排名**: 第 14 名 / 库内 KEEP 实验
- **val MAE**: 18.37
- **val RMSE**: —
- **参数量**: 545.2K
- **实验 id**: 518   **签名**: `07f2a42faffe8ac56c59794ea47941ada33bc6b4`
- **run_tag**: exp/sota-24h   **时间**: 2026-06-30T18:09:05.081028+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +0.57
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 5   hidden: 128   邻接: none
- 时空块:
  0. spatial=`identity` temporal=`identity` fusion=`iterative`
  1. spatial=`adaptive` temporal=`gru` fusion=`cross`
  2. spatial=`synth_SequentialRandomGCN` temporal=`identity` fusion=`iterative` 🔬
  3. spatial=`synth_ssm_additive_residual` temporal=`attn` fusion=`cross` 🔬
  4. spatial=`synth_PotentialResidual` temporal=`identity` fusion=`residual` 🔬
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

### `synth_PotentialResidual`
- **组合方式**: additive_residual
- **源机制 (跨域)**: learnable_potential_field
- **算子代码**: [`operators/synth_PotentialResidual.py`](operators/synth_PotentialResidual.py)

## 超参 (HPO 择优)

```json
{
  "lr": 0.0011535473443601459,
  "weight_decay": 1.2674729295260836e-06,
  "dropout": 0.29354905575658274,
  "batch_size": 16,
  "lr_schedule": "plateau",
  "num_heads": 1
}
```

## 复现

- 架构: [`genotype.json`](genotype.json) → `Genotype.from_dict(...)` → `build_model(...)`
- 超参: [`hparams.json`](hparams.json)
- 合成算子: `operators/*.py` 经 `OperatorRegistry.load_persisted()` 注入后方可构建
