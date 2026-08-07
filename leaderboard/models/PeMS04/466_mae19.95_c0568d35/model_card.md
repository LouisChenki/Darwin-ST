# 模型卡: darwin-st/c0568d35

- **数据集**: PeMS04
- **排名**: 第 466 名 / 库内 KEEP 实验
- **val MAE**: 19.95
- **val RMSE**: 32.16
- **参数量**: 149.1K
- **实验 id**: 95   **签名**: `c0568d354c0925edf8e627ba804634d3076ed6c3`
- **run_tag**: exp/tier2-v1   **时间**: 2026-07-26T08:14:53.642185+00:00
- **vs SOTA** (STD-MAE 17.80): 距 SOTA 差 +2.15
- **评测协议**: masked metric; MAE averaged over 12 horizons; split 6/2/2

## 架构

- 深度 (blocks): 4   hidden: 64   邻接: sym
- 时空块:
  0. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  1. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
  2. spatial=`gat` temporal=`synth_ParallelMinimalSufficiencyAttn` fusion=`residual` 🔬
  3. spatial=`gat` temporal=`series_decomp_attn` fusion=`residual`
- 身份嵌入: node=True(64) tod=True(64) dow=True(64)

## 🔬 跨域合成算子 (LLM 创造)

本模型用到以下由 LLM 跨域类比合成、经验证门通过的算子:

### `synth_ParallelMinimalSufficiencyAttn`
- **组合方式**: parallel
- **源机制 (跨域)**: st_graph_attn, minimal_predictive_sufficiency
- **算子代码**: [`operators/synth_ParallelMinimalSufficiencyAttn.py`](operators/synth_ParallelMinimalSufficiencyAttn.py)

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
