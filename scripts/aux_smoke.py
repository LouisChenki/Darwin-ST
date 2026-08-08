"""B7 辅助任务端到端冒烟 (真实 DeepSeek 合成 → 防泄漏验证门 → 挂载 → 真实 PeMS04 训练对照)。

一条命令验证 B7 整条链:
  1) 用 masked_autoencoding + decoupled_st_masked_pretraining 两张机制卡组 FusionRequest
  2) 真实 DeepSeek (默认 flash) plan-then-code 合成一个辅助任务 (LLM 直生代码, 不走 aider 简化冒烟)
  3) 防泄漏验证门 (静态+动态对抗) 裁决
  4) 过门则注册进临时 synth_dir, 构建带 aux 的模型, 在真实 PeMS04 上同架构同种子
     对照训练: aux_lambda=0.1 vs 0.0 (aux 前向但零权重), 看主任务 val MAE 差异

用法 (服务器, source darwin-secrets.env 后):
    EPOCHS=4 AUX_LAMBDA=0.1 DEVICE=cuda:0 python -u scripts/aux_smoke.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from darwin_st.data import prepare as P
from darwin_st.data.protocol import get_profile
from darwin_st.knowledge.qc import mechanisms_from_cards
from darwin_st.creation.contracts import FusionRequest
from darwin_st.creation.llm import OpenAICompatLLM
from darwin_st.creation.synthesizer import OperatorSynthesizer, SynthesisConfig
from darwin_st.creation.registry import OperatorRegistry
from darwin_st.search.genotype import Genotype, STBlock
from darwin_st.optim.train import train_one, limit_cpu_threads


def main():
    epochs = int(os.environ.get("EPOCHS", "4"))
    lam = float(os.environ.get("AUX_LAMBDA", "0.1"))
    device = os.environ.get("DEVICE", "cuda:0")
    dataset = os.environ.get("DATASET", "PeMS04")
    smoke_dir = os.path.join(os.environ.get("DARWIN_ST_CACHE", "."), "aux_smoke_ops")
    os.makedirs(smoke_dir, exist_ok=True)

    # 1) 机制卡
    cards_path = os.path.join(os.path.dirname(__file__), "..", "src", "darwin_st",
                              "knowledge", "data", "mechanism_cards.json")
    cards = json.load(open(cards_path, encoding="utf-8"))["mechanisms"]
    mechs = {m.name: m for m in mechanisms_from_cards(cards)}
    picked = [mechs[n] for n in ("masked_autoencoding", "decoupled_st_masked_pretraining")
              if n in mechs]
    print(f"[1] 机制卡: {[m.name for m in picked]}")

    # 2) 合成
    llm = OpenAICompatLLM()  # DEEPSEEK_MODEL, 默认 deepseek-v4-flash
    synth = OperatorSynthesizer(llm, SynthesisConfig(max_retries=2, temperature=0.6))
    req = FusionRequest(
        bottleneck="模型欠拟合, 需要更强的时空表示能力 (辅助任务提升表征质量)",
        target_preconditions=["redundant_structure", "label_scarcity"],
        mechanisms=picked,
        baseline_operator="gcn")
    print(f"[2] 合成中 (模型={llm.model}) ...")
    res = synth.synthesize_aux_task(req)
    if not res.success or res.operator is None:
        print(f"❌ 合成失败: {res.last_error}")
        sys.exit(1)
    op = res.operator
    print(f"✅ 合成成功 ({res.attempts} 次尝试): {op.name}  门={op.validation_gate}")
    print(f"   plan: {op.plan.mask_pattern} ratio={op.plan.mask_ratio} "
          f"target={op.plan.recon_target} loss={op.plan.loss_form} decoder={op.plan.decoder_form}")
    print(f"   rationale: {op.plan.rationale[:200]}")

    # 3) 注册 + 构建带 aux 的模型
    registry = OperatorRegistry(persist_dir=smoke_dir)
    reg_name = registry.register_aux(op)
    print(f"[3] 注册为 {reg_name}")

    profile = get_profile(dataset)
    data_dir = P.prepare_dataset(dataset)
    adj = P.load_adj(data_dir)
    geno = Genotype(blocks=[STBlock(spatial_op="gcn", temporal_op="tcn")],
                    hidden=64, aux_op=reg_name)
    limit_cpu_threads(1)

    # 4) 对照训练: 同架构同种子, λ=0.1 vs 0.0
    hps_base = {"batch_size": 64, "lr": 1e-3, "weight_decay": 1e-4,
                "lr_schedule": "none", "loss": "mae"}
    for lam_i in (lam, 0.0):
        torch.manual_seed(42)
        hps = dict(hps_base, aux_lambda=lam_i)
        best, trace = train_one(geno, hps, data_dir, profile, adj, device,
                                max_epochs=epochs, time_budget_s=900)
        print(f"[4] λ={lam_i}: best val MAE={best:.4f} "
              f"(epochs={trace.n_epochs_run}, aux_op={trace.aux_op}, λ={trace.aux_lambda})")
    print("=== aux 冒烟完成 ===")


if __name__ == "__main__":
    main()
