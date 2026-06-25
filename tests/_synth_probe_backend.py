"""Tier-2 算子在进程后端被 Tier-1 评测的 fake eval (CPU, 无 GPU/数据).

验证最险的衔接点③: spawn worker 的 SPATIAL_OPS 全局**不含**主进程后注册的 synth_xxx,
worker 必须靠 EvalSpec.synth_persist_dir → load_persisted() 从盘补回, 否则 build 解析
genotype 里的 synth_xxx 会 KeyError → 整条 Tier-1↔Tier-2 闭环在多进程下断裂。

此 eval_fn 不训练, 只做关键一步: 检查 genotype 用到的 spatial_op 是否在【本 worker 进程的】
SPATIAL_OPS 里可解析, 并真正实例化它 (build_spatial_op 路径)。解析到 → OK; 否则 CRASH。
"""

from __future__ import annotations

import os

from darwin_st.optim.scheduler import EvalResult


def make_synth_probe_eval(dataset: str, hpo_cfg=None):
    def eval_fn(genotype, device: str) -> EvalResult:
        from darwin_st.search import operators as ops_mod

        op_name = genotype.blocks[0].spatial_op
        present = op_name in ops_mod.SPATIAL_OPS
        if not present:
            # 衔接断裂: worker 没拿到 synth 算子 (load_persisted 没生效)
            raise KeyError(f"{op_name} not in worker SPATIAL_OPS (load_persisted failed)")
        # 真正实例化一次 (build_spatial_op 用 factory(dim, num_nodes=...))
        factory = ops_mod.SPATIAL_OPS[op_name]
        mod = factory(8, num_nodes=10)
        n_params = sum(p.numel() for p in mod.parameters())
        return EvalResult(
            genotype=genotype, status="OK", mae=15.0, rmse=16.0,
            device=device, wall_seconds=0.0,
            extra={"num_params": n_params, "worker_pid": os.getpid(),
                   "resolved_synth": op_name, "synth_in_worker": present},
        )

    return eval_fn
