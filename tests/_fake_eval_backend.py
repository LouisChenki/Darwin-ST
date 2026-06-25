"""进程后端测试用的可导入 fake eval 工厂 (CPU, 无 GPU/网络)。

放在独立 helper 模块而非 test 文件: spawn 子进程导入它时只拉最小依赖, 不触 pytest。
EvalSpec.eval_factory = "_fake_eval_backend:make_fake_eval" 即让 worker 自建此 eval_fn,
从而在无 GPU 的 CI 上真实走通 spawn 进程后端的整条回路 (绑卡/回填/兜底 CRASH)。
"""

from __future__ import annotations

import os

from darwin_st.optim.scheduler import EvalResult

# 哨兵: hidden 命中即抛 → 验证 worker 兜底 CRASH (异常绝不逃出进程池)
CRASH_HIDDEN = 999


def make_fake_eval(dataset: str, hpo_cfg=None):
    """工厂签名与 make_eval_fn 一致: (dataset, hpo_cfg) -> eval_fn(genotype, device)。"""

    def eval_fn(genotype, device: str) -> EvalResult:
        if getattr(genotype, "hidden", 0) == CRASH_HIDDEN:
            raise RuntimeError("sentinel crash")
        # 由 signature 派生确定性有限 MAE (可断言); 记录 worker pid + device 以证跨进程 + 绑卡
        sig = genotype.signature()
        mae = 10.0 + (sum(ord(c) for c in sig) % 100) / 10.0
        return EvalResult(
            genotype=genotype, status="OK", mae=mae, rmse=mae + 1.0,
            device=device, wall_seconds=0.0,
            extra={"num_params": 123, "worker_pid": os.getpid(), "seen_device": device},
        )

    return eval_fn
