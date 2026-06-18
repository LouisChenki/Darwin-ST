"""P2 基线冒烟脚本: 用 builder 编译一个 genotype, 在真实数据上训练若干 epoch 并评测。

这是 P2-c 的端到端验证 (非最终 train.py —— 那将由 orchestrator 阶段产出)。
目的: 证明 genotype→builder→真实数据→GPU 训练→masked 评测整条链贯通, 且模型能逼近基线。

用法:
    DATASET=PeMS04 GITHUB_MIRROR=... DARWIN_ST_CACHE=... python scripts/baseline_smoke.py
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from darwin_st.data import prepare as P
from darwin_st.data.metrics import masked_mae
from darwin_st.data.protocol import get_profile
from darwin_st.baseline_registry import get_sota
from darwin_st.search.builder import build_model, count_params
from darwin_st.search.genotype import random_genotype


def main():
    ds = os.environ.get("DATASET", "PeMS04")
    epochs = int(os.environ.get("EPOCHS", "15"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prof = get_profile(ds)

    print(f"=== P2 基线冒烟: {ds} on {device} ===")
    data_dir = P.prepare_dataset(ds)
    adj = P.load_adj(data_dir)
    print(f"邻接矩阵: {None if adj is None else adj.shape}")

    # 一个合理的初始 genotype: adaptive 图 + 门控时序 + 节点嵌入(默认开)
    geno = random_genotype(depth=2, spatial="adaptive", temporal="tcn",
                           fusion="residual", hidden=64)
    model = build_model(geno, num_nodes=prof.num_nodes, in_channels=prof.num_channels,
                        seq_len_in=prof.seq_len_in, seq_len_out=prof.seq_len_out, adj=adj).to(device)
    print(f"模型参数量: {count_params(model):,}")

    train_loader = P.load_split(data_dir, "train", batch_size=64, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    start = time.time()
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = masked_mae(model(x), y)   # 归一化尺度训练
            if torch.isnan(loss):
                print("NaN! 中止"); return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪防爆
            opt.step()
            tot += loss.item(); nb += 1
        # 用 evaluate 在真实尺度上算 masked 指标
        met = P.evaluate(model, data_dir, "val", batch_size=64, device=device, null_val=prof.null_val)
        print(f"epoch {ep:2d}  train_loss(norm)={tot/nb:.4f}  val_MAE={met['mae']:.3f}  "
              f"RMSE={met['rmse']:.3f}  ({time.time()-start:.0f}s)")

    # 最终对照 SOTA / 基线
    final = P.evaluate(model, data_dir, "test", batch_size=64, device=device, null_val=prof.null_val)
    sota_name, sota_mae = get_sota(ds)
    print(f"\n=== 最终 test: MAE={final['mae']:.3f} RMSE={final['rmse']:.3f} ===")
    print(f"SOTA({sota_name})={sota_mae} | 这是 2-block 冒烟基线, 目标先验证管道而非精度")


if __name__ == "__main__":
    main()
