"""Stage2 新算子真训练冒烟: 每个新算子起一个 genotype, 在真实 PeMS04 上训练几 epoch,
确认训练出**有限** masked MAE (不 NaN 不发散), 尤其 SSM 顺序扫描在 GPU 上稳定。

用法:
    DATASET=PeMS04 EPOCHS=5 DARWIN_ST_CACHE=/root/autodl-tmp/darwin-st-cache \
      python scripts/stage2_newop_smoke.py
"""

from __future__ import annotations

import os
import time

import torch

from darwin_st.data import prepare as P
from darwin_st.data.metrics import masked_mae
from darwin_st.data.protocol import get_profile
from darwin_st.search.builder import build_model, count_params
from darwin_st.search.genotype import Genotype, STBlock

# 每个新算子 → 放对槽的 genotype (spatial/temporal → S/T 槽; joint → joint 槽)
NEW_OP_GENOS = {
    "mixhop":             lambda: Genotype(blocks=[STBlock("mixhop", "tcn", "residual")], hidden=64),
    "gwnet_adp":          lambda: Genotype(blocks=[STBlock("gwnet_adp", "tcn", "residual")], hidden=64),
    "multiscale_tcn":     lambda: Genotype(blocks=[STBlock("gcn", "multiscale_tcn", "sequential")], hidden=64),
    "series_decomp_attn": lambda: Genotype(blocks=[STBlock("gcn", "series_decomp_attn", "sequential")], hidden=64),
    "ssm":                lambda: Genotype(blocks=[STBlock("gcn", "ssm", "sequential")], hidden=64),
    "dynamic_gat":        lambda: Genotype(blocks=[STBlock("identity", "identity", joint_op="dynamic_gat")], hidden=64),
    "stjoint_conv":       lambda: Genotype(blocks=[STBlock("identity", "identity", joint_op="stjoint_conv")], hidden=64),
}


def train_one(name, make_geno, data_dir, prof, adj, device, epochs):
    geno = make_geno()
    geno.validate()
    model = build_model(geno, num_nodes=prof.num_nodes, in_channels=prof.num_channels,
                        seq_len_in=prof.seq_len_in, seq_len_out=prof.seq_len_out, adj=adj).to(device)
    n_params = count_params(model)
    train_loader = P.load_split(data_dir, "train", batch_size=64, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)

    t0 = time.time()
    last_val = float("nan")
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = masked_mae(model(x), y)
            if torch.isnan(loss) or torch.isinf(loss):
                return {"op": name, "status": "NAN", "params": n_params,
                        "val_mae": float("nan"), "epoch": ep, "secs": time.time() - t0}
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); nb += 1
        met = P.evaluate(model, data_dir, "val", batch_size=64, device=device, null_val=prof.null_val)
        last_val = met["mae"]
    finite = bool(torch.isfinite(torch.tensor(last_val)))
    return {"op": name, "status": "OK" if finite else "BAD", "params": n_params,
            "val_mae": last_val, "epoch": epochs, "secs": time.time() - t0}


def main():
    ds = os.environ.get("DATASET", "PeMS04")
    epochs = int(os.environ.get("EPOCHS", "5"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prof = get_profile(ds)
    print(f"=== Stage2 新算子真训练冒烟: {ds} on {device}, {epochs} epochs/op ===", flush=True)
    data_dir = P.prepare_dataset(ds)
    adj = P.load_adj(data_dir)

    results = []
    for name, make_geno in NEW_OP_GENOS.items():
        print(f"\n--- 训练 {name} ---", flush=True)
        try:
            r = train_one(name, make_geno, data_dir, prof, adj, device, epochs)
        except Exception as e:
            r = {"op": name, "status": "ERROR", "params": 0, "val_mae": float("nan"),
                 "epoch": 0, "secs": 0.0, "err": str(e)[:200]}
        results.append(r)
        print(f"    {name}: status={r['status']} val_MAE={r['val_mae']:.3f} "
              f"params={r['params']:,} ({r['secs']:.0f}s)"
              + (f" ERR={r.get('err')}" if r.get("err") else ""), flush=True)

    print("\n=== 汇总 ===", flush=True)
    ok = sum(1 for r in results if r["status"] == "OK")
    for r in results:
        print(f"  {r['op']:20s} {r['status']:6s} val_MAE={r['val_mae']:.3f}", flush=True)
    print(f"\n{ok}/{len(results)} 新算子训练出有限 MAE", flush=True)
    if ok == len(results):
        print("ALL_NEWOPS_OK", flush=True)
    else:
        print("SOME_NEWOPS_FAILED", flush=True)


if __name__ == "__main__":
    main()
