import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v16(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and synth_StGraphAttnWithMpsGated_v7.

    V16 retains the successful gate-on-input philosophy from v10 but combines
    raw input x with post‑attention features h to form a joint gating signal.
    Gate: LayerNorm([x;h]) -> Linear -> GELU -> Linear -> Sigmoid, with hidden
    width channels//2.  All other components (ST‑Attn, MPS bottleneck) are unchanged.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio‑temporal graph attention: linear + graph aggregation + ReLU
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: richer MLP on concatenated [x, h] for joint information
        gate_hidden = max(1, channels // 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels * 2),           # input: [x, h] stacked along last dim
            nn.Linear(channels * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Residual scale, zero‑initialized for stable early training
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph‑attention transform
        h = self.lin(x)                # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)       # [B*T, N, C]
            adj_3d = adj.unsqueeze(0)             # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)  # [B*T, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                           # [B, T, N, C]

        # 2. MPS compressed branch (from activated baseline)
        h_mps = self.compress(h)                  # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                # [B, T, N, C]

        # 3. Joint gate from both raw input and transformed features
        joint = torch.cat([x, h], dim=-1)         # [B, T, N, 2*C]
        gate = self.gate_net(joint)               # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                          # [B, T, N, C]
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch         # [B, T, N, C]

        return out
