import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v18(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and synth_StGraphAttnWithMpsGated_v7.
    
    V18 builds on v10's successful gate‑on‑input design and strengthens the gate network
    by adding an extra LayerNorm after the GELU activation, giving a smoother training signal.
    The rest (ST‑attn baseline, MPS bottleneck, gate‑controlled residual) is kept identical to v10.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio-temporal graph attention: linear + graph aggregation + ReLU
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: richer MLP on input x, with GELU and an extra LayerNorm for stability
        gate_hidden = max(1, channels // 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, gate_hidden),
            nn.GELU(),
            nn.LayerNorm(gate_hidden),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Residual scale, zero-initialized for stable early training
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

        # 3. Dynamic gate from raw input (v18: wider MLP with extra LayerNorm)
        gate = self.gate_net(x)                   # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                          # [B, T, N, C]
        new_branch = (1.0 - gate) * diff          # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch         # [B, T, N, C]

        return out
