import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v21(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and synth_StGraphAttnWithMpsGated_v7.
    
    V21 inherits the gate‑on‑input design (proven effective in v10) and further strengthens
    the gate network with a deeper MLP (two hidden layers) and an extra LayerNorm for
    improved gradient flow. All other components (ST‑Attn, MPS bottleneck) remain efficient.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio‑temporal graph attention
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck (compress -> ReLU -> expand)
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: deeper MLP on raw input x, with GELU activations and two hidden layers
        h1 = max(1, channels // 2)
        h2 = max(1, channels // 4)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, h1),
            nn.GELU(),
            nn.LayerNorm(h1),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Linear(h2, channels),
            nn.Sigmoid()
        )

        # Residual scale, zero‑initialized
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph‑attention transform
        h = self.lin(x)                # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)        # [B*T, N, C]
            adj_3d = adj.unsqueeze(0)              # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)   # [B*T, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                            # [B, T, N, C]

        # 2. MPS compressed branch (from activated baseline)
        h_mps = self.compress(h)                   # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                 # [B, T, N, C]

        # 3. Dynamic gate (v21: deeper MLP, still on raw input x)
        gate = self.gate_net(x)                    # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch          # [B, T, N, C]

        return out
