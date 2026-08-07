import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v23(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v22 and synth_StGraphAttnWithMpsGated_v7.

    V23 retains the strong MPS residual design from v22 but uses concatenation instead of addition
    to fuse raw input x and projected h for the gate. This allows the gate to access both modalities
    without forcing them into a common additive subspace, increasing the model's capacity to learn
    nuanced fusion strategies while avoiding the overly simple gate of v7.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio-temporal graph attention
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck: compress → ReLU → expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Projection for h to complement x in the gate
        self.h_proj = nn.Linear(channels, channels)

        # Gate network: now takes concatenated [x, h_proj(h)] as input
        gate_hidden = max(1, channels // 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Residual scale, zero-initialized for stable early training
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph-attention transform
        h = self.lin(x)                          # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)            # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat) # [B*T, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                          # [B, T, N, C]

        # 2. MPS compressed branch (from the activated baseline features)
        h_mps = self.compress(h)                 # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)               # [B, T, N, C]

        # 3. Dynamic gate: concatenate raw input x and a learned projection of h
        gate_input = torch.cat([x, self.h_proj(h)], dim=-1)  # [B, T, N, 2C]
        gate = self.gate_net(gate_input)                     # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                         # [B, T, N, C]
        new_branch = (1.0 - gate) * diff         # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch        # [B, T, N, C]

        return out
