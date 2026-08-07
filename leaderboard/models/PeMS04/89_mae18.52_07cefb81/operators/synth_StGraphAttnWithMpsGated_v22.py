import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v22(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and synth_StGraphAttnWithMpsGated_v7.

    V22 blends the gate-on-input design of v10 with a refined gate that also sees the post-attention
    representation h through a learned projection. This avoids the overly simple gate of v7 while
    enriching the gating signal with transformed graph attention features.
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

        # Projection for h so that the gate can condition on both x and h
        self.h_proj = nn.Linear(channels, channels)

        # Gate network: mixes raw input x and projected h, then applies a powerful MLP
        gate_hidden = max(1, channels // 2)          # inherited from v10
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, gate_hidden),
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
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                          # [B, T, N, C]

        # 2. MPS compressed branch (from activated baseline)
        h_mps = self.compress(h)                 # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)               # [B, T, N, C]

        # 3. Dynamic gate: mix raw input x and a learned projection of h
        gate_input = x + self.h_proj(h)          # [B, T, N, C]
        gate = self.gate_net(gate_input)         # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                         # [B, T, N, C]
        new_branch = (1.0 - gate) * diff         # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch        # [B, T, N, C]

        return out
