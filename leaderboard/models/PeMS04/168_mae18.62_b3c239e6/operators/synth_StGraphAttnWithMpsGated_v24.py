import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v24(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v22 and synth_StGraphAttnWithMpsGated_v7.

    V24 absorbs the powerful gating mechanism of v22 (mixing raw input and projected
    graph attention features) and further enhances the MPS correction branch by adopting
    an inverted bottleneck structure (expand → activation → compress) for richer representations.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio-temporal graph attention: linear + graph aggregation + ReLU
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS inverted bottleneck branch: expand -> GELU -> compress
        expand_factor = 2
        self.expand_mps = nn.Linear(channels, channels * expand_factor)
        self.compress_mps = nn.Linear(channels * expand_factor, channels)

        # Projection for h to be mixed with raw input for gating
        self.h_proj = nn.Linear(channels, channels)

        # Gate network: mixes raw input x and projected h through a powerful MLP
        gate_hidden = max(1, channels // 2)
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

        # 1. Baseline spatio-temporal graph attention
        h = self.lin(x)                          # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)            # [1, N, N] -> broadcast to [B*T, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                          # [B, T, N, C]

        # 2. MPS correction branch (inverted bottleneck from h)
        h_mps = self.expand_mps(h)               # [B, T, N, expand_factor * C]
        h_mps = torch.nn.functional.gelu(h_mps)  # non-linearity
        h_mps = self.compress_mps(h_mps)          # [B, T, N, C]

        # 3. Dynamic gate: mix raw input x and a learned projection of h
        gate_input = x + self.h_proj(h)          # [B, T, N, C]
        gate = self.gate_net(gate_input)         # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                         # [B, T, N, C]
        new_branch = (1.0 - gate) * diff          # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch        # [B, T, N, C]

        return out
