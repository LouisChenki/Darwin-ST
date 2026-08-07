import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v7(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v4 and synth_StGraphAttnWithMpsGated_v5.
    
    V7 keeps the simple and stable gate design from v4 (LayerNorm + single linear + Sigmoid),
    but derives the gate from the post-attention features h instead of the raw input x.
    This allows the gate to adapt based on the actual aggregated representation,
    which is more directly related to the MPS correction branch, while avoiding the
    deeper (and empirically worse) gate network of v5.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: simple LN + linear + sigmoid, now takes h as input
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. baseline spatio-temporal graph attention
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                     # [B, T, N, C]

        # 2. MPS compressed branch (from the activated baseline features)
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # 3. Dynamic gate from post-attention features (v7: uses h instead of x)
        gate = self.gate_net(h)             # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                    # [B, T, N, C]
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch   # [B, T, N, C]

        return out
