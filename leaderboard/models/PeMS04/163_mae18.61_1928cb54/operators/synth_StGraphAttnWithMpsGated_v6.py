import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v6(nn.Module):
    """
    V6: Inherits the simple and effective gating of v4 (LayerNorm + Linear + Sigmoid)
    and introduces a residual connection in the MPS bottleneck branch to ease
    information flow, while avoiding the overly deep gate network of v5 that hurt performance.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # baseline spatio-temporal graph attention
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck with residual connection
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # gate network: light and stable (v4 style)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # learned mixing weight, initialized to 0 for stable early training
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
        h = self.act(h)

        # 2. MPS compressed branch with residual connection
        h_mps = self.compress(h)
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps) + h        # residual boosts gradient flow

        # 3. Dynamic gate from input statistics (v4's stable design)
        gate = self.gate_net(x)

        # 4. Gated fusion
        diff = h_mps - h                      # effectively the MPS residual
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch

        return out
