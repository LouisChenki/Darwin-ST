import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v12(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and v7.

    V12 inherits the stronger gating design from v10 (GELU, wider hidden layer) and
    the baseline MPS correction branch. The key advance is making the gate aware of the
    graph structure: it now operates on graph‑aggregated input features. This provides
    spatially informed gating while keeping all other components unchanged.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio-temporal graph attention
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Richer gate with graph awareness (v12 improvement)
        gate_hidden = max(1, channels // 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Learnable residual scale
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph‑attention transform
        h = self.lin(x)                # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)  # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)

        # 2. MPS compressed branch
        h_mps = self.compress(h)
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)

        # 3. Graph‑aware gate: first aggregate input x along the graph (if available)
        if adj is not None:
            x_flat = x.reshape(B * T, N, C)
            x_graph = torch.matmul(adj.unsqueeze(0), x_flat).view(B, T, N, C)
        else:
            x_graph = x
        gate = self.gate_net(x_graph)   # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch

        return out
