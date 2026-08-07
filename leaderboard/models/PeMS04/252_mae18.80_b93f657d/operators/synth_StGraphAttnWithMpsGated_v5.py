import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v5(nn.Module):
    """
    V5: combines the stable Layernorm-gating of v4 with a deeper two-layer
    gate network (from v2) for richer dynamic routing, while keeping
    the overall style and core components unchanged.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # enriched gate: LayerNorm -> hidden -> ReLU -> output -> Sigmoid
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. baseline spatio-temporal graph attention
        h = self.lin(x)
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)

        # 2. MPS compressed branch
        h_mps = self.compress(h)
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)

        # 3. dynamic gate from input statistics (stabilized, deeper)
        gate = self.gate_net(x)

        # 4. gated correction
        diff = h_mps - h
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch

        return out
