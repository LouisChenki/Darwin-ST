import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v15(nn.Module):
    """Improved version that combines the strengths of v10 (richer gate on input)
    and v7 (gate influenced by transformed features) by feeding both raw input x
    and the activated graph‑attention features h into the gate network.
    The gate uses a concatenated input [x, h] -> LayerNorm -> Linear -> GELU ->
    Linear -> Sigmoid, allowing a more informed modulation of the MPS correction
    while keeping a stable residual learning with zero‑initialised alpha.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio‑temporal graph attention
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: richer MLP taking both raw input and transformed features
        gate_hidden = channels
        self.gate_net = nn.Sequential(
            nn.LayerNorm(2 * channels),
            nn.Linear(2 * channels, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Residual correction scale, zero‑initialized
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph‑attention transform
        h = self.lin(x)                         # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)     # [B*T, N, C]
            adj_3d = adj.unsqueeze(0)           # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                         # [B, T, N, C]

        # 2. MPS compressed branch
        h_mps = self.compress(h)                # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)              # [B, T, N, C]

        # 3. Dynamic gate from concatenation of raw input and transformed features
        gate_input = torch.cat([x, h], dim=-1)  # [B, T, N, 2C]
        gate = self.gate_net(gate_input)        # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                        # [B, T, N, C]
        new_branch = (1.0 - gate) * diff        # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch       # [B, T, N, C]

        return out
