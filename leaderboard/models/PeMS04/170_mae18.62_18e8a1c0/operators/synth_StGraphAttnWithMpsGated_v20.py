import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v20(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v10 and synth_StGraphAttnWithMpsGated_v7.

    V20 retains the strong baseline and MPS branch from v10 while making the gate
    more expressive: it now uses both the raw input x and the aggregated baseline
    features h, combining their strengths. The gate network is deepened to two
    linear layers with GELU activation, taking a 2C‑dimensional concatenation.
    This design avoids the pitfalls of relying solely on h (v7) and provides richer
    context for soft gating than the pure x‑based gate of v10.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # Baseline spatio‑temporal graph attention: linear + graph aggregation + ReLU
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Gate: richer MLP on [x, h], combining raw and aggregated signals
        gate_input_dim = 2 * channels
        gate_hidden = channels  # keep expressive capacity reasonable
        self.gate_net = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        # Residual scale, zero‑initialized for stable early training
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline graph‑attention transform
        h = self.lin(x)                # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)   # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                # [B, T, N, C]

        # 2. MPS compressed branch (from activated baseline)
        h_mps = self.compress(h)       # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)     # [B, T, N, C]

        # 3. Dynamic gate from concatenation of raw input and baseline features
        gate_input = torch.cat([x, h], dim=-1)   # [B, T, N, 2C]
        gate = self.gate_net(gate_input)          # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                         # [B, T, N, C]
        new_branch = (1.0 - gate) * diff         # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch        # [B, T, N, C]

        return out
