import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v25(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v22 and synth_StGraphAttnWithMpsGated_v7.

    V25 refines the gate design further by using a sigmoid-gated modulation of the raw input x
    with a projection of the post-attention features h, instead of the simple additive mixture in v22.
    This allows h to selectively scale x before feeding into the gate MLP, yielding a more context-aware
    gating signal.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # h_proj generates a modulation mask from h, applied to x via element-wise multiplication
        self.h_proj = nn.Linear(channels, channels)

        gate_hidden = max(1, channels // 2)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # Baseline spatio-temporal graph attention
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                     # [B, T, N, C]

        # MPS compressed branch (from activated baseline features)
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # Dynamic gate: modulate x by a sigmoid of projected h, then pass through MLP
        gate_mod = torch.sigmoid(self.h_proj(h))   # [B, T, N, C]
        gate_input = x * gate_mod                   # selective scaling of input
        gate = self.gate_net(gate_input)            # [B, T, N, C]

        # Gated difference and residual correction
        diff = h_mps - h                     # [B, T, N, C]
        new_branch = (1.0 - gate) * diff     # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch    # [B, T, N, C]

        return out
