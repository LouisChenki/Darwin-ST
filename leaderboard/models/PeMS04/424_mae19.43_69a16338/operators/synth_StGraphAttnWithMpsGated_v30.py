import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v30(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v26 and synth_StGraphAttnWithMpsGated_v7.

    V30 refines the bidirectional gating scheme from v26 by:
    - narrowing the MPS bottleneck (channels // 4) with an extra LayerNorm for stability,
    - using SiLU activation in the gate MLP for smoother gradient flow,
    - keeping cross-modulation between raw input x and post-attention h for context-aware gating.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 4)               # tighter compression
        self.compress = nn.Linear(channels, bottleneck)
        self.compress_norm = nn.LayerNorm(bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Bidirectional modulation projections (same as v26)
        self.h_proj = nn.Linear(channels, channels)      # h -> modulation for x
        self.x_proj = nn.Linear(channels, channels)      # x -> modulation for h

        gate_hidden = max(1, channels // 2)
        self.gate_input_norm = nn.LayerNorm(2 * channels)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * channels, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, channels),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # 1. Baseline spatio-temporal graph attention
        h = self.lin(x)
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)

        # 2. MPS compressed branch with bottleneck + LayerNorm + activation
        h_mps = self.compress(h)
        h_mps = self.compress_norm(h_mps)
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                      # [B, T, N, C]

        # 3. Bidirectional feature modulation (x modulated by h, h modulated by x)
        gate_mod_x = torch.sigmoid(self.h_proj(h))      # [B, T, N, C]
        gate_mod_h = torch.sigmoid(self.x_proj(x))      # [B, T, N, C]
        x_mod = x * gate_mod_x
        h_mod = h * gate_mod_h
        gate_input = torch.cat([x_mod, h_mod], dim=-1)  # [B, T, N, 2*C]
        gate_input = self.gate_input_norm(gate_input)
        gate = self.gate_net(gate_input)                # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                                 # [B, T, N, C]
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch               # [B, T, N, C]

        return out
