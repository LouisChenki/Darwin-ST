import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v32(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v26 and synth_StGraphAttnWithMpsGated_v7.
    
    V32 extends the bidirectional feature modulation from V26 by incorporating
    a third modulated stream from the MPS-enhanced features h_mps. This allows
    the gate network to access a richer set of representations, including the
    already transformed bottleneck features, leading to more adaptive gating.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Projection layers for cross-modulation
        self.h_proj = nn.Linear(channels, channels)  # h -> modulation for x
        self.x_proj = nn.Linear(channels, channels)  # x -> modulation for h
        self.mps_proj = nn.Linear(channels, channels)  # h -> modulation for h_mps

        # Gate MLP: input now 3*C (modulated x, h, h_mps)
        gate_hidden = max(1, channels // 2)
        self.gate_input_norm = nn.LayerNorm(3 * channels)
        self.gate_net = nn.Sequential(
            nn.Linear(3 * channels, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, channels),
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

        # 2. MPS compressed branch
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # 3. Multi-stream modulation and gating
        gate_mod_x = torch.sigmoid(self.h_proj(h))   # modulate x by h
        gate_mod_h = torch.sigmoid(self.x_proj(x))   # modulate h by x
        gate_mod_mps = torch.sigmoid(self.mps_proj(h))  # modulate h_mps by h

        x_mod = x * gate_mod_x
        h_mod = h * gate_mod_h
        mps_mod = h_mps * gate_mod_mps

        gate_input = torch.cat([x_mod, h_mod, mps_mod], dim=-1)  # [B, T, N, 3*C]
        gate_input = self.gate_input_norm(gate_input)
        gate = self.gate_net(gate_input)               # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                     # [B, T, N, C]
        new_branch = (1.0 - gate) * diff
        out = h + self.alpha * new_branch    # [B, T, N, C]

        return out
