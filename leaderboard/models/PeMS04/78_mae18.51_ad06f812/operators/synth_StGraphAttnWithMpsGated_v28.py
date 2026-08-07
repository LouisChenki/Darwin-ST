import torch
import torch.nn as nn
import torch.nn.functional as F

class StGraphAttnWithMpsGated_v28(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v26 and synth_StGraphAttnWithMpsGated_v7.

    V28 retains the bidirectional modulation (x modulated by h, h modulated by x) and 
    gated difference structure from v26, but replaces the gate MLP's GELU with SiLU 
    and introduces intermediate LayerNorm for added training stability. 
    The MPS bottleneck activation is also switched from ReLU to SiLU for smoother feature transformation.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # Projection layers for bidirectional modulation
        self.h_proj = nn.Linear(channels, channels)
        self.x_proj = nn.Linear(channels, channels)

        # Gate MLP with SiLU and internal LayerNorm
        gate_hidden = max(1, channels // 2)
        self.gate_input_norm = nn.LayerNorm(2 * channels)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * channels, gate_hidden),
            nn.LayerNorm(gate_hidden),
            nn.SiLU(),
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
        h = self.act(h)                     # ReLU activated base features

        # 2. MPS compressed branch (SiLU instead of ReLU for smoothness)
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = F.silu(h_mps)               # SiLU activation
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # 3. Bidirectional modulation: x filtered by h, h filtered by x
        gate_mod_x = torch.sigmoid(self.h_proj(h))   # [B, T, N, C]
        gate_mod_h = torch.sigmoid(self.x_proj(x))   # [B, T, N, C]
        x_mod = x * gate_mod_x
        h_mod = h * gate_mod_h
        gate_input = torch.cat([x_mod, h_mod], dim=-1)  # [B, T, N, 2*C]
        gate_input = self.gate_input_norm(gate_input)
        gate = self.gate_net(gate_input)               # [B, T, N, C]

        # 4. Gated correction
        diff = h_mps - h                    # [B, T, N, C]
        new_branch = (1.0 - gate) * diff    # element-wise suppression of diff
        out = h + self.alpha * new_branch   # [B, T, N, C]

        return out
