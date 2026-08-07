import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v8(nn.Module):
    """Improved version of synth_StGraphAttnWithMpsGated_v4 and synth_StGraphAttnWithMpsGated_v7.
    
    V8 retains the effective gate design philosophy of v4 (gate derived from raw input x,
    with LayerNorm stabilization) but enhances the gate's capacity by using a two-layer
    network (LN -> Linear -> ReLU -> Linear -> Sigmoid) instead of a single Linear.
    This richer gate allows more nuanced blending of the baseline attention representation
    and the MPS bottleneck branch, while still avoiding the instability observed in v7
    when the gate was conditioned on the already-transformed features h.
    All other components (baseline ST-Attn, MPS bottleneck, residual scale alpha) are
    kept identical to v4.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes  # kept for potential future use

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # V8 gate: two-layer network for stronger dynamic routing, still based on x
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # learnable residual scale, zero-initialized for gradual training
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        """
        Args:
            x   : Tensor [B, T, N, C]
            adj : optional Tensor [N, N] (dense adjacency)
        Returns:
            Tensor [B, T, N, C] (same shape as x)
        """
        B, T, N, C = x.shape

        # 1. Baseline spatio-temporal graph attention transform
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)          # [B*T, N, C]
            adj_3d = adj.unsqueeze(0)                # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)     # [B*T, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                               # [B, T, N, C]

        # 2. MPS compressed representation (computed from the activated baseline)
        h_mps = self.compress(h)                      # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                    # [B, T, N, C]

        # 3. Dynamic gate from input statistics (v8: stronger two-layer gate)
        gate = self.gate_net(x)                       # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                              # [B, T, N, C]
        new_branch = (1.0 - gate) * diff              # gate controls MPS influence
        out = h + self.alpha * new_branch             # [B, T, N, C]

        return out
