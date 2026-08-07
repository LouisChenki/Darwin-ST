import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v4(nn.Module):
    """
    Improved version: retains the effective simple gating philosophy from the better
    parent version but adds a LayerNorm before the gate projection for more stable
    dynamic routing. The gate is derived from input x after normalization, helping
    the model to blend baseline attention output and MPS compressed branch more robustly.
    All core components (baseline ST-Attn, MPS bottleneck) are kept unchanged.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # baseline spatio-temporal graph attention: linear + graph aggregation + ReLU
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes  # retained for potential extensions

        # MPS bottleneck: compress -> ReLU -> expand
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # v4 gate: input LayerNorm provides stability, followed by single linear + sigmoid
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # residual scale, initialized to zero so early training emulates baseline
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        """
        Args:
            x   : Tensor [B, T, N, C]
            adj : optional Tensor [N, N] (dense adjacency)
        Returns:
            Tensor [B, T, N, C]
        """
        B, T, N, C = x.shape

        # 1. Baseline spatio‑temporal graph attention transform
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)          # [B*T, N, C]
            adj_3d = adj.unsqueeze(0)                # [1, N, N]
            # adjacency aggregation, broadcasts batch dimension
            h_agg = torch.matmul(adj_3d, h_flat)     # [B*T, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                               # [B, T, N, C]

        # 2. MPS compressed representation (from the activated baseline features)
        h_mps = self.compress(h)                      # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                    # [B, T, N, C]

        # 3. Dynamic gate from input statistics (v4: with LayerNorm stabilization)
        gate = self.gate_net(x)                       # [B, T, N, C]

        # 4. Gated difference and residual correction
        diff = h_mps - h                              # [B, T, N, C]
        new_branch = (1.0 - gate) * diff              # gate controls trust in baseline vs MPS
        out = h + self.alpha * new_branch             # [B, T, N, C]

        return out
