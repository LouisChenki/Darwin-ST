import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v3(nn.Module):
    """
    Improved version over v1 and v2:
      - Retains the simple single‑layer gate from v1 (proven better than v2’s deeper gate).
      - Adds LayerNorm before the gate to stabilise gate learning from raw input.
      - Inserts LayerNorm inside the MPS bottleneck for training stability.
      - Replaces the scalar alpha with a per‑channel learnable scaling vector,
        giving finer control over the residual correction.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # baseline st_graph_attn: linear + graph aggregation + activation
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes

        # MPS bottleneck (compression + norm + expansion)
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.mps_norm = nn.LayerNorm(bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # element‑wise gate from input statistics (norm stabilised single layer)
        self.gate_net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # per‑channel learnable residual scale (initialised to zero)
        self.alpha = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x, adj=None):
        """
        Args:
            x   : Tensor [B, T, N, C]
            adj : optional Tensor [N, N] (dense adjacency)
        Returns:
            Tensor [B, T, N, C]
        """
        B, T, N, C = x.shape

        # --- baseline spatio‑temporal graph attention ---
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)       # [1, N, N]
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)

        # --- MPS compressed representation with LayerNorm ---
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = self.mps_norm(h_mps)
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # --- dynamic gate from input (norm‑stabilised) ---
        gate = self.gate_net(x)             # [B, T, N, C]

        # --- gated difference ---
        diff = h_mps - h
        new_branch = (1.0 - gate) * diff    # [B, T, N, C]

        # --- additive residual with per‑channel scaling ---
        out = h + self.alpha * new_branch   # [B, T, N, C]
        return out
