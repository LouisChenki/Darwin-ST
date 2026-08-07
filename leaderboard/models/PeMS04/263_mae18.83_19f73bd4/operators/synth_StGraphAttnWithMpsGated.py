import torch
import torch.nn as nn

class StGraphAttnWithMpsGated(nn.Module):
    """
    Gated‑routed fusion between baseline spatio‑temporal graph attention output h
    and a minimal‑predictive‑sufficiency (MPS) compressed version h_mps.
    The gate dynamically weights trust in original attention vs. robust compressed
    representation, and a learnable alpha adds residual correction.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # baseline st_graph_attn: linear + graph aggregation + activation
        self.lin = nn.Linear(channels, channels)
        self.act = nn.ReLU()
        self.num_nodes = num_nodes  # kept for potential future use

        # MPS bottleneck
        bottleneck = max(1, channels // 2)
        self.compress = nn.Linear(channels, bottleneck)
        self.expand = nn.Linear(bottleneck, channels)

        # element‑wise gate derived from input statistics
        self.gate_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid()
        )

        # residual scale (starts at 0 so output begins as baseline)
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
        # --- baseline st_graph_attn transform ---
        h = self.lin(x)                     # [B, T, N, C]
        if adj is not None:
            # aggregate neighbour features using adjacency
            h_flat = h.reshape(B * T, N, C)          # [BT, N, C]
            adj_3d = adj.unsqueeze(0)                # [1, N, N]
            # matmul broadcasts batch dim (1 -> BT)
            h_agg = torch.matmul(adj_3d, h_flat)     # [BT, N, C]
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                               # still [B, T, N, C]

        # --- MPS compressed representation ---
        h_mps = self.compress(h)                      # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)                    # [B, T, N, C]

        # --- dynamic gating from input statistics ---
        gate = self.gate_net(x)                       # [B, T, N, C]

        # --- gated difference (new branch) ---
        diff = h_mps - h                              # [B, T, N, C]
        new_branch = (1.0 - gate) * diff              # gate controls trust

        # --- additive residual: h + alpha * new_branch ---
        out = h + self.alpha * new_branch             # [B, T, N, C]

        return out
