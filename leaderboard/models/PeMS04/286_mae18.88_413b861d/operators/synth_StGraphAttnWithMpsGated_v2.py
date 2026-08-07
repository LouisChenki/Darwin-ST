import torch
import torch.nn as nn

class StGraphAttnWithMpsGated_v2(nn.Module):
    """
    Gated‑routed fusion between baseline spatio‑temporal graph attention output h
    and a minimal‑predictive‑sufficiency (MPS) compressed version h_mps.
    V2: enhanced gate network with an extra hidden layer and ReLU activation,
        providing richer dynamic routing based on input statistics.
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

        # enhanced element‑wise gate (v2: two-layer MLP with hidden ReLU)
        self.gate_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
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
            h_flat = h.reshape(B * T, N, C)
            adj_3d = adj.unsqueeze(0)
            h_agg = torch.matmul(adj_3d, h_flat)
            h = h_agg.view(B, T, N, C)
        h = self.act(h)                     # [B, T, N, C]

        # --- MPS compressed representation ---
        h_mps = self.compress(h)            # [B, T, N, bottleneck]
        h_mps = torch.relu(h_mps)
        h_mps = self.expand(h_mps)          # [B, T, N, C]

        # --- dynamic gating from input statistics (enhanced v2 gate) ---
        gate = self.gate_net(x)             # [B, T, N, C]

        # --- gated difference (unchanged from original philosophy) ---
        diff = h_mps - h
        new_branch = (1.0 - gate) * diff

        # --- additive residual ---
        out = h + self.alpha * new_branch

        return out
