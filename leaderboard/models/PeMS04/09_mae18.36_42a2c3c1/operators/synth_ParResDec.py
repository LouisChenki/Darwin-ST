import torch
import torch.nn as nn

class ParResDec(nn.Module):
    """
    Parallel Residual Decomposition operator.

    Contract:
        __init__(self, channels, num_nodes, **kw)
        forward(self, x, adj=None) -> Tensor with shape [B, T, N, C]

    Design:
        - main_path: a non‑trivial learned transformation f(x) (two‑layer MLP).
        - new_branch: residual decomposition r = x - f(x).
        - output: main_path + alpha * new_branch, where alpha is a learnable scalar.
    """

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes

        # learnable alpha
        self.alpha = nn.Parameter(torch.zeros(1))

        # main transformation (non‑trivial, non‑identity)
        hidden = channels * 2
        self.f = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x, adj=None):
        # x shape: [B, T, N, C]
        B, T, N, C = x.shape

        # apply main transformation on the last dimension (channel)
        x_flat = x.view(-1, C)                  # (B*T*N, C)
        f_flat = self.f(x_flat)                 # (B*T*N, C)
        f_out = f_flat.view(B, T, N, C)         # back to original shape

        # residual decomposition branch: noise estimate = x - f(x)
        noise_est = x - f_out

        # parallel combination (additive residual)
        out = f_out + self.alpha * noise_est

        return out
