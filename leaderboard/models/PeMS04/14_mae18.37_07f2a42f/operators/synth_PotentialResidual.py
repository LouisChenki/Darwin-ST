import torch
import torch.nn as nn

class PotentialResidual(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.ReLU(),
            nn.Linear(channels * 2, channels)
        )
        # Learnable potential per node per channel
        self.node_potential = nn.Parameter(torch.zeros(num_nodes, channels))
        nn.init.normal_(self.node_potential, mean=0.0, std=0.02)
        # Learnable alpha for the additive residual
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        B, T, N, C = x.shape
        # Flatten to (B*T*N, C) for the main path
        x_flat = x.view(B * T * N, C)
        main_out = self.main_path(x_flat)
        main_out = main_out.view(B, T, N, C)

        # Branch: node potential expanded to full shape
        branch = self.node_potential.unsqueeze(0).unsqueeze(0)   # (1,1,N,C)
        branch = branch.expand(B, T, N, C)

        # Additive residual: main_path + alpha * branch
        out = main_out + self.alpha * branch
        return out
