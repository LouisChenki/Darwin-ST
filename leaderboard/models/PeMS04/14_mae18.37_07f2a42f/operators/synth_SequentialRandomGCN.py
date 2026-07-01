import torch
import torch.nn as nn

class SequentialRandomGCN(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        # learnable scaling factor for the additive residual
        self.alpha = nn.Parameter(torch.zeros(1))
        # domain‑randomization offset (learned per node and channel)
        self.domain_offset = nn.Parameter(torch.zeros(1, 1, num_nodes, channels))
        # main path: transform each node’s features with a linear layer + ReLU
        self.main_fc = nn.Linear(channels, channels)
        self.activation = nn.ReLU()

    def forward(self, x, adj=None):
        # x shape: [B, T, N, C]
        B, T, N, C = x.shape
        # apply domain‑randomization offset
        x_noisy = x + self.domain_offset
        # main path: reshape to (B*T*N, C), apply linear, activate, reshape back
        x_flat = x_noisy.view(-1, C)
        main_out = self.main_fc(x_flat)
        main_out = self.activation(main_out)
        main_out = main_out.view(B, T, N, C)
        # branch: identity of the original input
        branch = x
        # additive residual with learnable alpha
        out = main_out + self.alpha * branch
        return out
