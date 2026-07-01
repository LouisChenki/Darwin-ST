import torch
import torch.nn as nn

class DataDistillParallel(nn.Module):
    """
    Parallel fusion operator that combines main path (non‑trivial transformation)
    with a distillation branch via additive residual (α * branch).
    The distillation branch is inspired by dataset distillation concepts,
    compressing redundant information into a compact representation.
    """
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        # learnable mixing coefficient, initialized to zero
        self.alpha = nn.Parameter(torch.zeros(1))

        # main path: two linear layers with GELU activation, guarantees non‑trivial transform
        self.main_path = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )

        # distillation branch: a smaller bottleneck (channels → channels//2 → channels)
        hidden = max(channels // 2, 1)
        self.branch_path = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels)
        )

    def forward(self, x, adj=None):
        # x shape: [B, T, N, C]
        # Apply both paths on the last dimension (channels)
        main_out = self.main_path(x)          # [B, T, N, C]
        branch_out = self.branch_path(x)      # [B, T, N, C]

        output = main_out + self.alpha * branch_out
        return output
