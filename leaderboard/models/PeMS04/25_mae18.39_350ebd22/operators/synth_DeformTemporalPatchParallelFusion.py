import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformScale(nn.Module):
    """Single scale of deformable temporal patching."""
    def __init__(self, L: int, init_width: float = 0.5):
        super().__init__()
        self.L = L
        # centers in [0,1) will be scaled by (T-1) at runtime
        self.centers = nn.Parameter(torch.rand(L))
        # log_widths so that width = softplus(log_widths) + 1e-3
        self.log_widths = nn.Parameter(torch.full((L,), math.log(init_width)))

    def forward(self, x: torch.Tensor, T: int):
        """
        x : (B, T, N, C)
        returns (B, T, N, C)
        """
        device = x.device
        dtype  = x.dtype
        # time grid [0, 1, ..., T-1]
        time = torch.arange(T, device=device, dtype=dtype).unsqueeze(1)  # (T,1)
        # scale centers to actual time positions
        centers = self.centers * (T - 1)                        # (L,)
        widths  = F.softplus(self.log_widths) + 1e-3           # (L,)

        # squared distance
        dist = (time - centers.unsqueeze(0)) ** 2              # (T, L)
        logits = -dist / (2 * widths.unsqueeze(0) ** 2)        # (T, L)
        mask = F.softmax(logits, dim=1)                        # (T, L)

        # aggregate over time for each patch
        agg = torch.einsum('btnc,ti->binc', x, mask)           # (B, L, N, C)
        # reconstruct temporal signal using the same masks
        out = torch.einsum('binc,ti->btnc', agg, mask)         # (B, T, N, C)
        return out


class DeformTemporalPatchParallelFusion(nn.Module):
    """
    Parallel fusion of a learnable main path (1‑D Conv) and a deformable temporal
    patching branch.  The branch uses several scales of learned Gaussian windows
    to obtain patch‑level representations and then projects them back to the
    full temporal resolution via the same windows.

    Output shape matches input shape (B, T, N, C).
    """
    def __init__(self, channels: int, num_nodes: int, **kw):
        super().__init__()
        # alpha – learnable scalar for additive residual
        self.alpha = nn.Parameter(torch.zeros(1))

        # Main path: a non‑trivial temporal convolution
        self.main_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            bias=True
        )

        # Deformable branch: multiple scales with different numbers of patches
        patch_counts = [4, 8, 12]          # L_s values
        self.deform_scales = nn.ModuleList([
            DeformScale(L) for L in patch_counts
        ])

    def forward(self, x: torch.Tensor, adj=None) -> torch.Tensor:
        # x: (B, T, N, C)
        B, T, N, C = x.shape
        device = x.device
        dtype  = x.dtype

        # ---- Main path (non‑trivial temporal transform) ----
        # reshape to (B*N, C, T) for Conv1d
        x_main = x.permute(0, 2, 3, 1)          # (B, N, C, T)
        x_main = x_main.contiguous().view(B * N, C, T)   # (B*N, C, T)
        main_out = self.main_conv(x_main)                # (B*N, C, T)
        main_out = main_out.view(B, N, C, T)             # (B, N, C, T)
        main_out = main_out.permute(0, 3, 1, 2)          # (B, T, N, C)

        # ---- Deformable branch ----
        branch_out = torch.zeros_like(x)
        for scale_mod in self.deform_scales:
            branch_out = branch_out + scale_mod(x, T)

        # ---- Fusion ----
        out = main_out + self.alpha * branch_out
        return out
