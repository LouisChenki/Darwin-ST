import torch
import torch.nn as nn

class gated_routed_adaptive(nn.Module):
    """
    Implements a gated‑routed adaptive fusion of two mechanisms:
      - relational_spatial_encoding
      - masked_multi_modal_conditioning

    The fusion weight is obtained by a lightweight gating network that
    processes global input statistics.

    Additive residual (scalar α) is used: output = main_path + α * new_branch.

    forward(x, adj=None) -> Tensor of shape [B, T, N, C].
    """

    def __init__(self, channels: int, num_nodes: int, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes

        # non‑trivial main path (learnable linear transformation)
        self.main_linear = nn.Linear(channels, channels)

        # learnable alpha for additive residual
        self.alpha = nn.Parameter(torch.zeros(1))

        # lightweight gating network
        self.gate_net = nn.Sequential(
            nn.Linear(channels, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

        # relational branch: per‑node linear transformation
        self.rel_linear = nn.Linear(channels, channels)
        # additional linear for aggregated neighbour information (if adj present)
        self.rel_adj_linear = nn.Linear(channels, channels)

        # masked branch: per‑channel learnable mask (sigmoid bounded)
        self.mask_param = nn.Parameter(torch.ones(channels))
        self.mask_linear = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        B, T, N, C = x.shape

        # ----- main path (non‑trivial) -----
        x_flat = x.reshape(-1, C)
        main_out = self.main_linear(x_flat).reshape(B, T, N, C)

        # ----- global gating weights -----
        # average over temporal and node dimensions -> (B, C)
        x_avg = x.mean(dim=[1, 2])          # (B, C)
        gate_logits = self.gate_net(x_avg)  # (B, 2)
        gate_weights = torch.softmax(gate_logits, dim=-1)  # (B, 2)
        w_rel = gate_weights[:, 0:1]        # (B, 1)
        w_mask = gate_weights[:, 1:2]       # (B, 1)

        # ----- relational_spatial_encoding branch -----
        # per‑node transform
        rel_out = self.rel_linear(x_flat).reshape(B, T, N, C)

        if adj is not None:
            # adjacency is assumed to be [N, N] or [B, N, N]
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)          # [1, N, N]
            elif adj.dim() == 3 and adj.shape[0] == 1 and B > 1:
                adj = adj.expand(B, -1, -1)     # [B, N, N]
            elif adj.dim() != 3:
                # unexpected shape, ignore adjacency
                pass
            else:
                pass  # adj is [B, N, N] already

            if adj.dim() == 3 and adj.shape[0] == B:
                # aggregate neighbour information: e_i = sum_j A_ij * x_j
                # x_perm -> (B, T, C, N)
                x_perm = x.permute(0, 1, 3, 2)   # B, T, C, N
                # einsum 'bij,btcj->btci' -> (B, T, C, N)
                agg = torch.einsum('bij,btcj->btci', adj, x_perm)
                agg = agg.permute(0, 1, 3, 2)    # back to B, T, N, C
                agg_flat = agg.reshape(-1, C)
                rel_adj_out = self.rel_adj_linear(agg_flat).reshape(B, T, N, C)
                rel_out = rel_out + rel_adj_out   # combine self + neighbours

        # ----- masked_multi_modal_conditioning branch -----
        mask = torch.sigmoid(self.mask_param)         # (C,)
        x_masked = x * mask.view(1, 1, 1, C)          # B, T, N, C
        x_masked_flat = x_masked.reshape(-1, C)
        mask_out = self.mask_linear(x_masked_flat).reshape(B, T, N, C)

        # ----- weighted fusion -----
        w_rel_exp = w_rel.view(B, 1, 1, 1)      # broadcast over T, N, C
        w_mask_exp = w_mask.view(B, 1, 1, 1)
        new_branch = w_rel_exp * rel_out + w_mask_exp * mask_out  # B, T, N, C

        # additive residual
        out = main_out + self.alpha * new_branch
        return out
