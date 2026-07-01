import torch
import torch.nn as nn
import torch.nn.functional as F

class sparse_residual_subset(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.k = kw.get('k', num_nodes // 2)   # number of nodes to select
        self.tau = kw.get('tau', 1.0)          # temperature for Gumbel-Softmax
        # score projector (per node scalar)
        self.score_linear = nn.Linear(channels, 1, bias=False)
        # main transformation (non‑trivial)
        self.main_linear = nn.Linear(channels, channels, bias=True)
        # branch feature projection
        self.branch_linear = nn.Linear(channels, channels, bias=False)
        # additive residual weight
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, adj=None):
        # x shape: (B, T, N, C)
        B, T, N, C = x.shape

        # ---------- differentiable k‑subset sampling ----------
        # logits per node
        logits = self.score_linear(x).squeeze(-1)          # (B,T,N)
        # Gumbel noise
        noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        perturbed = logits + noise                          # (B,T,N)
        # soft weights (relaxed)
        soft_w = F.softmax(perturbed / self.tau, dim=-1)   # (B,T,N)
        # hard top‑k indices
        _, indices = torch.topk(perturbed, self.k, dim=-1) # (B,T,k)
        # hard binary mask
        hard_mask = torch.zeros_like(soft_w)
        hard_mask.scatter_(-1, indices, 1.0)                # (B,T,N)
        # straight‑through estimator
        mask = hard_mask - soft_w.detach() + soft_w         # (B,T,N)

        # ---------- branch feature aggregation ----------
        branch_feat = self.branch_linear(x)                # (B,T,N,C)
        # weighted average over selected nodes
        mask_exp = mask.unsqueeze(-1)                      # (B,T,N,1)
        weighted = mask_exp * branch_feat                  # (B,T,N,C)
        pooled = weighted.sum(dim=2, keepdim=True) / self.k  # (B,T,1,C)
        branch_out = pooled.expand(-1, -1, N, -1)          # (B,T,N,C)

        # ---------- main path (non‑trivial) ----------
        main_out = self.main_linear(x)                     # (B,T,N,C)

        # ---------- residual combination ----------
        out = main_out + self.alpha * branch_out
        return out
