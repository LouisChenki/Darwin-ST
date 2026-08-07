import torch
import torch.nn as nn
import torch.nn.functional as F


class st_graph_attn_with_vib_compression(nn.Module):
    """
    Spatio-temporal graph attention module combined with a Variational
    Information Bottleneck (VIB) compression stage.

    Contract:
        __init__(self, channels, num_nodes, **kw)
        forward(self, x, adj=None) -> tensor with same shape [B,T,N,C]

    The module never collapses the node dimension N.
    It is fully differentiable and numerically stable.
    """

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes

        # Whether to use stochastic reparameterisation (training) or
        # deterministic output (default – used for gradcheck safety).
        self.use_stochastic = bool(kw.pop('use_stochastic', False))

        # ----- main (non-trivial) branch: spatial graph + temporal conv -----
        self.feat_transform = nn.Linear(channels, channels, bias=True)
        self.temporal_conv = nn.Conv1d(
            channels, channels, kernel_size=3, padding=1
        )

        # ----- VIB compression branch (deterministic/stochastic) ------------
        self.vib_pre = nn.Linear(channels, channels, bias=True)
        self.vib_mu = nn.Linear(channels, channels, bias=True)
        self.vib_logvar = nn.Linear(channels, channels, bias=True)

        # additive residual weight (initialised to 0)
        self.alpha = nn.Parameter(torch.zeros(1))

    def _graph_aggregate(self, x, adj):
        # x: [B, T, N, C]
        if adj is None:
            return x

        B, T, N, C = x.shape
        if adj.dim() == 2:                # [N, N]
            adj = adj.unsqueeze(0).unsqueeze(0)   # [1, 1, N, N]
        elif adj.dim() == 3:              # [B, N, N]
            adj = adj.unsqueeze(1)        # [B, 1, N, N]
        else:
            raise ValueError("adj must be 2‑ or 3‑dimensional")

        # Broadcast adjacency across batch and time dimensions
        adj_bt = adj.expand(B, T, N, N).reshape(B * T, N, N)
        x_bt = x.reshape(B * T, N, C)

        out = torch.bmm(adj_bt, x_bt)      # [B*T, N, C]
        return out.reshape(B, T, N, C)

    def forward(self, x, adj=None):
        """
        Args:
            x:   [B, T, N, C] input features
            adj: optional adjacency matrix [N, N] or [B, N, N]
        Returns:
            y:   [B, T, N, C] output features (same shape as x)
        """
        B, T, N, C = x.shape

        # ---- non‑trivial main path -----------------------------------------
        h = self.feat_transform(x)                # [B, T, N, C]
        h = self._graph_aggregate(h, adj)          # optionally mix across nodes
        h = h.permute(0, 2, 3, 1).contiguous()    # [B, N, C, T]
        h = h.view(B * N, C, T)                   # [B*N, C, T]
        h = self.temporal_conv(h)                 # [B*N, C, T]
        h = h.view(B, N, C, T).permute(0, 3, 1, 2).contiguous()  # [B, T, N, C]
        main_out = h                               # main branch output

        # ---- VIB compression branch ---------------------------------------
        vib_h = F.relu(self.vib_pre(main_out))     # [B, T, N, C]
        mu = self.vib_mu(vib_h)                    # [B, T, N, C]
        logvar = self.vib_logvar(vib_h)            # [B, T, N, C]

        # Stochastic reparameterisation only when requested AND in training.
        if self.use_stochastic and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu   # deterministic path (used for gradcheck / inference)

        # ---- additive residual --------------------------------------------
        # alpha is a learnable scalar (starts at 0)
        out = main_out + self.alpha * z

        return out
