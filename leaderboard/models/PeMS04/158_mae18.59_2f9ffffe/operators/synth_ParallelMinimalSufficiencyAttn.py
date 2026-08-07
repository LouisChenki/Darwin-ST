import torch
import torch.nn as nn


class ParallelMinimalSufficiencyAttn(nn.Module):
    """
    ParallelMinimalSufficiencyAttn follows a parallel fusion architecture:
      - Main path: spatial-temporal graph attention.
      - Minimal predictive sufficiency branch: information-bottleneck-style compression.
      - Additive residual: output = main_path + α * sufficiency_branch, α learnable.

    Parameters
    ----------
    channels : int
        Number of feature channels (C).
    num_nodes : int
        Number of nodes (N).
    **kw :
        n_heads : int, default 4   – number of attention heads (channels must be divisible).
    """

    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes

        # Learnable residual weight – starts at 0 so the branch is gradually introduced.
        self.alpha = nn.Parameter(torch.zeros(1))

        # ---------- Graph attention (spatial) ----------
        self.n_heads = kw.get('n_heads', 4)
        assert channels % self.n_heads == 0, "channels must be divisible by n_heads"
        self.d_k = channels // self.n_heads

        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        # ---------- Temporal convolution ----------
        # Applies a 1x3 convolution across time for each node independently.
        self.temp_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

        # ---------- Minimal predictive sufficiency branch ----------
        # Compresses the representation via a bottleneck to discard noise.
        hidden = max(1, channels // 2)
        self.bottleneck = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
        )

    def forward(self, x, adj=None):
        """
        x : (B, T, N, C)    batch, time steps, nodes, features
        adj : (N, N)        optional binary/weighted adjacency matrix

        returns : (B, T, N, C)
        """
        B, T, N, C = x.shape

        # ==================================================================
        #  Main path – spatial-temporal graph attention
        # ==================================================================

        # (1) Node‑wise multi‑head self‑attention (graph attention)
        # Flatten batch & time to treat (b,t) as independent samples.
        x_flat = x.view(B * T, N, C)                     # (BT, N, C)

        q = self.q_proj(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)  # (BT, h, N, d_k)
        k = self.k_proj(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x_flat).view(B * T, N, self.n_heads, self.d_k).transpose(1, 2)

        attn_raw = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)  # (BT, h, N, N)

        if adj is not None:
            # mask non‑edges with -inf before softmax
            mask = (adj.to(torch.float) == 0).unsqueeze(0).unsqueeze(0)      # (1, 1, N, N)
            mask = mask.expand(B * T, self.n_heads, N, N)                    # (BT, h, N, N)
            attn_raw = attn_raw.masked_fill(mask, float('-inf'))

        attn = torch.softmax(attn_raw, dim=-1)            # (BT, h, N, N)
        out = torch.matmul(attn, v)                       # (BT, h, N, d_k)
        out = out.transpose(1, 2).contiguous()            # (BT, N, h, d_k)
        out = out.view(B * T, N, C)                       # (BT, N, C)
        out = self.out_proj(out)                          # (BT, N, C)

        main_graph = out.view(B, T, N, C)                 # (B, T, N, C)

        # (2) Temporal convolution – applied per node across time
        # Permute to (B, N, C, T) -> reshape to (B*N, C, T)
        main_graph_perm = main_graph.permute(0, 2, 3, 1)   # (B, N, C, T)
        main_graph_perm = main_graph_perm.reshape(B * N, C, T)

        main_temporal = self.temp_conv(main_graph_perm)    # (B*N, C, T)

        # Go back to (B, T, N, C)
        main_temporal = main_temporal.reshape(B, N, C, T).permute(0, 3, 1, 2)  # (B, T, N, C)

        main_out = main_temporal

        # ==================================================================
        #  Minimal predictive sufficiency branch – bottleneck compression
        # ==================================================================
        # Operates independently on each (B,T,N) sample; no cross‑node or
        # cross‑time mixing, because the goal is to filter per‑element noise.
        x_bn = x.view(-1, C)               # (B*T*N, C)
        bn_out = self.bottleneck(x_bn)     # (B*T*N, C)
        bn_out = bn_out.view(B, T, N, C)   # (B, T, N, C)

        # ==================================================================
        #  Parallel additive fusion
        # ==================================================================
        return main_out + self.alpha * bn_out
