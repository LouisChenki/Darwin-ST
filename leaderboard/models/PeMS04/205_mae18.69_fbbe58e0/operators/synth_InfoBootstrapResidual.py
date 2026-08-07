import torch
import torch.nn as nn

class InfoBootstrapResidual(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super(InfoBootstrapResidual, self).__init__()
        self.channels = channels
        self.num_nodes = num_nodes

        # main path (non-trivial transformation)
        self.main = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
        )

        # zero-initialised mixing weight (additive residual)
        self.alpha = nn.Parameter(torch.zeros(1))

        # bottleneck size
        hidden_dim = max(channels // 4, 1)

        # compression and expansion for the residual branch
        self.compress = nn.Linear(channels, hidden_dim)
        self.expand = nn.Linear(hidden_dim, channels)
        self.activation = nn.ReLU()

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # -------- main path --------
        flat_x = x.reshape(-1, C)
        main_out = self.main(flat_x).view(B, T, N, C)

        # -------- residual branch (info bottleneck style) --------
        if adj is not None:
            # aggregate across nodes using the supplied adjacency matrix
            # adj is expected to have shape [N, N]
            x_bt = x.reshape(B * T, N, C)                 # (BT, N, C)
            adj_exp = adj.unsqueeze(0).expand(B * T, -1, -1)  # (BT, N, N)
            x_agg = torch.bmm(adj_exp, x_bt)              # (BT, N, C)
            res = x_agg.view(B, T, N, C)
        else:
            # without adjacency, simply use the original input
            res = x

        # bottleneck compression + expansion
        res_flat = res.reshape(-1, C)
        z = self.compress(res_flat)
        z = self.activation(z)
        res_out = self.expand(z).view(B, T, N, C)

        # additive residual combination
        out = main_out + self.alpha * res_out
        return out
