import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveDilatedConv(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.channels = channels
        self.num_nodes = num_nodes
        self.alpha = nn.Parameter(torch.zeros(1))
        kernel_size = kw.get('kernel_size', 3)
        num_layers = kw.get('num_layers', 4)
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # dilated causal convolution layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            conv = nn.Conv1d(channels, channels, kernel_size,
                             dilation=dilation, bias=True)
            self.convs.append(conv)
        self.act = nn.ReLU()

    def forward(self, x, adj=None):
        B, T, N, C = x.shape

        # reshape for 1D convolution along time: (B*N, C, T)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * N, C, T)

        # ---------- main path ----------
        h = x_reshaped
        for conv in self.convs:
            # causal left‑padding
            dilation = conv.dilation[0]
            pad = (self.kernel_size - 1) * dilation
            h_padded = F.pad(h, (pad, 0))
            h_out = conv(h_padded)
            h_out = self.act(h_out)
            h = h + h_out   # residual

        out_main = h.view(B, N, C, T).permute(0, 3, 1, 2)   # (B, T, N, C)

        # ---------- augmented path (stop gradient) ----------
        # small random temporal perturbation
        noise_std = 0.05
        x_aug = x + torch.randn_like(x) * noise_std
        x_aug_reshaped = x_aug.permute(0, 2, 3, 1).reshape(B * N, C, T)

        h_aug = x_aug_reshaped
        for conv in self.convs:
            dilation = conv.dilation[0]
            pad = (self.kernel_size - 1) * dilation
            h_aug_padded = F.pad(h_aug, (pad, 0))
            h_aug_out = conv(h_aug_padded)
            h_aug_out = self.act(h_aug_out)
            h_aug = h_aug + h_aug_out

        out_aug = h_aug.view(B, N, C, T).permute(0, 3, 1, 2)
