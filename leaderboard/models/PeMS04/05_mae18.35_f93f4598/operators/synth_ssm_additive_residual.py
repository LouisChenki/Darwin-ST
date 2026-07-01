import torch
import torch.nn as nn

class ssm_additive_residual(nn.Module):
    def __init__(self, channels, num_nodes, **kw):
        super().__init__()
        self.num_nodes = num_nodes
        d_state = 4  # small state dimension
        self.d_state = d_state

        # Main path: non‑trivial transformation (Linear + ReLU)
        self.main_net = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU()
        )

        # Alpha parameter for additive residual
        self.alpha = nn.Parameter(torch.zeros(1))

        # SSM parameters
        # Diagonal A: A = -exp(A_log) to enforce negative values with magnitude ≤1
        self.A_log = nn.Parameter(torch.full((d_state,), -2.0))
        # Input matrix B: (d_state, channels)
        self.B = nn.Parameter(torch.randn(d_state, channels))
        # Output matrix C: (channels, d_state)
        self.C = nn.Parameter(torch.randn(channels, d_state))

    def forward(self, x, adj=None):
        # x shape: (B, T, N, C)
        # Ensure consistent dtype
        param_dtype = self.A_log.dtype
        if x.dtype != param_dtype:
            x = x.to(param_dtype)

        B, T, N, C = x.shape
        device = x.device

        # Flatten batch and node dimensions
        x_flat = x.contiguous().view(B * N, T, C)  # (B*N, T, C)

        # Main path transformation
        main_out = self.main_net(x_flat)  # (B*N, T, C)

        # SSM branch (discrete‑time state space model)
        h = torch.zeros(B * N, self.d_state, device=device, dtype=param_dtype)
        A_diag = -torch.exp(self.A_log)  # (d_state,)

        out_ssm = torch.zeros_like(x_flat)  # (B*N, T, C)

        for t in range(T):
            x_t = x_flat[:, t, :]  # (B*N, C)
            # State update: h = A * h + B * x_t
            # Bx = x_t @ B.T   (B*N, d_state)
            Bx = x_t @ self.B.T  # (B*N, d_state)
            h = A_diag * h + Bx   # element‑wise diagonal A
            # Output: y_t = C * h
            # y_t = h @ C.T   (B*N, C)
            y_t = h @ self.C.T  # (B*N, C)
            out_ssm[:, t, :] = y_t

        # Additive residual: output = main_path + alpha * ssm_branch
        output = main_out + self.alpha * out_ssm

        # Restore original shape
        output = output.view(B, T, N, C)
        return output
