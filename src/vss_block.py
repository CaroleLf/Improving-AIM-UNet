import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SS2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # separate scan weights per direction, matching the official VMamba SS2D
        # (only A_log/D are shared there; we approximate with 4 independent Mamba
        # modules rather than hand-splitting mamba_ssm's internals)
        self.scans = nn.ModuleList([
            Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(4)
        ])

    @staticmethod
    def _to_row_major(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _to_col_major(x):
        return x.transpose(2, 3).flatten(2).transpose(1, 2)

    @staticmethod
    def _from_row_major(seq, H, W):
        B, L, C = seq.shape
        return seq.transpose(1, 2).view(B, C, H, W)

    @staticmethod
    def _from_col_major(seq, H, W):
        B, L, C = seq.shape
        return seq.transpose(1, 2).view(B, C, W, H).transpose(2, 3)

    def forward(self, x):
        B, C, H, W = x.shape

        seq_row = self._to_row_major(x)
        seq_col = self._to_col_major(x)
        seq_row_rev = torch.flip(seq_row, dims=[1])
        seq_col_rev = torch.flip(seq_col, dims=[1])

        out_row = self._from_row_major(self.scans[0](seq_row), H, W)
        out_col = self._from_col_major(self.scans[1](seq_col), H, W)
        out_row_rev = self._from_row_major(torch.flip(self.scans[2](seq_row_rev), dims=[1]), H, W)
        out_col_rev = self._from_col_major(torch.flip(self.scans[3](seq_col_rev), dims=[1]), H, W)

        return out_row + out_col + out_row_rev + out_col_rev


class VSSBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = x.flatten(2).transpose(1, 2)          # (B, L, C)
        x_norm = self.norm(x_seq)                      # LN(X)

        u1 = self.act(self.w1(x_norm))                   # U1 = SiLU(W1 . LN(X))
        u1 = u1.transpose(1, 2).view(B, C, H, W)

        u2 = self.w2(x_norm)                               # W2 . LN(X)
        u2 = u2.transpose(1, 2).view(B, C, H, W)
        u2 = self.dwconv(u2)                                 # DWConv(...)
        u2 = self.act(u2)                                      # SiLU(...)
        u2 = self.ss2d(u2)                                       # SS2D(...)

        fused = u1 * u2                                            # U1 (x) U2
        fused_seq = fused.flatten(2).transpose(1, 2)                 # (B, L, C)
        out = self.out_proj(fused_seq)
        out = out.transpose(1, 2).view(B, C, H, W)

        return x + out                                                 # residual


if __name__ == "__main__":
    x = torch.randn(2, 32, 16, 16).to("cuda")

    ss2d = SS2D(dim=32).to("cuda")
    y = ss2d(x)
    print("SS2D:", y.shape)

    vss = VSSBlock(dim=32).to("cuda")
    z = vss(x)
    print("VSSBlock:", z.shape)
