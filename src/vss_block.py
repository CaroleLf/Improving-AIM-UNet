import torch
import torch.nn as nn
from mamba_ssm import Mamba


class SS2D(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.scan = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

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

        out_row = self._from_row_major(self.scan(seq_row), H, W)
        out_col = self._from_col_major(self.scan(seq_col), H, W)
        out_row_rev = self._from_row_major(torch.flip(self.scan(seq_row_rev), dims=[1]), H, W)
        out_col_rev = self._from_col_major(torch.flip(self.scan(seq_col_rev), dims=[1]), H, W)

        return out_row + out_col + out_row_rev + out_col_rev


if __name__ == "__main__":
    x = torch.randn(2, 32, 16, 16).to("cuda")
    ss2d = SS2D(dim=32).to("cuda")
    y = ss2d(x)
    print(y.shape)
