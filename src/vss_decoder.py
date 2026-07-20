import torch
import torch.nn as nn

from patch_ops import PatchExpand
from vss_block import VSSBlock


class VSSDecoder(nn.Module):
    def __init__(self, base_dim=64, depths=(2, 2, 2), out_channels=1,
                 d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        dims = [base_dim * (2 ** i) for i in range(len(depths) + 1)]   # [64,128,256,512]
        rev_dims = dims[::-1]                                            # [512,256,128,64]

        self.expands = nn.ModuleList([PatchExpand(dim) for dim in rev_dims[:-1]])
        self.reduces = nn.ModuleList([nn.Linear(dim, dim // 2) for dim in rev_dims[:-1]])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                VSSBlock(dim // 2, d_state=d_state, d_conv=d_conv, expand=expand_factor)
                for _ in range(depth)
            ])
            for dim, depth in zip(rev_dims[:-1], depths)
        ])

        self.final_expand = nn.ConvTranspose2d(base_dim, out_channels, kernel_size=4, stride=4)

    def forward(self, x, skips):
        rev_skips = skips[:-1][::-1]   # deepest-first, excluding the bottleneck (skips[-1] == x)

        for expand, reduce, stage, skip in zip(self.expands, self.reduces, self.stages, rev_skips):
            x = expand(x)                          # halve channels, double resolution
            x = torch.cat([x, skip], dim=1)          # concat on channel dim (simple skip, no SAB/CAB yet)
            x = x.permute(0, 2, 3, 1)                  # (B, H, W, C) for Linear
            x = reduce(x)
            x = x.permute(0, 3, 1, 2)                    # back to (B, C, H, W)
            x = stage(x)

        x = self.final_expand(x)
        return torch.sigmoid(x)


if __name__ == "__main__":
    from vss_encoder import VSSEncoder

    x = torch.randn(2, 1, 256, 256).to("cuda")
    encoder = VSSEncoder(in_channels=1).to("cuda")
    bottleneck, skips = encoder(x)

    decoder = VSSDecoder(base_dim=64, out_channels=1).to("cuda")
    out = decoder(bottleneck, skips)
    print("output:", out.shape)
