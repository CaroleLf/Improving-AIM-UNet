import torch
import torch.nn as nn

from aim_module import AIMModule
from patch_ops import PatchEmbed, PatchMerging
from vss_block import VSSBlock


class VSSEncoder(nn.Module):
    def __init__(self, in_channels=1, base_dim=64, depths=(2, 2, 2, 2),
                 d_state=16, d_conv=4, expand=2, use_aim=False):
        super().__init__()
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=base_dim, patch_size=4)
        self.use_aim = use_aim

        dims = [base_dim * (2 ** i) for i in range(len(depths))]
        if use_aim:
            self.aims = nn.ModuleList([AIMModule(dim) for dim in dims])
        self.stages = nn.ModuleList([
            nn.Sequential(*[
                VSSBlock(dim, d_state=d_state, d_conv=d_conv, expand=expand)
                for _ in range(depth)
            ])
            for dim, depth in zip(dims, depths)
        ])
        self.merges = nn.ModuleList([PatchMerging(dim) for dim in dims[:-1]])

    def forward(self, x):
        x = self.patch_embed(x)
        skips = []
        for i, stage in enumerate(self.stages):
            if self.use_aim:
                x = self.aims[i](x)
            x = stage(x)
            skips.append(x)
            if i < len(self.stages) - 1:
                x = self.merges[i](x)
        return x, skips


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256).to("cuda")
    encoder = VSSEncoder(in_channels=1).to("cuda")
    out, skips = encoder(x)
    print("bottleneck:", out.shape)
    for i, s in enumerate(skips):
        print(f"skip {i+1}:", s.shape)
