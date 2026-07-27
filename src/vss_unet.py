import torch
import torch.nn as nn

from vss_decoder import VSSDecoder
from vss_encoder import VSSEncoder


class VSSUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_dim=64,
                 encoder_depths=(2, 2, 2, 2), decoder_depths=(2, 2, 2),
                 d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.encoder = VSSEncoder(in_channels=in_channels, base_dim=base_dim,
                                   depths=encoder_depths, d_state=d_state, d_conv=d_conv, expand=expand)
        self.decoder = VSSDecoder(base_dim=base_dim, depths=decoder_depths, out_channels=out_channels,
                                   d_state=d_state, d_conv=d_conv, expand_factor=expand)

    def forward(self, x):
        bottleneck, skips, _ = self.encoder(x)
        return self.decoder(bottleneck, skips)


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256).to("cuda")
    model = VSSUNet(in_channels=1, out_channels=1).to("cuda")
    y = model(x)
    print(y.shape)
    print("params:", sum(p.numel() for p in model.parameters()))
