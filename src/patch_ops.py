import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)                          # (B, C, H/4, W/4)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)            # (B, L, C)
        x = self.norm(x)
        x = x.transpose(1, 2).view(B, C, H, W)        # back to (B, C, H/4, W/4)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                  # (B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)      # (B, H/2, W/2, 4C)

        x = self.norm(x)
        x = self.reduction(x)                          # (B, H/2, W/2, 2C)
        x = x.permute(0, 3, 1, 2)                        # (B, 2C, H/2, W/2)
        return x


class PatchExpand(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                    
        x = self.expand(x)                            
        x = x.view(B, H, W, 2, 2, C // 2)                
        x = x.permute(0, 1, 3, 2, 4, 5)                    
        x = x.reshape(B, H * 2, W * 2, C // 2)                 
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)                               
        return x


if __name__ == "__main__":
    x = torch.randn(2, 1, 256, 256)
    patch_embed = PatchEmbed(in_channels=1, embed_dim=64, patch_size=4)
    y = patch_embed(x)
    print("PatchEmbed:", y.shape)

    merge = PatchMerging(dim=64)
    z = merge(y)
    print("PatchMerging:", z.shape)

    expand = PatchExpand(dim=128)
    w = expand(z)
    print("PatchExpand:", w.shape)
