import torch
import torch.nn as nn


class CAB(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        s = x.mean(dim=(2, 3))                    # (B, C) global descriptor
        z = self.act(self.fc1(s))
        z = torch.sigmoid(self.fc2(z)).view(B, C, 1, 1)
        return x * z


class SAB(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=pad)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)          # (B, 1, H, W)
        max_out, _ = x.max(dim=1, keepdim=True)          # (B, 1, H, W)
        pooled = torch.cat([avg_out, max_out], dim=1)      # (B, 2, H, W)
        attn = torch.sigmoid(self.conv(pooled))              # (B, 1, H, W)
        return x * attn


class AttentionBridge(nn.Module):
    def __init__(self, channels, reduction=4, spatial_kernel=7):
        super().__init__()
        self.cab = CAB(channels, reduction=reduction)
        self.sab = SAB(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.cab(x)
        x = self.sab(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    bridge = AttentionBridge(channels=64)
    y = bridge(x)
    print(y.shape)
