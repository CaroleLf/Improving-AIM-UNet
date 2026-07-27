import torch
import torch.nn as nn


class AsymConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(1, kernel_size), padding=(0, pad))
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(kernel_size, 1), padding=(pad, 0))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class AIMModule(nn.Module):
    def __init__(self, channels, kernel_sizes=(3, 5, 7), reduction=4):
        super().__init__()
        self.M = len(kernel_sizes)
        self.channels = channels

        self.branches = nn.ModuleList([AsymConv(channels, k) for k in kernel_sizes])

        hidden = max(channels // reduction, 4)
        self.gate_fc1 = nn.Linear(channels, hidden)
        self.gate_fc2 = nn.Linear(hidden, channels * self.M)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, return_branches=False):
        B, C, H, W = x.shape
        branch_outputs = [branch(x) for branch in self.branches]
        s = x.mean(dim=(2, 3))
        z = self.act(self.gate_fc1(s))
        z = self.gate_fc2(z).view(B, self.M, C)
        alpha = torch.softmax(z, dim=1)

        stacked = torch.stack(branch_outputs, dim=1)
        alpha_ = alpha.unsqueeze(-1).unsqueeze(-1)
        fused = (alpha_ * stacked).sum(dim=1)

        if return_branches:
            return fused, branch_outputs
        return fused


if __name__ == "__main__":
    x = torch.randn(2, 64, 32, 32)
    aim = AIMModule(channels=64)
    y = aim(x)
    print(y.shape)
