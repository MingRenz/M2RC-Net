from torch import nn
import torch.nn.functional as F

def cna(in_channels, out_channels, kernel_size, groups):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2, bias=False,
        ),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(),
    )

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = cna(32, 32, 3, 8)
        self.second = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
        )

    def forward(self, x):
        return F.silu(x + self.second(self.first(x)))
