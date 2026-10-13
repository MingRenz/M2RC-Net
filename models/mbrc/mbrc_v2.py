import torch
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


class MBRCV2(nn.Module):
    def __init__(self, f1_channels=32):
        super().__init__()
        self.image_branch = cna(1, 16, 3, 4)
        self.f1_branch = cna(f1_channels, 16, 1, 4)
        self.logit_branch = cna(1, 8, 3, 4)

        self.shared = nn.Sequential(
            cna(40, 32, 3, 8),
            cna(32, 32, 3, 8),
            ResidualBlock(),
            ResidualBlock(),
        )

        self.boundary_features = nn.Sequential(
            cna(32, 16, 3, 4),
            cna(16, 16, 3, 4),
        )
        self.boundary_output = nn.Conv2d(16, 1, 1)

        self.action_head = nn.Sequential(
            cna(48, 32, 3, 8),
            cna(32, 16, 3, 4),
            nn.Conv2d(16, 3, 1),
        )
        self.magnitude_head = nn.Sequential(
            cna(48, 32, 3, 8),
            cna(32, 16, 3, 4),
            nn.Conv2d(16, 2, 1),
        )

        with torch.no_grad():
            nn.init.zeros_(self.action_head[-1].weight)
            self.action_head[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 2.0])
            )
            nn.init.zeros_(self.magnitude_head[-1].weight)
            nn.init.zeros_(self.magnitude_head[-1].bias)

    def forward(self, image, f1, z_base):
        if image.shape[-2:] != (224, 224):
            raise RuntimeError(f"BAR输入尺寸异常：{tuple(image.shape)}")

        if f1.shape[-2:] != image.shape[-2:]:
            f1 = F.interpolate(
                f1, size=image.shape[-2:], mode="nearest"
            )

        shared = self.shared(torch.cat([
            self.image_branch(image),
            self.f1_branch(f1),
            self.logit_branch(z_base),
        ], dim=1))

        f_edge = self.boundary_features(shared)
        edge_logits = self.boundary_output(f_edge)
        features = torch.cat([shared, f_edge], dim=1)

        action_logits = self.action_head(features)
        action_prob = torch.softmax(action_logits, dim=1)
        magnitude = F.softplus(self.magnitude_head(features))

        delta_z = (
            action_prob[:, 0:1] * magnitude[:, 0:1]
            - action_prob[:, 1:2] * magnitude[:, 1:2]
        )
        z_final = z_base + delta_z

        return (
            z_final, edge_logits, delta_z,
            action_logits, action_prob, magnitude,
        )
