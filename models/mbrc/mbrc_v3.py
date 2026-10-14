import torch
from torch import nn
import torch.nn.functional as F

from .mbrc_v2 import MBRCV2, ResidualBlock, cna


class MBRCV3(MBRCV2):
    def __init__(self, f1_channels=32, seed=0):
        # 先按V2.1原顺序创建全部共有模块。
        super().__init__(f1_channels=f1_channels)
        old_first = self.shared[0]
        narrow_first = cna(8, 32, 3, 8)
        with torch.no_grad():
            narrow_first[0].weight.copy_(old_first[0].weight[:, 32:40])
            narrow_first[1].weight.copy_(old_first[1].weight)
            narrow_first[1].bias.copy_(old_first[1].bias)
        self.shared[0] = narrow_first
        del self.image_branch
        del self.f1_branch

        # 新模块使用独立CPU随机流，不改变共有模块或后续训练的随机序列。
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(20260929 + seed)
            self.recovery3 = nn.Sequential(
                cna(64, 32, 1, 8), ResidualBlock()
            )
            self.project2 = cna(32, 32, 1, 8)
            self.merge2 = cna(64, 32, 3, 8)
            self.recovery2 = ResidualBlock()
            self.project1 = cna(16, 32, 1, 8)
            self.merge1 = cna(64, 32, 3, 8)
            self.recovery1 = ResidualBlock()
            self.fusion = cna(64, 32, 3, 8)

    def forward(self, image, f1, z_base, d3, d2, d1):
        batch = z_base.shape[0]
        expected = (
            (z_base, (batch, 1, 224, 224), 'Z_base'),
            (d3, (batch, 64, 56, 56), 'D3'),
            (d2, (batch, 32, 112, 112), 'D2'),
            (d1, (batch, 16, 224, 224), 'D1'),
        )
        for tensor, shape, name in expected:
            if tuple(tensor.shape) != shape:
                raise RuntimeError(f'{name} shape {tuple(tensor.shape)} != {shape}')

        shared = self.shared(self.logit_branch(z_base))

        r3 = self.recovery3(d3)
        r2 = self.recovery2(self.merge2(torch.cat([
            F.interpolate(r3, size=d2.shape[-2:], mode="nearest"),
            self.project2(d2),
        ], dim=1)))
        r1 = self.recovery1(self.merge1(torch.cat([
            F.interpolate(r2, size=d1.shape[-2:], mode="nearest"),
            self.project1(d1),
        ], dim=1)))
        recovery = F.interpolate(
            r1, size=z_base.shape[-2:], mode="nearest"
        )
        fused = self.fusion(
            torch.cat([shared, recovery], dim=1)
        )

        f_edge = self.boundary_features(fused)
        edge_logits = self.boundary_output(f_edge)
        features = torch.cat([fused, f_edge], dim=1)

        action_logits = self.action_head(features)
        action_prob = torch.softmax(action_logits, dim=1)
        magnitude = F.softplus(self.magnitude_head(features))

        delta_z = (
            action_prob[:, 0:1] * magnitude[:, 0:1]
            - action_prob[:, 1:2] * magnitude[:, 1:2]
        )
        z_final = z_base.detach() + delta_z
        return (
            z_final, edge_logits, delta_z,
            action_logits, action_prob, magnitude,
        )
