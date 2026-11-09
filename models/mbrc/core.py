import torch
from torch import nn
from .common import cna, ResidualBlock
from .mfr import recover
from .bfg import boundary_features
from .amrc import decide

class MBRC(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        cna(1, 16, 3, 4)
        cna(32, 16, 1, 4)
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
        old_first = self.shared[0]
        narrow_first = cna(8, 32, 3, 8)
        with torch.no_grad():
            narrow_first[0].weight.copy_(old_first[0].weight[:, 32:40])
            narrow_first[1].weight.copy_(old_first[1].weight)
            narrow_first[1].bias.copy_(old_first[1].bias)
        self.shared[0] = narrow_first

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
        self.multi = True
        self.use_boundary = True
        self.use_action = True

    def forward(self, z_base, d1, d2=None, d3=None):
        if tuple(z_base.shape[1:]) != (1, 224, 224):
            raise RuntimeError("Z_base must have shape [B,1,224,224]")
        if tuple(d1.shape[1:]) != (16, 224, 224):
            raise RuntimeError("D1 must have shape [B,16,224,224]")
        if self.multi and (
            tuple(d2.shape[1:]) != (32, 112, 112)
            or tuple(d3.shape[1:]) != (64, 56, 56)
        ):
            raise RuntimeError("D2/D3 feature shapes differ from frozen backbone")

        shared = self.shared(self.logit_branch(z_base))
        recovery = recover(self, d1, d2, d3)
        fused = self.fusion(torch.cat([shared, recovery], dim=1))
        features, edge_logits = boundary_features(self, fused)
        delta_z, action_logits = decide(self, features)
        z_final = z_base.detach() + delta_z
        return z_final, delta_z, action_logits, edge_logits
