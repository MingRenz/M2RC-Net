"""MBRC assembly. Disabled mechanisms have no modules or parameters."""

import torch

from .mbrc_v3 import MBRCV3
from .recovery import remove_multistage_modules, recover
from .decision import set_decision, decide
from .boundary import set_boundary, boundary_features


class VariantMBRC(MBRCV3):
    def __init__(self, variant, f1_channels=32):
        super().__init__(f1_channels=f1_channels)
        self.multi = variant.multi
        if not self.multi:
            remove_multistage_modules(self)
        set_decision(self, variant.action, variant.boundary)
        set_boundary(self, variant.boundary)

    def forward(self, z_base, d1, d2=None, d3=None):
        if tuple(z_base.shape[1:]) != (1, 224, 224):
            raise RuntimeError("Z_base must have shape [B,1,224,224]")
        if tuple(d1.shape[1:]) != (16, 224, 224):
            raise RuntimeError("D1 must have shape [B,16,224,224]")
        if self.multi and (
            tuple(d2.shape[1:]) != (32, 112, 112)
            or tuple(d3.shape[1:]) != (64, 56, 56)
        ):
            raise RuntimeError("D2/D3 feature shapes differ from strict Mamba")

        shared = self.shared(self.logit_branch(z_base))
        recovery = recover(self, d1, d2, d3)
        fused = self.fusion(torch.cat([shared, recovery], dim=1))
        features, edge_logits = boundary_features(self, fused)
        delta_z, action_logits = decide(self, features)
        z_final = z_base.detach() + delta_z
        return z_final, delta_z, action_logits, edge_logits
