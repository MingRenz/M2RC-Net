import torch
from torch import nn
from .backbone import SSMBase, VSSBlock
from .mscf_guided import STAGES, scale_convolutions, build_caliber_head, build_vesselness_head


class MSCFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stage):
        super().__init__()
        self.stage = stage
        self.scale3 = scale_convolutions(in_channels, out_channels, 3)
        self.scale5 = scale_convolutions(in_channels, out_channels, 5)
        self.scale7 = scale_convolutions(in_channels, out_channels, 7)
        self.vss3 = VSSBlock(hidden_dim=out_channels)
        self.vss5 = VSSBlock(hidden_dim=out_channels)
        self.vss7 = VSSBlock(hidden_dim=out_channels)
        self.fusion = nn.Conv2d(in_channels + 3*out_channels, out_channels, 1)

    @staticmethod
    def _vss(module, value):
        value = module(value.permute(0, 2, 3, 1).contiguous())
        return value.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        f3 = self._vss(self.vss3, self.scale3(x))
        f5 = self._vss(self.vss5, self.scale5(x))
        f7 = self._vss(self.vss7, self.scale7(x))
        return self.fusion(torch.cat((x, f3, f5, f7), dim=1))


class MSCFBackbone(SSMBase):
    def __init__(self):
        super().__init__()
        channels = self.embedding[0].out_channels
        with torch.random.fork_rng(devices=[]):
            temporary_caliber = build_caliber_head(channels)
            temporary_vesselness = build_vesselness_head(channels)
            del temporary_caliber, temporary_vesselness
            for stage in STAGES:
                block = getattr(self, stage).feature_block
                block.conv = MSCFBlock(
                    block.in_c, block.residual_conv.out_channels, stage)
