"""CAMS: caliber/vesselness-guided multi-scale SSM representation."""
import torch
from torch import nn

from .backbone import VSSBlock


def deterministic_resize224(x, size):
    """Same fixed-pixel sampling as the verified 224->112/56 CAMS implementation."""
    target = tuple(size)
    if tuple(x.shape[-2:]) == target:
        return x
    if tuple(x.shape[-2:]) != (224, 224) or target not in ((112, 112), (56, 56)):
        raise RuntimeError(f"Unsupported CAMS resize {tuple(x.shape[-2:])} -> {target}")
    step = 224 // target[0]
    offset = step // 2 - 1
    a = x[..., offset::step, offset::step]
    b = x[..., offset::step, offset + 1::step]
    c = x[..., offset + 1::step, offset::step]
    d = x[..., offset + 1::step, offset + 1::step]
    return (a + b + c + d) * 0.25


def scale_convolutions(in_channels, out_channels, kernel):
    half = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (1, kernel), padding=(0, half)),
        nn.Conv2d(out_channels, out_channels, (kernel, 1), padding=(half, 0)),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
    )


class CAMSBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stage):
        super().__init__()
        self.stage = stage
        self.scale3 = scale_convolutions(in_channels, out_channels, 3)
        self.scale5 = scale_convolutions(in_channels, out_channels, 5)
        self.scale7 = scale_convolutions(in_channels, out_channels, 7)
        # Independent, unchanged stable VSS/SS2D instances for the three branches.
        self.vss3 = VSSBlock(hidden_dim=out_channels)
        self.vss5 = VSSBlock(hidden_dim=out_channels)
        self.vss7 = VSSBlock(hidden_dim=out_channels)
        self.fusion = nn.Conv2d(in_channels + 3 * out_channels, out_channels, 1)
        self.a3 = nn.Parameter(torch.zeros(1))
        self.a5 = nn.Parameter(torch.zeros(1))
        self.a7 = nn.Parameter(torch.zeros(1))
        self.caliber = None
        self.vesselness = None
        self.capture_shapes = False
        self.shapes = {}

    @staticmethod
    def _vss(module, value):
        value = value.permute(0, 2, 3, 1).contiguous()
        value = module(value)
        return value.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        if self.caliber is None or self.vesselness is None:
            raise RuntimeError(f"{self.stage}: predicted R/V not provided")
        r = deterministic_resize224(self.caliber, x.shape[-2:])
        v = deterministic_resize224(self.vesselness, x.shape[-2:])
        g = v * (1.0 - r)
        f3 = self.scale3(x)
        f5 = self.scale5(x)
        f7 = self.scale7(x)
        fcams3 = self._vss(self.vss3, f3 * torch.exp(self.a3.view(1, 1, 1, 1) * g))
        fcams5 = self._vss(self.vss5, f5 * torch.exp(self.a5.view(1, 1, 1, 1) * g))
        fcams7 = self._vss(self.vss7, f7 * torch.exp(self.a7.view(1, 1, 1, 1) * g))
        merged = torch.cat((x, fcams3, fcams5, fcams7), dim=1)
        output = self.fusion(merged)
        if self.capture_shapes:
            self.shapes = {
                name: tuple(tensor.shape)
                for name, tensor in (
                    ("x", x), ("R", r), ("V", v), ("F3", f3), ("F5", f5),
                    ("F7", f7), ("Fcams3", fcams3), ("Fcams5", fcams5),
                    ("Fcams7", fcams7), ("Concat", merged), ("CAMS output", output),
                )
            }
        return output


"""Stable Clean SSM Base with six independent CAMS stages."""
import torch
from torch import nn

from .backbone import StableCleanSSMBase



STAGES = ("e1", "e2", "e3", "d3", "d2", "d1")


def _caliber_head(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, 1, 1),
        nn.Sigmoid(),
    )


def _vesselness_head(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, 1, 1),
    )


class StableCleanSSMBaseCAMS(StableCleanSSMBase):
    def __init__(self):
        # Construct the unchanged Base first. Its own initialization stream is preserved.
        super().__init__()
        channels = self.embedding[0].out_channels
        with torch.random.fork_rng(devices=[]):
            self.caliber_head = _caliber_head(channels)
            self.vesselness_head = _vesselness_head(channels)
            for stage in STAGES:
                block = getattr(self, stage).octamamba
                block.conv = CAMSBlock(
                    block.in_c, block.residual_conv.out_channels, stage
                )

    def guided(self):
        return {
            stage: getattr(self, stage).octamamba.conv
            for stage in STAGES
        }

    def forward(self, image):
        f0 = self.embedding(image)
        r = self.caliber_head(f0)
        vesselness_logits = self.vesselness_head(f0)
        v = torch.sigmoid(vesselness_logits)
        modules = self.guided()
        for module in modules.values():
            module.caliber = r
            module.vesselness = v
        try:
            x, skip1 = self.e1(f0)
            x, skip2 = self.e2(x)
            x, skip3 = self.e3(x)
            x = self.d3(x, skip3)
            x = self.d2(x, skip2)
            x = self.d1(x, skip1)
            segmentation_logits = self.conv_out(x)
            return segmentation_logits, r, vesselness_logits
        finally:
            for module in modules.values():
                module.caliber = None
                module.vesselness = None
