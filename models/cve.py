from torch import nn


def build_caliber_head(channels):
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


def build_vesselness_head(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(channels),
        nn.GELU(),
        nn.Conv2d(channels, 1, 1),
    )
