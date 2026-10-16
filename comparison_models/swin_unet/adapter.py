"""Adapter for the copied SwinUnet implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "swin_unet")
    from .Swin_UNet import SwinUnet

    return ComparisonModelBundle(
        SwinUnet(img_size=size[0], num_classes=1),
        OCTAProbabilityAdapter(size),
    )
