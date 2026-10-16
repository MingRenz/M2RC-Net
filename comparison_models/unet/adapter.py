"""Adapter for the copied ResNet18-based U-Net implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "unet")
    from .Unet import Unet

    return ComparisonModelBundle(Unet(n_class=1), OCTAProbabilityAdapter(size))
