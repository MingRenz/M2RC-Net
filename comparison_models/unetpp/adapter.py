"""Adapter for the copied ResNet34UnetPlus implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "unetpp")
    from .unetpp import ResNet34UnetPlus

    return ComparisonModelBundle(
        ResNet34UnetPlus(num_channels=1, num_class=1),
        OCTAProbabilityAdapter(size),
    )
