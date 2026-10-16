"""Adapter for the copied single-class H_vmunet implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "h_vmunet")
    from .H_vmunet import H_vmunet

    # The copied source stores `pretrained` but does not load it.
    return ComparisonModelBundle(
        H_vmunet(num_classes=1, input_channels=1, pretrained=None),
        OCTAProbabilityAdapter(size),
    )
