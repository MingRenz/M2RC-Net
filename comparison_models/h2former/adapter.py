"""Adapter for the copied res34_swin_MS implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "h2former")
    from .H2Former import res34_swin_MS

    return ComparisonModelBundle(
        res34_swin_MS(image_size=size[0], num_class=1),
        OCTAProbabilityAdapter(size),
    )
