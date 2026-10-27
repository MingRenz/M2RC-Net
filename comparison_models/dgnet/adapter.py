"""DGNet uses its native single-channel sigmoid output unchanged."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "dgnet")
    from .DGNet import DGNet
    model = DGNet(inp_c=1, input_resolution=size, n_classes=1)
    return ComparisonModelBundle(model, OCTAProbabilityAdapter(size))
