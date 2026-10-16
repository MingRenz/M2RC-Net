"""Adapter for the copied single-class VMUNet implementation."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "vm_unet")
    from .VM_UNet import VMUNet

    # No checkpoint is loaded here; the source's load_from() is opt-in.
    return ComparisonModelBundle(
        VMUNet(input_channels=1, num_classes=1, load_ckpt_path=None),
        OCTAProbabilityAdapter(size),
    )
