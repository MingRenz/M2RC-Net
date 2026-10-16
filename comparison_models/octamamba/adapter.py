"""Thin adapter for the copied model source."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "octamamba")
    from .OCTAMamba import OCTAMamba
    return ComparisonModelBundle(OCTAMamba(), OCTAProbabilityAdapter(size))
