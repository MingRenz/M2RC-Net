"""Thin adapter for the copied model source."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


def build(config):
    size = require_octa_config(config, "hv_octamamba")
    from .HV_OCTAMamba import HV_OCTAMamba
    return ComparisonModelBundle(HV_OCTAMamba(), OCTAProbabilityAdapter(size))
