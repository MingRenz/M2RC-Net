"""Keep the official two-class head and expose its foreground softmax."""
import torch
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


class PMDCAdapter(OCTAProbabilityAdapter):
    def extract_output(self, raw_output):
        if not isinstance(raw_output, torch.Tensor) or raw_output.ndim != 4 or raw_output.shape[1] != 2:
            raise TypeError("PMDC-Net must return [B,2,H,W] raw logits")
        probability = torch.softmax(raw_output, dim=1)[:, 1:2]
        return super().extract_output(probability)


def build(config):
    size = require_octa_config(config, "pmdc_net")
    from .pmdc_net import PMDC_Net
    model = PMDC_Net(n_channels=1, n_classes=2, bilinear=False,
                     use_msfam=True, use_cwm=True, use_pdcm=True, use_dafm=True)
    return ComparisonModelBundle(model, PMDCAdapter(size))
