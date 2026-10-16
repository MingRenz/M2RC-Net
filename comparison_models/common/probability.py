"""Explicit OCTA input and probability-output contract for copied baselines."""
from collections.abc import Mapping

import torch

from .adapter import ComparisonAdapter, SegmentationOutput


OCTA_IMAGE_SIZE = (224, 224)


def require_octa_config(config, model_name):
    """Reject accidental use of a proposed-model config or a different protocol."""
    if not isinstance(config, Mapping):
        raise TypeError("comparison config must be a mapping")
    image_size = config.get("image_size")
    if not isinstance(image_size, (list, tuple)) or tuple(image_size) != OCTA_IMAGE_SIZE:
        raise ValueError("comparison image_size must be [224, 224]")
    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("name") != model_name:
        raise ValueError(f"model.name must be {model_name!r}")
    extra = set(model) - {"name"}
    if extra:
        raise ValueError(f"unsupported model options: {sorted(extra)}")
    return OCTA_IMAGE_SIZE


class OCTAProbabilityAdapter(ComparisonAdapter):
    """Keep the source model's sigmoid probability and 224x224 grid unchanged."""

    def __init__(self, image_size=OCTA_IMAGE_SIZE):
        self.image_size = tuple(image_size)

    def prepare_input(self, image):
        if not isinstance(image, torch.Tensor) or image.ndim != 4:
            raise TypeError("input must be a [B,1,H,W] tensor")
        if image.shape[1] != 1 or tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(f"input must be [B,1,{self.image_size[0]},{self.image_size[1]}]")
        return image

    def extract_output(self, raw_output):
        output = SegmentationOutput(raw_output, "probability")
        if tuple(output.prediction.shape[-2:]) != self.image_size:
            raise ValueError(f"model output must keep spatial size {self.image_size}")
        return output
