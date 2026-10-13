"""Interface contracts for external models; no model mathematics are implemented here."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SegmentationOutput:
    """Raw binary segmentation output and any official auxiliary outputs.

    representation states whether prediction is logits or probability. Adapters
    must not silently apply sigmoid, interpolation, thresholding, or discard
    auxiliary outputs used by the official training objective.
    """
    prediction: torch.Tensor
    representation: str
    auxiliary: Any = None

    def __post_init__(self):
        if self.representation not in ("logits", "probability"):
            raise ValueError("representation must be logits or probability")
        if not isinstance(self.prediction, torch.Tensor) or self.prediction.ndim != 4:
            raise TypeError("prediction must be a [B,C,H,W] tensor")
        if self.prediction.shape[1] != 1:
            raise ValueError("binary segmentation prediction must have one channel")


class ComparisonAdapter(ABC):
    """Model-specific interface adaptation, separate from official model code."""

    def prepare_input(self, image: torch.Tensor) -> torch.Tensor:
        """Identity by default. A model adapter may explicitly document changes."""
        return image

    @abstractmethod
    def extract_output(self, raw_output: Any) -> SegmentationOutput:
        """Select the primary prediction and preserve relevant auxiliary output."""


@dataclass(frozen=True)
class ComparisonModelBundle:
    model: nn.Module
    adapter: ComparisonAdapter

    def __post_init__(self):
        if not isinstance(self.model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not isinstance(self.adapter, ComparisonAdapter):
            raise TypeError("adapter must implement ComparisonAdapter")
