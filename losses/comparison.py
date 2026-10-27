"""Comparison objectives: shared Dice by default, core method losses by exception."""
from contextlib import contextmanager, nullcontext
import torch
from .segmentation import DiceLoss


def build_comparison_loss(model_name, seed=0):
    if model_name == "s3_octa":
        from .s3_octa.criterion import S3OCTALoss
        return S3OCTALoss(evaluation_seed=seed)
    return DiceLoss()


def comparison_loss_description(model_name):
    if model_name == "s3_octa":
        return "S3-OCTA original IBP_clDice: 0.5 original Dice + 0.5 IBP-clDice; per-image mean"
    return "Base DiceLoss on source sigmoid probability"


def comparison_train_loss_reporting(model_name):
    if model_name == "s3_octa":
        return "mean(per-image-mean IBP_clDice / batch size); historical logging scale only"
    return "mean(batch DiceLoss / batch size), matching Base"


def comparison_evaluation_context(criterion, device):
    if not getattr(criterion, "stochastic", False):
        return nullcontext()
    return _isolated_diagnostic_rng(criterion.evaluation_seed, torch.device(device))


@contextmanager
def _isolated_diagnostic_rng(seed, device):
    import random
    import numpy as np
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            # Seed CPU and only the selected CUDA generator; do not alter other GPUs.
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
