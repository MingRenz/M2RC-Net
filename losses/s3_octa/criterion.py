"""Apply the original batch-one IBP-clDice independently to each sample."""
import torch
from .clDice import IBP_clDice


class S3OCTALoss(torch.nn.Module):
    stochastic = True

    def __init__(self, evaluation_seed=0):
        super().__init__()
        self.evaluation_seed = int(evaluation_seed)
        self.original = IBP_clDice(alpha=0.5, smooth=1.0)

    def forward(self, prediction, target):
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("S3-OCTA loss requires matching [B,1,H,W] tensors")
        if prediction.shape[0] < 1 or prediction.shape[1] != 1:
            raise ValueError("S3-OCTA loss requires a nonempty single-channel batch")
        # Preserve the actual upstream training API and stochastic call order.
        values = [
            self.original(prediction[i:i + 1], target[i:i + 1])
            for i in range(prediction.shape[0])
        ]
        return torch.stack(values).mean()
