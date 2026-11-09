import torch
from torch import nn
import torch.nn.functional as F

LOSS_CONFIG = {"smooth_l1_beta": 0.1, "background_weight": 0.1, "s_min": -5.0, "s_max": 5.0}


def region_mean(loss, mask):
    mask = mask.to(loss.dtype)
    count = mask.flatten(1).sum(1)
    values = (loss * mask).flatten(1).sum(1)
    selected = count > 0
    if selected.any():
        return (values[selected] / count[selected]).mean()
    return loss.sum() * 0.0



def caliber_loss(pred, target, segmentation_gt, valid, c):
    error = F.smooth_l1_loss(
        pred, target,
        reduction="none",
        beta=c["smooth_l1_beta"],
    )
    vessel = (segmentation_gt > 0.5) & (valid > 0.5)
    background = segmentation_gt <= 0.5

    lv = region_mean(error, vessel)
    lb = region_mean(
        F.smooth_l1_loss(
            pred, torch.zeros_like(pred),
            reduction="none",
            beta=c["smooth_l1_beta"],
        ),
        background,
    )
    total = lv + c["background_weight"] * lb
    return total, lv, lb



class LossWeights(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.s_seg = nn.Parameter(torch.zeros(()))
        self.s_cal = nn.Parameter(torch.zeros(()))
        self.s_vessel = nn.Parameter(torch.zeros(()))
        self.low = c["s_min"]
        self.high = c["s_max"]

    def effective(self):
        return (
            self.s_seg.clamp(self.low, self.high),
            self.s_cal.clamp(self.low, self.high),
            self.s_vessel.clamp(self.low, self.high),
        )

    def forward(self, segmentation, caliber, vesselness):
        ss, sc, sv = self.effective()

        return (
            torch.exp(-ss) * segmentation + ss
            + torch.exp(-sc) * caliber + sc
            + torch.exp(-sv) * vesselness + sv
        )

    @torch.no_grad()
    def project(self):
        self.s_seg.clamp_(self.low, self.high)
        self.s_cal.clamp_(self.low, self.high)
        self.s_vessel.clamp_(self.low, self.high)

    def values(self):
        ss, sc, sv = self.effective()

        return {
            "s_seg": float(ss.detach()),
            "s_cal": float(sc.detach()),
            "s_vessel": float(sv.detach()),
            "segmentation_weight": float(
                torch.exp(-ss).detach()
            ),
            "caliber_weight": float(
                torch.exp(-sc).detach()
            ),
            "vesselness_weight": float(
                torch.exp(-sv).detach()
            ),
        }
