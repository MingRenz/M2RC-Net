"""Boundary feature path and its 3x3 internal-boundary target."""

import torch
import torch.nn.functional as F


def set_boundary(mbrc, enabled):
    mbrc.use_boundary = enabled
    if not enabled:
        del mbrc.boundary_features
        del mbrc.boundary_output


def boundary_features(mbrc, fused):
    if not mbrc.use_boundary:
        return fused, None
    edge = mbrc.boundary_features(fused)
    return torch.cat([fused, edge], dim=1), mbrc.boundary_output(edge)


def inner_boundary(gt):
    background = F.pad(1.0 - gt, (1, 1, 1, 1), value=1.0)
    eroded = 1.0 - F.max_pool2d(background, 3, stride=1)
    return (gt - eroded).clamp(0.0, 1.0)
