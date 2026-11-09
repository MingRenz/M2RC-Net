import torch
import torch.nn.functional as F

def boundary_features(mbrc, fused):
    edge = mbrc.boundary_features(fused)
    return (torch.cat([fused, edge], dim=1), mbrc.boundary_output(edge))

def inner_boundary(gt):
    background = F.pad(1.0 - gt, (1, 1, 1, 1), value=1.0)
    eroded = 1.0 - F.max_pool2d(background, 3, stride=1)
    return (gt - eroded).clamp(0.0, 1.0)
