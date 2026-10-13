"""Shared segmentation, Action and boundary objectives."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from models.mbrc.boundary import inner_boundary


def binary_from_probability(probability):
    scaled = cv2.normalize(
        np.asarray(probability, dtype=np.float32), None,
        0, 255, cv2.NORM_MINMAX,
    ).astype("uint8")
    _, binary = cv2.threshold(
        scaled, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary.astype(bool)


def dynamic_actions(base_probability, labels):
    """Targets use only the current training batch's GT."""
    with torch.no_grad():
        probabilities = base_probability.detach().cpu().numpy()
        references = labels.detach().cpu().numpy()
        targets = []
        for index in range(probabilities.shape[0]):
            gt = references[index, 0] > 0.5
            base = binary_from_probability(probabilities[index, 0])
            target = np.full(gt.shape, 2, dtype=np.int64)
            target[gt & ~base] = 0
            target[~gt & base] = 1
            targets.append(target)
        return torch.from_numpy(np.stack(targets)).to(labels.device)


def balanced_action_loss(logits, target, gt):
    ce = F.cross_entropy(logits, target, reduction="none")
    vessel = gt[:, 0] > 0.5
    groups = (
        target == 0, target == 1,
        (target == 2) & vessel, (target == 2) & ~vessel,
    )
    values = [
        (ce * mask).sum() / mask.sum()
        for mask in groups if bool(mask.any().item())
    ]
    if not values:
        raise RuntimeError("Action labels have no valid group")
    return torch.stack(values).mean()


def compute_parts(output, gt, criterion, variant):
    if variant.name == "mamba":
        loss = criterion(output, gt)
        return {"total": loss, "base": loss, "final": loss}, output, output

    base = output["base_prob"]
    final = torch.sigmoid(output["z_final"])
    base_loss = criterion(base, gt)
    final_loss = criterion(final, gt)
    total = base_loss + final_loss
    if variant.action:
        target = dynamic_actions(base, gt)
        total = total + 0.1 * balanced_action_loss(
            output["action_logits"], target, gt,
        )
    if variant.boundary:
        total = total + 0.1 * F.binary_cross_entropy_with_logits(
            output["edge_logits"], inner_boundary(gt),
        )
    return {"total": total, "base": base_loss, "final": final_loss}, base, final
