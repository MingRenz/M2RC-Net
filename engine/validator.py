"""Validation using the exact frozen per-image minmax/uint8/Otsu metric."""
import sys
import numpy as np
import torch
from tqdm import tqdm
from utils.metrics import calc_result
from losses.mbrc import compute_parts as compute_mbrc_parts
from engine.trainer import cams_batch_loss, joint_calculate
def evaluate_base(model, loader, criterion, device, epoch=None):
    model.eval()
    losses = []
    dices = []
    phase = loader.dataset.image_dir.parent.name
    description = (
        f"{phase} epoch {epoch:03d}"
        if epoch is not None else f"{phase} final"
    )
    with torch.no_grad(), tqdm(
        loader, desc=description, total=len(loader),
        file=sys.stdout, disable=False, ncols=110,
        mininterval=1.0, leave=True,
    ) as progress:
        for _names, images, labels in progress:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            output = model(images)
            probabilities = torch.sigmoid(output)
            losses.append(float(criterion(probabilities, labels).item()))

            predictions = probabilities.cpu().numpy()
            references = labels.cpu().numpy()
            for index in range(images.shape[0]):
                dices.append(float(calc_result(
                    predictions[index, 0],
                    references[index, 0],
                    thresh_value=None,
                )["dice"]))
            progress.set_postfix_str(
                f"dice={np.mean(dices):.6f} loss={np.mean(losses):.6f}",
                refresh=False,
            )
    return float(np.mean(losses)), float(np.mean(dices))

@torch.no_grad()
def evaluate_cams(model, weights, loader, criterion, device, name):
    model.eval()
    weights.eval()
    totals = []
    dices = []
    with tqdm(loader, desc=name, total=len(loader),
              file=sys.stdout, ncols=110, mininterval=1.0) as progress:
        for batch in progress:
            total, lseg, lcal, lves, probabilities, labels = cams_batch_loss(
                model, weights, criterion, batch, device
            )
            totals.append([float(x) for x in (total, lseg, lcal, lves)])
            p = probabilities.cpu().numpy()
            y = labels.cpu().numpy()
            for i in range(p.shape[0]):
                dices.append(float(calc_result(p[i, 0], y[i, 0])["dice"]))
            progress.set_postfix_str(f"dice={np.mean(dices):.6f}", refresh=False)
    means = np.mean(totals, axis=0)
    return (
        {key: float(value) for key, value in zip(
            ("total", "segmentation", "caliber", "vesselness"), means
        )},
        float(np.mean(dices)),
    )

@torch.no_grad()
def evaluate_mbrc(model, loader, criterion, variant, device, split):
    model.eval()
    losses = []
    dices = []
    for _names, image, gt in tqdm(
        loader, total=len(loader), desc=split.upper(), unit="image",
        file=sys.stdout, ncols=110, mininterval=1.0,
    ):
        image = image.to(device=device, dtype=torch.float32)
        gt = gt.to(device=device, dtype=torch.float32)
        output = model(image)
        parts, _base, final = compute_mbrc_parts(output, gt, criterion, variant)
        losses.extend([float(parts["total"])] * image.shape[0])
        probabilities = final.detach().cpu().numpy()
        references = gt.detach().cpu().numpy()
        for index in range(image.shape[0]):
            dices.append(float(calc_result(
                probabilities[index, 0], references[index, 0],
                thresh_value=None,
            )["dice"]))
    return {
        "loss": float(np.mean(losses)),
        "dice": float(np.mean(dices)),
        "images": len(dices),
    }

@torch.no_grad()
def evaluate_joint(model, weights, dice, loader, device, label):
    model.eval()
    weights.eval()
    sums = {}
    base_dice = []
    final_dice = []
    with tqdm(loader, desc=label, file=sys.stdout,
              ncols=110, mininterval=1.0) as progress:
        for batch in progress:
            _names, output, gt, parts = joint_calculate(
                model, weights, dice, batch, device
            )
            n = gt.shape[0]
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value) * n
            b = output["base_prob"].cpu().numpy()
            f = output["final_prob"].cpu().numpy()
            y = gt.cpu().numpy()
            for i in range(n):
                base_dice.append(float(calc_result(b[i, 0], y[i, 0])["dice"]))
                final_dice.append(float(calc_result(f[i, 0], y[i, 0])["dice"]))
            progress.set_postfix_str(
                f"base={np.mean(base_dice):.6f} final={np.mean(final_dice):.6f}",
                refresh=False,
            )
    result = {key: value / len(base_dice) for key, value in sums.items()}
    result["base_dice"] = float(np.mean(base_dice))
    result["final_dice"] = float(np.mean(final_dice))
    return result



def is_new_best(current_val_dice, best_val_dice):
    return current_val_dice > best_val_dice


def should_stop(epoch, best_epoch, patience):
    return epoch - best_epoch >= patience
