"""One training engine; variant-specific loss arithmetic is unchanged."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from losses.segmentation import DiceLoss
from losses.cams import LOSS_CONFIG, caliber_loss
from losses.mbrc import compute_parts as compute_mbrc_parts
from losses.joint import compute_parts as compute_joint_parts
MAX_EPOCHS = 1000
DEVICE = torch.device("cuda:0")
def train_base_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    logged_losses = []
    with tqdm(
        loader, desc=f"Train epoch {epoch:03d}", total=len(loader),
        file=sys.stdout, disable=False, ncols=110,
        mininterval=1.0, leave=True,
    ) as progress:
        for _names, images, labels in progress:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(torch.sigmoid(output), labels)
            loss.backward()
            optimizer.step()

            logged_losses.append(float(loss.item()) / images.shape[0])
            progress.set_postfix_str(
                f"loss={np.mean(logged_losses):.6f}", refresh=False
            )
    return float(np.mean(logged_losses))

def cams_batch_loss(model, weights, criterion, batch, device):
    _names, images, labels, caliber_gt, valid = batch
    images = images.to(device=device, dtype=torch.float32)
    labels = labels.to(device=device, dtype=torch.float32)
    caliber_gt = caliber_gt.to(device=device, dtype=torch.float32)
    valid = valid.to(device=device, dtype=torch.float32)
    logits, r, vesselness_logits = model(images)
    probabilities = torch.sigmoid(logits)
    lseg = criterion(probabilities, labels)
    lcal, _, _ = caliber_loss(r, caliber_gt, labels, valid, LOSS_CONFIG)
    lves = F.binary_cross_entropy_with_logits(vesselness_logits, labels)
    total = weights(lseg, lcal, lves)
    if not torch.isfinite(total):
        raise RuntimeError("Non-finite CAMS loss")
    return total, lseg, lcal, lves, probabilities, labels

def train_cams_epoch(model, weights, loader, criterion, optimizer, device, epoch):
    model.train()
    weights.train()
    totals = []
    with tqdm(loader, desc=f"Train epoch {epoch:03d}", total=len(loader),
              file=sys.stdout, ncols=110, mininterval=1.0) as progress:
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            total, lseg, lcal, lves, _prob, _labels = cams_batch_loss(
                model, weights, criterion, batch, device
            )
            total.backward()
            optimizer.step()
            weights.project()
            totals.append([float(x.detach()) for x in (total, lseg, lcal, lves)])
            progress.set_postfix_str(
                f"total={np.mean(totals, axis=0)[0]:.6f}", refresh=False
            )
    means = np.mean(totals, axis=0)
    return {key: float(value) for key, value in zip(
        ("total", "segmentation", "caliber", "vesselness"), means
    )}

def mbrc_unpack(batch):
    _names, image, gt = batch
    return image.to(DEVICE, dtype=torch.float32), gt.to(DEVICE, dtype=torch.float32)

def train_mbrc_epoch(model, optimizer, loader, criterion, variant, epoch):
    model.train()
    total = 0.0
    images = 0
    progress = tqdm(
        loader, total=len(loader), desc=f"Train {epoch}/{MAX_EPOCHS}",
        unit="batch", file=sys.stdout, ncols=110, mininterval=1.0,
    )
    for batch in progress:
        image, gt = mbrc_unpack(batch)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        parts, _base, _final = compute_mbrc_parts(output, gt, criterion, variant)
        if not bool(torch.isfinite(parts["total"]).item()):
            raise RuntimeError("Nonfinite training loss at epoch " + str(epoch))
        parts["total"].backward()
        optimizer.step()
        n = image.shape[0]
        total += float(parts["total"].detach()) * n
        images += n
        progress.set_postfix_str(f"loss={total/images:.6f}", refresh=False)
    return total / images

def joint_unpack(batch, device):
    names, image, gt, caliber, valid = batch
    return names, *(
        x.to(device=device, dtype=torch.float32)
        for x in (image, gt, caliber, valid)
    )

def joint_calculate(model, weights, dice, batch, device):
    names, image, gt, caliber, valid = joint_unpack(batch, device)
    output = model(image)
    parts = compute_joint_parts(output, gt, caliber, valid, weights, dice)
    if not torch.isfinite(parts["total"]):
        raise RuntimeError("Nonfinite joint loss")
    return names, output, gt, parts

def train_joint_epoch(model, weights, optimizer, dice, loader, device, epoch):
    model.train()
    weights.train()
    sums = {}
    count = 0
    with tqdm(loader, desc=f"Train {epoch}/{MAX_EPOCHS}", file=sys.stdout,
              ncols=110, mininterval=1.0) as progress:
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            _names, _output, _gt, parts = joint_calculate(
                model, weights, dice, batch, device
            )
            parts["total"].backward()
            optimizer.step()
            weights.project()
            n = len(batch[0])
            count += n
            for key, value in parts.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach()) * n
            progress.set_postfix_str(
                f"loss={sums['total']/count:.6f}", refresh=False
            )
    return {key: value / count for key, value in sums.items()}

