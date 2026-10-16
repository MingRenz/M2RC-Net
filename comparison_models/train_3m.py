"""OCTA500-3M comparison training, using the project's frozen Base protocol.

The six copied model sources return probabilities. This runner keeps that
representation, applies the same DiceLoss and per-image Otsu validation as the
Base experiment, and leaves proposed-model training code untouched.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from comparison_models import build_comparison_model, registered_models
from comparison_models.common.paths import seed_log_dir
from datasets import build_datasets, make_loaders
from engine.validator import is_new_best, should_stop
from losses.segmentation import DiceLoss
from utils.checkpoint import save_json, save_weights
from utils.logger import append_history
from utils.metrics import calc_result


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "octa500_3m.yaml"
EXPECTED = {
    "dataset": "OCTA500_3M",
    "split": {"train": 140, "val": 10, "test": 50},
    "image_size": [224, 224],
    "batch_size": 2,
    "optimizer": "Adam",
    "lr": 1e-4,
    "weight_decay": 1e-3,
    "max_epoch": 400,
    "patience": 40,
}


def comparison_seed(seed):
    """Fix RNGs while allowing source CUDA ops that fail strict mode."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(False)


def load_config():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key, expected in EXPECTED.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Frozen 3M protocol mismatch: {key}")
    data_root = Path(config["data_root"]).expanduser().resolve(strict=True)
    if data_root != ROOT / "data" / "OCTA500_3M":
        raise RuntimeError(f"Unexpected 3M data root: {data_root}")
    return config, data_root


def probabilities(bundle, images):
    result = bundle.adapter.extract_output(
        bundle.model(bundle.adapter.prepare_input(images))
    )
    if result.representation != "probability" or result.auxiliary is not None:
        raise RuntimeError("Comparison output contract changed")
    return result.prediction


def train_epoch(bundle, loader, criterion, optimizer, device, epoch):
    bundle.model.train()
    logged_losses = []
    with tqdm(loader, desc=f"Train epoch {epoch:03d}", total=len(loader),
              ncols=110, mininterval=1.0, leave=True) as progress:
        for _names, images, labels in progress:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = probabilities(bundle, images)
            loss = criterion(prediction, labels)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            # Match Base's historical train-loss logging scale exactly.
            logged_losses.append(float(loss.item()) / images.shape[0])
            progress.set_postfix_str(
                f"loss={np.mean(logged_losses):.6f}", refresh=False
            )
    return float(np.mean(logged_losses))


@torch.no_grad()
def evaluate(bundle, loader, criterion, device, label):
    bundle.model.eval()
    losses, dices = [], []
    with tqdm(loader, desc=label, total=len(loader), ncols=110,
              mininterval=1.0, leave=True) as progress:
        for _names, images, labels in progress:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)
            prediction = probabilities(bundle, images)
            losses.append(float(criterion(prediction, labels).item()))
            pred = prediction.cpu().numpy()
            truth = labels.cpu().numpy()
            for index in range(images.shape[0]):
                dices.append(float(calc_result(
                    pred[index, 0], truth[index, 0], thresh_value=None
                )["dice"]))
            progress.set_postfix_str(
                f"dice={np.mean(dices):.6f} loss={np.mean(losses):.6f}",
                refresh=False,
            )
    return {"loss": float(np.mean(losses)),
            "dice": float(np.mean(dices)), "images": len(dices)}


def run(model_name, seed, result_dir):
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    config, data_root = load_config()
    seed_group = seed_log_dir(
        ROOT, "OCTA500_3M", "comparison", model_name, seed
    ).resolve()
    run_dir = Path(result_dir).expanduser().resolve(strict=True)
    if (run_dir.parent != seed_group or not run_dir.name.startswith("run_")
            or len(run_dir.name) < 7 or not run_dir.name[4:].isdigit()):
        raise RuntimeError(f"Run directory must be {seed_group}/run_NNN")
    protected = ("metadata.json", "history.jsonl", "model_best.pth",
                 "summary.json", "finished.flag")
    if any((run_dir / name).exists() for name in protected):
        raise RuntimeError(f"Existing training output in {run_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    comparison_seed(seed)
    device = torch.device("cuda:0")
    datasets = build_datasets("OCTA500_3M", data_root, False, seed=seed)
    loaders = make_loaders(datasets, config["batch_size"], seed)
    bundle = build_comparison_model(
        model_name, {"image_size": config["image_size"],
                     "model": {"name": model_name}}
    )
    bundle.model.to(device)
    criterion = DiceLoss()
    optimizer = torch.optim.Adam(
        bundle.model.parameters(), lr=config["lr"],
        weight_decay=config["weight_decay"]
    )
    parameters = sum(p.numel() for p in bundle.model.parameters())
    trainable = sum(p.numel() for p in bundle.model.parameters()
                    if p.requires_grad)
    metadata = {
        "family": "comparison", "model": model_name, "run": run_dir.name,
        "dataset": "OCTA500_3M", "data_root": str(data_root),
        "split": {split: len(dataset) for split, dataset in datasets.items()},
        "seed": seed, "resolution": [224, 224], "batch_size": 2,
        "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-3,
        "max_epochs": 400, "patience": 40,
        "parameters": parameters, "trainable_parameters": trainable,
        "initialization": "random; no pretrained checkpoint",
        "loss": "Base DiceLoss on source sigmoid probability",
        "train_loss_reporting": "mean(batch DiceLoss / batch size), matching Base",
        "checkpoint_metric": "mean per-image minmax/uint8/Otsu Val Dice",
        "test_policy": "once after final best checkpoint reload",
        "strict_deterministic_algorithms": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "split_manifest.json", {
        split: dataset.names for split, dataset in datasets.items()
    })
    print("START " + json.dumps(metadata, ensure_ascii=False), flush=True)

    best_epoch, best_val, last_epoch = -1, -1.0, -1
    checkpoint = run_dir / "model_best.pth"
    started = time.time()
    for epoch in range(config["max_epoch"]):
        tick = time.time()
        train_loss = train_epoch(
            bundle, loaders["train"], criterion, optimizer, device, epoch
        )
        val = evaluate(
            bundle, loaders["val"], criterion, device, f"VAL epoch {epoch:03d}"
        )
        improved = is_new_best(val["dice"], best_val)
        if improved:
            best_epoch, best_val = epoch, val["dice"]
            save_weights(checkpoint, bundle.model.state_dict())
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val["loss"], "current_val_dice": val["dice"],
            "saved_best_epoch": best_epoch,
            "saved_best_val_dice": best_val,
            "saved_best_test_dice": None,
            "saved_this_epoch": improved,
            "epoch_seconds": time.time() - tick,
        }
        append_history(run_dir, record)
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val['loss']:.6f} "
            f"current_val_dice={val['dice']:.9f} "
            f"saved_best_epoch={best_epoch} "
            f"saved_best_val_dice={best_val:.9f} "
            f"saved_this_epoch={'YES' if improved else 'NO'} "
            f"epoch_seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )
        last_epoch = epoch
        if should_stop(epoch, best_epoch, config["patience"]):
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    bundle.model.load_state_dict(
        torch.load(checkpoint, map_location=device), strict=True
    )
    reloaded = evaluate(bundle, loaders["val"], criterion, device, "RELOAD VAL")
    if reloaded["dice"] != best_val:
        raise RuntimeError("Best Val Dice changed after checkpoint reload")
    test = evaluate(bundle, loaders["test"], criterion, device, "TEST BEST")
    summary = {
        "family": "comparison", "model": model_name, "seed": seed,
        "last_epoch": last_epoch, "saved_best_epoch": best_epoch,
        "saved_best_val_dice": best_val,
        "final_test_dice": test["dice"], "final_test_loss": test["loss"],
        "stop_reason": (
            "patience40" if should_stop(last_epoch, best_epoch,
                                        config["patience"]) else "max400"
        ),
        "elapsed_seconds": time.time() - started,
        "checkpoint": str(checkpoint),
    }
    save_json(run_dir / "summary.json", summary)
    (run_dir / "finished.flag").write_text("finished\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=registered_models())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()
    run(args.model, args.seed, args.result_dir)


if __name__ == "__main__":
    main()
