"""Single training orchestration with the four frozen objective/checkpoint policies."""
import json
import time
from pathlib import Path
import torch

from datasets import build_datasets, make_loaders, COUNTS
from losses.segmentation import DiceLoss
from losses.cams import LossWeights, LOSS_CONFIG
from models import build_model
from models.mbrc.variants import VARIANTS
from utils.deterministic import seed_all
from utils.checkpoint import save_json, save_weights, load_legacy_checkpoint
from utils.logger import next_run, append_history, log_record
from .trainer import (
    train_base_epoch, train_cams_epoch, train_mbrc_epoch, train_joint_epoch,
)
from .validator import evaluate_base, evaluate_cams, evaluate_mbrc, evaluate_joint, is_new_best, should_stop


VARIANT_FLAGS = {
    "base": (False, False),
    "cams": (True, False),
    "mbrc": (False, True),
    "cams_mbrcnet": (True, True),
}
DISPLAY_NAMES = {
    "base": "base",
    "cams": "base_cams",
    "mbrc": "base_mbrc",
    "cams_mbrcnet": "CAMS-MBRCNet",
}
EXPECTED_PARAMS = {
    "base": 420880,
    "cams": 3120628,
    "mbrc": 628814,
    "cams_mbrcnet": 3328562,
}


def resolve_variant(config_path, override=None):
    if override is not None:
        if override not in VARIANT_FLAGS:
            raise ValueError(override)
        return override
    model = json.loads(Path(config_path).read_text(encoding="utf-8"))["model"]
    flags = (bool(model["use_cams"]), bool(model["use_mbrc"]))
    return next(name for name, value in VARIANT_FLAGS.items() if value == flags)


def config_and_flags(config_path, variant):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unknown variant {variant}")
    dataset = config["dataset"]
    if dataset not in COUNTS or config["split"] != COUNTS[dataset]:
        raise RuntimeError("Dataset/split differs from frozen protocol")
    expected = {
        "image_size": [224, 224], "batch_size": 2, "seed": 0,
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-3,
        "max_epoch": 400, "patience": 40,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"Frozen protocol mismatch: {key}")
    return config, VARIANT_FLAGS[variant]


def build_training_objects(config, variant, device):
    use_cams, use_mbrc = VARIANT_FLAGS[variant]
    # Source Base+MBRC constructs its model before data; other sources construct
    # the frozen datasets before the model. Both orders remain explicit here.
    if variant == "mbrc":
        model = build_model(use_cams, use_mbrc).to(device)
        datasets = build_datasets(config["dataset"], config["data_root"], False)
    else:
        datasets = build_datasets(
            config["dataset"], config["data_root"], use_cams,
            config["map_root"] if use_cams else None,
        )
        model = build_model(use_cams, use_mbrc).to(device)
    count = sum(p.numel() for p in model.parameters())
    if count != EXPECTED_PARAMS[variant]:
        raise RuntimeError(f"Parameter count changed: {variant} {count}")
    weights = LossWeights(LOSS_CONFIG).to(device) if use_cams else None
    criterion = DiceLoss()
    parameters = list(model.parameters()) + list(weights.parameters()) if use_cams else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    loaders = make_loaders(datasets, config["batch_size"], config["seed"])
    return datasets, loaders, model, weights, criterion, optimizer


def epoch_train(variant, model, weights, criterion, optimizer, loader, device, epoch):
    if variant == "base":
        return train_base_epoch(model, loader, criterion, optimizer, device, epoch)
    if variant == "cams":
        return train_cams_epoch(model, weights, loader, criterion, optimizer, device, epoch)
    if variant == "mbrc":
        return train_mbrc_epoch(model, optimizer, loader, criterion, VARIANTS["full"], epoch)
    return train_joint_epoch(model, weights, optimizer, criterion, loader, device, epoch)


def epoch_evaluate(variant, model, weights, criterion, loader, device, epoch=None, phase="VAL"):
    if variant == "base":
        loss, dice = evaluate_base(model, loader, criterion, device, epoch)
        return {"loss": loss, "dice": dice}
    if variant == "cams":
        losses, dice = evaluate_cams(model, weights, loader, criterion, device, phase)
        return {"loss": losses, "dice": dice}
    if variant == "mbrc":
        return evaluate_mbrc(model, loader, criterion, VARIANTS["full"], device, phase.lower())
    return evaluate_joint(model, weights, criterion, loader, device, phase)


def save_best(variant, path, model, weights, epoch, dice):
    if variant == "base":
        save_weights(path, model.state_dict())
    elif variant == "mbrc":
        save_weights(path, {
            "variant": "full", "epoch": epoch, "val_dice": dice,
            "model": {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()},
        })
    else:
        save_weights(path, {
            "model_state_dict": model.state_dict(),
            "loss_weights_state_dict": weights.state_dict(),
            "best_epoch": epoch, "best_val_dice": dice,
        })


def train(config_path, variant, result_dir=None):
    config, (use_cams, use_mbrc) = config_and_flags(config_path, variant)
    seed_all(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    root = Path(__file__).resolve().parents[1]
    if result_dir is None:
        run = next_run(root, config["dataset"], DISPLAY_NAMES[variant])
    else:
        run = Path(result_dir).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
    protected = ("metadata.json", "history.jsonl", "model_best.pth", "summary.json", "finished.flag")
    if any((run / name).exists() for name in protected):
        raise RuntimeError(f"Run already contains training output: {run}")
    datasets, loaders, model, weights, criterion, optimizer = build_training_objects(config, variant, device)
    metadata = {
        "model": DISPLAY_NAMES[variant], "variant": variant,
        "dataset": config["dataset"], "data_root": str(Path(config["data_root"]).resolve()),
        "map_root": str(Path(config["map_root"]).resolve()) if use_cams else None,
        "split": {s: len(d) for s, d in datasets.items()},
        "seed": 0, "resolution": [224, 224], "batch_size": 2,
        "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-3,
        "max_epochs": 400, "patience": 40,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_metric": "mean per-image Otsu Val Dice",
        "test_policy": "each new best" if use_mbrc else "final best",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    save_json(run / "metadata.json", metadata)
    save_json(run / "split_manifest.json", {
        split: (dataset.original.names if use_cams else dataset.names)
        for split, dataset in datasets.items()
    })
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    best_epoch = -1 if not use_mbrc else 0
    best_val = -1.0
    best_test = float("nan")
    last_epoch = best_epoch
    checkpoint = run / "model_best.pth"
    started = time.time()
    epochs = range(1, 401) if use_mbrc else range(400)
    for epoch in epochs:
        tick = time.time()
        train_loss = epoch_train(variant, model, weights, criterion, optimizer, loaders["train"], device, epoch)
        val = epoch_evaluate(variant, model, weights, criterion, loaders["val"], device, epoch, "VAL")
        current = val["final_dice"] if variant == "cams_mbrcnet" else val["dice"]
        improved = is_new_best(current, best_val)
        if improved:
            best_epoch, best_val = epoch, current
            save_best(variant, checkpoint, model, weights, epoch, current)
            if use_mbrc:
                test = epoch_evaluate(variant, model, weights, criterion, loaders["test"], device, phase="TEST BEST")
                best_test = test["final_dice"] if variant == "cams_mbrcnet" else test["dice"]
                save_json(run / "best_result.json", {
                    "epoch": best_epoch, "val": val, "test": test, "checkpoint": str(checkpoint),
                })
        record = {
            "epoch": epoch,
            "train_loss": train_loss["total"] if isinstance(train_loss, dict) else train_loss,
            "train": train_loss,
            "val_loss": val["total"] if variant == "cams_mbrcnet" else val["loss"],
            "current_val_dice": current,
            "val": val,
            "saved_best_epoch": best_epoch,
            "saved_best_val_dice": best_val,
            "saved_best_test_dice": best_test if use_mbrc else None,
            "saved_this_epoch": improved,
            "epoch_seconds": time.time() - tick,
        }
        append_history(run, record)
        log_record(record, best_test)
        last_epoch = epoch
        if should_stop(epoch, best_epoch, config["patience"]):
            print(f"EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break
    load_legacy_checkpoint(checkpoint, model, weights)
    if not use_mbrc:
        test = epoch_evaluate(variant, model, weights, criterion, loaders["test"], device, phase="TEST BEST")
        best_test = test["dice"]
    else:
        reload_val = epoch_evaluate(variant, model, weights, criterion, loaders["val"], device, phase="RELOAD VAL")
        reloaded = reload_val["final_dice"] if variant == "cams_mbrcnet" else reload_val["dice"]
        if reloaded != best_val:
            raise RuntimeError("Best checkpoint Val Dice changed after reload")
    summary = {
        "variant": variant, "last_epoch": last_epoch, "saved_best_epoch": best_epoch,
        "saved_best_val_dice": best_val, "final_test_dice": best_test,
        "stop_reason": "patience40" if should_stop(last_epoch, best_epoch, config["patience"]) else "max400",
        "elapsed_seconds": time.time()-started, "checkpoint": str(checkpoint),
    }
    save_json(run / "summary.json", summary)
    (run / "finished.flag").write_text("finished\n", encoding="utf-8")
    print("FINAL " + json.dumps(summary, ensure_ascii=False), flush=True)
    return summary
