import json
import time
from pathlib import Path
import torch

from datasets import build_datasets, make_loaders, COUNTS
from losses.segmentation import DiceLoss
from losses.representation import LossWeights, LOSS_CONFIG
from models import build_model
from models.mbrc import FULL_MBRC
from utils.deterministic import seed_all
from utils.checkpoint import save_json, save_weights, load_checkpoint
from utils.logger import next_run, append_history, log_record
from .trainer import (
    train_base_epoch, train_representation_epoch, train_mbrc_epoch, train_joint_epoch,
)
from .validator import evaluate_base, evaluate_representation, evaluate_mbrc, evaluate_joint, is_new_best, should_stop


VARIANT_FLAGS = {
    "base": (False, False, False),
    "mscf": (True, False, False),
    "mscf_cve": (True, True, False),
    "mbrc": (False, False, True),
    "m2rcnet_wo_cve": (True, False, True),
    "m2rcnet": (True, True, True),
}
DISPLAY_NAMES = {"base":"base", "mscf":"base_mscf", "mscf_cve":"base_mscf_cve",
                 "mbrc":"base_mbrc", "m2rcnet_wo_cve":"M2RC-Net_wo_CVE", "m2rcnet":"M2RC-Net"}
EXPECTED_PARAMS = {"base":420880, "mscf":3111232, "mscf_cve":3120628,
                   "mbrc":628814, "m2rcnet_wo_cve":3319166, "m2rcnet":3328562}
AVAILABLE_VARIANTS = tuple(VARIANT_FLAGS)

def resolve_variant(config_path, override=None):
    if override is not None:
        if override not in VARIANT_FLAGS:
            raise ValueError(override)
        return override
    from utils.configuration import read_config
    model = read_config(config_path)["model"]
    flags = tuple(bool(model[k]) for k in ("use_mscf","use_cve","use_mbrc"))
    return next(name for name,value in VARIANT_FLAGS.items() if value==flags)


def config_and_flags(config_path, variant):
    from utils.configuration import read_config
    config = dict(config_path) if isinstance(config_path, dict) else read_config(config_path)
    if variant not in VARIANT_FLAGS:
        raise ValueError(f"Unknown variant {variant}")
    dataset = config["dataset"]
    if dataset not in COUNTS or config["split"] != COUNTS[dataset]:
        raise RuntimeError("Dataset/split differs from frozen protocol")
    expected = {
        "image_size": [224, 224], "batch_size": 2,
        "optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-3,
        "max_epoch": 400, "patience": 40,
    }
    for key, value in expected.items():
        if config[key] != value:
            raise RuntimeError(f"Frozen protocol mismatch: {key}")
    if isinstance(config["seed"], bool) or not isinstance(config["seed"], int) or config["seed"] < 0:
        raise RuntimeError("Training seed must be a nonnegative integer")
    return config, VARIANT_FLAGS[variant]


def build_training_objects(config, variant, device, model_factory=None, expected_parameters=None):
    use_mscf, use_cve, use_mbrc = VARIANT_FLAGS[variant]
    builder = build_model if model_factory is None else model_factory
    if variant == "mbrc":
        model = builder(use_mscf, use_cve, use_mbrc, seed=config["seed"]).to(device)
        datasets = build_datasets(config["dataset"], config["data_root"], False, seed=config["seed"])
    else:
        datasets = build_datasets(
            config["dataset"], config["data_root"], use_cve,
            config["map_root"] if use_cve else None,
            seed=config["seed"],
        )
        model = builder(use_mscf, use_cve, use_mbrc, seed=config["seed"]).to(device)
    count = sum(p.numel() for p in model.parameters())
    expected_count = EXPECTED_PARAMS[variant] if expected_parameters is None else int(expected_parameters)
    if count != expected_count:
        raise RuntimeError(f"Parameter count changed: {variant} {count}")
    weights = LossWeights(LOSS_CONFIG).to(device) if use_cve else None
    criterion = DiceLoss()
    parameters = list(model.parameters()) + list(weights.parameters()) if use_cve else model.parameters()
    optimizer = torch.optim.Adam(parameters, lr=config["lr"], weight_decay=config["weight_decay"])
    loaders = make_loaders(datasets, config["batch_size"], config["seed"])
    return datasets, loaders, model, weights, criterion, optimizer


def epoch_train(variant, model, weights, criterion, optimizer, loader, device, epoch):
    if variant == "base":
        return train_base_epoch(model, loader, criterion, optimizer, device, epoch)
    if variant == "mscf_cve":
        return train_representation_epoch(model, weights, loader, criterion, optimizer, device, epoch)
    if variant == "mbrc":
        return train_mbrc_epoch(model, optimizer, loader, criterion, FULL_MBRC, epoch)
    return train_joint_epoch(model, weights, optimizer, criterion, loader, device, epoch)


def epoch_evaluate(variant, model, weights, criterion, loader, device, epoch=None, phase="VAL"):
    if variant == "base":
        loss, dice = evaluate_base(model, loader, criterion, device, epoch)
        return {"loss": loss, "dice": dice}
    if variant == "mscf_cve":
        losses, dice = evaluate_representation(model, weights, loader, criterion, device, phase)
        return {"loss": losses, "dice": dice}
    if variant == "mbrc":
        return evaluate_mbrc(model, loader, criterion, FULL_MBRC, device, phase.lower())
    return evaluate_joint(model, weights, criterion, loader, device, phase)


def save_best(variant, path, model, weights, epoch, dice):
    if variant == "base":
        save_weights(path, model.state_dict())
    elif variant == "mbrc":
        save_weights(path, {
            "variant": "mbrc", "epoch": epoch, "val_dice": dice,
            "model_config": dict(zip(("use_mscf", "use_cve", "use_mbrc"), VARIANT_FLAGS[variant])),
            "model": {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()},
        })
    else:
        save_weights(path, {
            "model_state_dict": model.state_dict(),
            "loss_weights_state_dict": weights.state_dict(),
            "best_epoch": epoch, "best_val_dice": dice,
        })


def train(config_path, variant, result_dir=None, *, model_factory=None,
          model_name=None, expected_parameters=None, config=None):
    if variant in ("mscf", "m2rcnet_wo_cve"):
        raise ValueError("Use the dedicated engine selected by the root training entry")
    config, (use_mscf, use_cve, use_mbrc) = config_and_flags(config if config is not None else config_path, variant)
    display_name = DISPLAY_NAMES[variant] if model_name is None else model_name
    seed_all(config["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device("cuda:0")
    root = Path(__file__).resolve().parents[1]
    if result_dir is None:
        run = next_run(root, config["dataset"], display_name, seed=config["seed"])
    else:
        run = Path(result_dir).expanduser().resolve()
        run.mkdir(parents=True, exist_ok=True)
    protected = ("metadata.json", "history.jsonl", "model_best.pth", "summary.json", "finished.flag")
    if any((run / name).exists() for name in protected):
        raise RuntimeError(f"Run already contains training output: {run}")
    datasets, loaders, model, weights, criterion, optimizer = build_training_objects(
        config, variant, device, model_factory=model_factory,
        expected_parameters=expected_parameters,
    )
    metadata = {
        "model": display_name, "variant": variant,
        "dataset": config["dataset"], "data_root": str(Path(config["data_root"]).resolve()),
        "map_root": str(Path(config["map_root"]).resolve()) if use_cve else None,
        "split": {s: len(d) for s, d in datasets.items()},
        "seed": config["seed"], "resolution": [224, 224], "batch_size": 2,
        "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 1e-3,
        "max_epochs": 400, "patience": 40,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_metric": "mean per-image Otsu Val Dice",
        "test_policy": "each new best" if use_mbrc else "final best",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
    }
    if model_factory is not None:
        metadata["model_class"] = model.__class__.__module__ + "." + model.__class__.__name__
    save_json(run / "metadata.json", metadata)
    save_json(run / "split_manifest.json", {
        split: (dataset.original.names if use_cve else dataset.names)
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
        current = val["final_dice"] if variant == "m2rcnet" else val["dice"]
        improved = is_new_best(current, best_val)
        if improved:
            best_epoch, best_val = epoch, current
            save_best(variant, checkpoint, model, weights, epoch, current)
            if use_mbrc:
                test = epoch_evaluate(variant, model, weights, criterion, loaders["test"], device, phase="TEST BEST")
                best_test = test["final_dice"] if variant == "m2rcnet" else test["dice"]
                save_json(run / "best_result.json", {
                    "epoch": best_epoch, "val": val, "test": test, "checkpoint": str(checkpoint),
                })
        record = {
            "epoch": epoch,
            "train_loss": train_loss["total"] if isinstance(train_loss, dict) else train_loss,
            "train": train_loss,
            "val_loss": val["total"] if variant == "m2rcnet" else val["loss"],
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
    load_checkpoint(checkpoint, model, weights)
    if not use_mbrc:
        test = epoch_evaluate(variant, model, weights, criterion, loaders["test"], device, phase="TEST BEST")
        best_test = test["dice"]
    else:
        reload_val = epoch_evaluate(variant, model, weights, criterion, loaders["val"], device, phase="RELOAD VAL")
        reloaded = reload_val["final_dice"] if variant == "m2rcnet" else reload_val["dice"]
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
