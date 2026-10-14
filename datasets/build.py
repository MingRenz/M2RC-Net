"""One dataset and DataLoader entry, preserving each frozen split and RNG order."""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .octa500 import OCTA3M, OCTAMulti, seed_worker
from .rossa import ROSSADataset
from .caliber import CaliberDataset

COUNTS = {
    "OCTA500_3M": {"train": 140, "val": 10, "test": 50},
    "OCTA500_6M": {"train": 180, "val": 20, "test": 100},
    "ROSSA": {"train": 718, "val": 100, "test": 100},
}


def build_dataset(dataset, root, split, use_cams=False, map_root=None, seed=0):
    root = Path(root).expanduser().resolve(strict=True)
    original = (OCTA3M(root, split, seed=seed) if dataset == "OCTA500_3M"
                else OCTAMulti(root, split, seed=seed) if dataset == "OCTA500_6M"
                else ROSSADataset(root, split, seed=seed))
    if len(original) != COUNTS[dataset][split]:
        raise RuntimeError("Frozen split count changed")
    if use_cams:
        if map_root is None:
            raise ValueError("CAMS requires existing map root")
        return CaliberDataset(original, Path(map_root) / "caliber" / split)
    return original


def build_datasets(dataset, root, use_cams=False, map_root=None, seed=0):
    sets = {
        split: build_dataset(dataset, root, split, use_cams, map_root, seed=seed)
        for split in ("train", "val", "test")
    }
    for split in sets:
        sets[split][0]
    return sets


def make_loaders(sets, batch_size=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            sets["train"], batch_size=batch_size, shuffle=True,
            drop_last=False, num_workers=3, generator=generator,
            worker_init_fn=seed_worker,
        ),
        "val": DataLoader(sets["val"], batch_size=1, shuffle=False, num_workers=0),
        "test": DataLoader(sets["test"], batch_size=1, shuffle=False, num_workers=0),
    }
    return loaders


def build_dataloaders(dataset, root, use_cams=False, map_root=None, batch_size=2, seed=0):
    sets = build_datasets(dataset, root, use_cams, map_root, seed=seed)
    return sets, make_loaders(sets, batch_size, seed)
