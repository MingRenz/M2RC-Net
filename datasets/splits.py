import random
from pathlib import Path


def validate_split(dataset, name, split, seed):
    path = Path(__file__).resolve().parents[1]/"splits"/name/f"{split}.txt"
    names = path.read_text(encoding="utf-8").splitlines()
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate names in frozen split: {path}")
    random.Random(seed).shuffle(names)
    if dataset.names != names:
        raise RuntimeError(f"Dataset membership/order differs from frozen split: {name}/{split}")
