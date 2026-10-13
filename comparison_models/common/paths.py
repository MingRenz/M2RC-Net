"""Future experiment paths. This module never moves existing runs."""
from pathlib import Path
import re

DATASETS = ("OCTA500_3M", "OCTA500_6M", "ROSSA")
FAMILIES = ("proposed", "comparison")
_COMPONENT = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def seed_log_dir(project_root, dataset, family, model, seed):
    if dataset not in DATASETS:
        raise ValueError("unknown dataset: " + str(dataset))
    if family not in FAMILIES:
        raise ValueError("unknown model family: " + str(family))
    if not isinstance(model, str) or not _COMPONENT.fullmatch(model):
        raise ValueError("invalid model directory name")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return Path(project_root) / "log" / dataset / family / model / f"seed_{seed}"


def reserve_seed_log_dir(project_root, dataset, family, model, seed):
    """Create a fresh directory; refuse to overwrite an existing seed run."""
    path = seed_log_dir(project_root, dataset, family, model, seed)
    path.mkdir(parents=True, exist_ok=False)
    return path


def results_dir(project_root, dataset):
    if dataset not in DATASETS:
        raise ValueError("unknown dataset: " + str(dataset))
    return Path(project_root) / "results" / dataset
