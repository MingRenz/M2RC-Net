import hashlib
import importlib.metadata
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random

import cv2
import numpy as np
from tqdm import tqdm

from .caliber import GEOMETRY, local_caliber


SPLITS = ("train", "val", "test")
EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
LAYOUTS = {
    "OCTA500_3M": {s: (s,) for s in SPLITS},
    "OCTA500_6M": {s: (s,) for s in SPLITS},
    "ROSSA": {"train": ("train_manual", "train_sam"), "val": ("val",), "test": ("test",)},
}


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_samples(dataset, data_root, split_root=None):
    if dataset not in LAYOUTS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    data_root = Path(data_root).expanduser().resolve(strict=True)
    split_root = Path(split_root or Path(__file__).resolve().parents[1]/"splits"/dataset)
    rows, seen = {}, set()
    for split in SPLITS:
        names = (split_root/f"{split}.txt").read_text(encoding="utf-8").splitlines()
        if not names or len(names) != len(set(names)):
            raise ValueError(f"Empty or duplicate frozen split: {split}")
        if any(Path(name).name != name or Path(name).suffix.lower() not in EXTENSIONS for name in names):
            raise ValueError(f"Invalid image filename in {split}")
        if seen.intersection(names):
            raise ValueError("Frozen splits overlap")
        seen.update(names)
        if len({Path(name).stem for name in names}) != len(names):
            raise ValueError(f"Duplicate cache stems in {split}")
        labels = {}
        for source in LAYOUTS[dataset][split]:
            image_dir, label_dir = data_root/source/"image", data_root/source/"label"
            if not image_dir.is_dir() or not label_dir.is_dir():
                raise FileNotFoundError(f"Missing image/label directories in {source}")
            images = {p.name for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
            masks = {p.name for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS}
            if images != masks:
                raise ValueError(f"Unpaired image/GT files in {source}")
            if set(labels).intersection(masks):
                raise ValueError(f"Duplicate sample names across source directories: {split}")
            labels.update({name: label_dir/name for name in masks})
        if set(names) != set(labels):
            raise ValueError(f"Dataset membership differs from frozen {split}.txt")
        random.Random(0).shuffle(names)
        rows[split] = [(name, labels[name]) for name in names]
    return rows


def claim_output(output_dir, data_root):
    requested = Path(output_dir).expanduser()
    if requested.is_symlink():
        raise ValueError("Output directory must not be a symbolic link")
    output_dir = requested.resolve()
    data_root = Path(data_root).expanduser().resolve(strict=True)
    if output_dir == data_root or data_root in output_dir.parents or output_dir in data_root.parents:
        raise ValueError("Output directory must be separate from the input dataset")
    for parent in output_dir.parents:
        if (parent/"prepared.json").is_file() and (parent/"caliber").is_dir():
            raise ValueError("Output directory must not be inside an existing CVE cache")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Output must be a new or empty directory; existing caches are never overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/"preparing.lock").open("x", encoding="utf-8") as stream:
        stream.write("CVE cache generation in progress\n")
    return output_dir


def compute_sample(task):
    name, mask_path = task
    mask_path = Path(mask_path)
    source = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if source is None:
        raise ValueError(f"Cannot read GT: {name}")
    label = cv2.resize(source.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)/255.0
    mask = label > 0.5
    raw, valid, samples = local_caliber(mask, GEOMETRY)
    if raw.shape != (224,224) or raw.dtype != np.float32 or valid.dtype != np.bool_ or samples.dtype != np.float32:
        raise RuntimeError(f"Unexpected geometry output format: {name}")
    if not np.isfinite(raw).all() or not np.isfinite(samples).all() or np.any(raw[~mask] != 0) or np.any(valid & ~mask):
        raise RuntimeError(f"Invalid caliber output: {name}")
    quality = {"name": name, "centerline_samples": int(len(samples)),
               "valid_vessel_fraction": float(valid.sum()/max(1,mask.sum()))}
    return name, raw, valid, samples, quality


def fit_training_p99(samples_by_split):
    nonempty = [s for s in samples_by_split["train"] if len(s)]
    if not nonempty:
        raise RuntimeError("No valid training caliber samples")
    samples = np.concatenate(nonempty)
    if samples.dtype != np.float32 or not np.isfinite(samples).all() or np.any(samples <= 0):
        raise RuntimeError("Invalid training centerline samples")
    p99 = float(np.percentile(samples, GEOMETRY["percentile"]))
    if not np.isfinite(p99) or p99 <= 0:
        raise RuntimeError("Invalid training P99")
    return p99, int(len(samples)), float(np.mean(samples > p99))


def normalize_caliber(raw, p99):
    if raw.dtype != np.float32 or not np.isfinite(raw).all() or not np.isfinite(p99) or p99 <= 0:
        raise ValueError("Normalization requires finite float32 caliber and positive P99")
    normalized = np.clip(raw / p99, 0, 1)
    return normalized.astype(np.float32)


def array_digest(directory):
    digest = hashlib.sha256()
    directory = Path(directory)
    for path in sorted(directory.rglob("*.npy")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def prepare_cache(dataset, data_root, output_dir, workers=1):
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    rows = load_samples(dataset, data_root)
    output = claim_output(output_dir, data_root)
    qualities = []
    samples_by_split = {split: [] for split in SPLITS}
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for split in SPLITS:
            directory = output/"caliber"/split
            directory.mkdir(parents=True)
            results = executor.map(compute_sample, rows[split], chunksize=1) if executor else map(compute_sample, rows[split])
            for name, raw, valid, samples, quality in tqdm(results, total=len(rows[split]), desc=f"CVE {split}", unit="image", mininterval=1.0):
                stem = Path(name).stem
                np.save(directory/f"{stem}.raw.npy", raw, allow_pickle=False)
                np.save(directory/f"{stem}.valid.npy", valid, allow_pickle=False)
                np.save(directory/f"{stem}.samples.npy", samples, allow_pickle=False)
                samples_by_split[split].append(samples)
                qualities.append({"split": split, **quality})
        p99, count, clipped = fit_training_p99(samples_by_split)
        for split in SPLITS:
            directory = output/"caliber"/split
            for name, _ in rows[split]:
                stem = Path(name).stem
                raw = np.load(directory/f"{stem}.raw.npy", allow_pickle=False)
                np.save(directory/f"{stem}.caliber.npy", normalize_caliber(raw,p99), allow_pickle=False)
        manifest = {"dataset": dataset, "names": {s: [name for name,_ in rows[s]] for s in SPLITS},
                    "effective_resolution": [224,224], "label_resize": "cv2.INTER_NEAREST",
                    "label_scaling": "float32 / 255.0; foreground > 0.5", "processing_seed": 0}
        save_json(output/"data_manifest.json", manifest)
        save_json(output/"caliber_quality.json", qualities)
        prepared = {"dataset": dataset, "geometry": GEOMETRY, "p99_train": p99,
                    "training_centerline_samples": count, "training_clipped_fraction": clipped,
                    "split": {s:len(rows[s]) for s in SPLITS}, "p99_source": "valid train centerline samples only",
                    "normalization": "clip(raw / p99_train, 0, 1), float32",
                    "caliber_digest": array_digest(output/"caliber"),
                    "versions": {package:importlib.metadata.version(package) for package in
                                 ["numpy","scipy","scikit-image","opencv-python-headless"]}}
        save_json(output/"prepared.json", prepared)
        (output/"preparing.lock").unlink()
        return prepared
    finally:
        if executor:
            executor.shutdown(wait=True)
