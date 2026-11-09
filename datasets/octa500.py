import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

SEED = 0
EXPECTED = {"train": 140, "val": 10, "test": 50}


class OCTA3M(Dataset):
    def __init__(self, root, split, seed=SEED):
        self.image_dir = root / split / "image"
        self.label_dir = root / split / "label"
        if not self.image_dir.is_dir() or not self.label_dir.is_dir():
            raise FileNotFoundError(f"缺少 {split}/image 或 {split}/label")

        names = sorted(
            p.name for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower()
            in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        )
        if len(names) != EXPECTED[split]:
            raise RuntimeError(
                f"{split} 图像数量为 {len(names)}，预期 {EXPECTED[split]}"
            )
        missing = [name for name in names if not (self.label_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"{split} 缺少同名标签：{missing[:5]}")

        random.Random(seed).shuffle(names)
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = cv2.imread(str(self.image_dir / name), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(self.label_dir / name), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise RuntimeError(f"无法读取图像或标签：{name}")

        image = cv2.resize(
            image.astype(np.float32), (224, 224),
            interpolation=cv2.INTER_LINEAR
        ) / 255.0
        label = cv2.resize(
            label.astype(np.float32), (224, 224),
            interpolation=cv2.INTER_NEAREST
        ) / 255.0
        return (
            name,
            torch.from_numpy(np.ascontiguousarray(image[None])),
            torch.from_numpy(np.ascontiguousarray(label[None])),
        )


def seed_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


LAYOUTS = {
    "OCTA500_6M": {
        "train": (("train",), 180),
        "val": (("val",), 20),
        "test": (("test",), 100),
    },
    "ROSSA": {
        "train": (("train_manual", "train_sam"), 718),
        "val": (("val",), 100),
        "test": (("test",), 100),
    },
}
EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OCTAMulti(Dataset):
    def __init__(self, root, split, seed=SEED):
        root = Path(root).expanduser().resolve(strict=True)
        if root.name not in LAYOUTS:
            raise ValueError(f"Dataset root must be OCTA500_6M or ROSSA: {root}")
        if split not in LAYOUTS[root.name]:
            raise ValueError(split)
        source_splits, expected = LAYOUTS[root.name][split]
        self.root, self.split = root, split
        self.image_dir = root / source_splits[0] / "image"
        entries = []
        seen = set()
        for source_split in source_splits:
            images = root / source_split / "image"
            labels = root / source_split / "label"
            if not images.is_dir() or not labels.is_dir():
                raise FileNotFoundError(f"Missing image/label directory: {source_split}")
            image_names = {p.name for p in images.iterdir()
                           if p.is_file() and p.suffix.lower() in EXTENSIONS}
            label_names = {p.name for p in labels.iterdir()
                           if p.is_file() and p.suffix.lower() in EXTENSIONS}
            if image_names != label_names:
                raise RuntimeError(f"Unpaired files in {source_split}: "
                                   f"images-only={sorted(image_names-label_names)[:5]}, "
                                   f"labels-only={sorted(label_names-image_names)[:5]}")
            for name in sorted(image_names):
                if name in seen:
                    raise RuntimeError(f"Duplicate sample name: {name}")
                seen.add(name)
                entries.append((name, images / name, labels / name))
        if len(entries) != expected:
            raise RuntimeError(f"{root.name}/{split}: {len(entries)} != {expected}")
        random.Random(seed).shuffle(entries)
        self.entries = entries
        self.names = [entry[0] for entry in entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, image_path, label_path = self.entries[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None or image.shape != label.shape:
            raise RuntimeError(f"Unreadable or spatially mismatched pair: {name}")
        image = cv2.resize(image.astype(np.float32), (224, 224),
                           interpolation=cv2.INTER_LINEAR) / 255.0
        label = cv2.resize(label.astype(np.float32), (224, 224),
                           interpolation=cv2.INTER_NEAREST) / 255.0
        return (name,
                torch.from_numpy(np.ascontiguousarray(image[None])),
                torch.from_numpy(np.ascontiguousarray(label[None])))
