from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


class CaliberDataset(Dataset):
    def __init__(self, original, directory):
        self.original = original
        self.directory = Path(directory)

    def __len__(self):
        return len(self.original)

    def __getitem__(self, index):
        name, image, label = self.original[index]
        stem = Path(name).stem
        caliber = np.load(
            self.directory / f"{stem}.caliber.npy"
        )
        valid = np.load(
            self.directory / f"{stem}.valid.npy"
        )
        if caliber.dtype != np.float32:
            raise RuntimeError("Caliber must be float32")
        if caliber.shape != label.shape[-2:]:
            raise RuntimeError("Caliber shape mismatch")
        if valid.shape != caliber.shape:
            raise RuntimeError("Validity shape mismatch")
        if (
            not np.isfinite(caliber).all()
            or caliber.min() < 0
            or caliber.max() > 1
        ):
            raise RuntimeError("Invalid caliber values")
        if np.any(caliber[label[0] == 0] != 0):
            raise RuntimeError("Nonzero background caliber GT")

        return (
            name, image, label,
            caliber[None], valid[None].astype(np.float32),
        )
