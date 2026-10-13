"""ROSSA preserves the source train_manual/train_sam split via the shared reader."""
from .octa500 import OCTAMulti


class ROSSADataset(OCTAMulti):
    pass
