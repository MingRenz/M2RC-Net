from dataclasses import dataclass
from .core import MBRC

@dataclass(frozen=True)
class MBRCObjective:
    name: str = "full"
    action: bool = True
    boundary: bool = True

FULL_MBRC = MBRCObjective()
