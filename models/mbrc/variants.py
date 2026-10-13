"""Six structural ablations of the strict, direct-detach MBRC model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Variant:
    name: str
    multi: bool
    action: bool
    boundary: bool


VARIANTS = {
    "mamba": Variant("mamba", False, False, False),
    "full": Variant("full", True, True, True),
    "multi_only": Variant("multi_only", True, False, False),
    "action_only": Variant("action_only", False, True, False),
    "boundary_only": Variant("boundary_only", False, False, True),
    "multi_action": Variant("multi_action", True, True, False),
    "multi_boundary": Variant("multi_boundary", True, False, True),
    "action_boundary": Variant("action_boundary", False, True, True),
}
