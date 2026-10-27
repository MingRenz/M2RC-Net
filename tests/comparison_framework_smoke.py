"""CPU-only contract smoke test; intentionally ignored by Git."""
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch import nn

from comparison_models import (
    build_comparison_model, register_comparison_model, registered_models,
)
from comparison_models.common import (
    ComparisonAdapter, ComparisonModelBundle, SegmentationOutput,
    reserve_seed_log_dir, results_dir,
)


class DummyAdapter(ComparisonAdapter):
    def extract_output(self, raw_output):
        return SegmentationOutput(raw_output, "logits", auxiliary=())


def dummy_builder(config):
    assert config == {"channels": 1}
    return ComparisonModelBundle(nn.Conv2d(1, 1, 1), DummyAdapter())


assert registered_models() == ()
try:
    build_comparison_model("octamamba", {})
except LookupError:
    pass
else:
    raise AssertionError("unintegrated model unexpectedly available")

register_comparison_model("dummy", __name__ + ":dummy_builder")
bundle = build_comparison_model("dummy", {"channels": 1})
image = torch.ones(1, 1, 8, 8)
output = bundle.adapter.extract_output(bundle.model(bundle.adapter.prepare_input(image)))
assert output.prediction.shape == (1, 1, 8, 8)
assert output.representation == "logits"
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    path = reserve_seed_log_dir(root, "OCTA500_3M", "comparison", "Dummy", 0)
    assert path == root / "log/OCTA500_3M/comparison/Dummy/seed_0"
    try:
        reserve_seed_log_dir(root, "OCTA500_3M", "comparison", "Dummy", 0)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing seed directory was overwritten")
    assert results_dir(root, "ROSSA") == root / "results/ROSSA"
print("COMPARISON_FRAMEWORK_SMOKE PASS")
