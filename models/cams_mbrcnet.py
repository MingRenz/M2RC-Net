"""Assembly only. Backbone, CAMS and MBRC mathematics live in their modules."""
import torch
from torch import nn
from .backbone import StableCleanSSMBase
from .cams import StableCleanSSMBaseCAMS
from .mbrc import VariantMBRC, VARIANTS


class CAMS_MBRCNet(nn.Module):
    def __init__(self, use_cams=True):
        super().__init__()
        self.use_cams = bool(use_cams)
        self.base = StableCleanSSMBaseCAMS() if self.use_cams else StableCleanSSMBase()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            self.mbrc = VariantMBRC(VARIANTS["full"], f1_channels=32)
        self._features = {}
        self._handles = [
            module.register_forward_hook(self._capture(name))
            for name, module in (
                ("d3", self.base.d3),
                ("d2", self.base.d2),
                ("d1", self.base.d1),
                ("z_base", self.base.conv_out),
            )
        ]

    def _capture(self, name):
        def hook(_module, _inputs, output):
            if name in self._features or not isinstance(output, torch.Tensor):
                raise RuntimeError("Invalid decoder feature capture: " + name)
            self._features[name] = output.clone()
        return hook

    def forward(self, image):
        self._features.clear()
        try:
            if self.use_cams:
                z_base, caliber, vesselness_logits = self.base(image)
            else:
                z_base = self.base(image)
            if set(self._features) != {"d3", "d2", "d1", "z_base"}:
                raise RuntimeError("Decoder feature capture incomplete")
            if not torch.equal(z_base, self._features["z_base"]):
                raise RuntimeError("Captured logits differ from base output")
            z_final, delta_z, action_logits, edge_logits = self.mbrc(
                self._features["z_base"],
                self._features["d1"],
                self._features["d2"],
                self._features["d3"],
            )
            if self.use_cams:
                return {
                    "z_base": z_base,
                    "z_final": z_final,
                    "base_prob": torch.sigmoid(z_base),
                    "final_prob": torch.sigmoid(z_final),
                    "caliber": caliber,
                    "vesselness_logits": vesselness_logits,
                    "delta_z": delta_z,
                    "action_logits": action_logits,
                    "edge_logits": edge_logits,
                }
            return {
                "base_prob": torch.sigmoid(z_base),
                "z_base": self._features["z_base"],
                "z_final": z_final,
                "delta_z": delta_z,
                "action_logits": action_logits,
                "edge_logits": edge_logits,
            }
        finally:
            self._features.clear()


def build_model(use_cams=False, use_mbrc=False):
    if use_mbrc:
        return CAMS_MBRCNet(use_cams=use_cams)
    if use_cams:
        return StableCleanSSMBaseCAMS()
    return StableCleanSSMBase()
