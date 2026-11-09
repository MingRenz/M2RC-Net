import torch
from torch import nn
from .backbone import SSMBase
from .mscf_guided import MSCFBackboneWithCVE
from .mbrc import MBRC


class M2RCNet(nn.Module):
    def __init__(self, use_cve=True, seed=0):
        super().__init__()
        self.use_cve = bool(use_cve)
        self.base = MSCFBackboneWithCVE() if self.use_cve else SSMBase()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.mbrc = MBRC(seed=seed)
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
            if self.use_cve:
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
            if self.use_cve:
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


def build_model(use_mscf=False, use_cve=False, use_mbrc=False, seed=0):
    if use_cve and not use_mscf:
        raise ValueError("CVE requires MSCF")
    if use_mbrc:
        if use_mscf and not use_cve:
            from .m2rcnet_without_cve import M2RCNetWithoutCVE
            return M2RCNetWithoutCVE(seed=seed)
        return M2RCNet(use_cve=use_cve, seed=seed)
    if use_mscf:
        if use_cve:
            return MSCFBackboneWithCVE()
        from .base_mscf import BaseMSCF
        return BaseMSCF()
    return SSMBase()


def build_m2rcnet(seed=0):
    return M2RCNet(use_cve=True, seed=seed)
