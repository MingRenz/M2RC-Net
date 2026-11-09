import torch
from torch import nn
from .m2rcnet import M2RCNet
from .mscf import MSCFBackbone
from .mbrc import MBRC

EXPECTED_PARAMETERS = 3319166


class M2RCNetWithoutCVE(M2RCNet):
    def __init__(self, seed=0):
        nn.Module.__init__(self)
        self.use_cve = False
        self.base = MSCFBackbone()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.mbrc = MBRC(seed=seed)
        self._features = {}
        self._handles = [module.register_forward_hook(self._capture(name))
                         for name, module in (('d3',self.base.d3),
                                              ('d2',self.base.d2),
                                              ('d1',self.base.d1),
                                              ('z_base',self.base.conv_out))]

    def forward(self, image):
        output = super().forward(image)
        output['final_prob'] = torch.sigmoid(output['z_final'])
        return output


def build_m2rcnet_without_cve(seed=0):
    return M2RCNetWithoutCVE(seed=seed)
