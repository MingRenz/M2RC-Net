import torch
import torch.nn.functional as F

def recover(mbrc, d1, d2=None, d3=None):
    if d2 is None or d3 is None:
        raise RuntimeError('Multistage Recovery requires both D2 and D3')
    r3 = mbrc.recovery3(d3)
    r2 = mbrc.recovery2(mbrc.merge2(torch.cat([F.interpolate(r3, size=d2.shape[-2:], mode='nearest'), mbrc.project2(d2)], dim=1)))
    r1 = mbrc.recovery1(mbrc.merge1(torch.cat([F.interpolate(r2, size=d1.shape[-2:], mode='nearest'), mbrc.project1(d1)], dim=1)))
    return F.interpolate(r1, size=(224, 224), mode='nearest')
