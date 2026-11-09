from .mscf import MSCFBackbone

EXPECTED_PARAMETERS = 3111232


class BaseMSCF(MSCFBackbone):
    pass


def build_base_mscf():
    return BaseMSCF()
