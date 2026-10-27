"""Single-projection adapter; the core IBP objective is selected independently."""
from ..common.adapter import ComparisonModelBundle
from ..common.probability import OCTAProbabilityAdapter, require_octa_config


class S3OCTAAdapter(OCTAProbabilityAdapter):
    def prepare_input(self, image):
        image = super().prepare_input(image)
        # Same input information as the other models, NOT three independent views.
        return image.repeat(1, 3, 1, 1)


def build(config):
    size = require_octa_config(config, "s3_octa")
    from .S3OCTA import S3OCTA
    # Match the supplied train builder, including its effective extend_scope=3.
    # Build on CPU; a caller may subsequently move the model with .to(device).
    model = S3OCTA(in_channels=3, out_channels=1, kernel_size=11,
                   extend_scope=3, layer_depth=3, rate=72,
                   ga="MiT", device_id="cpu")
    return ComparisonModelBundle(model, S3OCTAAdapter(size))
