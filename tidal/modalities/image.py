"""Image front end -- PLANNED (not implemented). Interface only, so the encoder/data code can be written against it.

Plan (docs/architecture.md): a tiny conv/patch stem (e.g. 4x4 patches of a 96-128 px thumbnail -> depthwise-separable
convs or a 2-layer patch transformer, <0.5M params) distilled offline from a larger image encoder (e.g. a Chinese-CLIP
class teacher) on non-private images; int8 ONNX on CPU. Stickers/emoji get an extra learned id embedding."""
from tidal.modalities import register

@register
class ImageFrontEnd:
    name = "image"; dim = 128
    def available(self): return False
    def encode(self, images):
        raise NotImplementedError("image front end is planned; see docs/architecture.md")
