"""CPU image front end.

The project deliberately avoids shipping a large vision checkpoint.  This module therefore provides a
small deterministic thumbnail encoder that is useful in production as a fallback and as a stable
pre-training target.  Inputs may be paths, bytes, PIL images, or numpy arrays in HWC/CHW layout.
"""
from __future__ import annotations
import io
import os
import numpy as np
from tidal.modalities import register


def _image_array(image) -> np.ndarray:
    if isinstance(image, (str, os.PathLike)):
        with open(image, "rb") as f:
            image = f.read()
    if isinstance(image, (bytes, bytearray, memoryview)):
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(bytes(image)))
        except Exception as e:
            raise ValueError("image bytes require Pillow") from e
    if hasattr(image, "convert") and hasattr(image, "size"):
        image = np.asarray(image.convert("RGB"))
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3:
        raise ValueError("image must be HxW, HxWxC, or CxHxW")
    if a.shape[-1] not in (1, 3, 4) and a.shape[0] in (1, 3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] not in (1, 3, 4):
        raise ValueError("image channel dimension must have 1, 3, or 4 channels")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("image dimensions must be non-zero")
    a = a[..., :3].astype(np.float32, copy=False)
    if a.max(initial=0) > 1.5: a /= 255.0
    return np.clip(a, 0.0, 1.0)


def _resize(a, size=16):
    # Bilinear-free area sampling keeps this dependency-free and deterministic.
    h, w = a.shape[:2]
    ys = np.minimum((np.arange(size) * h // size), h - 1)
    xs = np.minimum((np.arange(size) * w // size), w - 1)
    return a[ys[:, None], xs[None, :]]


@register
class ImageFrontEnd:
    name = "image"; dim = 128
    def available(self):
        # A learned checkpoint is optional; encode() always has a deterministic CPU fallback.
        return False
    def encode(self, images):
        out = []
        for image in images:
            a = _resize(_image_array(image), 16)
            gray = a.mean(-1)
            hist = np.concatenate([np.histogram(a[..., c], bins=16, range=(0, 1), density=True)[0] for c in range(3)])
            gx = np.diff(gray, axis=1).ravel(); gy = np.diff(gray, axis=0).ravel()
            stats = np.array([a[..., c].mean() for c in range(3)] + [a[..., c].std() for c in range(3)] +
                             [gray.mean(), gray.std(), np.abs(gx).mean(), np.abs(gy).mean()], np.float32)
            pix = gray.ravel()[:64]
            v = np.concatenate([hist, stats, pix]).astype(np.float32)
            if v.size < self.dim: v = np.pad(v, (0, self.dim - v.size))
            out.append(v[:self.dim])
        return np.stack(out) if out else np.empty((0, self.dim), np.float32)
