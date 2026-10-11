"""Dependency-free video front end built from :mod:`tidal.modalities.image`.

A clip is a sequence of image-like frames (or a T×H×W×C numpy array). Frames are uniformly sampled,
encoded independently, and pooled with mean/last/motion statistics.
"""
from __future__ import annotations
import numpy as np
from tidal.modalities import register
from tidal.modalities.image import ImageFrontEnd, _image_array

@register
class VideoFrontEnd:
    name = "video"; dim = 128
    def __init__(self, max_frames=8): self.max_frames = int(max_frames)
    def available(self): return False
    def encode(self, clips):
        enc = ImageFrontEnd(); out = []
        for clip in clips:
            if isinstance(clip, np.ndarray) and clip.ndim == 4:
                frames = [clip[i] for i in range(clip.shape[0])]
            elif isinstance(clip, np.ndarray) and clip.ndim in (2, 3):
                frames = [clip]
            else:
                frames = list(clip) if not isinstance(clip, (str, bytes, bytearray)) else [clip]
            if not frames: raise ValueError("video clip must contain at least one frame")
            idx = np.linspace(0, len(frames) - 1, min(self.max_frames, len(frames))).round().astype(int)
            z = enc.encode([frames[i] for i in idx])
            mean, last = z.mean(0), z[-1]
            motion = np.abs(np.diff(z, axis=0)).mean(0) if len(z) > 1 else np.zeros(self.dim, np.float32)
            out.append((0.5 * mean + 0.3 * last + 0.2 * motion).astype(np.float32))
        return np.stack(out) if out else np.empty((0, self.dim), np.float32)
