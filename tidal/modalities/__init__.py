"""Per-modality front ends for tidal's (planned) native multimodal encoder.

Design (see docs/architecture.md): every incoming event becomes one token for the shared causal temporal
encoder = timing/role features  +  sum of the projections of whatever modality payloads the event carries.

    front end            status        output (per event)
    meta  (message type) implemented   one-hot/multi-hot media kinds + counts (no content)  -> meta.py
    text  (bge-small-zh) implemented   512-d frozen sentence embedding                      -> text.py
    image (patch/conv)   planned       D-d embedding of one image / sticker                  -> image.py
    video (frames)       planned       pooled embedding of 4-8 uniformly sampled frames      -> video.py
    audio (log-mel)      planned       speech/music embedding (+ optional ASR / acoustic EOT) -> audio.py

Every front end exposes the same tiny interface (`FrontEnd`): `dim`, `available()` and `encode(items) -> [N, dim]`,
so the temporal model only ever sees fixed-size vectors and missing modalities are simply zero + a mask bit.
"""
from typing import Protocol, Sequence
import numpy as np

class FrontEnd(Protocol):
    name: str
    dim: int
    def available(self) -> bool: ...
    def encode(self, items: Sequence) -> np.ndarray: ...

REGISTRY = {}
def register(cls):
    REGISTRY[cls.name] = cls
    return cls
