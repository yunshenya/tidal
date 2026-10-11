"""Per-modality front ends for tidal's native multimodal encoder.

Design (see docs/architecture.md): every incoming event becomes one token for the shared causal temporal
encoder = timing/role features  +  sum of the projections of whatever modality payloads the event carries.

    front end            status        output (per event)
    meta  (message type) implemented   one-hot/multi-hot media kinds + counts (no content)  -> meta.py
    text  (bge-small-zh) implemented   512-d frozen sentence embedding                      -> text.py
    image (thumbnail)    fallback      D-d deterministic CPU embedding                        -> image.py
    video (frames)       fallback      pooled embedding of uniformly sampled frames            -> video.py
    audio (log-spectrum) fallback      D-d deterministic CPU embedding                        -> audio.py

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
