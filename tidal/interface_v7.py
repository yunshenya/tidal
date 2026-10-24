"""Frozen growbrain multimodal packet contract for ``interface_version`` 7.

This module is type stubs and named constants only. Live cue export stays
:data:`tidal.llm_memory.CUE_INTERFACE_VERSION` 6. ``cue_fields`` does not
emit these fields. Exporters do not bump the live version. Wiring lands in
a later pull request after text stage-1.

Agreed with growbrain / 两仪式 on 2026-10-25. growbrain mirrors the same
shape in ``TIDAL_INTERFACE`` v7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Protocol, Sequence, TypedDict

import numpy as np
from numpy.typing import NDArray

# Planned packet version. Not assigned to CUE_INTERFACE_VERSION.
INTERFACE_VERSION_V7 = 7

# Perceptual fingerprint width. Aligned with growbrain mem_key_dim. Hypothesis
# until a fingerprint encoder is trained.
FINGERPRINT_DIM = 32

# fusion_weights slots. Order is text, audio, image. fused and none do not
# occupy a slot.
FUSION_WEIGHT_DIM = 3
FUSION_WEIGHT_ORDER: tuple[str, str, str] = ("text", "audio", "image")

Modality = Literal["text", "audio", "image", "fused", "none"]
MODALITY_VALUES: tuple[Modality, ...] = ("text", "audio", "image", "fused", "none")

Fingerprint = NDArray[np.float32]
FusionWeights = NDArray[np.float32]


class CueV7Additions(TypedDict):
    """Fields added on the cue winner / packet when live export becomes 7.

    ``fingerprint`` is memory-level ``(F,)``, not per-token. ``F`` is
    :data:`FINGERPRINT_DIM`. ``fusion_confidence`` is one float in ``[0, 1]``
    per sample, last-token / sample-level, the same rule as the v6 gate
    scalars. No cross-modal fusion stores 0. ``fusion_weights`` is not a
    cue-packet field.
    """

    fingerprint: Fingerprint
    fusion_confidence: float


class MemoryStoreV7(TypedDict):
    """Per-memory store fields in the v7 contract.

    ``vec`` already exists: float32, width ``d_model`` (stage0: 192).
    ``fingerprint`` is produced by growbrain modality encoders. tidal stores
    and returns it. tidal does not compute it. Live writes stay text-only
    until the wiring pull request. Unknown modality is ``none``. ``fused``
    is a memory after tidal low-level auto-fusion. A single-modality memory
    keeps its original label. Touch and proprioception are out of scope.
    """

    vec: NDArray[np.float32]
    fingerprint: Fingerprint
    modality: Modality


class FusionWeightsWrite(TypedDict):
    """growbrain → tidal control payload. Not part of cue_fields.

    ``weights`` is float32 ``(M,)`` with ``M`` :data:`FUSION_WEIGHT_DIM` and
    order :data:`FUSION_WEIGHT_ORDER`. Sleep replay and low-level fusion
    read these weights. Semantic new pairings use the same write.
    """

    weights: FusionWeights
    pair: Optional[object]


class FusionControl(Protocol):
    """Signature stub for the growbrain → tidal write API.

    Implementations land later. This Protocol documents the call shape.
    It is not registered on ``LLMMemory``.
    """

    def set_fusion_weights(
        self,
        weights: FusionWeights,
        *,
        pair: Optional[object] = None,
    ) -> None:
        """Store ``(M,)`` fusion weights, optionally with a new pairing."""


@dataclass(frozen=True)
class FusionWeightsStub:
    """Concrete shape for the write API. No store side effects."""

    weights: FusionWeights
    pair: Optional[object] = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.weights, dtype=np.float32)
        if arr.shape != (FUSION_WEIGHT_DIM,):
            raise ValueError(f"fusion_weights must have shape ({FUSION_WEIGHT_DIM},)")
        object.__setattr__(self, "weights", arr)


def zero_fingerprint() -> Fingerprint:
    """Placeholder while text has no encoder. Does not enter the v6 packet."""
    return np.zeros((FINGERPRINT_DIM,), dtype=np.float32)


def empty_fusion_weights() -> FusionWeights:
    """Uniform-zero ``(M,)`` weights. Does not enter the v6 packet."""
    return np.zeros((FUSION_WEIGHT_DIM,), dtype=np.float32)


def check_modality(value: str) -> Modality:
    if value not in MODALITY_VALUES:
        raise ValueError(f"unknown modality {value!r}; use none")
    return value  # type: ignore[return-value]


def check_fingerprint(value: Sequence[float] | Fingerprint) -> Fingerprint:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (FINGERPRINT_DIM,):
        raise ValueError(f"fingerprint must have shape ({FINGERPRINT_DIM},)")
    return arr
