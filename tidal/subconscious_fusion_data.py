"""Fit and held ids for fusion scorer v0. Numeric fixtures, not sentences.

No torch. The ids keep the banks disjoint. Every id runs the same
pre-registered fixtures, so the canonical scores do not depend on the id.
"""
from __future__ import annotations

import hashlib

import numpy as np

from tidal.fusion import (
    CAUSAL_FLOOR,
    CAUSAL_TAU,
    FP_FLOOR,
    FUSION_WINDOW,
    RELIABILITY_SAT,
    TIME_PRIOR,
    FusionObservation,
    score_fusion,
)
from tidal.interface_v7 import FINGERPRINT_DIM, FUSION_WEIGHT_ORDER, empty_fusion_weights, zero_fingerprint

PROTOCOL = "v7-fusion-scorer-v0"
FIT_N = 8
HELD_N = 50
GATES = (
    "v7-fusion-scorer-multi",
    "v7-fusion-scorer-single",
    "v7-fusion-scorer-empty",
    "v7-fusion-scorer-causal",
    "v7-fusion-scorer-time",
    "v7-fusion-scorer-window",
    "v7-fusion-scorer-range",
    "v7-fusion-scorer-weights",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("fusion-fit", FIT_N)
HELD_IDS = _ids("fusion-held", HELD_N)


def _mark(scale: float) -> np.ndarray:
    return (np.linspace(1.0, 2.0, FINGERPRINT_DIM, dtype=np.float32) * np.float32(scale))


def matching_pair() -> tuple[FusionObservation, FusionObservation]:
    mark = _mark(1.0)
    return (
        FusionObservation(t=1, modality="text", fingerprint=mark, strength=1.0),
        FusionObservation(t=2, modality="audio", fingerprint=mark.copy(), strength=1.0),
    )


def conflicting_pair() -> tuple[FusionObservation, FusionObservation]:
    return (
        FusionObservation(t=1, modality="text", fingerprint=_mark(1.0), strength=1.0),
        FusionObservation(t=2, modality="audio", fingerprint=-_mark(1.0), strength=1.0),
    )


def time_only_pair() -> tuple[FusionObservation, FusionObservation]:
    zeros = zero_fingerprint()
    return (
        FusionObservation(t=1, modality="text", fingerprint=zeros, strength=1.0),
        FusionObservation(t=2, modality="audio", fingerprint=zeros.copy(), strength=1.0),
    )


def far_pair() -> tuple[FusionObservation, FusionObservation]:
    mark = _mark(1.0)
    return (
        FusionObservation(t=1, modality="text", fingerprint=mark, strength=1.0),
        FusionObservation(t=1 + int(FUSION_WINDOW) + 1, modality="audio", fingerprint=mark.copy(), strength=1.0),
    )


def canonical_scores() -> dict:
    """Scores this fixture set must produce. Identical for every id."""
    match = score_fusion(matching_pair())
    time_only = score_fusion(time_only_pair())
    conflict = score_fusion(conflicting_pair())
    single = score_fusion(matching_pair()[:1])
    empty = score_fusion(())
    far = score_fusion(far_pair(), now=far_pair()[1].t)
    growbrain = np.array([1.0, 0.25, 0.0], dtype=np.float32)
    weighted = score_fusion(matching_pair(), weights=growbrain)
    return {
        "protocol": PROTOCOL,
        "matching": float(match),
        "time_only": float(time_only),
        "conflict": float(conflict),
        "single": float(single),
        "empty": float(empty),
        "far": float(far),
        "weighted": float(weighted),
        "growbrain_weights": [float(item) for item in growbrain],
        "weight_order": list(FUSION_WEIGHT_ORDER),
        "empty_weights": [float(item) for item in empty_fusion_weights()],
        "window": int(FUSION_WINDOW),
        "fp_floor": float(FP_FLOOR),
        "time_prior": float(TIME_PRIOR),
        "causal_tau": float(CAUSAL_TAU),
        "causal_floor": float(CAUSAL_FLOOR),
        "reliability_sat": float(RELIABILITY_SAT),
    }


def gate_row() -> dict:
    """One item's gate outcomes. The id is not an input to the score."""
    scores = canonical_scores()
    matching = float(scores["matching"])
    time_only = float(scores["time_only"])
    weighted = float(scores["weighted"])
    gates = {
        "v7-fusion-scorer-multi": 0.0 < matching <= 1.0,
        "v7-fusion-scorer-single": float(scores["single"]) == 0.0 and float(scores["empty"]) == 0.0,
        "v7-fusion-scorer-empty": float(scores["empty"]) == 0.0,
        "v7-fusion-scorer-causal": float(scores["conflict"]) == 0.0 and time_only < matching,
        "v7-fusion-scorer-time": 0.0 < time_only < matching,
        "v7-fusion-scorer-window": float(scores["far"]) == 0.0,
        "v7-fusion-scorer-range": matching == 1.0 and 0.0 < weighted <= 1.0,
        "v7-fusion-scorer-weights": weighted < matching and scores["empty_weights"] == [0.0, 0.0, 0.0],
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "scores": scores,
    }


def bank_sha256() -> str:
    scores = canonical_scores()
    lines = [
        *FIT_IDS,
        *HELD_IDS,
        PROTOCOL,
        str(scores["window"]),
        str(scores["fp_floor"]),
        str(scores["time_prior"]),
        str(scores["causal_tau"]),
        str(scores["causal_floor"]),
        str(scores["reliability_sat"]),
        *GATES,
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
