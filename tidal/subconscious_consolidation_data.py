"""Fit and held ids for McGaugh consolidation gain. Numeric schedules.

No torch. Every id runs the same pre-registered schedule, so the
canonical gains do not depend on the id.
"""
from __future__ import annotations

import hashlib

from tidal.arousal import AROUSAL_BASELINE
from tidal.consolidation import CONSOLIDATION_SPAN, consolidation_gain

PROTOCOL = "mcgaugh-consolidation-v1"
FIT_N = 8
HELD_N = 50
AROUSAL_MID = 0.5
AROUSAL_HIGH = 1.0
VALENCE_LOW = 0.5
VALENCE_HIGH = 1.5
GATES = (
    "identity",
    "arousal_mono",
    "valence_mono",
    "sign",
    "write",
    "replay",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("consolidation-fit", FIT_N)
HELD_IDS = _ids("consolidation-held", HELD_N)


def canonical_gains() -> dict:
    """Locked formula values. Identical for every id."""
    mid_low = consolidation_gain(AROUSAL_MID, VALENCE_LOW)
    mid_high = consolidation_gain(AROUSAL_MID, VALENCE_HIGH)
    high_low = consolidation_gain(AROUSAL_HIGH, VALENCE_LOW)
    high_high = consolidation_gain(AROUSAL_HIGH, VALENCE_HIGH)
    return {
        "baseline_zero": 1.0,
        "baseline_high": 1.0,
        "high_zero": 1.0,
        "mid_low": mid_low,
        "mid_high": mid_high,
        "high_low": high_low,
        "high_high": high_high,
        "high_high_neg": consolidation_gain(AROUSAL_HIGH, -VALENCE_HIGH),
        "span": float(CONSOLIDATION_SPAN),
        "baseline": float(AROUSAL_BASELINE),
    }


def formula_gates() -> dict:
    """Pure-function gates. Live write / replay / export are scored in tests."""
    gains = canonical_gains()
    identity = (
        consolidation_gain(AROUSAL_BASELINE, 0.0) == 1.0
        and consolidation_gain(AROUSAL_BASELINE, VALENCE_LOW) == 1.0
        and consolidation_gain(AROUSAL_BASELINE, VALENCE_HIGH) == 1.0
        and consolidation_gain(AROUSAL_MID, 0.0) == 1.0
        and consolidation_gain(AROUSAL_HIGH, 0.0) == 1.0
        and consolidation_gain(0.0, VALENCE_HIGH) == 1.0
    )
    arousal_mono = (
        1.0
        == consolidation_gain(AROUSAL_BASELINE, VALENCE_HIGH)
        < gains["mid_high"]
        < gains["high_high"]
        <= 1.0 + float(CONSOLIDATION_SPAN)
    )
    valence_mono = (
        1.0
        == consolidation_gain(AROUSAL_HIGH, 0.0)
        < gains["high_low"]
        < gains["high_high"]
        <= 1.0 + float(CONSOLIDATION_SPAN)
    )
    sign = (
        gains["high_high"] == gains["high_high_neg"]
        and consolidation_gain(AROUSAL_MID, VALENCE_LOW)
        == consolidation_gain(AROUSAL_MID, -VALENCE_LOW)
    )
    return {
        "identity": bool(identity),
        "arousal_mono": bool(arousal_mono),
        "valence_mono": bool(valence_mono),
        "sign": bool(sign),
        "gains": gains,
    }


def bank_sha256() -> str:
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            str(AROUSAL_BASELINE),
            str(AROUSAL_MID),
            str(AROUSAL_HIGH),
            str(VALENCE_LOW),
            str(VALENCE_HIGH),
            str(CONSOLIDATION_SPAN),
            str(canonical_gains()["high_high"]),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
