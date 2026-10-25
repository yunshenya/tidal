"""Fit and held ids for weak-cue spontaneous recollection. Numeric fixtures.

No torch. Every id runs the same pre-registered fixtures, so the
canonical overlap and mode-scaled pressure numbers do not depend on
the id.
"""
from __future__ import annotations

import hashlib

import numpy as np

from tidal.arousal import AROUSAL_BASELINE
from tidal.memory_mode import ENCODE_SPONTANEOUS_KEEP, apply_sleep_operators, memory_mode, spontaneous_pressure
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_PERIOD, SPONTANEOUS_S_MIN, wake_after
from tidal.weak_cue import (
    WEAK_CEILING,
    WEAK_FLOOR,
    WEAK_OVERLAP,
    context_fragment,
    in_weak_band,
    overlap_score,
    weak_winner,
)

PROTOCOL = "weak-cue-recollection-v0"
FIT_N = 8
HELD_N = 50
WAKE_DURATION = 4.0
AROUSAL_HIGH = 1.0
PROBE_DIM = 8
GATES = (
    "identity",
    "weak",
    "strong",
    "consolidate",
    "encode",
    "compose",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("weak-cue-fit", FIT_N)
HELD_IDS = _ids("weak-cue-held", HELD_N)


def _axis(index: int, dim: int = PROBE_DIM) -> np.ndarray:
    row = np.zeros(int(dim), dtype=np.float64)
    row[int(index)] = 1.0
    return row


def schedule_scalars() -> dict:
    """Locked process-S / overlap / mode points."""
    after_wake = wake_after(float(SLEEP_FLOOR), float(WAKE_DURATION))
    key = _axis(0)
    fragment = context_fragment(key)
    return {
        "floor": float(SLEEP_FLOOR),
        "baseline": float(AROUSAL_BASELINE),
        "after_wake": float(after_wake),
        "spontaneous_min": float(SPONTANEOUS_S_MIN),
        "arousal_high": float(AROUSAL_HIGH),
        "encode_pressure": float(after_wake * float(ENCODE_SPONTANEOUS_KEEP)),
        "weak_floor": float(WEAK_FLOOR),
        "weak_overlap": float(WEAK_OVERLAP),
        "weak_ceiling": float(WEAK_CEILING),
        "period": int(SPONTANEOUS_PERIOD),
        "rate": 1.0 / float(SPONTANEOUS_PERIOD),
        "fragment_cosine": float(overlap_score(fragment, key)),
        "strong_cosine": 1.0,
        "miss_cosine": 0.0,
    }


def formula_gates() -> dict:
    """Pure-function gates. Live remember / ticks are scored in tests."""
    scalars = schedule_scalars()
    key = _axis(0)
    decoy = _axis(2)
    fragment = context_fragment(key)
    overlaps = [overlap_score(fragment, key), overlap_score(fragment, decoy)]
    identity = (
        scalars["weak_overlap"] == 0.25
        and scalars["weak_floor"] == 0.10
        and scalars["weak_ceiling"] == 0.32
        and scalars["fragment_cosine"] == scalars["weak_overlap"]
        and scalars["period"] == 16
        and scalars["rate"] == 0.0625
    )
    weak = (
        in_weak_band(scalars["fragment_cosine"]) is True
        and in_weak_band(scalars["weak_floor"]) is True
        and in_weak_band(scalars["weak_ceiling"]) is False
        and in_weak_band(scalars["miss_cosine"]) is False
        and weak_winner(overlaps, [0.1, 0.9], [1.0, 1.0]) == 0
    )
    strong = (
        in_weak_band(scalars["strong_cosine"]) is False
        and scalars["strong_cosine"] >= scalars["weak_ceiling"]
        and scalars["fragment_cosine"] < scalars["weak_ceiling"]
        and weak_winner([1.0, 0.25]) == 1
    )
    wake_mode = memory_mode(scalars["after_wake"], scalars["baseline"], 0.0)
    aroused_mode = memory_mode(scalars["after_wake"], scalars["arousal_high"], 0.0)
    consolidate = (
        wake_mode == "consolidate"
        and spontaneous_pressure(wake_mode, scalars["after_wake"]) == scalars["after_wake"]
        and scalars["after_wake"] >= scalars["spontaneous_min"]
        and apply_sleep_operators(wake_mode) is True
    )
    encode = (
        aroused_mode == "encode"
        and spontaneous_pressure(aroused_mode, scalars["after_wake"]) == scalars["encode_pressure"]
        and scalars["encode_pressure"] < scalars["spontaneous_min"]
        and apply_sleep_operators(aroused_mode) is True
    )
    compose = (
        apply_sleep_operators("encode") is True
        and apply_sleep_operators("consolidate") is True
        and memory_mode(scalars["floor"], scalars["baseline"], 0.0) == "encode"
        and weak_winner([0.0, 0.0]) is None
    )
    export = (
        scalars["weak_ceiling"] == 0.32
        and scalars["weak_overlap"] == 0.25
        and scalars["rate"] < 0.10
    )
    return {
        "identity": bool(identity),
        "weak": bool(weak),
        "strong": bool(strong),
        "consolidate": bool(consolidate),
        "encode": bool(encode),
        "compose": bool(compose),
        "export": bool(export),
        "scalars": scalars,
    }


def bank_sha256() -> str:
    scalars = schedule_scalars()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            str(scalars["after_wake"]),
            str(scalars["encode_pressure"]),
            str(scalars["fragment_cosine"]),
            str(scalars["weak_floor"]),
            str(scalars["weak_overlap"]),
            str(scalars["weak_ceiling"]),
            str(scalars["rate"]),
            *GATES,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
