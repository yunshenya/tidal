"""Fit and held ids for LC-NE arousal. Numeric schedules, not sentences.

No torch. The schedules live in the eval script. These ids only keep the
banks disjoint.
"""
from __future__ import annotations

import hashlib

PROTOCOL = "arousal-lcne-v1"
FIT_N = 8
HELD_N = 50
SURPRISE_SEEDS = (1.0, 1.0, 1.0, 1.0)
SURPRISE_HIGH = 4.0
QUIET_SHORT = 1
QUIET_LONG = 4
VALENCE_LOW = 0.5
VALENCE_HIGH = 1.5
TRIGGER_LOW = 1
TRIGGER_HIGH = 4


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("arousal-fit", FIT_N)
HELD_IDS = _ids("arousal-held", HELD_N)


def bank_sha256() -> str:
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            ",".join(str(item) for item in SURPRISE_SEEDS),
            str(SURPRISE_HIGH),
            str(QUIET_SHORT),
            str(QUIET_LONG),
            str(VALENCE_LOW),
            str(VALENCE_HIGH),
            str(TRIGGER_LOW),
            str(TRIGGER_HIGH),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
