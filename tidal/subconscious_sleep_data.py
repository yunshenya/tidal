"""Fit and held ids for process S. Numeric schedules, not sentences.

No torch. The schedules live in the eval script. These ids only keep the
banks disjoint.
"""
from __future__ import annotations

import hashlib

PROTOCOL = "sleep-pressure-v1"
FIT_N = 8
HELD_N = 50
WAKE_DURATIONS = (1.0, 2.0, 4.0)
SLEEP_DURATIONS = (1.0, 3.0)
OFFLINE_TICKS = 16


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("sleep-fit", FIT_N)
HELD_IDS = _ids("sleep-held", HELD_N)


def bank_sha256() -> str:
    payload = "\n".join((*FIT_IDS, *HELD_IDS, PROTOCOL)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
