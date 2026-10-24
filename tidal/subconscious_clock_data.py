"""Fit and held ids and numeric schedules for the wall clock.

No torch. The ids keep the banks disjoint. The durations are the
pre-registered schedule, not fitted constants.
"""
from __future__ import annotations

import hashlib

PROTOCOL = "clock-drift-v1"
FIT_N = 8
HELD_N = 50
WALL_SHORT = (1.0, 4.0)
WALL_CLEAR = 300.0
TOKEN_STEPS = (0, 1, 4)
BOTH_WALL_STRONG = (1, 30.0)
BOTH_TOKEN_STRONG = (8, 1.0)
ENTRAIN_SECONDS = 8.0


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("clock-fit", FIT_N)
HELD_IDS = _ids("clock-held", HELD_N)


def bank_sha256() -> str:
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            ",".join(str(item) for item in WALL_SHORT),
            str(WALL_CLEAR),
            ",".join(str(item) for item in TOKEN_STEPS),
            ",".join(str(item) for item in BOTH_WALL_STRONG),
            ",".join(str(item) for item in BOTH_TOKEN_STRONG),
            str(ENTRAIN_SECONDS),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
