"""Fit and held ids for reconstructive phone and birthday recall.

No torch. The ids keep the banks disjoint. The phone, the date, and the
schedules are the pre-registered probe, not fitted constants. Every id runs
that same probe, so the canonical spelling does not depend on the id.
"""
from __future__ import annotations

import hashlib

from tidal.reconstruct import (
    EXACT_AGE,
    EXACT_STRENGTH,
    EXACT_VALENCE,
    LONG_AGE,
    LOSS_LINE_OFFSET,
    LOSS_STRENGTH,
    recall_surface,
    residue_from_answer,
)

PROTOCOL = "reconstruct-spell-v1"
FIT_N = 8
HELD_N = 50
PHONE = "+1-617-555-4826"
BIRTHDAY = "April 21, 1984"
LURE_PHONE = "+1-617-555-9090"
LURE_BIRTHDAY = "June 9, 1972"
PHONE_PROMPT = "What is Ada's phone number?"
BIRTHDAY_PROMPT = "When is Ada's birthday?"
OFF_PROMPT = "Which city does Ada live in?"
HIGH = (1.0, 0, 0.0)
LOW = (0.2, 0, 0.0)
LONG = (1.0, 4, 0.0)
EMOTION = (1.0, 0, 1.5)
SLIP = (0.5, 0, 0.0)
SCHEDULES = (
    ("high", HIGH),
    ("low", LOW),
    ("long", LONG),
    ("emotion", EMOTION),
    ("slip", SLIP),
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("reconstruct-fit", FIT_N)
HELD_IDS = _ids("reconstruct-held", HELD_N)


def canonical_surfaces() -> dict:
    """Spellings this probe must produce. Identical for every id."""
    phone = residue_from_answer(PHONE)
    birthday = residue_from_answer(BIRTHDAY)
    out = {"off_relation": None, "interface_version": 6}
    for label, schedule in SCHEDULES:
        out[f"phone_{label}"] = recall_surface(phone, *schedule)
        out[f"birthday_{label}"] = recall_surface(birthday, *schedule)
    return out


def bank_sha256() -> str:
    surfaces = canonical_surfaces()
    lines = [
        *FIT_IDS,
        *HELD_IDS,
        PROTOCOL,
        PHONE,
        BIRTHDAY,
        LURE_PHONE,
        LURE_BIRTHDAY,
        PHONE_PROMPT,
        BIRTHDAY_PROMPT,
        OFF_PROMPT,
        str(EXACT_STRENGTH),
        str(EXACT_AGE),
        str(EXACT_VALENCE),
        str(LOSS_STRENGTH),
        str(LONG_AGE),
        str(LOSS_LINE_OFFSET),
    ]
    for label, schedule in SCHEDULES:
        lines.append(label + ":" + ",".join(str(item) for item in schedule))
        lines.append(str(surfaces[f"phone_{label}"]))
        lines.append(str(surfaces[f"birthday_{label}"]))
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
