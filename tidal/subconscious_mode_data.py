"""Fit and held ids for encode vs consolidate mode. Numeric schedules.

No torch. Every id runs the same pre-registered fixtures, so the
canonical write-keep and sleep numbers do not depend on the id.
"""
from __future__ import annotations

import hashlib

from tidal.arousal import AROUSAL_BASELINE
from tidal.memory_mode import (
    CONSOLIDATE_WRITE_KEEP,
    ENCODE_SPONTANEOUS_KEEP,
    ENCODE_WRITE_KEEP,
    apply_sleep_operators,
    memory_mode,
    spontaneous_pressure,
    write_keep,
)
from tidal.reconstruct import recall_surface, residue_from_answer
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_S_MIN, sleep_after, wake_after
from tidal.subconscious_reconstruct_data import BIRTHDAY, PHONE

PROTOCOL = "encode-consolidate-mode-v0"
FIT_N = 8
HELD_N = 50
WAKE_DURATION = 4.0
AROUSAL_HIGH = 1.0
GATES = (
    "identity",
    "encode",
    "consolidate",
    "write",
    "sleep",
    "compose",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("mode-fit", FIT_N)
HELD_IDS = _ids("mode-held", HELD_N)


def schedule_scalars() -> dict:
    """Locked process-S / arousal / circadian points."""
    after_wake = wake_after(float(SLEEP_FLOOR), float(WAKE_DURATION))
    after_sleep = sleep_after(after_wake, 1.0)
    return {
        "floor": float(SLEEP_FLOOR),
        "baseline": float(AROUSAL_BASELINE),
        "after_wake": float(after_wake),
        "after_sleep": float(after_sleep),
        "spontaneous_min": float(SPONTANEOUS_S_MIN),
        "arousal_high": float(AROUSAL_HIGH),
    }


def canonical_surfaces() -> dict:
    """Same phone under encode keep vs consolidate keep."""
    phone = residue_from_answer(PHONE)
    birthday = residue_from_answer(BIRTHDAY)
    encode_strength = float(ENCODE_WRITE_KEEP)
    consolidate_strength = float(CONSOLIDATE_WRITE_KEEP)
    return {
        "phone_encode": recall_surface(phone, encode_strength, 0, 0.0),
        "phone_consolidate": recall_surface(phone, consolidate_strength, 0, 0.0),
        "birthday_encode": recall_surface(birthday, encode_strength, 0, 0.0),
        "birthday_consolidate": recall_surface(birthday, consolidate_strength, 0, 0.0),
        "phone_residue": phone,
        "birthday_residue": birthday,
        "encode_strength": encode_strength,
        "consolidate_strength": consolidate_strength,
    }


def formula_gates() -> dict:
    """Pure-function gates. Live remember / sleep are scored in tests."""
    scalars = schedule_scalars()
    surfaces = canonical_surfaces()
    default_mode = memory_mode(scalars["floor"], scalars["baseline"], 0.0)
    wake_mode = memory_mode(scalars["after_wake"], scalars["baseline"], 0.0)
    aroused_mode = memory_mode(scalars["after_wake"], scalars["arousal_high"], 0.0)
    day_mode = memory_mode(scalars["after_wake"], scalars["baseline"], 1.0)
    identity = (
        default_mode == "encode"
        and write_keep("encode") == float(ENCODE_WRITE_KEEP)
        and surfaces["phone_encode"] == PHONE
        and surfaces["birthday_encode"] == BIRTHDAY
        and apply_sleep_operators("encode") is True
        and apply_sleep_operators("consolidate") is True
    )
    encode = (
        aroused_mode == "encode"
        and day_mode == "encode"
        and write_keep(aroused_mode) == 1.0
        and spontaneous_pressure(aroused_mode, scalars["after_wake"])
        == scalars["after_wake"] * float(ENCODE_SPONTANEOUS_KEEP)
        and spontaneous_pressure(aroused_mode, scalars["after_wake"]) < scalars["spontaneous_min"]
    )
    consolidate = (
        wake_mode == "consolidate"
        and write_keep(wake_mode) == float(CONSOLIDATE_WRITE_KEEP)
        and surfaces["phone_consolidate"] != PHONE
        and surfaces["birthday_consolidate"] != BIRTHDAY
        and spontaneous_pressure(wake_mode, scalars["after_wake"]) == scalars["after_wake"]
        and scalars["after_wake"] >= scalars["spontaneous_min"]
    )
    write = (
        write_keep("encode") > write_keep("consolidate")
        and surfaces["phone_encode"] != surfaces["phone_consolidate"]
        and surfaces["encode_strength"] == 1.0
        and surfaces["consolidate_strength"] == 0.75
    )
    sleep = (
        spontaneous_pressure("encode", scalars["after_wake"]) < scalars["spontaneous_min"]
        and spontaneous_pressure("consolidate", scalars["after_wake"]) >= scalars["spontaneous_min"]
        and spontaneous_pressure("encode", scalars["floor"]) == scalars["floor"] * float(ENCODE_SPONTANEOUS_KEEP)
    )
    compose = (
        apply_sleep_operators(wake_mode) is True
        and apply_sleep_operators(aroused_mode) is True
        and memory_mode(scalars["floor"], scalars["baseline"], -1.0) == "encode"
    )
    return {
        "identity": bool(identity),
        "encode": bool(encode),
        "consolidate": bool(consolidate),
        "write": bool(write),
        "sleep": bool(sleep),
        "compose": bool(compose),
        "scalars": scalars,
        "surfaces": {
            "phone_encode": surfaces["phone_encode"],
            "phone_consolidate": surfaces["phone_consolidate"],
            "birthday_encode": surfaces["birthday_encode"],
            "birthday_consolidate": surfaces["birthday_consolidate"],
        },
        "keeps": {
            "encode_write": float(ENCODE_WRITE_KEEP),
            "consolidate_write": float(CONSOLIDATE_WRITE_KEEP),
            "encode_spontaneous": float(ENCODE_SPONTANEOUS_KEEP),
        },
    }


def bank_sha256() -> str:
    scalars = schedule_scalars()
    surfaces = canonical_surfaces()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            PHONE,
            BIRTHDAY,
            str(scalars["after_wake"]),
            str(surfaces["phone_encode"]),
            str(surfaces["phone_consolidate"]),
            str(surfaces["birthday_encode"]),
            str(surfaces["birthday_consolidate"]),
            str(ENCODE_WRITE_KEEP),
            str(CONSOLIDATE_WRITE_KEEP),
            str(ENCODE_SPONTANEOUS_KEEP),
            *GATES,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
