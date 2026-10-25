"""Fit and held ids for sleep two-phase consolidation. Numeric schedules.

No torch. Every id runs the same pre-registered fixtures, so the
canonical gist and REM numbers do not depend on the id.
"""
from __future__ import annotations

import hashlib

import numpy as np

from tidal.consolidation import consolidation_gain
from tidal.reconstruct import recall_surface, residue_from_answer
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_S_MIN, sleep_after, wake_after
from tidal.sleep_phases import (
    REM_BIND,
    REM_VALENCE_KEEP,
    gist_stronger_peripheral_weaker,
    rem_bind_pair,
    rem_refresh_valence,
    sleep_phase,
    sws_age,
)
from tidal.subconscious_reconstruct_data import BIRTHDAY, PHONE

PROTOCOL = "sleep-two-phase-v0"
FIT_N = 8
HELD_N = 50
VALENCE_HIGH = 1.5
AROUSAL_HIGH = 1.0
WAKE_DURATION = 4.0
SWS_SLEEP = 1.0
REM_SLEEP = 3.0
GATES = (
    "identity",
    "sws",
    "rem_emotion",
    "rem_bind",
    "compose",
    "export",
    "replay",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("sleep-phase-fit", FIT_N)
HELD_IDS = _ids("sleep-phase-held", HELD_N)


def _axis(index: int, dim: int = 4) -> np.ndarray:
    row = np.zeros(int(dim), dtype=np.float64)
    row[int(index)] = 1.0
    return row


def schedule_pressures() -> dict:
    """Locked process-S points from the sleep hypothesis taus."""
    after_wake = wake_after(float(SLEEP_FLOOR), float(WAKE_DURATION))
    after_sws = sleep_after(after_wake, float(SWS_SLEEP))
    after_rem = sleep_after(after_wake, float(REM_SLEEP))
    return {
        "floor": float(SLEEP_FLOOR),
        "after_wake": float(after_wake),
        "after_sws": float(after_sws),
        "after_rem": float(after_rem),
        "spontaneous_min": float(SPONTANEOUS_S_MIN),
    }


def canonical_surfaces() -> dict:
    """SWS age-1 slip vs the no-sleep exact spelling."""
    phone = residue_from_answer(PHONE)
    birthday = residue_from_answer(BIRTHDAY)
    return {
        "phone_exact": recall_surface(phone, 1.0, 0, 0.0),
        "phone_sws": recall_surface(phone, 1.0, sws_age(0), 0.0),
        "birthday_exact": recall_surface(birthday, 1.0, 0, 0.0),
        "birthday_sws": recall_surface(birthday, 1.0, sws_age(0), 0.0),
        "phone_residue": phone,
        "birthday_residue": birthday,
    }


def canonical_rem() -> dict:
    """Locked emotion-tag and same-group bind numbers."""
    left, right = rem_bind_pair(_axis(0), _axis(1), float(REM_BIND))
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    refreshed = rem_refresh_valence(float(VALENCE_HIGH))
    return {
        "valence_before": float(VALENCE_HIGH),
        "valence_after": float(refreshed),
        "valence_neg_after": rem_refresh_valence(-float(VALENCE_HIGH)),
        "gain_before": consolidation_gain(float(AROUSAL_HIGH), float(VALENCE_HIGH)),
        "gain_after": consolidation_gain(float(AROUSAL_HIGH), float(refreshed)),
        "bind_cosine": cosine,
        "keep": float(REM_VALENCE_KEEP),
        "bind": float(REM_BIND),
    }


def formula_gates() -> dict:
    """Pure-function gates. Live sleep / export are scored in tests."""
    pressures = schedule_pressures()
    surfaces = canonical_surfaces()
    rem = canonical_rem()
    identity = (
        sleep_phase(pressures["floor"], 0.0) == "none"
        and rem_refresh_valence(0.0) == 0.0
        and sws_age(0) == 1
        and surfaces["phone_exact"] == PHONE
        and surfaces["birthday_exact"] == BIRTHDAY
    )
    sws = (
        sleep_phase(pressures["after_wake"], 0.0) == "sws"
        and sleep_phase(pressures["after_sws"], 0.0) == "sws"
        and surfaces["phone_sws"] != PHONE
        and surfaces["birthday_sws"] != BIRTHDAY
        and gist_stronger_peripheral_weaker(surfaces["phone_residue"], 1.0, 1, 0.0)
        and gist_stronger_peripheral_weaker(surfaces["birthday_residue"], 1.0, 1, 0.0)
        and not gist_stronger_peripheral_weaker(surfaces["phone_residue"], 1.0, 0, 0.0)
    )
    rem_emotion = (
        rem["valence_after"] == rem["valence_before"] * rem["keep"]
        and rem["valence_neg_after"] == -rem["valence_after"]
        and rem["gain_after"] < rem["gain_before"]
        and rem["gain_after"] == consolidation_gain(float(AROUSAL_HIGH), rem["valence_after"])
    )
    rem_bind = rem["bind_cosine"] > 0.0 and rem["bind"] == 0.05
    compose = (
        sleep_phase(pressures["after_rem"], 0.0) == "rem"
        and sleep_phase(pressures["floor"], 1.0) == "rem"
        and consolidation_gain(0.2, float(VALENCE_HIGH)) == 1.0
    )
    return {
        "identity": bool(identity),
        "sws": bool(sws),
        "rem_emotion": bool(rem_emotion),
        "rem_bind": bool(rem_bind),
        "compose": bool(compose),
        "pressures": pressures,
        "surfaces": {
            "phone_exact": surfaces["phone_exact"],
            "phone_sws": surfaces["phone_sws"],
            "birthday_exact": surfaces["birthday_exact"],
            "birthday_sws": surfaces["birthday_sws"],
        },
        "rem": rem,
    }


def bank_sha256() -> str:
    surfaces = canonical_surfaces()
    rem = canonical_rem()
    pressures = schedule_pressures()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            PHONE,
            BIRTHDAY,
            str(pressures["after_wake"]),
            str(pressures["after_sws"]),
            str(pressures["after_rem"]),
            str(surfaces["phone_sws"]),
            str(surfaces["birthday_sws"]),
            str(rem["valence_after"]),
            str(rem["bind_cosine"]),
            str(REM_VALENCE_KEEP),
            str(REM_BIND),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
