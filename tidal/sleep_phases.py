"""SWS-like gist and REM-like integrative operators on existing sleep.

Diekelmann and Born (2010) give SWS a systems-consolidation role
(replay, redistribute, qualitative change) and REM a complementary
stabilizing / integrative role. Klinzing, Niethard, and Born (2019)
name the qualitative product: abstracted, gist-like representations.
https://doi.org/10.1038/nrn2762
https://doi.org/10.1038/s41593-019-0467-3

tidal already owns process S, a circadian phase, prediction-error
replay, reconstructive residue, valence, arousal, and McGaugh
``consolidation_gain``. This module schedules two offline passes from
those existing scalars. It does not simulate slow oscillations,
spindles, or ripples. It does not add a cue-export field.
``replay_phase`` is internal and is not written onto the cue wire.

Phase (HYPOTHESIS; no grid was run)
-----------------------------------
High process S (the same 0.2 floor that admits spontaneous recall) is
SWS: early-night homeostatic pressure. After S has dropped below that
floor, or when circadian phase is nonzero at low S, the pass is REM.
Sleep from the process-S floor with a zero circadian phase is ``none``:
prediction-error replay only.

SWS pass
--------
Increment the slot's consolidation ``age`` by one. Strength, keys,
values, residue parts, ``text``, and ``answer`` stay. Reconstructive
spelling already maps age 1 onto a peripheral slip and a kept gist.
That is the product. It is not a second residue format.

REM pass
--------
Refresh a nonzero emotion tag by ``REM_VALENCE_KEEP``, then recompute
``consolidation_gain`` with the existing McGaugh formula. Weakly bind
same-group value vectors (not keys) by ``REM_BIND``. Different groups
do not bind. This is not a second memory system.
"""
from __future__ import annotations

import math
from typing import Literal, Never, Optional

import numpy as np

from tidal.reconstruct import birthday_parts, phone_parts, recall_surface
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_S_MIN

Phase = Literal["sws", "rem", "none"]
PHASES = ("sws", "rem", "none")

# Emotion-tag keep on one REM pass. HYPOTHESIS. Not a depotentiation dose.
REM_VALENCE_KEEP = 0.9
# Same-group value blend. Half of sleep.REPLAY_BLEND. HYPOTHESIS.
REM_BIND = 0.05
# Treat a vector at or below this L2 norm as missing.
NORM_ZERO = 1e-12


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _pressure(pressure: float) -> float:
    value = _finite(pressure, "sleep pressure")
    if value < float(SLEEP_FLOOR) or value > 1.0:
        raise ValueError("sleep pressure must be in [SLEEP_FLOOR, 1]")
    return value


def _unknown_phase(phase: Never) -> Never:
    raise RuntimeError(f"unhandled sleep phase: {phase}")


def sleep_phase(pressure: float, circadian_phase: float) -> Phase:
    """Internal scheduler. Not a cue field.

    High process S is SWS. Low S with a leftover drop, or a nonzero
    circadian phase at low S, is REM. The process-S floor plus a zero
    circadian phase is ``none``.
    """
    current = _pressure(pressure)
    clock = _finite(circadian_phase, "circadian phase")
    if current >= float(SPONTANEOUS_S_MIN):
        return "sws"
    if current > float(SLEEP_FLOOR) or clock != 0.0:
        return "rem"
    return "none"


def apply_entry_phase(phase: str) -> bool:
    """True when the sleep-onset phase should run the SWS gist pass."""
    if phase == "sws":
        return True
    if phase == "rem":
        return False
    if phase == "none":
        return False
    _unknown_phase(phase)


def apply_exit_phase(phase: str) -> bool:
    """True when the post-sleep phase should run the REM pass."""
    if phase == "rem":
        return True
    if phase == "sws":
        return False
    if phase == "none":
        return False
    _unknown_phase(phase)


def sws_age(age: int) -> int:
    """One systems-consolidation cycle. Strength is not this counter."""
    if isinstance(age, bool) or not isinstance(age, int):
        raise TypeError("age must be an int")
    if age < 0:
        raise ValueError("age must be >= 0")
    return int(age) + 1


def rem_refresh_valence(valence: float, keep: float = REM_VALENCE_KEEP) -> float:
    """Scale a nonzero tag. Zero stays exactly 0. Sign is unchanged."""
    feeling = _finite(valence, "valence")
    factor = _finite(keep, "keep")
    if factor <= 0.0 or factor > 1.0:
        raise ValueError("keep must be in (0, 1]")
    if feeling == 0.0:
        return 0.0
    return float(feeling * factor)


def rem_bind_pair(left, right, weight: float = REM_BIND):
    """Symmetric value blend. Restores each vector's L2 norm.

    The update is simultaneous. A missing vector is copied through.
    Keys are not an argument.
    """
    blend = _finite(weight, "weight")
    if blend < 0.0 or blend > 0.25:
        raise ValueError("weight must be in [0, 0.25]")
    first = np.asarray(left, dtype=np.float64).reshape(-1).copy()
    second = np.asarray(right, dtype=np.float64).reshape(-1).copy()
    if first.shape != second.shape or first.size == 0:
        raise ValueError("values must align")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("values must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= float(NORM_ZERO) or second_norm <= float(NORM_ZERO) or blend == 0.0:
        return first, second
    mixed_first = (1.0 - blend) * first + blend * (second / second_norm) * first_norm
    mixed_second = (1.0 - blend) * second + blend * (first / first_norm) * second_norm
    mixed_first_norm = float(np.linalg.norm(mixed_first))
    mixed_second_norm = float(np.linalg.norm(mixed_second))
    if mixed_first_norm <= float(NORM_ZERO) or mixed_second_norm <= float(NORM_ZERO):
        return first, second
    return mixed_first * (first_norm / mixed_first_norm), mixed_second * (second_norm / mixed_second_norm)


def gist_holds(residue, surface: Optional[str]) -> bool:
    """True when country/area/exchange or month/year still match the residue."""
    if not isinstance(residue, tuple) or not residue or not isinstance(surface, str):
        return False
    kind = residue[0]
    if kind == "phone":
        parts = phone_parts(surface)
        if parts is None or len(residue) != 4:
            return False
        return int(parts[0]) == int(residue[1]) and int(parts[1]) == int(residue[2])
    if kind == "birthday":
        parts = birthday_parts(surface)
        if parts is None or len(residue) != 4:
            return False
        return int(parts[0]) == int(residue[1]) and int(parts[2]) == int(residue[3])
    return False


def peripheral_changed(residue, surface: Optional[str]) -> bool:
    """True when the line or the day differs from the residue."""
    if not isinstance(residue, tuple) or not residue or not isinstance(surface, str):
        return False
    kind = residue[0]
    if kind == "phone":
        parts = phone_parts(surface)
        if parts is None or len(residue) != 4:
            return False
        return int(parts[2]) != int(residue[3])
    if kind == "birthday":
        parts = birthday_parts(surface)
        if parts is None or len(residue) != 4:
            return False
        return int(parts[1]) != int(residue[2])
    return False


def gist_stronger_peripheral_weaker(residue, strength: float, age: int, valence: float) -> bool:
    """Reconstructed surface keeps gist and changes the peripheral field."""
    surface = recall_surface(residue, strength, age, valence)
    return bool(gist_holds(residue, surface) and peripheral_changed(residue, surface))
