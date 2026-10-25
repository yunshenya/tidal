"""Encode vs consolidate mode from live gate scalars.

Hasselmo and McGaughy (2004) argue that high acetylcholine sets
cortical dynamics for attention and encoding (afferent input over
internal feedback) and that low acetylcholine sets dynamics for
consolidation (recurrent / endogenous spread).
https://doi.org/10.1016/S0079-6123(03)45015-2

Gerstner and Yin (2010) review time-of-day effects on
neurophysiology and memory formation. Circadian phase is that
kind of lean, not a second clock gene model.
https://doi.org/10.1038/nrn2881

tidal already owns process S, an LC-NE arousal scalar, and a
store circadian phase. This module derives one internal mode from
those existing numbers. It does not add a cue-export field. It
does not reimplement ``sleep_for`` SWS/REM; those stay on
:func:`tidal.sleep_phases.sleep_phase`.

Rule (HYPOTHESIS; no grid was run)
----------------------------------
Above-baseline arousal is ACh-like afferent drive and is encode.
A positive circadian phase is a time-of-day encode lean. Otherwise
process S at or above the spontaneous / SWS floor is
consolidate-favoring (quiet wake / sleep pressure). The process-S
floor plus baseline arousal plus a nonpositive phase is encode.

Encode mode keeps a new ``remember`` write at full clock strength
and lowers the pressure that admits spontaneous offline recall.
Consolidate mode scales that write by ``CONSOLIDATE_WRITE_KEEP``
and leaves spontaneous pressure unchanged so the existing sleep
operators can run.
"""
from __future__ import annotations

import math
from typing import Literal, Never

from tidal.arousal import AROUSAL_BASELINE
from tidal.episodic import CLOCK_SPAN
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_S_MIN

Mode = Literal["encode", "consolidate"]
MODES = ("encode", "consolidate")

# Full intake. HYPOTHESIS.
ENCODE_WRITE_KEEP = 1.0
# Weaker encoding intake. Same 0.25 span family as clock / McGaugh.
# HYPOTHESIS. Strength 0.75 is below reconstruct EXACT_STRENGTH 0.8.
CONSOLIDATE_WRITE_KEEP = 1.0 - float(CLOCK_SPAN)
# Encode multiplies process S before the spontaneous floor check.
# After wake duration 4, 0.5 * S stays below 0.2. HYPOTHESIS.
ENCODE_SPONTANEOUS_KEEP = 0.5


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


def _arousal(level: float) -> float:
    value = _finite(level, "arousal")
    if value < 0.0 or value > 1.0:
        raise ValueError("arousal must be in [0, 1]")
    return value


def _unknown_mode(mode: Never) -> Never:
    raise RuntimeError(f"unhandled memory mode: {mode}")


def memory_mode(pressure: float, arousal: float, circadian_phase: float) -> Mode:
    """Internal encode vs consolidate gate. Not a cue field.

    High afferent / ACh-like arousal, or a positive circadian phase, is
    encode. Else high process S is consolidate. The process-S floor
    with baseline arousal and a nonpositive phase is encode.
    """
    current = _pressure(pressure)
    level = _arousal(arousal)
    clock = _finite(circadian_phase, "circadian phase")
    if level > float(AROUSAL_BASELINE):
        return "encode"
    if clock > 0.0:
        return "encode"
    if current >= float(SPONTANEOUS_S_MIN):
        return "consolidate"
    return "encode"


def write_keep(mode: str) -> float:
    """Clock-strength multiplier on a new ``remember`` write."""
    if mode == "encode":
        return float(ENCODE_WRITE_KEEP)
    if mode == "consolidate":
        return float(CONSOLIDATE_WRITE_KEEP)
    _unknown_mode(mode)


def spontaneous_pressure(mode: str, pressure: float) -> float:
    """Process S seen by the spontaneous floor. Sleep ``S`` itself stays."""
    current = _pressure(pressure)
    if mode == "encode":
        return float(current * float(ENCODE_SPONTANEOUS_KEEP))
    if mode == "consolidate":
        return current
    _unknown_mode(mode)


def apply_sleep_operators(mode: str) -> bool:
    """SWS/REM stay on ``sleep_phase``. Mode does not block them."""
    if mode == "encode":
        return True
    if mode == "consolidate":
        return True
    _unknown_mode(mode)
