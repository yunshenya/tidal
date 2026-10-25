"""Process S and offline reconsolidation.

Process S rises during wake and falls during sleep (Daan, Beersma, and
Borbély 1984; Borbély 2016; Borbély 2022). The 1982 Human Neurobiology
paper has no Crossref DOI and is not cited. Circadian process C stays on
the store clock. This module does not compute growbrain's planned
``theta0 + a*S - b*C``.

A reactivated slot is rewritten only when prediction error is above a
threshold (Sevenster, Beckers, and Kindt 2013). The audit row is numbers
and a bool. It has no fact text.

A no-cue offline tick can recall one slot (Berntsen 2020). That paper
says this is not mind-wandering. The period and the other rates below are
a hypothesis: abstract duration units, not the papers' hours, and not a
grid search.
"""
from __future__ import annotations

import math

import numpy as np

# Homeostatic floor. Sleep approaches it and does not assign 0.
# HYPOTHESIS. Not an EEG time constant.
SLEEP_FLOOR = 0.1
# Wake and sleep time constants in abstract duration units. HYPOTHESIS.
TAU_WAKE = 16.0
TAU_SLEEP = 4.0
# 1 - cosine(key, value). Parallel traces are 0 and are not rewritten.
# Orthogonal traces are 1. HYPOTHESIS. Not a propranolol dose.
PE_THRESHOLD = 0.25
# Embedding blend toward the unit key, keeping the value's norm as a scale.
# HYPOTHESIS.
REPLAY_BLEND = 0.1
# Strength multiplier exp(-step * min(error, 1)). HYPOTHESIS.
STRENGTH_STEP = 0.05
# Offline tick period. Tick 0 fires, so 16 ticks fire once. HYPOTHESIS.
SPONTANEOUS_PERIOD = 16
# Process S must be at least this high. Wake of duration 4 from the floor
# clears it at the hypothesis taus. HYPOTHESIS.
SPONTANEOUS_S_MIN = 0.2
# Strict ceiling. The cue false-trigger ceiling is 0.10. A rate of 0.10 fails.
SPONTANEOUS_CEILING = 0.10

AUDIT_FIELDS = ("index", "error", "updated", "strength_before", "strength_after")


def _duration(duration: float) -> float:
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        raise TypeError("duration must be a real number")
    value = float(duration)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("duration must be finite and >= 0")
    return value


def _pressure(pressure: float) -> float:
    if isinstance(pressure, bool) or not isinstance(pressure, (int, float)):
        raise TypeError("sleep pressure must be a real number")
    value = float(pressure)
    if not math.isfinite(value) or value < float(SLEEP_FLOOR) or value > 1.0:
        raise ValueError("sleep pressure must be in [SLEEP_FLOOR, 1]")
    return value


def _clamp(pressure: float) -> float:
    value = float(pressure)
    if value < float(SLEEP_FLOOR):
        return float(SLEEP_FLOOR)
    if value > 1.0:
        return 1.0
    return value


def wake_after(pressure: float, duration: float) -> float:
    """``1 - (1 - S) * exp(-t / tau_wake)``. Zero duration leaves ``S``."""
    current = _pressure(pressure)
    span = _duration(duration)
    if span == 0.0:
        return current
    updated = 1.0 - (1.0 - current) * math.exp(-span / float(TAU_WAKE))
    return _clamp(updated)


def sleep_after(pressure: float, duration: float) -> float:
    """``floor + (S - floor) * exp(-D / tau_sleep)``. Finite ``D`` stays above 0.

    When the exponential underflows, the result is the floor, which is still
    above 0. The function never assigns 0.
    """
    current = _pressure(pressure)
    span = _duration(duration)
    if span == 0.0:
        return current
    updated = float(SLEEP_FLOOR) + (current - float(SLEEP_FLOOR)) * math.exp(-span / float(TAU_SLEEP))
    updated = _clamp(updated)
    if updated <= 0.0:
        raise RuntimeError("sleep pressure collapsed to 0")
    return updated


def prediction_error(key, value) -> float:
    """``1 - cosine(key, value)``, in ``[0, 2]``."""
    left = np.asarray(key, dtype=np.float64).reshape(-1)
    right = np.asarray(value, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("key and value must align")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("key and value must be finite")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm < 1e-12 or right_norm < 1e-12:
        return 1.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    if cosine > 1.0:
        cosine = 1.0
    elif cosine < -1.0:
        cosine = -1.0
    return float(1.0 - cosine)


def reconsolidate(key, value, strength: float) -> dict:
    """Rewrite a slot only when prediction error exceeds the threshold.

    The returned value is a float64 copy. ``text`` is not an argument.
    Strength stays positive. Below threshold, strength and the value copy
    are unchanged.
    """
    if isinstance(strength, bool) or not isinstance(strength, (int, float)):
        raise TypeError("strength must be a real number")
    current = float(strength)
    if not math.isfinite(current) or current < 0.0:
        raise ValueError("strength must be finite and >= 0")
    error = prediction_error(key, value)
    original = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if not (error > float(PE_THRESHOLD)):
        return {
            "value": original,
            "strength": current,
            "error": float(error),
            "updated": False,
        }
    direction = np.asarray(key, dtype=np.float64).reshape(-1)
    key_norm = float(np.linalg.norm(direction))
    value_norm = float(np.linalg.norm(original))
    if key_norm < 1e-12 or value_norm < 1e-12 or direction.shape != original.shape:
        return {
            "value": original,
            "strength": current,
            "error": float(error),
            "updated": False,
        }
    blended = (1.0 - float(REPLAY_BLEND)) * original + float(REPLAY_BLEND) * (direction / key_norm) * value_norm
    updated_strength = current * math.exp(-float(STRENGTH_STEP) * min(float(error), 1.0))
    if updated_strength <= 0.0:
        raise RuntimeError("reconsolidation collapsed strength to 0")
    return {
        "value": blended,
        "strength": float(updated_strength),
        "error": float(error),
        "updated": True,
    }


def audit_row(index: int, error: float, updated: bool, strength_before: float, strength_after: float) -> dict:
    """One reconsolidation record. Numbers and a bool. No fact text."""
    if isinstance(updated, bool) is False:
        raise TypeError("updated must be a bool")
    row = {
        "index": int(index),
        "error": float(error),
        "updated": bool(updated),
        "strength_before": float(strength_before),
        "strength_after": float(strength_after),
    }
    if tuple(row) != AUDIT_FIELDS:
        raise RuntimeError("audit row carries extra fields")
    return row


def spontaneous_due(tick: int, pressure: float, n_slots: int) -> bool:
    """True on a rate-limit tick when the store is nonempty and S is high enough."""
    if isinstance(tick, bool) or not isinstance(tick, int):
        raise TypeError("tick must be an int")
    if isinstance(n_slots, bool) or not isinstance(n_slots, int):
        raise TypeError("n_slots must be an int")
    if tick < 0 or n_slots < 0:
        raise ValueError("tick and n_slots must be >= 0")
    if n_slots < 1:
        return False
    if float(pressure) < float(SPONTANEOUS_S_MIN):
        return False
    return tick % int(SPONTANEOUS_PERIOD) == 0


def spontaneous_winner(salience, strength, priority=None) -> int:
    """Index of the greatest ``(salience, strength, priority, -index)``.

    Missing priority is 1, so the order matches the old triple.
    """
    weights = np.asarray(salience, dtype=np.float64).reshape(-1)
    levels = np.asarray(strength, dtype=np.float64).reshape(-1)
    if weights.shape != levels.shape or weights.size == 0:
        raise ValueError("salience and strength must be nonempty and aligned")
    if not np.isfinite(weights).all() or not np.isfinite(levels).all():
        raise ValueError("salience and strength must be finite")
    if priority is None:
        ranks = np.ones(weights.shape, dtype=np.float64)
    else:
        ranks = np.asarray(priority, dtype=np.float64).reshape(-1)
        if ranks.shape != weights.shape:
            raise ValueError("priority must align with salience and strength")
        if not np.isfinite(ranks).all():
            raise ValueError("priority must be finite")
    best_key = None
    best_index = 0
    for index in range(int(weights.size)):
        key = (float(weights[index]), float(levels[index]), float(ranks[index]), -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    return int(best_index)
