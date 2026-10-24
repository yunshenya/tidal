"""LC-NE arousal in ``[0, 1]``.

Aston-Jones and Cohen (2005) describe locus coeruleus–norepinephrine
activity as an arousal signal with a phasic mode and a tonic mode.
https://doi.org/10.1146/annurev.neuro.28.061604.135709
Phasic activation follows the outcome of a task decision and is proposed
to facilitate the behavior that follows. When utility falls, tonic activity
is associated with disengagement from the current task. This module does
not choose exploitation or exploration, and it does not choose to speak or
to wait. Those decisions belong to the brain layer.

The scalar rises on outcome-locked signals tidal already computes, then
decays toward a baseline on quiet tokens. That decay is not the paper's
tonic rise during disengagement. The weights, the baseline, the time
constant, and the window are a hypothesis. No grid was run.
"""
from __future__ import annotations

import math

# Resting level in ``[0, 1]``. Quiet tokens move toward it. HYPOTHESIS.
AROUSAL_BASELINE = 0.2
# Quiet-token time constant, in tokens. HYPOTHESIS. Not an EEG constant.
AROUSAL_TAU = 8.0
# Phasic weight on surprise above the recent mean. HYPOTHESIS.
W_SURPRISE = 0.35
# Weight on ``|valence|``. HYPOTHESIS. Utility magnitude, not a decision.
W_VALENCE = 0.25
# Weight on a positive trigger count. HYPOTHESIS.
W_TRIGGER = 0.20
# Prior surprises kept for the phasic comparison. HYPOTHESIS.
# This is not the write-gate window.
AROUSAL_WINDOW = 8


def _level(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise TypeError("arousal must be a real number")
    value = float(level)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("arousal must be in [0, 1]")
    return value


def _clip(level: float) -> float:
    value = float(level)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def quiet_step(level: float) -> float:
    """One quiet token: ``baseline + (level - baseline) * exp(-1 / tau)``.

    At the baseline the value stays. Above the baseline it decreases and
    stays strictly above the baseline.
    """
    current = _level(level)
    factor = math.exp(-1.0 / float(AROUSAL_TAU))
    return _clip(float(AROUSAL_BASELINE) + (current - float(AROUSAL_BASELINE)) * factor)


def after_surprise(level: float, surprise: float, prior) -> float:
    """Phasic bump when ``surprise`` is strictly above the prior mean.

    The prior needs at least two finite samples. A shorter prior, or a
    surprise that is not strictly above that mean, leaves the level. The
    excess is ``(surprise - mean) / (|mean| + 1)``. The bump is
    ``W_SURPRISE * tanh(excess)``.
    """
    current = _level(level)
    if isinstance(surprise, bool) or not isinstance(surprise, (int, float)):
        raise TypeError("surprise must be a real number")
    value = float(surprise)
    if not math.isfinite(value):
        raise ValueError("surprise must be finite")
    samples = []
    for item in prior:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError("surprise prior must be real numbers")
        sample = float(item)
        if not math.isfinite(sample):
            raise ValueError("surprise prior must be finite")
        samples.append(sample)
    if len(samples) < 2:
        return current
    mean = sum(samples) / float(len(samples))
    if not value > mean:
        return current
    excess = (value - mean) / (abs(mean) + 1.0)
    return _clip(current + float(W_SURPRISE) * math.tanh(excess))


def after_valence(level: float, valence: float) -> float:
    """``W_VALENCE * tanh(|valence|)``. Zero valence adds nothing."""
    current = _level(level)
    if isinstance(valence, bool) or not isinstance(valence, (int, float)):
        raise TypeError("valence must be a real number")
    magnitude = abs(float(valence))
    if not math.isfinite(magnitude):
        raise ValueError("valence must be finite")
    if magnitude == 0.0:
        return current
    return _clip(current + float(W_VALENCE) * math.tanh(magnitude))


def after_triggers(level: float, count: int) -> float:
    """``W_TRIGGER * tanh(count)`` for a positive count.

    A count of 0 is a no-op. It is not a quiet step.
    """
    current = _level(level)
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("trigger count must be an int")
    if count < 0:
        raise ValueError("trigger count must be >= 0")
    if count == 0:
        return current
    return _clip(current + float(W_TRIGGER) * math.tanh(float(count)))


def apply_trace(level: float, flags) -> list:
    """Step one token at a time. True is one trigger. False is one quiet token.

    Returns the level after each token. Does not mutate the caller.
    """
    current = _level(level)
    series = []
    for flag in flags:
        if bool(flag):
            current = after_triggers(current, 1)
        else:
            current = quiet_step(current)
        series.append(current)
    return series
