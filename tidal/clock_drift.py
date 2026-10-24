"""Token decay and wall-clock decay, plus a drifting internal clock.

The token factor is ``exp(-dt / tau)`` with the shipped token tau. Joordens
and Becker (1997) studied short and long semantic priming in lexical
decision. The token half-life stays 30 tokens. Their unit is not seconds.
This module does not change that tau.
https://doi.org/10.1037/0278-7393.23.5.1083

Wall time is counted in seconds. The tick is 1 second. Buhusi and Meck
(2005) review functional and neural mechanisms of interval timing.
https://doi.org/10.1038/nrn1764 Allman, Teki, Griffiths, and Meck (2014)
review properties of the internal clock and of subjective time.
https://doi.org/10.1146/annurev-psych-010213-115117 Crossref recorded no
abstract for the 2005 or 2014 paper on 2026-10-24, so this module does not
restate a result from them. The seconds half-life, the drift rate, and the
entrainment gain below are a hypothesis. They are not numbers from those
papers. No grid was run.

At an update the remaining factor is the lower of the token factor and the
wall factor. That is the stronger decay. It is not the product, so one
clock is not applied twice. ``dt`` counts tokens already seen after the
trigger. Elapsed wall time before the trigger is 0, so a stamp behind the
trigger cannot raise activation. A new winner resets both clocks.
Continuing the same winner never raises activation.

An utterance trace holds the wall reading fixed. A tick that has not been
applied is invisible inside that forward. Wall time moves only through an
explicit tick. This module does not read the operating-system clock.

The drifted internal clock is not an input to the decay. Decay reads wall
seconds since the trigger. Calibration moves only the internal estimate.
"""
from __future__ import annotations

import math

# One tick, in seconds. HYPOTHESIS. The agreed growbrain tick.
WALL_TICK = 1.0
# Wall half-life, in seconds. HYPOTHESIS. Not the 30-token half-life.
WALL_HALF_LIFE = 60.0
WALL_TAU = WALL_HALF_LIFE / math.log(2.0)
# Internal seconds advanced per wall second. Above 1, the clock runs fast.
# HYPOTHESIS. Not a dopamine dose.
DRIFT_RATE = 1.25
# Pull toward a provided timestamp. One exposure leaves half the error.
# HYPOTHESIS. Not a circadian phase shift.
ENTRAIN_GAIN = 0.5


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def elapsed_since(trigger_wall: float, now) -> float:
    """Wall seconds since the trigger. Missing or earlier ``now`` is 0."""
    if now is None:
        return 0.0
    current = _real(now, "wall time")
    start = _real(trigger_wall, "trigger wall time")
    span = current - start
    if span < 0.0:
        return 0.0
    return span


def clock_activation(peak: float, dt: int, elapsed: float, tau: float, wall_tau: float) -> float:
    """Peak times the lower decay factor.

    ``elapsed <= 0`` returns ``peak * exp(-dt / tau)`` and does not take a
    minimum, so a frozen wall clock leaves the token formula unchanged.
    """
    level = _real(peak, "peak")
    if level < 0.0 or level > 1.0:
        raise ValueError("peak must be in [0, 1]")
    if isinstance(dt, bool) or not isinstance(dt, int):
        raise TypeError("dt must be an int")
    if dt < 0:
        raise ValueError("dt must be >= 0")
    token_tau = _real(tau, "token tau")
    if token_tau <= 0.0:
        raise ValueError("token tau must be positive")
    span = _real(elapsed, "elapsed")
    if span < 0.0:
        span = 0.0
    token_factor = math.exp(-float(dt) / token_tau)
    if span <= 0.0:
        return level * token_factor
    slow = _real(wall_tau, "wall tau")
    if slow <= 0.0:
        raise ValueError("wall tau must be positive")
    wall_factor = math.exp(-span / slow)
    factor = token_factor if token_factor <= wall_factor else wall_factor
    return level * factor


def advance_clock(wall: float, internal: float, seconds: float, rate: float = DRIFT_RATE) -> tuple:
    """Advance wall by ``seconds`` and the internal clock by ``rate * seconds``."""
    current_wall = _real(wall, "wall time")
    current_internal = _real(internal, "internal time")
    span = _real(seconds, "seconds")
    if span < 0.0:
        raise ValueError("seconds must be >= 0")
    speed = _real(rate, "drift rate")
    if speed <= 0.0:
        raise ValueError("drift rate must be positive")
    return current_wall + span, current_internal + speed * span


def entrain(internal: float, timestamp: float, gain: float = ENTRAIN_GAIN) -> float:
    """Move the internal clock toward ``timestamp`` by ``gain`` of the error."""
    current = _real(internal, "internal time")
    stamp = _real(timestamp, "timestamp")
    step = _real(gain, "entrainment gain")
    if step <= 0.0 or step > 1.0:
        raise ValueError("entrainment gain must be in (0, 1]")
    return current + step * (stamp - current)
