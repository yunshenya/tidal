"""Weak-overlap spontaneous recollection (Berntsen 2020).

Berntsen reviews involuntary autobiographical memory: memories that
come to mind with no conscious initiation of retrieval. That paper
says this cannot be equated with mind-wandering, and that most
involuntary memories still have a cue — often a weak internal or
external overlap with the remembered episode, not a strong
deliberate probe.
https://doi.org/10.1098/rstb.2019.0693

tidal already rate-limits no-external-cue offline ticks and already
lowers that pressure in encode mode. This module is the missing
overlap check. A partial key, a residual, or a recent context
fragment can surface a stored slot when cosine sits in a weak band
below cue theta. A strong cue stays on the cue path
(``source=cue``, ``triggered=True``). A weak-band offline fire stays
``source=spontaneous`` and ``triggered=False``. No new cue-export
field is added. This is not a default-mode-network model.

Formula (HYPOTHESIS; no grid was run)
-------------------------------------
1. Fragment. ``context_fragment(x)`` mixes ``x`` with its
   deterministic complement so cosine(x, fragment) is
   ``WEAK_OVERLAP``. That is the same 0.25 span family as clock /
   McGaugh / pattern separation. Cosine 0.25 is below cue theta
   0.32, so it is not a strong cue hit.

2. Band. Cosine at or above ``WEAK_FLOOR`` and strictly below
   ``WEAK_CEILING`` (cue theta) is weak overlap. Cosine 1 is a
   strong cue. Cosine 0 is no overlap.

3. Winner. Among slots in that band, the greatest
   ``(overlap, salience, strength, priority, -index)`` surfaces.
   Missing salience / strength / priority is 1. Nobody in the band
   returns None, and the existing salience offline ranking stays.

Live hook
---------
:meth:`tidal.llm_memory.LLMMemory.remember` plants a partial-key
fragment of the written key. A cue trace plants the last cue row as
a recent context fragment. ``offline_ticks`` still needs the period
and the mode-scaled process-S floor. Encode mode still multiplies
that pressure by 0.5. The period, the ceiling, and the sleep
operators stay.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from tidal.episodic import CLOCK_SPAN
from tidal.pattern_separation import NORM_ZERO, complement, key_cosine

ArrayLike = Union[np.ndarray, Sequence[float]]

# Below this is no overlap. Same number as the offline-rate ceiling.
# HYPOTHESIS.
WEAK_FLOOR = 0.10
# Target cosine of a degraded fragment. Same 0.25 span family.
# HYPOTHESIS. Below cue theta 0.32.
WEAK_OVERLAP = float(CLOCK_SPAN)
# Exclusive strong-cue ceiling. Matches live CUE_THETA. HYPOTHESIS.
WEAK_CEILING = 0.32


def _finite_span(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _as_vec(value: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError("empty vector")
    if not np.isfinite(arr).all():
        raise ValueError("non-finite vector")
    return arr


def _norm(value: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(value))


def _aligned(values, n: int, name: str) -> NDArray[np.float64]:
    if values is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must align with overlaps")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def overlap_score(fragment: ArrayLike, key: ArrayLike) -> float:
    """Cosine in ``[-1, 1]``. A missing vector returns 0."""
    return float(key_cosine(fragment, key))


def in_weak_band(
    sim: float,
    *,
    floor: float = WEAK_FLOOR,
    ceiling: float = WEAK_CEILING,
) -> bool:
    """True when cosine is weak overlap, not a miss and not a strong cue."""
    cosine = _finite_span(sim, "cosine")
    low = _finite_span(floor, "floor")
    high = _finite_span(ceiling, "ceiling")
    if high <= low:
        raise ValueError("ceiling must be > floor")
    return bool(low <= cosine < high)


def context_fragment(vec: ArrayLike, overlap: float = WEAK_OVERLAP) -> NDArray[np.float64]:
    """Partial key / residual / recent-context fragment at a target cosine.

    Mixes ``vec`` with its deterministic complement and restores the
    incoming L2 norm. Cosine with the original is ``overlap`` when both
    are nonzero. A zero vector is a copy. Overlap 1 is a copy. Overlap
    0 is the complement at the same norm.
    """
    raw = _as_vec(vec)
    target = _finite_span(overlap, "overlap")
    if target < -1.0 or target > 1.0:
        raise ValueError("overlap must be in [-1, 1]")
    scale = _norm(raw)
    if scale <= float(NORM_ZERO):
        return raw.copy()
    unit = raw / scale
    if abs(target) >= 1.0:
        return raw.copy() if target > 0.0 else (-raw).copy()
    lateral = complement(unit)
    mix = target * unit + math.sqrt(max(0.0, 1.0 - target * target)) * lateral
    mix_norm = _norm(mix)
    if mix_norm <= float(NORM_ZERO):
        return raw.copy()
    return mix * (scale / mix_norm)


def weak_winner(
    overlaps,
    salience=None,
    strength=None,
    priority=None,
    *,
    floor: float = WEAK_FLOOR,
    ceiling: float = WEAK_CEILING,
) -> Optional[int]:
    """Index of the greatest weak-band ``(overlap, salience, strength, priority, -index)``.

    Missing ranks are 1. ``None`` when nobody sits in the weak band.
    """
    scores = np.asarray(overlaps, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("overlaps must be nonempty")
    if not np.isfinite(scores).all():
        raise ValueError("overlaps must be finite")
    n = int(scores.size)
    weights = _aligned(salience, n, "salience")
    levels = _aligned(strength, n, "strength")
    ranks = _aligned(priority, n, "priority")
    best_key = None
    best_index = None
    for index in range(n):
        sim = float(scores[index])
        if not in_weak_band(sim, floor=floor, ceiling=ceiling):
            continue
        key = (sim, float(weights[index]), float(levels[index]), float(ranks[index]), -index)
        if best_key is None or key > best_key:
            best_key = key
            best_index = index
    if best_index is None:
        return None
    return int(best_index)
