"""Lightweight dentate-gyrus-like pattern separation (v0).

Yassa and Stark (2011) review evidence that the dentate gyrus transforms
similar experiences into more distinct representations (pattern
separation), complementary to CA3 pattern completion. tidal's live cue
path is already CA3-like: one winner from raw key cosine. As the store
grows, near-neighbor keys collide. This module orthogonalizes those
neighbors on the existing key geometry. It is not a dentate gyrus.
https://doi.org/10.1016/j.tins.2011.06.006
Wixted (2004) reviews interference as a cause of forgetting. Reducing
near-neighbor overlap is that kind of anti-interference, not a
reconsolidation window and not a sleep gist phase.
https://doi.org/10.1146/annurev.psych.55.090902.141555

The residual operates on the stored key, which already lives at
``d_model`` (stage0: 192). The same function accepts ``mem_key_dim`` /
fingerprint width 32. It does not write ``fingerprint``,
``fusion_confidence``, or ``consolidation_gain``. growbrain still
produces fingerprints. ``cue_fields`` keeps the v7 packet.

Formula (HYPOTHESIS; no grid was run)
-------------------------------------
1. Collision. Cosine at or above ``SEPARATION_TAU`` is a near-neighbor.
   Below that, the incoming vector is returned unchanged (identity).

2. Weight. ``w = SPAN * (sim - TAU) / (1 - TAU)``. Cosine 1 is ``SPAN``.
   Cosine at TAU is 0.

3. Unique part. Subtract the neighbor projection. If that residual is
   ~0 (parallel traces), take a deterministic complementary axis so
   exact duplicates can still move apart when only one side is pushed.
   When both sides of an equal pair are re-encoded against each other,
   they take the same complement and stay equal.

4. Push. ``y = x + sum w * unit(unique) * ||x||``. Restore ``||x||``.
   Salience and inject gain keep their scale. Empty neighbors, a
   zero vector, or every neighbor below TAU is a copy of ``x``.

Live hook
---------
:func:`separate_key` is the encode transform. Live ``remember`` stores
the raw key, so coexist geometry, RIF similarity, and the equal-key
tie stay. :meth:`tidal.llm_memory.LLMMemory._trace_numpy` applies
:func:`separate_keys` as the cue-ranking view. Theta, fan, and the
candidate set still read raw key cosine. Ranking and the exported gap
read the separated cosine. Residual inject still uses the stored key.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

from tidal.episodic import cosine_scores

ArrayLike = Union[np.ndarray, Sequence[float]]

# Cosine at or above this is a near-neighbor collision. HYPOTHESIS.
SEPARATION_TAU = 0.90
# Push span. Phase-0 clock and McGaugh gain use the same 0.25. HYPOTHESIS.
SEPARATION_SPAN = 0.25
# Treat a vector at or below this L2 norm as missing. Same floor as fusion.
NORM_ZERO = 1e-12


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


def key_cosine(left: ArrayLike, right: ArrayLike) -> float:
    """Cosine in ``[-1, 1]``. A missing vector returns 0."""
    a = _as_vec(left)
    b = _as_vec(right)
    na = _norm(a)
    nb = _norm(b)
    if na <= float(NORM_ZERO) or nb <= float(NORM_ZERO):
        return 0.0
    if np.array_equal(a, b):
        return 1.0
    if np.array_equal(a, -b):
        return -1.0
    sim = float(np.dot(a, b) / (na * nb))
    if sim > 1.0:
        return 1.0
    if sim < -1.0:
        return -1.0
    return sim


def overlap_weight(sim: float, *, tau: float = SEPARATION_TAU, span: float = SEPARATION_SPAN) -> float:
    """0 below ``tau``. ``span`` at cosine 1."""
    cosine = _finite_span(sim, "cosine")
    threshold = _finite_span(tau, "tau")
    width = _finite_span(span, "span")
    if width < 0.0:
        raise ValueError("span must be >= 0")
    if threshold >= 1.0:
        raise ValueError("tau must be < 1")
    if cosine < threshold:
        return 0.0
    return width * (cosine - threshold) / (1.0 - threshold)


def complement(vec: ArrayLike) -> NDArray[np.float64]:
    """Deterministic unit vector orthogonal to ``vec``.

    Tries standard-basis axes in increasing ``|unit_i|`` order. A 1-d
    vector has no lateral room and returns its own unit direction.
    """
    raw = _as_vec(vec)
    scale = _norm(raw)
    if scale <= float(NORM_ZERO):
        out = np.zeros_like(raw)
        out[0] = 1.0
        return out
    unit = raw / scale
    order = np.argsort(np.abs(unit))
    for axis in order.tolist():
        probe = np.zeros_like(unit)
        probe[int(axis)] = 1.0
        residual = probe - float(np.dot(probe, unit)) * unit
        residual_norm = _norm(residual)
        if residual_norm > float(NORM_ZERO):
            return residual / residual_norm
    return unit.copy()


def separate_key(
    incoming: ArrayLike,
    neighbors: Sequence[ArrayLike] = (),
    *,
    tau: float = SEPARATION_TAU,
    span: float = SEPARATION_SPAN,
) -> NDArray[np.float64]:
    """Contrastive residual of ``incoming`` against ``neighbors``.

    Identity when the neighbor list is empty, the incoming vector is
    ~0, or every neighbor cosine is below ``tau``. Restores the
    incoming L2 norm.
    """
    incoming_vec = _as_vec(incoming)
    scale = _norm(incoming_vec)
    if scale <= float(NORM_ZERO):
        return incoming_vec.copy()
    push = np.zeros_like(incoming_vec)
    for neighbor in neighbors:
        other = _as_vec(neighbor)
        weight = overlap_weight(key_cosine(incoming_vec, other), tau=tau, span=span)
        if weight == 0.0:
            continue
        other_norm = _norm(other)
        if other_norm <= float(NORM_ZERO):
            continue
        unit = other / other_norm
        unique = incoming_vec - float(np.dot(incoming_vec, unit)) * unit
        unique_norm = _norm(unique)
        if unique_norm <= float(NORM_ZERO):
            direction = complement(unit)
        else:
            direction = unique / unique_norm
        push = push + weight * direction * scale
    out = incoming_vec + push
    out_norm = _norm(out)
    if out_norm <= float(NORM_ZERO):
        return incoming_vec.copy()
    return out * (scale / out_norm)


def separate_keys(
    keys: ArrayLike,
    *,
    tau: float = SEPARATION_TAU,
    span: float = SEPARATION_SPAN,
) -> NDArray[np.float64]:
    """Each row separated against the others. One row, or none, is a copy."""
    bank = np.asarray(keys, dtype=np.float64)
    if bank.ndim != 2:
        raise ValueError("keys must be [slots, dim]")
    if bank.shape[0] <= 1:
        return bank.copy()
    if not np.isfinite(bank).all():
        raise ValueError("non-finite vector")
    rows = []
    for index in range(int(bank.shape[0])):
        others = [bank[other] for other in range(int(bank.shape[0])) if other != index]
        rows.append(separate_key(bank[index], others, tau=tau, span=span))
    return np.stack(rows, 0)


def match_scores(
    query: ArrayLike,
    keys: ArrayLike,
    *,
    tau: float = SEPARATION_TAU,
    span: float = SEPARATION_SPAN,
) -> NDArray[np.float64]:
    """Cosine of ``query`` to :func:`separate_keys`. Distinct keys match raw cosine."""
    return cosine_scores(separate_keys(keys, tau=tau, span=span), query)
