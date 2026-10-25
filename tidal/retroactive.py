"""Retroactive interference from a new similar write (Wixted 2004).

Wixted reviews everyday forgetting as driven more by retroactive
interference from new learning than by unused time. tidal already ages
traces on token and wall clocks, and it already orthogonalizes
near-neighbor keys as a retrieve ranking view. This module is the
encoding-time hit: a successful new slot ages older neighbors on the
same key geometry. Residual detail already reads that age. It is not
unused-time decay, not retrieval-induced forgetting, not a
reconsolidation window, and not a sleep gist phase.
https://doi.org/10.1146/annurev.psych.55.090902.141555

The hit increments ``slot.age``. Reconstructive spelling already maps
age 1 onto a peripheral slip and a kept gist. Replay
``recompute_decay`` is still ``decay ** age``, so a later cycle is
weaker than a time-matched neighbor that was not hit. Strength is not
rewritten here, so ``fusion_confidence``, cue ranking, and RIF still
see the clock / RIF scale until that cycle. Salience, residue parts,
``consolidation_gain``, keys, and values stay. No cue-export field is
added.

Formula (HYPOTHESIS; no grid was run)
-------------------------------------
1. Collision. Cosine at or above ``INTERFERENCE_TAU`` is a near-neighbor.
   That is the same band :data:`tidal.pattern_separation.SEPARATION_TAU`
   uses. Below that, the older slot is unchanged.

2. Weight. ``w = SPAN * (sim - TAU) / (1 - TAU)`` via
   :func:`tidal.pattern_separation.overlap_weight`. Cosine 1 is ``SPAN``.
   Cosine at TAU is 0. ``keep = 1 - w``. Cosine 1 is ``1 - SPAN``.

3. Apply. When ``keep < 1``, ``age = age + 1`` on the older slot. The
   new slot is skipped. Same-identity overwrite and a surface-repeat
   merge do not run this pass. Distinct keys below TAU do not blanket-punish
   the store.

Live hook
---------
:meth:`tidal.episodic.EpisodicStore.write` applies the hit after a
successful append. Live ``remember`` uses that write, then stamps clock
strength and McGaugh gain on the new slot only. Neighbors keep the
write-time age. A refresh still sets the replayed slot's age to 0 and
strength to 1.
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np

from tidal.pattern_separation import (
    SEPARATION_SPAN,
    SEPARATION_TAU,
    key_cosine,
    overlap_weight,
)

ArrayLike = Union[np.ndarray, Sequence[float]]

# Same collision band as pattern-separation retrieve ranking. HYPOTHESIS.
INTERFERENCE_TAU = float(SEPARATION_TAU)
# Max weight at cosine 1. Same 0.25 as clock / McGaugh / separation.
# HYPOTHESIS. Age steps by 1 whenever keep is below 1.
INTERFERENCE_SPAN = float(SEPARATION_SPAN)


def _finite_span(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def neighbor_keep(
    sim: float,
    *,
    tau: float = INTERFERENCE_TAU,
    span: float = INTERFERENCE_SPAN,
) -> float:
    """``1`` below ``tau``. ``1 - span`` at cosine 1."""
    weight = overlap_weight(
        _finite_span(sim, "cosine"),
        tau=_finite_span(tau, "tau"),
        span=_finite_span(span, "span"),
    )
    keep = 1.0 - float(weight)
    if keep < 0.0:
        return 0.0
    if keep > 1.0:
        return 1.0
    return keep


def apply_retroactive(
    slots: Sequence[object],
    incoming: ArrayLike,
    skip_index: int,
    *,
    tau: float = INTERFERENCE_TAU,
    span: float = INTERFERENCE_SPAN,
) -> int:
    """Age older near-neighbors. Returns how many ages changed.

    ``skip_index`` is the new slot. A missing key, an out-of-range skip,
    or an empty store is a no-op. Strength is not rewritten.
    """
    if isinstance(skip_index, bool) or not isinstance(skip_index, int):
        raise TypeError("skip index must be an int")
    rows = list(slots)
    if skip_index < 0 or skip_index >= len(rows):
        return 0
    incoming_key = np.asarray(incoming, dtype=np.float64).reshape(-1)
    if incoming_key.size == 0 or not np.isfinite(incoming_key).all():
        return 0
    changed = 0
    for index, slot in enumerate(rows):
        if index == skip_index:
            continue
        key = getattr(slot, "k", None)
        if key is None:
            continue
        keep = neighbor_keep(key_cosine(incoming_key, key), tau=tau, span=span)
        if keep == 1.0:
            continue
        before = int(getattr(slot, "age", 0))
        slot.age = before + 1
        changed += 1
    return changed
