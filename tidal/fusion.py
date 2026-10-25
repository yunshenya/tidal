"""Low-level automatic fusion scoring (v0 heuristic).

Ernst and Banks (2002) show that humans combine cues by reliability
(inverse variance). This module treats strength, salience, and fingerprint
presence as an inverse-noise proxy and takes a reliability-weighted mix.
https://doi.org/10.1038/415429a
Körding, Beierholm, Ma, Quartz, Tenenbaum, and Shams (2007) show that
observers also infer whether two cues share a cause, and only then take
that mix; otherwise they keep the estimates separate. Weak common-cause
evidence pulls the score down. A single modality never invents a fused
score.
https://doi.org/10.1371/journal.pbio.0050260

This is a named heuristic, not a trained model. It reads existing episodic
slots (write step ``t``, modality, fingerprint, strength, salience). It
does not create a parallel memory, and it does not write
``fusion_weights``. growbrain ``TidalWrite → set_fusion_weights`` stays
authoritative. Suggested weights are used only inside the score when
growbrain has not written a positive mix on a present source. The only
tidal write is :meth:`tidal.llm_memory.LLMMemory.set_fusion_confidence`.

Formula (HYPOTHESIS; no grid was run)
-------------------------------------
1. Window. Keep slots with ``0 <= now - t <= FUSION_WINDOW``. ``now`` is
   the store step. Window 4 is a hypothesis in store-step units, not a
   millisecond superior-colliculus window.

2. Sources. Only ``text``, ``audio``, and ``image`` count. ``fused`` is
   already a fusion product. ``none`` is unknown. Fewer than two distinct
   source modalities yields ``c = 0``.

3. Reliability (Ernst and Banks inverse-noise). For each observation::

       presence = 1 if ||fp|| > FP_ZERO else 0
       r        = max(strength, 0) * (1 + max(salience, 0))
                  * (FP_FLOOR + (1 - FP_FLOOR) * presence)

   Per modality, keep the max ``r``. ``FP_FLOOR`` is the reliability of a
   labeled modality whose fingerprint is still the zero placeholder.
   Fingerprint magnitude is not calibrated, so presence is binary.

4. Mix. If growbrain ``fusion_weights`` has positive mass on a present
   source, those weights modulate that source (``r_eff = r * max(w, 0)``).
   Otherwise the suggested mix is ``r`` itself. Combined precision is
   ``R = sum r_eff``. The reliability term is ``min(1, R / RELIABILITY_SAT)``.
   Two unit-reliability sources saturate at 1.

5. Causal soft gate (Körding). For each pair of per-modality winners::

       if either fingerprint is ~0:
           p_pair = TIME_PRIOR          # time coincidence only
       else:
           sim = cosine(fp_i, fp_j)
           if sim >= CAUSAL_TAU:
               p_pair = TIME_PRIOR + (1 - TIME_PRIOR)
                        * (sim - TAU) / (1 - TAU)
           else:
               p_pair = TIME_PRIOR * (sim + 1) / (TAU + 1)

   ``P(C=1)`` is the mean of ``p_pair``. Matching fingerprints give 1.
   Opposite fingerprints give 0. Missing fingerprints give TIME_PRIOR.
   If ``P(C=1) < CAUSAL_FLOOR``, return 0 (do not invent fusion).

6. ``c = P(C=1) * reliability_term``, clipped to ``[0, 1]``. Finite.
   ``1.0`` is allowed. ``0`` means no fusion.

Live hook
---------
:meth:`tidal.llm_memory.LLMMemory.refresh_fusion_confidence` reads the
store, scores, and calls ``set_fusion_confidence``. ``remember`` calls it
after an accepted write. ``cue_fields`` does not rescore; it exports the
last setter write, the same last-token path as #75. Fixtures that write
through ``store.write`` call ``refresh_fusion_confidence``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Never, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from tidal.interface_v7 import (
    FUSION_WEIGHT_ORDER,
    FusionWeights,
    Modality,
    check_fingerprint,
    check_fusion_confidence,
    check_fusion_weights,
    check_modality,
    empty_fusion_weights,
    zero_fingerprint,
)

# Store-step coincidence window. HYPOTHESIS. Not a millisecond SC window.
FUSION_WINDOW = 4
# Reliability of a labeled modality with a zero fingerprint. HYPOTHESIS.
FP_FLOOR = 0.25
# Treat a fingerprint at or below this L2 norm as missing. HYPOTHESIS.
FP_ZERO = 1e-12
# P(common cause) from time-window co-occurrence alone. HYPOTHESIS.
TIME_PRIOR = 0.55
# Cosine at or above this is same-cause evidence. HYPOTHESIS.
CAUSAL_TAU = 0.20
# Below this P(C=1), return 0 rather than invent fusion. HYPOTHESIS.
CAUSAL_FLOOR = 0.15
# Combined precision that saturates the reliability term at 1. HYPOTHESIS.
RELIABILITY_SAT = 2.0
SOURCE_MODALITIES: tuple[Modality, ...] = ("text", "audio", "image")


@dataclass(frozen=True)
class FusionObservation:
    """One store slot as the scorer sees it. Not a second memory."""

    t: int
    modality: Modality
    fingerprint: NDArray[np.float32]
    strength: float = 1.0
    salience: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.t, bool) or not isinstance(self.t, int):
            raise TypeError("t must be an int")
        object.__setattr__(self, "modality", check_modality(str(self.modality)))
        object.__setattr__(self, "fingerprint", check_fingerprint(self.fingerprint))
        object.__setattr__(self, "strength", _finite(self.strength, "strength"))
        object.__setattr__(self, "salience", _finite(self.salience, "salience"))


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _is_source_modality(modality: Modality) -> bool:
    if modality == "text":
        return True
    if modality == "audio":
        return True
    if modality == "image":
        return True
    if modality == "fused":
        return False
    if modality == "none":
        return False
    unused: Never = modality
    raise ValueError(f"unknown modality {unused!r}")


def _fingerprint_norm(fingerprint: NDArray[np.float32]) -> float:
    return float(np.linalg.norm(np.asarray(fingerprint, dtype=np.float32)))


def _presence(fingerprint: NDArray[np.float32]) -> float:
    """1 when an encoder vector is present, 0 for the zero placeholder."""
    if _fingerprint_norm(fingerprint) <= float(FP_ZERO):
        return 0.0
    return 1.0


def observation_reliability(observation: FusionObservation) -> float:
    """Inverse-noise proxy in ``[0, inf)``. Strength * salience * presence."""
    strength = max(float(observation.strength), 0.0)
    salience = max(float(observation.salience), 0.0)
    presence = _presence(observation.fingerprint)
    signal = strength * (1.0 + salience) * (float(FP_FLOOR) + (1.0 - float(FP_FLOOR)) * presence)
    if not math.isfinite(signal) or signal <= 0.0:
        return 0.0
    return signal


def fingerprint_cosine(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    """Cosine in ``[-1, 1]``. Missing fingerprints return 0."""
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= float(FP_ZERO) or nb <= float(FP_ZERO):
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


def pair_common_cause(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    """Soft P(common cause) for one pair. Missing fingerprints use TIME_PRIOR."""
    na = _fingerprint_norm(left)
    nb = _fingerprint_norm(right)
    prior = float(TIME_PRIOR)
    tau = float(CAUSAL_TAU)
    if na <= float(FP_ZERO) or nb <= float(FP_ZERO):
        return prior
    sim = fingerprint_cosine(left, right)
    if sim >= tau:
        return prior + (1.0 - prior) * (sim - tau) / (1.0 - tau)
    return prior * (sim + 1.0) / (tau + 1.0)


def suggested_fusion_weights(reliabilities: Mapping[str, float]) -> FusionWeights:
    """Internal ``(3,)`` mix from per-modality reliability. Not a store write."""
    out = empty_fusion_weights()
    total = 0.0
    for index, name in enumerate(FUSION_WEIGHT_ORDER):
        raw = reliabilities.get(name, 0.0)
        level = max(float(raw), 0.0) if math.isfinite(float(raw)) else 0.0
        out[index] = level
        total += level
    if total > 0.0:
        out /= np.float32(total)
    return out


def observations_from_slots(slots: Sequence[object]) -> tuple[FusionObservation, ...]:
    """Project existing store slots. Does not copy values or keys."""
    out: list[FusionObservation] = []
    for slot in slots:
        fingerprint = getattr(slot, "fingerprint", None)
        if fingerprint is None:
            fingerprint = zero_fingerprint()
        out.append(
            FusionObservation(
                t=int(slot.t),
                modality=check_modality(str(slot.modality)),
                fingerprint=check_fingerprint(fingerprint),
                strength=float(getattr(slot, "strength", 1.0)),
                salience=float(getattr(slot, "salience", 0.0)),
            )
        )
    return tuple(out)


def _effective_reliability(
    reliabilities: Mapping[str, float],
    weights: Optional[Union[Sequence[float], FusionWeights]],
) -> dict[str, float]:
    if weights is None:
        return {name: float(reliabilities[name]) for name in reliabilities}
    mix = check_fusion_weights(weights)
    present = {
        name: max(float(mix[index]), 0.0)
        for index, name in enumerate(FUSION_WEIGHT_ORDER)
        if name in reliabilities
    }
    if sum(present.values()) <= 0.0:
        return {name: float(reliabilities[name]) for name in reliabilities}
    return {name: float(reliabilities[name]) * present[name] for name in reliabilities}


def score_fusion(
    observations: Sequence[FusionObservation],
    *,
    now: Optional[int] = None,
    weights: Optional[Union[Sequence[float], FusionWeights]] = None,
    window: int = FUSION_WINDOW,
) -> float:
    """Finite fusion confidence in ``[0, 1]``. ``0`` when there is no fusion.

    ``weights`` is an optional growbrain ``(3,)`` mix used only to compute
    ``c``. This function does not write ``fusion_weights``.
    """
    if isinstance(window, bool) or not isinstance(window, int):
        raise TypeError("window must be an int")
    if window < 0:
        raise ValueError("window must be >= 0")
    if now is not None:
        if isinstance(now, bool) or not isinstance(now, int):
            raise TypeError("now must be an int")
    rows = tuple(observations)
    if not rows:
        return 0.0
    stamp = int(now) if now is not None else max(item.t for item in rows)
    windowed = [item for item in rows if 0 <= stamp - item.t <= window]
    best: dict[str, tuple[float, FusionObservation]] = {}
    for item in windowed:
        if not _is_source_modality(item.modality):
            continue
        reliability = observation_reliability(item)
        prior = best.get(item.modality)
        if prior is None or reliability > prior[0]:
            best[item.modality] = (reliability, item)
    if len(best) < 2:
        return 0.0
    pairs = []
    names = list(best)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            pairs.append(pair_common_cause(best[left][1].fingerprint, best[right][1].fingerprint))
    common = float(sum(pairs) / len(pairs))
    if common < float(CAUSAL_FLOOR):
        return 0.0
    reliabilities = {name: pair[0] for name, pair in best.items()}
    combined = sum(_effective_reliability(reliabilities, weights).values())
    sat = float(RELIABILITY_SAT)
    if sat <= 0.0:
        raise ValueError("reliability saturation must be positive")
    reliability_term = 1.0 if combined >= sat else combined / sat
    level = common * reliability_term
    if not math.isfinite(level) or level <= 0.0:
        return 0.0
    if level > 1.0:
        level = 1.0
    return check_fusion_confidence(level)
