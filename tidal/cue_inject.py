"""Cue-triggered retrieval. One winner, then a decaying per-token residual.

The cue is one token's hidden state. Its width is ``cue_dim``. Nothing in
this module names a decoder index. The match vector is the cue minus the
query center, which is the space stored keys already live in.

A candidate must clear ``theta`` on cosine alone. Among those candidates the
winner is the greatest ``(log(strength), cosine, -index)``. Cosine is only
the tie break. One winner is injected. There is no average across neighbors.

On the trigger token, ``dt`` is 0 and the exported activation is
``a = a_trig * fan_factor``. ``n`` is the number of candidates that clear
``theta``. ``fan_factor`` is ``max(0, (S - ln n) / S)`` with ``S`` fixed at
1.45, so fan 1 leaves ``a_trig`` unchanged. The fan factor only scales that
activation down. A token with candidates past ``theta`` still triggers when
the product is below ``floor``, including when the factor is 0. A later
token that does not trigger keeps that fan factor and uses
``a = a_trig * fan_factor * exp(-dt / tau)`` when no wall time has passed
since the trigger. ``dt`` is the number of tokens after the trigger. If wall
seconds have passed, the remaining factor is the lower of that token factor
and ``exp(-elapsed / wall_tau)``. A frozen wall clock leaves the token
formula. A new winner resets ``dt`` and the wall stamp and recomputes the
fan factor. On that decay path, ``a`` below ``floor`` clears the carried
trace. Continuing the same winner never raises ``a``. The
same-token add is applied after the cue is copied, so the cue does not
depend on the add and no earlier position sees it. This module does not
apply the language-model gain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tidal.clock_drift import WALL_TAU, clock_activation, elapsed_since
from tidal.episodic import MEMORY_MODALITY, cosine_scores
from tidal.interface_v7 import check_fingerprint, check_modality, zero_fingerprint

# Anderson & Reder 1999 estimated S = 1.45 for the Anderson 1974 fan data.
# Not a fitted knob. fan_factor divides by S so one candidate stays 1.
FAN_S = 1.45


def fan_factor(n: int, s: float = FAN_S) -> float:
    """``max(0, (S - ln n) / S)``. Decreases in ``n``. Fan 1 is exactly 1."""
    if int(n) < 1:
        return 0.0
    if not math.isfinite(float(s)) or float(s) <= 0.0:
        raise ValueError("fan S must be positive")
    raw = (float(s) - math.log(float(n))) / float(s)
    if raw <= 0.0:
        return 0.0
    return float(min(1.0, raw))


@dataclass
class CueState:
    """Decay carried across tokens the way a cache carries past keys.

    ``memory_id`` is the slot index chosen at the trigger, or -1 when nothing
    is active. ``a`` is the activation in ``[0, 1]``. ``dt`` is tokens since
    the trigger. ``fan_factor`` is frozen at the trigger. Strength,
    confidence, valence offset, and the stored value are frozen there too.
    ``trigger_wall`` is the wall time stamped at the trigger. It is not the
    drifted internal clock.
    """

    a: float = 0.0
    dt: int = 0
    memory_id: int = -1
    confidence: float = 0.0
    strength: float = 0.0
    valence: float = 0.0
    gap: float = float("nan")
    winner_cosine: float = float("nan")
    fan_factor: float = 1.0
    modality: str = MEMORY_MODALITY
    vec: Optional[np.ndarray] = None
    fingerprint: Optional[np.ndarray] = None
    trigger_wall: float = 0.0

    def inactive(self) -> bool:
        return int(self.memory_id) < 0 or float(self.a) <= 0.0


def _confidence(n_positive: int, gap: Optional[float], gap_tau: float) -> float:
    """Same scale as ``retrieval_confidence``. One positive slot is unscaled."""
    if int(n_positive) < 2:
        return 1.0
    if float(gap_tau) <= 0.0:
        raise ValueError("gap_tau must be positive")
    if gap is None:
        return 0.0
    return float(min(1.0, max(0.0, float(gap) / float(gap_tau))))


def select_cue_winner(cosines, strengths, theta: float) -> Optional[dict]:
    """One slot with cosine at least ``theta``, ranked by log strength.

    ``theta`` does not read strength. Nonpositive strength is not a candidate.
    Equal log strength yields to the higher cosine, then to the lower index.
    The gap is the winner's cosine minus the next positive-strength cosine,
    including slots that missed ``theta``. ``None`` when nobody clears it.
    """
    cosine = np.asarray(cosines, dtype=np.float64).reshape(-1)
    strength = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if cosine.shape != strength.shape:
        raise ValueError("cosines and strengths must have the same shape")
    if not math.isfinite(float(theta)):
        raise ValueError("theta must be finite")
    best = None
    best_index = None
    positive = []
    n_candidates = 0
    for index, (cos, level) in enumerate(zip(cosine.tolist(), strength.tolist())):
        if not math.isfinite(float(level)) or float(level) <= 0.0:
            continue
        positive.append(index)
        if not math.isfinite(float(cos)) or float(cos) < float(theta):
            continue
        n_candidates += 1
        key = (math.log(float(level)), float(cos), -int(index))
        if best is None or key > best:
            best = key
            best_index = int(index)
    if best_index is None:
        return None
    if len(positive) < 2:
        gap = None
    else:
        others = [float(cosine[index]) for index in positive if index != best_index]
        gap = float(cosine[best_index]) - max(others)
    # Same log strength and the same cosine is a tie. The lower index is the
    # recorded winner and the confidence is 0, so the tie adds nothing.
    tied = False
    best_log, best_cos, _neg = best
    for index in positive:
        if index == best_index:
            continue
        if float(cosine[index]) < float(theta):
            continue
        other = (math.log(float(strength[index])), float(cosine[index]))
        if other == (best_log, best_cos):
            tied = True
            break
    candidates = []
    for index in positive:
        if not math.isfinite(float(cosine[index])) or float(cosine[index]) < float(theta):
            continue
        candidates.append(int(index))
    return {
        "index": int(best_index),
        "cosine": float(cosine[best_index]),
        "gap": gap,
        "strength": float(strength[best_index]),
        "tied": bool(tied),
        "n": int(n_candidates),
        "candidates": candidates,
    }


def kwta_weights(cosines, strengths, k: int) -> np.ndarray:
    """Keep the top ``k`` positive-strength slots and inhibit the rest.

    Rank is ``(log strength, cosine, -index)``, the same order as one winner.
    The threshold is the cosine of the first excluded positive-strength slot.
    A kept slot's raw weight is ``max(0, cosine - threshold)``. The weights
    then sum to 1. When every positive slot is kept, the threshold is
    ``min(0, lowest kept cosine)``, so a positive cosine is not cancelled by
    the absence of a loser. A raw-weight sum of 0 shares the weight equally
    across the kept slots. This is a k-winner-take-all diagnostic. It is not
    an oscillating network, and :func:`step_cue` does not call it.
    """
    cosine = np.asarray(cosines, dtype=np.float64).reshape(-1)
    strength = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if cosine.shape != strength.shape:
        raise ValueError("cosines and strengths must have the same shape")
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    weights = np.zeros(cosine.shape[0], dtype=np.float64)
    ranked = []
    for index, (cos, level) in enumerate(zip(cosine.tolist(), strength.tolist())):
        if not math.isfinite(float(level)) or float(level) <= 0.0:
            continue
        if not math.isfinite(float(cos)):
            continue
        ranked.append((math.log(float(level)), float(cos), -int(index), int(index)))
    if not ranked:
        return weights
    ranked.sort(reverse=True)
    active = [item[3] for item in ranked[: int(k)]]
    if len(ranked) > len(active):
        threshold = float(ranked[len(active)][1])
    else:
        threshold = min(0.0, min(float(cosine[index]) for index in active))
    for index in active:
        raw = float(cosine[index]) - threshold
        if raw > 0.0:
            weights[index] = raw
    total = float(weights.sum())
    if total <= 0.0:
        share = 1.0 / float(len(active))
        for index in active:
            weights[index] = share
        return weights
    weights /= total
    return weights


def rank_auc(related, unrelated) -> float:
    """Probability a related score exceeds an unrelated score. Ties count half."""
    left = [float(value) for value in related]
    right = [float(value) for value in unrelated]
    if not left or not right:
        raise ValueError("auc needs both classes")
    if any(not math.isfinite(value) for value in left + right):
        raise ValueError("auc scores must be finite")
    wins = 0.0
    total = 0
    for good in left:
        for bad in right:
            total += 1
            if good > bad:
                wins += 1.0
            elif good == bad:
                wins += 0.5
    return float(wins / total)


def _check_schedule(tau: float, a_trig: float, floor: float) -> None:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    if not math.isfinite(float(a_trig)) or not 0.0 < float(a_trig) <= 1.0:
        raise ValueError("a_trig must be in (0, 1]")
    if not math.isfinite(float(floor)) or not 0.0 <= float(floor) < 1.0:
        raise ValueError("floor must be in [0, 1)")


def _modality_of(modalities, index: int, n: int) -> str:
    """The stored label. Accepted values are the v7 modality enum."""
    if modalities is None:
        return MEMORY_MODALITY
    mods = list(modalities)
    if len(mods) != int(n):
        raise ValueError("modalities must align with slots")
    value = mods[int(index)]
    if not isinstance(value, str):
        raise ValueError("unknown modality; use none")
    return check_modality(value)


def _fingerprint_of(fingerprints, index: int, n: int) -> np.ndarray:
    """The stored ``(F,)`` fingerprint, or zeros when the writer omitted it."""
    if fingerprints is None:
        return zero_fingerprint()
    rows = list(fingerprints)
    if len(rows) != int(n):
        raise ValueError("fingerprints must align with slots")
    return check_fingerprint(rows[int(index)])


def _state_fingerprint(state: CueState) -> np.ndarray:
    if state.inactive() or state.fingerprint is None:
        return zero_fingerprint()
    return check_fingerprint(state.fingerprint)


def _blank(cue_dim: int) -> dict:
    return {
        "memory_id": -1,
        "vec": np.zeros(int(cue_dim), dtype=np.float32),
        "activation": 0.0,
        "strength_confidence": 0.0,
        "valence": 0.0,
        "modality": MEMORY_MODALITY,
        "triggered": False,
        "gap": float("nan"),
        "winner_cosine": float("nan"),
        "candidates": [],
        "fingerprint": zero_fingerprint(),
    }


def _row_from_state(state: CueState, cue_dim: int, *, triggered: bool) -> dict:
    vec = state.vec
    if vec is None:
        vec = np.zeros(int(cue_dim), dtype=np.float32)
    else:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape != (int(cue_dim),):
            raise ValueError(f"carried value dim {vec.shape[0]} != cue_dim {cue_dim}")
    check_modality(str(state.modality))
    return {
        "memory_id": int(state.memory_id),
        "vec": vec.copy(),
        "activation": float(state.a),
        "strength_confidence": float(state.strength) * float(state.confidence),
        "valence": float(state.valence),
        "modality": str(state.modality),
        "triggered": bool(triggered),
        "gap": float(state.gap),
        "winner_cosine": float(state.winner_cosine),
        "candidates": [],
        "fingerprint": _state_fingerprint(state),
    }


def step_cue(
    state: CueState,
    cosines,
    strengths,
    valences,
    values,
    *,
    theta: float,
    tau: float,
    a_trig: float,
    floor: float,
    gap_tau: float,
    valence_span: float,
    fan_s: float = FAN_S,
    modalities=None,
    fingerprints=None,
    now=None,
    wall_tau: float = WALL_TAU,
) -> tuple:
    """Advance one token. Returns ``(new_state, row)`` and does not mutate ``state``.

    ``now`` is the wall time in seconds. Omitted, the wall elapsed is 0 and
    the decay is the token formula. A new winner stamps ``now``.
    """
    _check_schedule(tau, a_trig, floor)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must be [slots, cue_dim]")
    cue_dim = int(values.shape[1])
    if cue_dim < 1:
        raise ValueError("cue_dim must be positive")
    choice = select_cue_winner(cosines, strengths, theta)
    if choice is not None:
        factor = fan_factor(int(choice["n"]), fan_s)
        activation = float(a_trig) * factor
        index = int(choice["index"])
        vec = np.asarray(values[index], dtype=np.float32).reshape(-1).copy()
        offset = float(valence_span) * math.tanh(float(valences[index]))
        gap = choice["gap"]
        n_positive = int(np.sum(np.asarray(strengths, dtype=np.float64) > 0.0))
        if choice.get("tied"):
            confidence = 0.0
        elif gap is not None and float(gap) < 0.0:
            # Log strength picked a slot that is not the cosine leader.
            # The cosine gate already fired. Do not zero that winner.
            confidence = 1.0
        else:
            confidence = _confidence(n_positive, gap, gap_tau)
        new = CueState(
            a=activation,
            dt=0,
            memory_id=index,
            confidence=confidence,
            strength=float(choice["strength"]),
            valence=offset,
            gap=float("nan") if gap is None else float(gap),
            winner_cosine=float(choice["cosine"]),
            fan_factor=factor,
            modality=_modality_of(modalities, index, int(values.shape[0])),
            vec=vec,
            fingerprint=_fingerprint_of(fingerprints, index, int(values.shape[0])),
            trigger_wall=0.0 if now is None else float(now),
        )
        row = _row_from_state(new, cue_dim, triggered=True)
        row["candidates"] = [int(item) for item in choice["candidates"]]
        return new, row
    if state.inactive():
        return CueState(), _blank(cue_dim)
    dt = int(state.dt) + 1
    factor = float(state.fan_factor)
    elapsed = elapsed_since(state.trigger_wall, now)
    activation = clock_activation(float(a_trig) * factor, dt, elapsed, tau, wall_tau)
    # A continued trace does not rise when a stamp sits behind the trigger.
    if activation > float(state.a):
        activation = float(state.a)
    if activation < float(floor):
        return CueState(), _blank(cue_dim)
    new = CueState(
        a=float(activation),
        dt=dt,
        memory_id=int(state.memory_id),
        confidence=float(state.confidence),
        strength=float(state.strength),
        valence=float(state.valence),
        gap=float(state.gap),
        winner_cosine=float(state.winner_cosine),
        fan_factor=factor,
        modality=str(state.modality),
        vec=None if state.vec is None else np.asarray(state.vec, dtype=np.float32).copy(),
        fingerprint=None if state.fingerprint is None else np.asarray(state.fingerprint, dtype=np.float32).copy(),
        trigger_wall=float(state.trigger_wall),
    )
    return new, _row_from_state(new, cue_dim, triggered=False)


def trace_sequence(
    cues,
    keys,
    strengths,
    valences,
    values,
    query_mean,
    *,
    theta: float,
    tau: float,
    a_trig: float,
    floor: float,
    gap_tau: float,
    valence_span: float,
    state: Optional[CueState] = None,
    fan_s: float = FAN_S,
    modalities=None,
    fingerprints=None,
    now=None,
    wall_tau: float = WALL_TAU,
) -> tuple:
    """Left-to-right trace. ``cues`` is ``[T, cue_dim]`` before the query center.

    ``state`` is the decay before the first row. A full forward starts from
    an empty :class:`CueState`. Stepping that same function one row at a time
    matches this trace.
    """
    rows = np.asarray(cues, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("cues must be [tokens, cue_dim]")
    steps, cue_dim = int(rows.shape[0]), int(rows.shape[1])
    key = np.asarray(keys, dtype=np.float64)
    if key.ndim != 2 or int(key.shape[1]) != cue_dim:
        raise ValueError("keys must be [slots, cue_dim]")
    value = np.asarray(values, dtype=np.float32)
    if value.shape != key.shape:
        raise ValueError("values must match keys")
    strength = np.asarray(strengths, dtype=np.float64).reshape(-1)
    valence = np.asarray(valences, dtype=np.float64).reshape(-1)
    if strength.shape[0] != key.shape[0] or valence.shape[0] != key.shape[0]:
        raise ValueError("strengths and valences must align with keys")
    center = np.asarray(query_mean, dtype=np.float64).reshape(-1)
    if center.shape != (cue_dim,):
        raise ValueError("query center dim must equal cue_dim")
    current = CueState() if state is None else state
    memory_id = np.full(steps, -1, dtype=np.int64)
    vec = np.zeros((steps, cue_dim), dtype=np.float32)
    activation = np.zeros(steps, dtype=np.float64)
    strength_confidence = np.zeros(steps, dtype=np.float64)
    valence_offset = np.zeros(steps, dtype=np.float64)
    triggered = np.zeros(steps, dtype=bool)
    gap = np.full(steps, np.nan, dtype=np.float64)
    winner_cosine = np.full(steps, np.nan, dtype=np.float64)
    modality = []
    candidates = []
    for index in range(steps):
        if key.shape[0] == 0:
            scores = np.zeros(0, dtype=np.float64)
        else:
            scores = cosine_scores(key, rows[index] - center)
        current, row = step_cue(
            current,
            scores,
            strength,
            valence,
            value if value.size else np.zeros((0, cue_dim), dtype=np.float32),
            theta=theta,
            tau=tau,
            a_trig=a_trig,
            floor=floor,
            gap_tau=gap_tau,
            valence_span=valence_span,
            fan_s=fan_s,
            modalities=modalities,
            fingerprints=fingerprints,
            now=now,
            wall_tau=wall_tau,
        )
        memory_id[index] = int(row["memory_id"])
        vec[index] = row["vec"]
        activation[index] = float(row["activation"])
        strength_confidence[index] = float(row["strength_confidence"])
        valence_offset[index] = float(row["valence"])
        triggered[index] = bool(row["triggered"])
        gap[index] = float(row["gap"])
        winner_cosine[index] = float(row["winner_cosine"])
        modality.append(str(row["modality"]))
        candidates.append([int(item) for item in row["candidates"]])
    return {
        "memory_id": memory_id,
        "vec": vec,
        "activation": activation,
        "strength_confidence": strength_confidence,
        "valence": valence_offset,
        "triggered": triggered,
        "gap": gap,
        "winner_cosine": winner_cosine,
        "modality": modality,
        "candidates": candidates,
        "cue_dim": int(cue_dim),
        "fingerprint": _state_fingerprint(current),
    }, current
