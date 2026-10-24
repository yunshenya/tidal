"""Cue-triggered retrieval. One winner, then a decaying per-token residual.

The cue is one token's hidden state. Its width is ``cue_dim``. Nothing in
this module names a decoder index. The match vector is the cue minus the
query center, which is the space stored keys already live in.

A candidate must clear ``theta`` on cosine alone. Among those candidates the
winner is the greatest ``(log(strength), cosine, -index)``. Cosine is only
the tie break. One winner is injected. There is no average across neighbors.

On the trigger token, ``dt`` is 0 and ``a = a_trig``. A later token that does
not trigger uses ``a = a_trig * exp(-dt / tau)``, where ``dt`` is the number
of tokens after the trigger. A new winner resets ``dt``. ``a`` below
``floor`` clears the trace. The same-token add is applied after the cue is
copied, so the cue does not depend on the add and no earlier position sees it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tidal.episodic import cosine_scores


@dataclass
class CueState:
    """Decay carried across tokens the way a cache carries past keys.

    ``memory_id`` is the slot index chosen at the trigger, or -1 when nothing
    is active. ``a`` is the activation in ``[0, 1]``. ``dt`` is tokens since
    the trigger. Strength, confidence, valence offset, and the stored value
    are frozen at the trigger.
    """

    a: float = 0.0
    dt: int = 0
    memory_id: int = -1
    confidence: float = 0.0
    strength: float = 0.0
    valence: float = 0.0
    gap: float = float("nan")
    winner_cosine: float = float("nan")
    vec: Optional[np.ndarray] = None

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
    for index, (cos, level) in enumerate(zip(cosine.tolist(), strength.tolist())):
        if not math.isfinite(float(level)) or float(level) <= 0.0:
            continue
        positive.append(index)
        if not math.isfinite(float(cos)) or float(cos) < float(theta):
            continue
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
    return {
        "index": int(best_index),
        "cosine": float(cosine[best_index]),
        "gap": gap,
        "strength": float(strength[best_index]),
        "tied": bool(tied),
    }


def _check_schedule(tau: float, a_trig: float, floor: float) -> None:
    if not math.isfinite(float(tau)) or float(tau) <= 0.0:
        raise ValueError("tau must be positive")
    if not math.isfinite(float(a_trig)) or not 0.0 < float(a_trig) <= 1.0:
        raise ValueError("a_trig must be in (0, 1]")
    if not math.isfinite(float(floor)) or not 0.0 <= float(floor) < 1.0:
        raise ValueError("floor must be in [0, 1)")


def _blank(cue_dim: int) -> dict:
    return {
        "memory_id": -1,
        "vec": np.zeros(int(cue_dim), dtype=np.float32),
        "activation": 0.0,
        "strength_confidence": 0.0,
        "valence": 0.0,
        "triggered": False,
        "gap": float("nan"),
        "winner_cosine": float("nan"),
    }


def _row_from_state(state: CueState, cue_dim: int, *, triggered: bool) -> dict:
    vec = state.vec
    if vec is None:
        vec = np.zeros(int(cue_dim), dtype=np.float32)
    else:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape != (int(cue_dim),):
            raise ValueError(f"carried value dim {vec.shape[0]} != cue_dim {cue_dim}")
    return {
        "memory_id": int(state.memory_id),
        "vec": vec.copy(),
        "activation": float(state.a),
        "strength_confidence": float(state.strength) * float(state.confidence),
        "valence": float(state.valence),
        "triggered": bool(triggered),
        "gap": float(state.gap),
        "winner_cosine": float(state.winner_cosine),
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
) -> tuple:
    """Advance one token. Returns ``(new_state, row)`` and does not mutate ``state``."""
    _check_schedule(tau, a_trig, floor)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("values must be [slots, cue_dim]")
    cue_dim = int(values.shape[1])
    if cue_dim < 1:
        raise ValueError("cue_dim must be positive")
    choice = select_cue_winner(cosines, strengths, theta)
    if choice is not None:
        activation = float(a_trig)
        if activation < float(floor):
            return CueState(), _blank(cue_dim)
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
            vec=vec,
        )
        return new, _row_from_state(new, cue_dim, triggered=True)
    if state.inactive():
        return CueState(), _blank(cue_dim)
    dt = int(state.dt) + 1
    activation = float(a_trig) * math.exp(-float(dt) / float(tau))
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
        vec=None if state.vec is None else np.asarray(state.vec, dtype=np.float32).copy(),
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
        )
        memory_id[index] = int(row["memory_id"])
        vec[index] = row["vec"]
        activation[index] = float(row["activation"])
        strength_confidence[index] = float(row["strength_confidence"])
        valence_offset[index] = float(row["valence"])
        triggered[index] = bool(row["triggered"])
        gap[index] = float(row["gap"])
        winner_cosine[index] = float(row["winner_cosine"])
    return {
        "memory_id": memory_id,
        "vec": vec,
        "activation": activation,
        "strength_confidence": strength_confidence,
        "valence": valence_offset,
        "triggered": triggered,
        "gap": gap,
        "winner_cosine": winner_cosine,
        "cue_dim": int(cue_dim),
    }, current
