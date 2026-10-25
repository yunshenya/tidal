"""Fit and held ids for retroactive-interference v0. Numeric fixtures.

No torch. Every id runs the same pre-registered fixtures, so the
canonical age and keep values do not depend on the id.
"""
from __future__ import annotations

import hashlib

from tidal.pattern_separation import key_cosine
from tidal.retroactive import (
    INTERFERENCE_SPAN,
    INTERFERENCE_TAU,
    apply_retroactive,
    neighbor_keep,
)
from tidal.subconscious_separation_data import distinct_pair, similar_pair

PROTOCOL = "retroactive-interference-v0"
FIT_N = 8
HELD_N = 50
REPLAY_DECAY = 0.5
GATES = (
    "identity",
    "similar",
    "control",
    "orthogonal",
    "write",
    "compose",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("retroactive-fit", FIT_N)
HELD_IDS = _ids("retroactive-held", HELD_N)


class _Slot:
    def __init__(self, key, strength: float = 1.0, age: int = 0):
        self.k = key
        self.strength = float(strength)
        self.age = int(age)


def _decay(slot: _Slot, decay: float = REPLAY_DECAY) -> None:
    slot.age = int(slot.age) + 1
    slot.strength = float(decay) ** int(slot.age)


def canonical_scores() -> dict:
    """Locked keep and age values. Identical for every id."""
    similar_left, similar_right = similar_pair()
    distinct_left, distinct_right = distinct_pair()
    similar_raw = float(key_cosine(similar_left, similar_right))
    distinct_raw = float(key_cosine(distinct_left, distinct_right))
    similar_keep = float(neighbor_keep(similar_raw))
    treated = _Slot(similar_left)
    apply_retroactive([treated, _Slot(similar_right)], similar_right, 1)
    control = _Slot(similar_left)
    _decay(treated)
    _decay(control)
    return {
        "protocol": PROTOCOL,
        "tau": float(INTERFERENCE_TAU),
        "span": float(INTERFERENCE_SPAN),
        "similar_raw": similar_raw,
        "similar_keep": similar_keep,
        "exact_keep": float(neighbor_keep(1.0)),
        "tau_keep": float(neighbor_keep(float(INTERFERENCE_TAU))),
        "below_keep": float(neighbor_keep(0.5)),
        "distinct_raw": distinct_raw,
        "distinct_keep": float(neighbor_keep(distinct_raw)),
        "similar_age": 1,
        "control_age": 0,
        "decay": float(REPLAY_DECAY),
        "decay_treated": float(treated.strength),
        "decay_control": float(control.strength),
    }


def formula_gates() -> dict:
    """Pure-function gates. Live remember / export are scored in tests."""
    scores = canonical_scores()
    similar_left, similar_right = similar_pair()
    distinct_left, distinct_right = distinct_pair()
    identity = (
        neighbor_keep(0.5) == 1.0
        and neighbor_keep(float(INTERFERENCE_TAU)) == 1.0
        and neighbor_keep(1.0) == 1.0 - float(INTERFERENCE_SPAN)
        and apply_retroactive([], similar_left, 0) == 0
    )
    treated = [_Slot(similar_left), _Slot(similar_right)]
    similar_changed = apply_retroactive(treated, similar_right, 1)
    similar = (
        scores["similar_raw"] >= float(INTERFERENCE_TAU)
        and similar_changed == 1
        and treated[0].age == 1
        and treated[1].age == 0
        and treated[0].strength == 1.0
        and treated[1].strength == 1.0
    )
    control = [_Slot(similar_left)]
    control_changed = apply_retroactive(control, similar_left, 0)
    control_ok = control_changed == 0 and control[0].age == 0 and control[0].strength == 1.0
    orthogonal = [_Slot(distinct_left), _Slot(distinct_right)]
    ortho_changed = apply_retroactive(orthogonal, distinct_right, 1)
    orthogonal_ok = (
        scores["distinct_raw"] == 0.0
        and scores["distinct_keep"] == 1.0
        and ortho_changed == 0
        and orthogonal[0].age == 0
        and orthogonal[0].strength == 1.0
        and orthogonal[1].strength == 1.0
    )
    write = [_Slot(similar_left), _Slot(similar_right)]
    write_changed = apply_retroactive(write, similar_right, 1)
    write_ok = write_changed == 1 and write[1].age == 0 and write[1].strength == 1.0
    prior = [_Slot(similar_left), _Slot(similar_right)]
    apply_retroactive(prior, similar_right, 1)
    baseline = _Slot(similar_left)
    _decay(prior[0])
    _decay(baseline)
    compose = (
        prior[0].strength == scores["decay_treated"]
        and baseline.strength == scores["decay_control"]
        and prior[0].strength < baseline.strength
        and prior[1].strength == 1.0
    )
    return {
        "identity": bool(identity),
        "similar": bool(similar),
        "control": bool(control_ok),
        "orthogonal": bool(orthogonal_ok),
        "write": bool(write_ok),
        "compose": bool(compose),
        "scores": scores,
    }


def bank_sha256() -> str:
    scores = canonical_scores()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            str(scores["tau"]),
            str(scores["span"]),
            str(scores["similar_raw"]),
            str(scores["similar_keep"]),
            str(scores["similar_age"]),
            str(scores["decay_treated"]),
            str(scores["decay_control"]),
            *GATES,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
