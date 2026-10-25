"""Fit and held ids for the post-retrieval labile window. Numeric schedules.

No torch. Every id runs the same pre-registered fixtures, so the
canonical window and write-back numbers do not depend on the id.
"""
from __future__ import annotations

import hashlib

from tidal.reconsolidation import (
    LABILE_TOKENS,
    LABILE_WALL,
    apply_writeback,
    open_window,
    reconstructive_writeback,
    window_open,
)
from tidal.reconstruct import recall_surface, residue_from_answer
from tidal.subconscious_reconstruct_data import LURE_PHONE, PHONE

PROTOCOL = "reconsolidation-window-v0"
FIT_N = 8
HELD_N = 50
GATES = (
    "identity",
    "hit",
    "update",
    "token",
    "wall",
    "miss",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("reconsolidation-fit", FIT_N)
HELD_IDS = _ids("reconsolidation-held", HELD_N)


class _Slot:
    def __init__(self, answer: str, strength: float = 1.0, age: int = 0, valence: float = 0.0):
        self.answer = answer
        self.residue = residue_from_answer(answer)
        self.strength = float(strength)
        self.age = int(age)
        self.valence = float(valence)
        self.labile_until_token = 0
        self.labile_until_wall = 0.0


def canonical_window() -> dict:
    """Locked deadlines and write-back surfaces. Identical for every id."""
    until_token, until_wall = open_window(0, 0.0)
    exact = reconstructive_writeback(residue_from_answer(PHONE), 1.0, 0, 0.0)
    lure = reconstructive_writeback(residue_from_answer(PHONE), 1.0, 0, 0.0, LURE_PHONE)
    slip = reconstructive_writeback(residue_from_answer(PHONE), 0.5, 0, 0.0)
    return {
        "until_token": int(until_token),
        "until_wall": float(until_wall),
        "tokens": int(LABILE_TOKENS),
        "wall": float(LABILE_WALL),
        "phone": PHONE,
        "lure": LURE_PHONE,
        "exact": None if exact is None else exact[0],
        "lure_surface": None if lure is None else lure[0],
        "slip": None if slip is None else slip[0],
        "slip_residue": None if slip is None else slip[1],
    }


def formula_gates() -> dict:
    """Pure-function gates. Live cue / write-back / export are scored in tests."""
    locked = canonical_window()
    closed = not window_open(0, 0.0, 0, 0.0)
    opened = window_open(locked["until_token"], locked["until_wall"], 0, 0.0)
    token_close = not window_open(locked["until_token"], locked["until_wall"], locked["tokens"], 0.0)
    token_open = window_open(locked["until_token"], locked["until_wall"], locked["tokens"] - 1, 0.0)
    wall_close = not window_open(locked["until_token"], locked["until_wall"], 0, locked["wall"])
    wall_open = window_open(locked["until_token"], locked["until_wall"], 0, locked["wall"] - 1.0)
    both_close = not window_open(
        locked["until_token"],
        locked["until_wall"],
        locked["tokens"],
        locked["wall"],
    )
    baseline_slot = _Slot(PHONE)
    baseline = apply_writeback(baseline_slot, 0, 0.0, LURE_PHONE)
    identity = (
        closed
        and not baseline
        and baseline_slot.answer == PHONE
        and locked["exact"] == PHONE
        and locked["until_token"] == int(LABILE_TOKENS)
        and locked["until_wall"] == float(LABILE_WALL)
    )
    live = _Slot(PHONE)
    live.labile_until_token, live.labile_until_wall = open_window(0, 0.0)
    updated = apply_writeback(live, 0, 0.0, LURE_PHONE)
    restabilized = apply_writeback(live, locked["tokens"], 0.0, PHONE)
    hit = opened and locked["tokens"] == 8 and locked["wall"] == 4.0
    update = bool(updated) and live.answer == LURE_PHONE and live.residue == residue_from_answer(LURE_PHONE)
    token = token_open and token_close and not restabilized and live.answer == LURE_PHONE
    wall = wall_open and wall_close
    miss = closed and both_close and reconstructive_writeback(None, 1.0, 0, 0.0, "city") is None
    return {
        "identity": bool(identity),
        "hit": bool(hit),
        "update": bool(update),
        "token": bool(token),
        "wall": bool(wall),
        "miss": bool(miss),
        "window": locked,
        "slip": recall_surface(residue_from_answer(PHONE), 0.5, 0, 0.0),
    }


def bank_sha256() -> str:
    locked = canonical_window()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            PHONE,
            LURE_PHONE,
            str(locked["until_token"]),
            str(locked["until_wall"]),
            str(locked["exact"]),
            str(locked["lure_surface"]),
            str(locked["slip"]),
            str(LABILE_TOKENS),
            str(LABILE_WALL),
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
