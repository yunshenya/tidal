"""Post-retrieval labile window (Nader, Schafe, and LeDoux 2000).

Nader, Schafe, and LeDoux showed that a consolidated fear memory, when
reactivated during retrieval, returns to a labile state that requires
protein synthesis to restabilize. Anisomycin in the amygdala shortly after
reactivation produces later amnesia. The same infusion without
reactivation leaves the memory intact. A delay of six hours after
reactivation produces no amnesia.
https://doi.org/10.1038/35021052

tidal already retrieves one cue winner and reconstructs a phone or
birthday from residue, strength, age, and valence. This module stamps a
short labile window on the matched slot after a cue hit, then lets that
existing reconstructive write-back update or distort the residue while
the window is open. After the window closes, the same write-back no-ops.
It does not add a cue-export field. It is not a growbrain overwrite API.
``remember`` still writes new facts. Sleep prediction-error replay stays
on ``sleep_for``.

Clocks (HYPOTHESIS; no grid was run)
------------------------------------
The window closes when the token budget expires or wall seconds expire.
That is the stronger constraint, the same ``min`` composition the cue
decay already uses. A frozen wall clock leaves the token budget. The
spans are abstract units. They are not the paper's six hours.
"""
from __future__ import annotations

import math
from typing import Optional

from tidal.reconstruct import recall_surface, residue_from_answer

# Tokens after a hit before the window closes. HYPOTHESIS. Not six hours.
LABILE_TOKENS = 8
# Wall seconds after a hit before the window closes. HYPOTHESIS.
LABILE_WALL = 4.0


def _token(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    count = int(value)
    if count < 0:
        raise ValueError(f"{name} must be >= 0")
    return count


def _wall(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


def open_window(
    token_now: int,
    wall_now: float,
    *,
    tokens: int = LABILE_TOKENS,
    wall: float = LABILE_WALL,
) -> tuple:
    """Return ``(until_token, until_wall)`` stamped at retrieval."""
    start_token = _token(token_now, "token")
    start_wall = _wall(wall_now, "wall time")
    span_tokens = _token(tokens, "labile tokens")
    span_wall = _wall(wall, "labile wall")
    if span_tokens < 1:
        raise ValueError("labile tokens must be >= 1")
    if span_wall <= 0.0:
        raise ValueError("labile wall must be > 0")
    return start_token + span_tokens, start_wall + span_wall


def window_open(
    until_token: int,
    until_wall: float,
    token_now: int,
    wall_now: float,
) -> bool:
    """True while both clocks still have remaining labile time.

    A never-opened stamp is ``until_token == 0``. Frozen wall time cannot
    close the window by itself; the token budget still can.
    """
    deadline_token = _token(until_token, "until token")
    deadline_wall = _wall(until_wall, "until wall")
    current_token = _token(token_now, "token")
    current_wall = _wall(wall_now, "wall time")
    if deadline_token <= 0:
        return False
    token_ok = current_token < deadline_token
    wall_ok = current_wall < deadline_wall
    return bool(token_ok and wall_ok)


def matched_indices(triggered, memory_id, candidates) -> tuple:
    """Slot indexes that cleared theta on a triggered token.

    A miss, or a forward with no trigger, is empty. The winner is included
    even when the candidate list is missing it.
    """
    fires = [bool(item) for item in list(triggered)]
    winners = [int(item) for item in list(memory_id)]
    if len(fires) != len(winners):
        raise ValueError("triggered and memory_id must align")
    rows = list(candidates)
    if len(rows) != len(fires):
        raise ValueError("candidates must align with triggered")
    found = []
    seen = set()
    for index, fired in enumerate(fires):
        if not fired:
            continue
        row = []
        winner = int(winners[index])
        if winner >= 0:
            row.append(winner)
        row.extend(int(item) for item in list(rows[index]))
        for item in row:
            if item < 0 or item in seen:
                continue
            seen.add(item)
            found.append(int(item))
    return tuple(found)


def reconstructive_writeback(residue, strength: float, age: int, valence: float, surface: Optional[str] = None):
    """Residue path only. ``None`` when the surface is not a phone or birthday.

    Omitted ``surface`` reads :func:`tidal.reconstruct.recall_surface`. A
    supplied surface must still parse as residue. Keys and values are not
    arguments.
    """
    if surface is None:
        text = recall_surface(residue, float(strength), int(age), float(valence))
    else:
        if not isinstance(surface, str) or surface == "":
            return None
        text = surface
    parts = residue_from_answer(text)
    if parts is None:
        return None
    return text, parts


def apply_writeback(
    slot,
    token_now: int,
    wall_now: float,
    surface: Optional[str] = None,
) -> bool:
    """Write reconstructed or supplied residue onto ``slot`` while labile.

    Keys, values, ``text``, strength, and valence stay. ``answer`` and
    ``residue`` take the new surface. A closed window, or a surface that is
    not reconstructable, is a no-op.
    """
    if not window_open(
        int(getattr(slot, "labile_until_token", 0)),
        float(getattr(slot, "labile_until_wall", 0.0)),
        token_now,
        wall_now,
    ):
        return False
    result = reconstructive_writeback(
        getattr(slot, "residue", None),
        float(slot.strength),
        int(slot.age),
        float(slot.valence),
        surface,
    )
    if result is None:
        return False
    text, parts = result
    slot.answer = text
    slot.residue = parts
    return True
