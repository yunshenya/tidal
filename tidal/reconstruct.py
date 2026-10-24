"""Strength-dependent reconstruction of a phone or a birthday.

The residue is the parsed parts, not the written string. Recall reads those
parts together with strength, age, and valence. A high, recent, calm trace
reassembles the spelling. A weak or old trace drops the peripheral field.
Anything between those bands slips that field's order. The thresholds are a
hypothesis. This module does not read a wall clock, process S, or arousal.

Bartlett's Remembering describes memory as reconstruction.
https://doi.org/10.1017/CBO9780511759185 Brewer and Treyens (1981), Role of
schemata in memory for places. https://doi.org/10.1016/0010-0285(81)90008-6
Kensinger (2009), Remembering the Details: Effects of Emotion.
https://doi.org/10.1177/1754073908100432 The gist-versus-peripheral split
below is a hypothesis mapped onto that paper's unequal, focal account. It is
not a quotation of the abstract.

No torch.
"""
from __future__ import annotations

import math
import re
from typing import Optional

from tidal.llm_memory_data import _MONTHS
from tidal.llm_memory_data import parse_date

# HYPOTHESIS. No grid. Not retuned on a held bank.
EXACT_STRENGTH = 0.8
EXACT_AGE = 0
EXACT_VALENCE = 0.75
LOSS_STRENGTH = 0.4
LONG_AGE = 4
LOSS_LINE_OFFSET = 1111

_PHONE_RE = re.compile(r"^\+1-(\d{3})-(\d{3})-(\d{4})$")


def grade(strength: float, age: int, valence: float) -> str:
    """``exact``, ``loss``, or ``slip``.

    Exact requires strength at least 0.8, age at most 0, and absolute valence
    strictly below 0.75. Loss is strength below 0.4 or age at least 4. Every
    other finite triple is a peripheral order slip. A non-finite input is loss.
    """
    level = float(strength)
    cycles = int(age)
    feeling = float(valence)
    if not math.isfinite(level) or not math.isfinite(feeling):
        return "loss"
    if level >= float(EXACT_STRENGTH) and cycles <= int(EXACT_AGE) and abs(feeling) < float(EXACT_VALENCE):
        return "exact"
    if level < float(LOSS_STRENGTH) or cycles >= int(LONG_AGE):
        return "loss"
    return "slip"


def residue_from_answer(answer: Optional[str]):
    """Parts of a ``+1-AAA-EEE-LLLL`` phone or a ``Month D, YYYY`` date.

    The tuple does not contain the raw string. Anything else, including a
    city name, returns None.
    """
    if not isinstance(answer, str) or answer == "":
        return None
    phone = _PHONE_RE.fullmatch(answer)
    if phone is not None:
        return (
            "phone",
            int(phone.group(1)),
            int(phone.group(2)),
            int(phone.group(3)),
        )
    parsed = parse_date(answer)
    if parsed is None:
        return None
    month, day, year, _spans = parsed
    return ("birthday", int(month), int(day), int(year))


def spell_phone(area: int, exchange: int, line: int) -> str:
    return f"+1-{int(area):03d}-{int(exchange):03d}-{int(line):04d}"


def spell_birthday(month: int, day: int, year: int) -> str:
    return f"{_MONTHS[int(month)]} {int(day)}, {int(year)}"


def slip_line(line: int) -> int:
    """Swap the last two line digits, or add one to the last digit when they match."""
    chars = f"{int(line) % 10000:04d}"
    if chars[-1] != chars[-2]:
        chars = chars[:-2] + chars[-1] + chars[-2]
    else:
        chars = chars[:-1] + str((int(chars[-1]) + 1) % 10)
    return int(chars)


def loss_line(line: int) -> int:
    """A different four-digit line derived from this line alone."""
    original = int(line) % 10000
    shifted = (original + int(LOSS_LINE_OFFSET)) % 10000
    if shifted == original:
        shifted = (original + 1) % 10000
    return int(shifted)


def slip_day(day: int) -> int:
    """Swap the two day digits when that day is still in 1–27 and is different.

    Otherwise step to the neighboring day that stays in 1–27.
    """
    current = int(day)
    if current >= 10:
        swapped = int(f"{current:02d}"[::-1])
        if 1 <= swapped <= 27 and swapped != current:
            return int(swapped)
    if current > 1:
        return int(current - 1)
    return int(current + 1)


def loss_day(day: int) -> int:
    """Drop the day to 1, or to 2 when the day is already 1."""
    return 1 if int(day) != 1 else 2


def _phone_line(residue, mode: str) -> int:
    line = int(residue[3])
    if mode == "exact":
        return line
    if mode == "slip":
        return slip_line(line)
    return loss_line(line)


def _birthday_day(residue, mode: str) -> int:
    day = int(residue[2])
    if mode == "exact":
        return day
    if mode == "slip":
        return slip_day(day)
    return loss_day(day)


def recall_surface(residue, strength: float, age: int, valence: float) -> Optional[str]:
    """Reconstructed spelling, or None when ``residue`` is empty.

    Phone gist is the ``+1`` country code and the exchange. Birthday gist is
    the month and the year. The line and the day are the peripheral fields.
    This split is a hypothesis. The closed digit and date vocabulary has no
    synonym, so the errors are an order slip and a detail loss.
    """
    if not isinstance(residue, tuple) or len(residue) < 2:
        return None
    mode = grade(strength, age, valence)
    kind = residue[0]
    if kind == "phone" and len(residue) == 4:
        return spell_phone(residue[1], residue[2], _phone_line(residue, mode))
    if kind == "birthday" and len(residue) == 4:
        return spell_birthday(residue[1], _birthday_day(residue, mode), residue[3])
    return None


def phone_parts(surface: Optional[str]):
    if not isinstance(surface, str):
        return None
    found = _PHONE_RE.fullmatch(surface)
    if found is None:
        return None
    return int(found.group(1)), int(found.group(2)), int(found.group(3))


def birthday_parts(surface: Optional[str]):
    parsed = parse_date(surface) if isinstance(surface, str) else None
    if parsed is None:
        return None
    month, day, year, _spans = parsed
    return int(month), int(day), int(year)
