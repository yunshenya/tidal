"""Dev split for a bare locative that is not a places ask.

``place`` / ``where`` / ``哪里`` / ``位置`` are not lexical cues, so a
question can mention a placement slot and still have no unique cue. The
abstain ridge reads that slot as places. This bank separates that slot from
a places question, and from a placement verb that still has a real cue
(``place a call``, ``交接放在哪里``).

A locative row is the placement verb ``place`` / ``placed`` / ``placing``
(not the noun ``place list``), or ``放在哪里`` / ``往哪里`` / ``朝哪里`` /
``位置先``. A keep row is a places ask worded with ``where``, ``place list``,
``在哪里``, ``位置是哪些``, or ``哪边``, or a cue-free question with none of
those slots. A guarded row matches the locative pattern and still has one
lexical cue, which stays the ask.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, the CJK dev bank, the nickname dev bank, the indirect
dev bank, the typo dev bank, the misleading-cue dev bank, and the frozen
adversarial test. The matcher does not import this module. The frozen test
is not an input to the class.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, CJK, nickname, indirect, cue,
# or adversarial rosters.
NAMES = {"en": ("Sibyl", "Leander", "Quince", "Romilda"), "zh": ("卞珩", "汲棠", "郗宁", "晏书")}

# kind, lang, rel, question. ``locative`` has no cue. ``guarded`` matches the
# slot and keeps one cue. ``keep`` is not the slot. Anchors (``brontide``,
# ``鼗鼙``) are not people.
_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("locative", "en", "birthday", "Sibyl's year-mark sits where the vestry has it placed, brontide pschent."),
    ("locative", "en", "birthday", "Leander grows a year older once that tick is placed, clinquant uraeus."),
    ("locative", "en", "phone", "Quince can be found by voice once I place the chit, sistrum menat."),
    ("locative", "en", "phone", "A filed slip, placed so a voice finds Romilda, reads as what, nefer djed."),
    ("locative", "en", "birthday", "The calendar square for placing Sibyl's added year is which, tyet ankh."),
    ("guarded", "en", "phone", "Place a call for Sibyl and read those digits aloud, was-sceptre scarab."),
    ("guarded", "en", "phone", "After dusk on the quay we place a voice call so Romilda can be raised, and those digits are which, khopesh naos."),
    ("guarded", "en", "places", "Read the places Leander set down, then place the card back, naos pylon."),
    ("keep", "en", "places", "Where ought Quince's courier to stand, earliest halt before the later one, clinquant pschent."),
    ("keep", "en", "places", "Leander left a place list on the inner card, uraeus sistrum."),
    ("keep", "en", "places", "Remind the vestry where Romilda said the meeting was, menat nefer."),
    ("keep", "en", "birthday", "Sibyl adds one year on which dawn, djed tyet."),
    ("keep", "en", "places", "Quince's ordered handoff stops stay on the inner card, brontide uraeus."),
    ("locative", "zh", "birthday", "卞珩那道岁记，执事该把它放在哪里，鼗鼙。"),
    ("locative", "zh", "phone", "汲棠要靠声音找人，该往哪里把条子送去，镛钲。"),
    ("locative", "zh", "phone", "夜里先找郗宁的声音条，位置先搁下，柜侧翻哪一页，鼗镛。"),
    ("locative", "zh", "birthday", "晏书又长一岁，柜侧执事该把那道印放在哪里，鼗鼙镛钲。"),
    ("locative", "zh", "phone", "朝哪里把声音递给卞珩，柜侧蓝册记着哪一条，鼗钲。"),
    ("guarded", "zh", "places", "卞珩的交接要放在哪里，执事照着念，鼗鼙镛。"),
    ("guarded", "zh", "phone", "柜侧要给汲棠打电话，朝哪里递先别管，钲鼗。"),
    ("keep", "zh", "places", "替卞珩露面的那一站，按先来后到该在哪里，鼗鼙镛钲。"),
    ("keep", "zh", "places", "汲棠按先后点过的位置是哪些，钟楼侧柜里那本蓝册子鼗鼙镛钲。"),
    ("keep", "zh", "places", "郗宁平时窝在哪一带，口述次序不要换，鼗钲。"),
    ("keep", "zh", "birthday", "晏书满岁落在哪一个清晨，钟楼侧柜的蓝册子，鼙镛钲。"),
    ("keep", "zh", "places", "钟楼侧柜问的是郗宁的地点是哪里，鼗鼙镛钲。"),
)


@dataclass(frozen=True)
class BareRow:
    kind: str
    lang: str
    rel: str
    question: str


def build_rows() -> Tuple[BareRow, ...]:
    rows = tuple(BareRow(*row) for row in _ROWS)
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate bare-locative dev question")
    return rows


def questions(rows: Sequence[BareRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[BareRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[BareRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
