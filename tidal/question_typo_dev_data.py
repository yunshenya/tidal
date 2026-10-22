"""Dev split for a one-character typo versus an absent near-duplicate.

The abstain dev bank (sha256 26889048…) uses one surface as both an
answerable typo and an absent person: ``Harols``, ``Juduth``, ``Marcas``,
``Anna Louize``, ``吴笛``, ``郑恺``, ``徐锋``. The frozen adversarial test
does the same for ``黄蕾``, ``周滔``, and ``陈靖``. A name rule that sees
only the surface and the stored roster cannot return both labels.

This bank rebuilds that collision with new people, and adds the cases a
store-context gate *can* separate: two stored names at edit distance 1
(abstain), and a longer span that still contains the stored name (abstain).
English closers are three frames. Chinese closers are ``时`` / ``的话`` /
``，`` / ``呢``. The matcher does not import this module. No rule is fit
here. The frozen test is not an input.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# kind, lang, surface, roster, closers.
# Collision kinds are emitted twice, once as match and once as abstain,
# with the same surface and the same roster. The closer is the only difference
# inside a label, and it is not a name feature.
_COLLISION = frozenset({
    "collision_adj",
    "collision_mn",
    "collision_vowel",
    "collision_del",
    "collision_homo",
    "collision_diffpy",
})

_Spec = Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]]

_SPECS: Tuple[_Spec, ...] = (
    # s↔d is the abstain-dev pair (Harols / Harold). m↔n is the other adjacent
    # pair. Both are a typo and an absent name. A denylist of the first pair
    # does not make the second pair a free typo.
    ("collision_adj", "en", "Hollid", ("Hollis", "Quillon", "Vesper"), ("slip", "ledger", "chit")),
    ("collision_mn", "en", "Branwell", ("Bramwell", "Isolde", "Lorcan"), ("slip", "ledger", "chit")),
    ("collision_vowel", "en", "Bramwall", ("Bramwell", "Barnaby", "Romilly"), ("slip", "ledger", "chit")),
    ("collision_del", "en", "Pembrok", ("Pembroke", "Marlow", "Quillon"), ("slip", "ledger", "chit")),
    ("collision_homo", "zh", "阮峥", ("阮铮", "庞岚", "闵廷"), ("shi", "dehua", "comma", "ne")),
    ("collision_diffpy", "zh", "庞琨", ("庞岚", "闵廷", "鲍麒"), ("shi", "dehua", "comma", "ne")),
    ("tie", "en", "Hollid", ("Hollis", "Hollin", "Quillon"), ("slip", "ledger", "chit")),
    ("tie", "zh", "阮峥", ("阮铮", "阮崎", "庞岚"), ("shi", "dehua", "comma", "ne")),
    ("hyphen", "en", "Hollis-Jane", ("Hollis", "Isolde", "Lorcan"), ("slip", "ledger", "chit")),
    ("space", "en", "Hollis Jane", ("Hollis", "Isolde", "Lorcan"), ("slip", "ledger", "chit")),
    ("extension", "en", "Bramwello", ("Bramwell", "Calder", "Winifred"), ("slip", "ledger", "chit")),
    ("prefix", "zh", "老阮铮", ("阮铮", "庞岚", "闵廷"), ("shi", "dehua", "comma", "ne")),
    ("middle", "zh", "阮小铮", ("阮铮", "庞岚", "闵廷"), ("shi", "dehua", "comma", "ne")),
    ("superstring", "zh", "阮铮然", ("阮铮", "庞岚", "闵廷"), ("shi", "dehua", "comma", "ne")),
)

_EN_FRAME = {
    "slip": "brumaire xylograph slip lists {who} beside {mark}.",
    "ledger": "brumaire xylograph ledger under {who} keeps {mark}.",
    "chit": "The brumaire chit for {who} is xylograph {mark}.",
}
_ZH_FRAME = {
    "shi": "鼷册写着{who}时，鼢页空着，参见鼩鼹{mark}。",
    "dehua": "若鼷册是{who}的话，鼢页没抄，参见鼩鼹{mark}。",
    "comma": "{who}，鼷鼢页仍空，参见鼩鼹{mark}。",
    "ne": "{who}呢，鼷页先翻鼢，参见鼩鼹{mark}。",
}


@dataclass(frozen=True)
class TypoRow:
    kind: str
    lang: str
    label: str
    closer: str
    surface: str
    roster: Tuple[str, ...]
    question: str


def _frame(lang: str, closer: str, who: str, mark: str) -> str:
    template = _EN_FRAME[closer] if lang == "en" else _ZH_FRAME[closer]
    return template.format(who=who, mark=mark)


def build_rows() -> Tuple[TypoRow, ...]:
    rows = []
    n = 0
    for kind, lang, surface, roster, closers in _SPECS:
        labels = ("match", "abstain") if kind in _COLLISION else ("abstain",)
        for label in labels:
            for closer in closers:
                n += 1
                question = _frame(lang, closer, surface, f"q{n:02d}")
                rows.append(TypoRow(kind, lang, label, closer, surface, roster, question))
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate typo-dev question")
    return tuple(rows)


def questions(rows: Sequence[TypoRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[TypoRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[TypoRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts


def label_counts(rows: Sequence[TypoRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return counts
