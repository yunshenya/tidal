"""Dev split for a last token read as a multiword given name.

``Ms. Crowe`` / ``Crowe`` against stored ``Philomena Crowe``, when Crowe is
not a stored person, is the shape the head currently accepts. The same shape
with the last-name person stored separately is already that person. A full
``Philomena Crowe`` ask has to stay that person, including when ``Crowe`` is
also stored. ``Jane`` stored alone, a diminutive, and a CJK superstring are
controls.

Names and sentences are disjoint from the probe, the surface cue corpus, the
abstain dev bank, the frozen adversarial test, the CJK dev bank, the nickname
dev bank, the indirect dev bank, the typo dev bank, the misleading-cue dev
bank, the bare-locative dev bank, the residual misleading bank, the two-name
dev bank, and the focus dev bank. The matcher does not import this module.
The frozen test is not an input.

Diminutive controls reuse #22 anchors (``Bob`` / ``Robert``). The question
and the stored sentence name the diminutive, not the full anchor, so those
full names are not a lexical hit in this bank.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the banks listed above. Distractors never appear in a question.
NAMES = {
    "en": (
        "Philomena Crowe", "Crowe",
        "Thaddeus Brabazon", "Brabazon",
        "Gwendolyn Hesketh", "Hesketh",
        "Lysander", "Jocasta", "Iphigenia",
    ),
    "zh": ("傅岐", "钟岫", "宋湄", "程岢"),
}

# (first, last, distractor). The last name is also a person in some rows.
_PAIRS = (
    ("Philomena", "Crowe", "Lysander"),
    ("Thaddeus", "Brabazon", "Jocasta"),
    ("Gwendolyn", "Hesketh", "Iphigenia"),
)
_TITLES = ("Ms.", "Mr.", "Mrs.", "Miss")
_RELS = ("phone", "birthday", "places")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "locations"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
# Surname plus given. Surnames are on the closed Baijiaxing list the head uses.
_ZH = (("傅", "岐"), ("钟", "岫"), ("宋", "湄"), ("程", "岢"))


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class Row:
    kind: str
    lang: str
    rel: str
    pattern: str
    who: str
    target: str
    near: str
    question: str
    expect: str
    golds: Tuple[str, ...]
    slots: Tuple[Slot, ...]


def _value(i: int, rel: str) -> str:
    if rel == "phone":
        return f"+47-220-{8000 + i:04d}"
    if rel == "birthday":
        year = 1610 + (i % 50)
        return f"{year:04d}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
    return f"skerry{i}-cairn{i}-tor{i}"


def _other(rel: str) -> str:
    return _RELS[(_RELS.index(rel) + 1) % 3]


class _Build:
    def __init__(self) -> None:
        self.rows: list[Row] = []
        self.n = 0
        self.v = 0

    def _en_q(self, who: str, rel: str) -> str:
        cue = _CUE["en"][rel]
        text = f"Nacreous orrery records {who} beside the {cue}, quern{self.n:02d}."
        self.n += 1
        return text

    def _zh_q(self, who: str, rel: str) -> str:
        cue = _CUE["zh"][rel]
        text = f"齂匣点到{who}，要查的是{cue}，齃{self.n:02d}。"
        self.n += 1
        return text

    def _stmt(self, lang: str, shown: str, rel: str, salt: str) -> str:
        cue = _CUE[lang][rel]
        if lang == "en":
            return f"On skerry tray {self.n:02d}{salt}, nacre holds the {cue} entered for {shown}."
        return f"齂匣{self.n:02d}{salt}把{shown}的{cue}压在齃页下。"

    def _one(self, lang: str, name: str, shown: str, rel: str, salt: str) -> Slot:
        value = _value(self.v, rel)
        self.v += 1
        return Slot(name, rel, value, self._stmt(lang, shown, rel, salt))

    def add(
        self,
        kind: str,
        lang: str,
        rel: str,
        pattern: str,
        who: str,
        target: str,
        near: str,
        expect: str,
        slots: Tuple[Slot, ...],
    ) -> None:
        question = self._en_q(who, rel) if lang == "en" else self._zh_q(who, rel)
        golds: Tuple[str, ...] = ()
        if expect == "answer":
            golds = tuple(slot.value for slot in slots if slot.name == target and slot.rel == rel)
            if len(golds) != 1:
                raise RuntimeError(f"{kind} {who} gold {golds}")
        self.rows.append(Row(
            kind, lang, rel, pattern, who, target, near, question, expect, golds, slots,
        ))


def build_rows() -> Tuple[Row, ...]:
    built = _Build()
    add = built.add

    def pair(i: int):
        first, last, distractor = _PAIRS[i]
        return first, last, f"{first} {last}", distractor

    for pair_i in (0, 1):
        first, last, full, distractor = pair(pair_i)
        for title in _TITLES:
            for rel in _RELS:
                slots = (
                    built._one("en", full, full, rel, "a"),
                    built._one("en", distractor, distractor, _other(rel), "b"),
                )
                add("honor_absent", "en", rel, title, f"{title} {last}", "", full, "absent", slots)
    for pair_i in (0, 1, 2):
        _first, last, full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", full, full, rel, "a"),
                built._one("en", distractor, distractor, _other(rel), "b"),
            )
            add("bare_absent", "en", rel, "bare", last, "", full, "absent", slots)

    first, last, full, distractor = pair(0)
    for title in _TITLES:
        for rel in _RELS:
            slots = (
                built._one("en", full, full, rel, "a"),
                built._one("en", last, last, rel, "b"),
                built._one("en", distractor, distractor, _other(rel), "c"),
            )
            add("honor_sep", "en", rel, title, f"{title} {last}", last, full, "answer", slots)
    for pair_i in (1, 2):
        _first, last, full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", full, full, rel, "a"),
                built._one("en", last, last, rel, "b"),
                built._one("en", distractor, distractor, _other(rel), "c"),
            )
            add("bare_sep", "en", rel, "bare", last, last, full, "answer", slots)

    for pair_i in (0, 1):
        _first, _last, full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", full, full, rel, "a"),
                built._one("en", distractor, distractor, _other(rel), "b"),
            )
            add("full_only", "en", rel, "full", full, full, full, "answer", slots)
    for pair_i in (0, 1):
        _first, last, full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", full, full, rel, "a"),
                built._one("en", last, last, rel, "b"),
                built._one("en", distractor, distractor, _other(rel), "c"),
            )
            add("full_both", "en", rel, "full", full, full, last, "answer", slots)

    for pair_i, title in ((0, ""), (1, "Ms. ")):
        _first, last, _full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", last, last, rel, "a"),
                built._one("en", distractor, distractor, _other(rel), "b"),
            )
            who = f"{title}{last}" if title else last
            pattern = "ms" if title else "bare"
            add("single_hit", "en", rel, pattern, who, last, last, "answer", slots)
    # The other title/person combination, so both last names get both forms.
    for pair_i, title in ((1, ""), (0, "Ms. ")):
        _first, last, _full, distractor = pair(pair_i)
        for rel in _RELS:
            slots = (
                built._one("en", last, last, rel, "a"),
                built._one("en", distractor, distractor, _other(rel), "b"),
            )
            who = f"{title}{last}" if title else last
            pattern = "ms" if title else "bare"
            add("single_hit", "en", rel, pattern, who, last, last, "answer", slots)

    _first, _last, full, distractor = pair(2)
    for rel in _RELS:
        slots = (
            built._one("en", full, full, rel, "a"),
            built._one("en", distractor, distractor, _other(rel), "b"),
        )
        add("given_keep", "en", rel, "given", "Gwendolyn", full, full, "answer", slots)

    # Shown text is the diminutive. The slot name is the #22 anchor.
    nick_hits = (
        ("abby", "Abby", "Abigail"),
        ("mr_mike", "Mr. Mike", "Michael"),
        ("bob", "Bob", "Robert"),
        ("ms_ginny", "Ms. Ginny", "Virginia"),
        ("kate", "Kate", "Katherine"),
    )
    for pattern, who, anchor in nick_hits:
        for rel in ("phone",):
            slots = (
                built._one("en", anchor, who, rel, "a"),
                built._one("en", "Lysander", "Lysander", _other(rel), "b"),
            )
            add("nick_hit", "en", rel, pattern, who, anchor, anchor, "answer", slots)
    nick_miss = (
        ("dot", "Dot", "Dorothy Quinn"),
        ("nell", "Nell", "Helen Grace"),
    )
    for pattern, who, anchor in nick_miss:
        rel = "phone"
        slots = (built._one("en", anchor, who, rel, "a"),)
        add("nick_miss", "en", rel, pattern, who, "", anchor, "absent", slots)

    for i, (sur, given) in enumerate(_ZH):
        full = sur + given
        forms = (
            ("cjk_super", "extra", f"{full}生的", ""),
            ("cjk_super", "prefix", f"老{full}", ""),
            ("cjk_hit", "title", f"{sur}{'先生' if i % 2 == 0 else '女士'}", full),
            ("cjk_hit", "nick", f"小{given}" if i % 2 == 0 else given + given, full),
        )
        for kind, pattern, who, target in forms:
            rel = _RELS[(i + len(built.rows)) % 3]
            slots = (built._one("zh", full, full, rel, "a"),)
            expect = "answer" if kind == "cjk_hit" else "absent"
            near = full
            add(kind, "zh", rel, pattern, who, target, near, expect, slots)

    questions = [row.question for row in built.rows]
    if len(questions) != len(set(questions)):
        dup = sorted({q for q in questions if questions.count(q) > 1})
        raise RuntimeError(f"duplicate last-name question: {dup[:2]}")
    return tuple(built.rows)


def questions(rows: Sequence[Row] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[Row] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def bank_strings(rows: Sequence[Row] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        for text in (row.question, *(slot.text for slot in row.slots)):
            if text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def kind_counts(rows: Sequence[Row] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
