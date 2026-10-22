"""Dev split for indirect hearer descriptions. Not an input to the frozen test.

An indirect description (``the hearer``, ``眼前这位``) is not a name. With
only flat episodic facts, Continuum has no speaker and no addressee. The
policy this bank labels:

* Resolve when exactly one stored person has a first-person surface
  (``I`` / ``my``, ``我`` not ``我们``). That person is the dialogue self.
  The same convention is what ``you`` / ``你`` already assume: the
  participant's fact is stored in the first person.
* Do not resolve when nobody is first person, or two people are. A single
  third-person participant is not enough. The last write is not the
  referent. The only person who holds the asked relation is not the
  referent when the self lacks it (``missing_rel``).
* A stored name in the question wins over the description. An unmatched
  name blocks it.
* A neighbor, a clerk, a bare ``this person``, ``隔壁那位``, ``另一位``
  are not hearer descriptions.

``hearer`` and ``listener`` are the role nouns checked here. ``addressee``
and ``interlocutor`` are the same class and are not question strings in
this file. ``跟前这位`` and ``对面那位`` are not question strings either;
the locatives checked here are ``眼前`` / ``面前`` / ``身边`` / ``身旁`` /
``对过``. The matcher does not import this module.

Question strings and rosters are disjoint from the probe, the surface cue
corpus, the abstain dev bank, the frozen adversarial test, the CJK dev
bank, the nickname dev bank, and the typo dev bank.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

RELS: Tuple[str, ...] = ("phone", "birthday", "places")
SELF = {"en": "Saffi", "zh": "倪棠"}
OTHER = {"en": "Torvald", "zh": "卞菽"}
THIRD = {"en": "Eulalie", "zh": "糜涧"}
ABSENT = {"en": "Ysolde", "zh": "郗茉"}

_EN_CUE = {"phone": "telephone", "birthday": "birthday", "places": "ordered locations"}
_ZH_CUE_BODY = {
    "phone": "的电话写在哪一格",
    "birthday": "的生日落在哪一天",
    "places": "常去的地方有哪些",
}

# Positive descriptions. Not the frozen strings and not the abstain-dev strings.
_EN_HEARER: Tuple[Tuple[str, str], ...] = (
    ("hearer", "the hearer"),
    ("listener", "the listener"),
    ("spoken_with", "the person being spoken with"),
    ("talked_to", "the person being talked to"),
    ("across_table", "the one across the table"),
    ("party_opposite", "the party opposite"),
    ("someone_opposite", "someone sitting opposite"),
    ("this_person_front", "this person in front"),
    ("person_facing", "the person facing"),
    ("talked_with", "the one being talked with"),
)
_ZH_HEARER: Tuple[Tuple[str, str], ...] = (
    ("yanqian_zhe", "眼前这位"),
    ("mianqian_na", "面前那位"),
    ("shenbian_ren", "身边的人"),
    ("shenpang_zhe", "身旁这位"),
    ("duiguo_na", "对过那位"),
    ("yanqian_ren", "眼前的人"),
    ("mianqian_de_zhe", "面前的这位"),
    ("shenbian_na", "身边那位"),
)
# Unique self in the store, and the description is still not a hearer.
_EN_OTHER: Tuple[Tuple[str, str], ...] = (
    ("neighbor", "the neighbor"),
    ("clerk", "the clerk"),
    ("guest", "the guest"),
    ("stranger", "the stranger"),
    ("colleague", "the colleague"),
    ("courier", "the courier"),
    ("this_person_bare", "this person"),
    ("someone_bare", "someone"),
    ("person_bare", "the person"),
)
_ZH_OTHER: Tuple[Tuple[str, str], ...] = (
    ("gebi", "隔壁那位"),
    ("lingyi", "另一位"),
    ("bare_na", "那位"),
    ("pangbian", "旁边的人"),
    ("tongshi", "同事"),
    ("keren", "客人"),
    ("zhiban", "值班的人"),
)


@dataclass(frozen=True)
class IndirectRow:
    lang: str
    rel: str
    setup: str
    kind: str
    label: str
    referent: str
    question: str
    names: Tuple[str, ...]
    rels: Tuple[str, ...]
    surfaces: Tuple[str, ...]


def _side(rel: str) -> str:
    return RELS[(RELS.index(rel) + 1) % 3]


def _en_surf(name: str, rel: str, style: str) -> str:
    # Wording is not the frozen first-person statements ("the morning I grow
    # a year older", "the handset I actually answer").
    if style == "self":
        body = {
            "phone": "I filed my own callback under the mark",
            "birthday": "I circled my own natal dawn on the mark",
            "places": "I recited my own halt sequence on the mark",
        }[rel]
        return f"Nephoscope ledger: {body}, yaffingale."
    if style == "self_b":
        body = {
            "phone": "I logged my own callback and I kept the copy",
            "birthday": "I marked my own natal dawn and I kept the copy",
            "places": "I recited my own halt sequence and I kept the copy",
        }[rel]
        return f"Yaffingale slate: {body}, nephoscope."
    body = {
        "phone": f"the callback kept under {name} stays on the yaffingale card",
        "birthday": f"the natal dawn kept under {name} stays on the yaffingale card",
        "places": f"the halt sequence kept under {name} stays on the yaffingale card",
    }[rel]
    return f"Nephoscope ledger: {body}."


def _zh_surf(name: str, rel: str, style: str) -> str:
    if style == "self":
        body = {
            "phone": "我自己的回拨串",
            "birthday": "我自己的生辰",
            "places": "我自己的歇脚顺序",
        }[rel]
        return f"龖页写着{body}，爩边没有第二份。"
    if style == "self_b":
        body = {
            "phone": "我记下的回拨串，我自己留的",
            "birthday": "我记下的生辰，我自己留的",
            "places": "我记下的歇脚顺序，我自己留的",
        }[rel]
        return f"爩页写着{body}，龖边另存。"
    body = {
        "phone": f"{name}的回拨串",
        "birthday": f"{name}的生辰",
        "places": f"{name}的歇脚顺序",
    }[rel]
    return f"龖页写着{body}，爩边没有第二份。"


def _surf(lang: str, name: str, rel: str, style: str) -> str:
    if lang == "en":
        return _en_surf(name, rel, style)
    return _zh_surf(name, rel, style)


def _unique(lang: str, rel: str) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], str]:
    """Third person, then the self, then the other person last with the same relation."""
    self_n, other, third = SELF[lang], OTHER[lang], THIRD[lang]
    side = _side(rel)
    names = (third, self_n, other)
    rels = (side, rel, rel)
    surfaces = (
        _surf(lang, third, side, "third"),
        _surf(lang, self_n, rel, "self"),
        _surf(lang, other, rel, "third"),
    )
    return names, rels, surfaces, self_n


def _en_q(desc: str, rel: str, mark: int) -> str:
    return (
        f"Which {_EN_CUE[rel]} is filed for {desc}? "
        f"nephoscope yaffingale m{mark:02d}"
    )


def _zh_q(desc: str, rel: str, mark: int) -> str:
    return f"龖册只问{desc}{_ZH_CUE_BODY[rel]}，爩页别混。m{mark:02d}"


def _hearer_desc(lang: str, kind: str) -> str:
    table = _EN_HEARER if lang == "en" else _ZH_HEARER
    return dict(table)[kind]


def build_rows() -> Tuple[IndirectRow, ...]:
    """Deterministic rows. No draw. The frozen test is not an input."""
    rows: list[IndirectRow] = []
    mark = 0

    def add(lang, rel, setup, kind, label, referent, question, names, rels, surfaces) -> None:
        nonlocal mark
        mark += 1
        if len(names) != len(rels) or len(names) != len(surfaces):
            raise RuntimeError(f"slot mismatch on {kind}")
        rows.append(IndirectRow(
            lang, rel, setup, kind, label, referent, question, names, rels, surfaces,
        ))

    for lang, table in (("en", _EN_HEARER), ("zh", _ZH_HEARER)):
        for i, (kind, desc) in enumerate(table):
            rel = RELS[i % 3]
            names, rels, surfaces, referent = _unique(lang, rel)
            question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
            add(lang, rel, "unique_self", kind, "answerable", referent, question, names, rels, surfaces)

    # Same hearer wording, different stores. One phrasing per language.
    for lang, kind in (("en", "hearer"), ("zh", "yanqian_zhe")):
        desc = _hearer_desc(lang, kind)
        self_n, other = SELF[lang], OTHER[lang]
        # No first person. Both people have the relation. Last write is not a self.
        rel = "phone"
        names = (self_n, other)
        rels = (rel, rel)
        surfaces = (
            _surf(lang, self_n, rel, "third"),
            _surf(lang, other, rel, "third"),
        )
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "no_self", kind, "absent_person", "", question, names, rels, surfaces)
        # Two first-person people.
        rel = "birthday"
        names = (self_n, other)
        rels = (rel, rel)
        surfaces = (
            _surf(lang, self_n, rel, "self"),
            _surf(lang, other, rel, "self_b"),
        )
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "two_selves", kind, "absent_person", "", question, names, rels, surfaces)
        # The only stored person is third person.
        rel = "places"
        names = (other,)
        rels = (rel,)
        surfaces = (_surf(lang, other, rel, "third"),)
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "single_third", kind, "absent_person", "", question, names, rels, surfaces)
        # The only stored person is first person.
        rel = "phone"
        names = (self_n,)
        rels = (rel,)
        surfaces = (_surf(lang, self_n, rel, "self"),)
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "single_self", kind, "answerable", self_n, question, names, rels, surfaces)
        # Self lacks the asked relation. The other person, written last, has it.
        rel = "phone"
        side = "birthday"
        names = (self_n, other)
        rels = (side, rel)
        surfaces = (
            _surf(lang, self_n, side, "self"),
            _surf(lang, other, rel, "third"),
        )
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "missing_rel", kind, "missing_rel", self_n, question, names, rels, surfaces)
        # Self holds two relations. The other person, last, holds the asked one too.
        rel = "birthday"
        names = (self_n, self_n, other)
        rels = ("phone", "birthday", "birthday")
        surfaces = (
            _surf(lang, self_n, "phone", "self"),
            _surf(lang, self_n, "birthday", "self"),
            _surf(lang, other, "birthday", "third"),
        )
        question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
        add(lang, rel, "multi_fact", kind, "answerable", self_n, question, names, rels, surfaces)

    # A stored name in the question beats the hearer description.
    for lang in ("en", "zh"):
        named, dialogue = SELF[lang], OTHER[lang]
        rel = "phone"
        names = (named, dialogue)
        rels = (rel, rel)
        surfaces = (
            _surf(lang, named, rel, "third"),
            _surf(lang, dialogue, rel, "self"),
        )
        if lang == "en":
            question = (
                f"Which telephone is filed for {named}, not the hearer? "
                f"nephoscope yaffingale m{mark + 1:02d}"
            )
        else:
            question = (
                f"龖册只问{named}的电话写在哪一格，眼前这位另说，爩页别混。m{mark + 1:02d}"
            )
        add(lang, rel, "name_wins", "name_wins", "answerable", named, question, names, rels, surfaces)
        # An unmatched name blocks the description even though a unique self exists.
        self_n, other = SELF[lang], OTHER[lang]
        names = (self_n, other)
        rels = (rel, rel)
        surfaces = (
            _surf(lang, self_n, rel, "self"),
            _surf(lang, other, rel, "third"),
        )
        absent = ABSENT[lang]
        if lang == "en":
            question = (
                f"Which telephone is filed for {absent}, not the hearer? "
                f"nephoscope yaffingale m{mark + 1:02d}"
            )
        else:
            question = (
                f"龖册先看，{absent}，再问眼前这位的电话写在哪一格，爩页别混。m{mark + 1:02d}"
            )
        add(
            lang, rel, "absent_name", "absent_name", "absent_person", "",
            question, names, rels, surfaces,
        )

    for lang, table in (("en", _EN_OTHER), ("zh", _ZH_OTHER)):
        for i, (kind, desc) in enumerate(table):
            rel = RELS[i % 3]
            names, rels, surfaces, _self = _unique(lang, rel)
            question = _en_q(desc, rel, mark + 1) if lang == "en" else _zh_q(desc, rel, mark + 1)
            add(
                lang, rel, "unique_self", kind, "absent_person", "",
                question, names, rels, surfaces,
            )

    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate indirect-dev question")
    return tuple(rows)


def questions(rows: Sequence[IndirectRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def bank_strings(rows: Sequence[IndirectRow] | None = None) -> Tuple[str, ...]:
    """Questions and stored surfaces. Disjointness uses this set."""
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        for text in (row.question, *row.surfaces):
            if text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def questions_sha256(rows: Sequence[IndirectRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def bank_sha256(rows: Sequence[IndirectRow] | None = None) -> str:
    """Sha of questions, labels, and store slots. A store edit moves this."""
    rows = build_rows() if rows is None else rows
    lines = []
    for row in rows:
        lines.append("\t".join((
            row.lang,
            row.rel,
            row.setup,
            row.kind,
            row.label,
            row.referent,
            row.question,
            ",".join(row.names),
            ",".join(row.rels),
            " || ".join(row.surfaces),
        )))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def label_counts(rows: Sequence[IndirectRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return counts


def setup_counts(rows: Sequence[IndirectRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.setup] = counts.get(row.setup, 0) + 1
    return counts
