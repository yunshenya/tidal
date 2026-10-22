"""Dev split for suppletive diminutives. Not an input to the frozen test.

The abstain dev bank already labels prefix nicknames (``Harry`` / ``Harold``,
``Ken`` / ``Kenneth``) and rejects a two-letter prefix (``Immy``). It has no
suppletive pair (``Ted`` / ``Edward``). This bank is that pair, both
directions, plus the misses a closed list must not create: a shared
diminutive, a diminutive of one token inside a longer name, a one-edit
absent name, a two-letter form, and a lowercase word.

Chinese given-name nicknames (``小宁``, ``宁宁``, ``阿砚``) already match on
the head. A surname diminutive (``小侯``) is ambiguous when two people share
the surname, so it is not proposed.

Question strings and rosters are disjoint from the probe, the surface cue
corpus, the abstain dev bank, the CJK superstring dev bank, and the frozen
adversarial test. The proposed dictionary names common full forms, including
ones that are stored people on those banks; a diminutive that *is* one of
those people is dropped. The matcher does not import this module.

``quinzaine`` / ``xeraphim`` / ``markNN`` and ``鹾`` / ``齾`` are anchors so
token Jaccard is not carried by shared function words. Not a list of people.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# (full given names, diminutives). Diminutives the filter drops are in here
# on purpose: shorter than 3, another bank's person, one edit of the full name.
PROPOSED: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
    (("robert",), ("bob", "bobby")),
    (("william",), ("bill", "billy", "liam")),
    (("margaret",), ("maggie", "meg", "peggy", "greta")),
    (("richard",), ("dick", "rick")),
    (("james",), ("jim", "jimmy")),
    (("john",), ("jack",)),
    (("edward",), ("ed", "ted", "teddy", "ned")),
    (("theodore",), ("ted", "teddy")),
    (("charles",), ("chuck", "charlie")),
    (("anthony",), ("tony",)),
    (("michael",), ("mike",)),
    (("joseph",), ("joe",)),
    (("catherine", "katherine"), ("kate", "katie", "kathy")),
    (("elizabeth",), ("beth", "betty")),
    (("rebecca",), ("becky",)),
    (("abigail",), ("abby",)),
    (("dorothy",), ("dolly", "dot")),
    (("virginia",), ("ginny",)),
    (("barbara",), ("babs",)),
    (("patricia",), ("tricia",)),
    (("andrew",), ("drew",)),
    (("susan",), ("susie",)),
    (("ann",), ("anne",)),
    (("albert",), ("al",)),
    (("helen",), ("nellie",)),
)
for _anchors, _dims in PROPOSED:
    if not isinstance(_anchors, tuple) or not isinstance(_dims, tuple):
        raise RuntimeError(f"proposed group is not a pair of tuples: {_anchors!r} {_dims!r}")
del _anchors, _dims

Group = Tuple[Tuple[str, ...], Tuple[str, ...]]
Rejection = Tuple[str, str, str]  # diminutive, anchor shown, reason


def _lev(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return abs(len(a) - len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def known_people() -> frozenset:
    """Single-token and full-string names from the banks this split must not use."""
    from tidal.question_abstain_dev_data import BANK as DEV
    from tidal.question_adversarial_data import BANK as TEST
    from tidal.question_cjk_dev_data import NAMES as CJK
    from tidal.question_heldout_data import ABSENT, PEOPLE
    from tidal.question_surface import ABSENT_NAMES, DEV_NAMES, FIT_NAMES

    names = set(CJK)
    for lang in ("en", "zh"):
        for bank in (PEOPLE, ABSENT, FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
            names.update(bank[lang])
        names.update(DEV["stored"][lang])
        names.update(DEV["absent"][lang])
        names.update(TEST["stored"][lang])
        names.update(TEST["absent"][lang])
    out = set()
    for name in names:
        out.add(name.lower())
        parts = re.findall(r"[A-Za-z]+", name)
        if len(parts) == 1:
            out.add(parts[0].lower())
    return frozenset(out)


def _reject(form: str, anchors: Sequence[str], known: frozenset) -> str:
    if len(form) < 3:
        return "shorter than 3"
    if form in known and form not in anchors:
        return "known person"
    for anchor in anchors:
        if form != anchor and abs(len(form) - len(anchor)) <= 1 and _lev(form, anchor) == 1:
            return f"edit-1 of {anchor}"
    return ""


def rejections(known: frozenset | None = None) -> Tuple[Rejection, ...]:
    """Diminutives the filter drops, in proposed order."""
    if known is None:
        known = known_people()
    dropped: list[Rejection] = []
    for anchors, dims in PROPOSED:
        for form in dims:
            reason = _reject(form, anchors, known)
            if reason:
                dropped.append((form, anchors[0], reason))
    return tuple(dropped)


def selected_groups(known: frozenset | None = None) -> Tuple[Group, ...]:
    """Groups that still have a diminutive after the filter."""
    if known is None:
        known = known_people()
    kept: list[Group] = []
    for anchors, dims in PROPOSED:
        forms = tuple(form for form in dims if not _reject(form, anchors, known))
        if forms:
            kept.append((anchors, forms))
    return tuple(kept)


@dataclass(frozen=True)
class NickRow:
    kind: str
    question: str
    roster: Tuple[str, ...]
    expect: Tuple[str, ...]


# kind, surface, roster, expect. The surface is the only title-case name.
# Rosters are not people from the probe, surface, abstain dev, CJK dev, or
# the frozen test. Anchors that *are* those people (Robert, Catherine, …)
# stay in the dictionary and are checked as pairs, not as questions.
_EN_SPEC: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("positive", "Ted", ("Edward", "Richard", "James"), ("Edward",)),
    ("positive", "Teddy", ("Edward", "Richard"), ("Edward",)),
    ("positive", "Ned", ("Edward", "Joseph"), ("Edward",)),
    ("reverse", "Edward", ("Ted", "Chuck", "Dolly"), ("Ted",)),
    ("honorific", "Mr. Tony", ("Anthony", "Michael"), ("Anthony",)),
    ("possessive", "Becky's", ("Rebecca", "Abigail"), ("Rebecca",)),
    ("positive", "Charlie", ("Charles", "James"), ("Charles",)),
    ("positive", "Chuck", ("Charles", "John"), ("Charles",)),
    ("positive", "Dolly", ("Dorothy", "Susan"), ("Dorothy",)),
    ("positive", "Dot", ("Dorothy", "Susan"), ("Dorothy",)),
    ("prefix_kept", "Kate", ("Katherine", "Rebecca"), ("Katherine",)),
    ("prefix_kept", "Kathy", ("Katherine", "Barbara"), ("Katherine",)),
    ("prefix_kept", "Katie", ("Katherine", "Abigail"), ("Katherine",)),
    ("positive", "Joe", ("Joseph", "Andrew"), ("Joseph",)),
    ("positive", "Jack", ("John", "James"), ("John",)),
    ("positive", "Dick", ("Richard", "Edward"), ("Richard",)),
    ("positive", "Rick", ("Richard", "Charles"), ("Richard",)),
    ("positive", "Jim", ("James", "John"), ("James",)),
    ("positive", "Jimmy", ("James", "Michael"), ("James",)),
    ("positive", "Susie", ("Susan", "Dorothy"), ("Susan",)),
    ("positive", "Drew", ("Andrew", "Joseph"), ("Andrew",)),
    ("positive", "Babs", ("Barbara", "Virginia"), ("Barbara",)),
    ("positive", "Abby", ("Abigail", "Rebecca"), ("Abigail",)),
    ("positive", "Ginny", ("Virginia", "Barbara"), ("Virginia",)),
    ("positive", "Becky", ("Rebecca", "Abigail"), ("Rebecca",)),
    ("positive", "Mike", ("Michael", "Anthony"), ("Michael",)),
    ("honorific", "Ms. Ginny", ("Virginia", "Susan"), ("Virginia",)),
    ("possessive", "Tony's", ("Anthony", "Michael"), ("Anthony",)),
    ("negative", "Ed", ("Edward", "Richard"), ()),
    ("negative", "Anne", ("Edward", "Charles"), ()),
    ("negative", "Edwardo", ("Edward", "James"), ()),
    ("negative", "Charls", ("Charles", "John"), ()),
    ("ambiguous", "Ted", ("Edward", "Theodore", "James"), ()),
    ("compound", "Dolly", ("Dorothy Quinn", "James"), ()),
    ("compound", "Dot", ("Dorothy Quinn",), ()),
    ("compound", "Nell", ("Helen Grace",), ()),
    ("negative", "Dick", ("Edward", "James"), ()),
)

_ZH_SPEC: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("zh_keep", "小宁。鹾齾册", ("侯宁", "侯柏", "蒋砚"), ("侯宁",)),
    ("zh_keep", "宁宁。鹾齾页", ("侯宁", "侯柏", "蒋砚"), ("侯宁",)),
    ("zh_keep", "阿砚。鹾齾条", ("侯宁", "侯柏", "蒋砚"), ("蒋砚",)),
    ("zh_drop", "小侯。鹾齾卷", ("侯宁", "侯柏"), ()),
    ("zh_drop", "老侯宁。鹾齾夹", ("侯宁",), ()),
)


def build_rows() -> Tuple[NickRow, ...]:
    rows = []
    for i, (kind, surface, roster, expect) in enumerate(_EN_SPEC, 1):
        question = f"quinzaine xeraphim slip lists {surface} beside mark{i:02d}."
        rows.append(NickRow(kind, question, roster, expect))
    # Lowercase word, isolated so the span is the word and not a glued phrase.
    rows.append(NickRow(
        "negative",
        "Quinzaine. ted. Xeraphim marklc.",
        ("Edward", "Richard"),
        (),
    ))
    for kind, question, roster, expect in _ZH_SPEC:
        rows.append(NickRow(kind, question, roster, expect))
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate nickname-dev question")
    return tuple(rows)


def questions(rows: Sequence[NickRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[NickRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[NickRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
