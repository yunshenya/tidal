"""Dev split for two-name asks whose target is not the cue-nearest person.

A question can name two stored people. The ask target is the person outside
a dismissed clause, or the person who possessively governs the live cue
(``Name's cue``, ``Name的cue``, ``is Name's``, ``是Name的``). The other
person may sit closer to the cue word and may store the asked relation.
That fact must not answer. When the target stores the relation, the answer
is the target's fact.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, the frozen adversarial test, the CJK dev bank, the
nickname dev bank, the indirect dev bank, the typo dev bank, the
misleading-cue dev bank, the bare-locative dev bank, and the residual
misleading bank. The matcher does not import this module. The frozen test
is not an input.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, adversarial, CJK, nickname,
# indirect, cue, bare-locative, or residual rosters.
NAMES = {
    "en": ("Cressida", "Balthasar", "Honoria", "Peregrine"),
    "zh": ("嵇沅", "逯荞", "酆棠", "禚岚"),
}
ABSENT = {"en": "Winifred", "zh": "扈茗"}

# Pair A is the missing-relation store. Pair B is the answerable store.
_PAIRS = {
    "en": (("Cressida", "Balthasar"), ("Peregrine", "Honoria")),
    "zh": (("嵇沅", "逯荞"), ("禚岚", "酆棠")),
}

_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "locations"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
_RELS = ("phone", "birthday", "places")

_STMT = {
    "en": {
        "phone": "A cedar slip in the west loft names the handset kept for {name}.",
        "birthday": "A cedar slip in the west loft names the morning {name} adds a year.",
        "places": "A cedar slip in the west loft lists the halts {name} dictated, earliest first.",
    },
    "zh": {
        "phone": "西柜杉笺记着{name}要留的手线。",
        "birthday": "西柜杉笺圈着{name}添一岁的清晨。",
        "places": "西柜杉笺按先后来记下{name}歇过的站。",
    },
}

# kind, lang, frame. ``{cue}`` is the relation word, ``{Cue}`` its capital
# (English) or the same word (Chinese). ``{target}`` / ``{other}`` are the pair.
_FRAMES: Tuple[Tuple[str, str, str], ...] = (
    ("dismiss_miss", "en", "Skip {other}. {Cue} kept in the cedar book is the line under {target}'s seal, {anchor}."),
    ("dismiss_miss", "en", "Ignore {other} completely. Which {cue} does the night clerk read for {target}, {anchor}."),
    ("dismiss_miss", "en", "Forget {other}. The {cue} on the cedar slip is the one under {target}, {anchor}."),
    ("dismiss_miss", "en", "Exclude {other} from the tally. The {cue} the cedar book cites is {target}'s, {anchor}."),
    ("dismiss_hit", "en", "Skip {other}. {Cue} kept in the cedar book is the line under {target}'s seal, {anchor}."),
    ("dismiss_hit", "en", "Ignore {other} completely. Which {cue} does the night clerk read for {target}, {anchor}."),
    ("dismiss_hit", "en", "Forget {other}. The {cue} on the cedar slip is the one under {target}, {anchor}."),
    ("dismiss_hit", "en", "Exclude {other} from the tally. The {cue} the cedar book cites is {target}'s, {anchor}."),
    ("poss_miss", "en", "Compared with {other}, the {cue} the cedar book keeps is {target}'s, {anchor}."),
    ("poss_miss", "en", "The {cue} for {other} sits in the margin while the cedar book keeps the line as {target}'s, {anchor}."),
    ("poss_hit", "en", "Compared with {other}, the {cue} the cedar book keeps is {target}'s, {anchor}."),
    ("poss_hit", "en", "The {cue} for {other} sits in the margin while the cedar book keeps the line as {target}'s, {anchor}."),
    ("one_hit", "en", "On the cedar slip, {target}'s {cue} is the string to cite, {anchor}."),
    ("one_hit", "en", "Read {target}'s {cue} off the cedar slip, {anchor}."),
    ("one_miss", "en", "The cedar slip never stored a {cue} for {target}, {anchor}."),
    ("one_miss", "en", "No {cue} was entered for {target} on the cedar slip, {anchor}."),
    ("one_dismiss_hit", "en", "Ignore {target}'s supper. Cite the {cue} the cedar slip holds, {anchor}."),
    ("absent", "en", "Which {cue} was written down for {absent} rather than {other}, {anchor}."),
    ("open_either", "en", "{target} alongside {other}, and the cedar book holds a {cue} for each, {anchor}."),
    ("dismiss_miss", "zh", "别提{other}。{cue}记在杉笺上最终算{target}的，{anchor}。"),
    ("dismiss_miss", "zh", "跳过{other}。{cue}最终落到{target}名下，{anchor}。"),
    ("dismiss_hit", "zh", "别提{other}。{cue}记在杉笺上最终算{target}的，{anchor}。"),
    ("dismiss_hit", "zh", "跳过{other}。{cue}最终落到{target}名下，{anchor}。"),
    ("poss_miss", "zh", "跟{other}相比，{cue}写了很长以后才点明是{target}的，{anchor}。"),
    ("poss_miss", "zh", "和{other}对照，{cue}翻过几页才算定{target}的，{anchor}。"),
    ("poss_hit", "zh", "跟{other}相比，{cue}写了很长以后才点明是{target}的，{anchor}。"),
    ("poss_hit", "zh", "和{other}对照，{cue}翻过几页才算定{target}的，{anchor}。"),
    ("one_hit", "zh", "杉笺里{target}的{cue}记在哪一条，{anchor}。"),
    ("one_miss", "zh", "杉笺里还欠{target}的哪一条{cue}，{anchor}。"),
    ("one_dismiss_hit", "zh", "别提{target}的晚饭。{cue}记在杉笺哪一条，{anchor}。"),
    ("absent", "zh", "{cue}记给{absent}，不是记给{other}，{anchor}。"),
    ("open_either", "zh", "{target}跟{other}一并在册，{cue}两条都写过，{anchor}。"),
)


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class TwoRow:
    kind: str
    lang: str
    rel: str
    target: str
    other: str
    question: str
    expect: str
    golds: Tuple[str, ...]
    slots: Tuple[Slot, ...]


def _value(i: int, rel: str) -> str:
    if rel == "phone":
        return f"+881-700-{3000 + i:04d}"
    if rel == "birthday":
        year = 1711 + (i % 80)
        return f"{year:04d}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
    return f"brine{i}-keel{i}-wale{i}"


def _anchor(i: int, lang: str) -> str:
    if lang == "en":
        return f"brinekeel{i}"
    # Two rare characters, then a digit, so token Jaccard is not all function words.
    return f"廌廑{i}"


def _expect(kind: str) -> str:
    if kind.endswith("_miss") or kind == "one_miss":
        return "missing_rel"
    if kind == "absent":
        return "absent_person"
    if kind == "open_either":
        return "either"
    return "answer"


def _people(kind: str, lang: str, rel_i: int) -> Tuple[str, str]:
    """Target, other. Missing rows use pair A. Answerable rows use pair B."""
    if kind in ("one_hit", "one_dismiss_hit"):
        return (NAMES[lang][rel_i % 4], "")
    if kind == "one_miss":
        return (NAMES[lang][(rel_i + 1) % 4], "")
    if kind == "absent":
        return ("", NAMES[lang][0])
    miss = kind.endswith("_miss")
    target, other = _PAIRS[lang][0 if miss else 1]
    return target, other


def _slots(kind: str, lang: str, rel: str, target: str, other: str, n: int) -> Tuple[Slot, ...]:
    def one(name: str, slot_rel: str) -> Slot:
        nonlocal n
        n += 1
        value = _value(n, slot_rel)
        return Slot(name, slot_rel, value, _STMT[lang][slot_rel].format(name=name))

    others = [lab for lab in _RELS if lab != rel]
    if kind in ("dismiss_miss", "poss_miss"):
        return (one(target, others[0]), one(target, others[1]), one(other, rel))
    if kind in ("dismiss_hit", "poss_hit", "open_either"):
        return (one(target, rel), one(other, rel))
    if kind in ("one_hit", "one_dismiss_hit"):
        return (one(target, rel),)
    if kind == "one_miss":
        return (one(target, others[0]), one(target, others[1]))
    if kind == "absent":
        return (one(other, rel),)
    raise RuntimeError(kind)


def build_rows() -> Tuple[TwoRow, ...]:
    rows = []
    serial = 0
    seen = {lang: {kind: 0 for kind, _lang, _frame in _FRAMES if _lang == lang} for lang in ("en", "zh")}
    # Count how many times each (kind, lang, frame) pattern is emitted: once per relation.
    for kind, lang, frame in _FRAMES:
        for rel_i, rel in enumerate(_RELS):
            cue = _CUE[lang][rel]
            target, other = _people(kind, lang, rel_i + seen[lang][kind])
            absent = ABSENT[lang] if kind == "absent" else ""
            # one_* uses a different person per relation so the four names all appear.
            if kind in ("one_hit", "one_miss", "one_dismiss_hit"):
                target = NAMES[lang][(rel_i + (0 if "hit" in kind else 1) + (2 if kind == "one_dismiss_hit" else 0)) % 4]
                other = ""
            question = frame.format(
                cue=cue,
                Cue=cue[:1].upper() + cue[1:] if lang == "en" else cue,
                target=target,
                other=other,
                absent=absent,
                anchor=_anchor(serial, lang),
            )
            slots = _slots(kind, lang, rel, target, other if kind != "absent" else other, serial * 10)
            if kind == "absent":
                target = ""
            golds: Tuple[str, ...] = ()
            if _expect(kind) == "answer":
                golds = tuple(slot.value for slot in slots if slot.name == target and slot.rel == rel)
            elif _expect(kind) == "either":
                golds = tuple(slot.value for slot in slots if slot.rel == rel)
            rows.append(TwoRow(
                kind, lang, rel, target, other, question, _expect(kind), golds, slots,
            ))
            serial += 1
        seen[lang][kind] += 1
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        dup = sorted({q for q in questions if questions.count(q) > 1})
        raise RuntimeError(f"duplicate two-name question: {dup[:2]}")
    return tuple(rows)


def questions(rows: Sequence[TwoRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[TwoRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def bank_strings(rows: Sequence[TwoRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        texts = [row.question, *(slot.text for slot in row.slots)]
        for text in texts:
            if text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def kind_counts(rows: Sequence[TwoRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
