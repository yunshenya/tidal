"""Dev split for a two-name ask that restricts the focus.

``only about Name`` / ``just Name`` / ``solely Name``, and ``只问`` / ``只看`` /
``单独问``, name the person the question is about. Another stored name is
still in the sentence, closer to the cue, and may be the only one who stores
the relation. That fact must not answer. An open ``between`` / ``among`` /
``和`` / ``与`` does not restrict. ``not only`` and a coordinated pair do
not either. Dismiss frames from the two-name rule stay controls.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, the frozen adversarial test, the CJK dev bank, the
nickname dev bank, the indirect dev bank, the typo dev bank, the
misleading-cue dev bank, the bare-locative dev bank, the residual
misleading bank, and the two-name dev bank. The matcher does not import
this module. The frozen test is not an input.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, adversarial, CJK, nickname,
# indirect, cue, bare-locative, residual, or two-name rosters.
NAMES = {
    "en": ("Griselda", "Hildebrand", "Cunigunde", "Roderick"),
    "zh": ("夔昭", "厍楠", "仉禾", "爨书"),
}

# Pair 0 is the missing-relation store. Pair 1 is the answerable store.
_PAIRS = {
    "en": (("Griselda", "Hildebrand"), ("Cunigunde", "Roderick")),
    "zh": (("夔昭", "厍楠"), ("仉禾", "爨书")),
}

_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "locations"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
_RELS = ("phone", "birthday", "places")

# The other's stored sentence repeats this clause. The question starts with
# it, so sentence cosine prefers the other person until the restrictor drops
# them. The target's sentence does not share the clause.
_ECHO_TAIL = {
    "en": " and keeps that line on the chit.",
    "zh": "，并在册尾留了底。",
}
_FAR = {
    "en": "A birch tally in the south alcove keeps a private {cue} line for {name}.",
    "zh": "南厢桦笺单记{name}留的{cue}底账。",
}

# kind, lang, pattern, frame. Focus frames start with ``{echo}``, which is
# the clause above. ``{target}`` is the focus. ``{other}`` is the extra name.
_FRAMES: Tuple[Tuple[str, str, str, str], ...] = (
    ("focus_hit", "en", "only_about", "{echo}. A later line names {target}. Cite only about {target}, {anchor}."),
    ("focus_hit", "en", "just_about", "{echo}. A later line names {target}. The night desk reads just about {target}, {anchor}."),
    ("focus_hit", "en", "solely", "{echo}. A later line names {target}. The night desk reads solely {target}, {anchor}."),
    ("focus_hit", "en", "just", "{echo}. A later line names {target}. Read just {target}, {anchor}."),
    ("focus_miss", "en", "only_about", "{echo}. A later line names {target}. Cite only about {target}, {anchor}."),
    ("focus_miss", "en", "just_about", "{echo}. A later line names {target}. The night desk reads just about {target}, {anchor}."),
    ("focus_miss", "en", "solely", "{echo}. A later line names {target}. The night desk reads solely {target}, {anchor}."),
    ("focus_miss", "en", "just", "{echo}. A later line names {target}. Read just {target}, {anchor}."),
    ("focus_hit", "zh", "zhiwen", "{echo}。册尾才写到{target}。只问{target}，{anchor}。"),
    ("focus_hit", "zh", "zhikan", "{echo}。册尾才写到{target}。只看{target}，{anchor}。"),
    ("focus_hit", "zh", "dandu", "{echo}。册尾才写到{target}。单独问{target}，{anchor}。"),
    ("focus_miss", "zh", "zhiwen", "{echo}。册尾才写到{target}。只问{target}，{anchor}。"),
    ("focus_miss", "zh", "zhikan", "{echo}。册尾才写到{target}。只看{target}，{anchor}。"),
    ("focus_miss", "zh", "dandu", "{echo}。册尾才写到{target}。单独问{target}，{anchor}。"),
    ("open_either", "en", "between", "Between {target} and {other}, the tin chit holds a {cue} for each of them, {anchor}."),
    ("open_either", "en", "among", "Among {target} and {other}, both lines on the tin chit carry a {cue}, {anchor}."),
    ("open_miss", "en", "among", "Among {target} and {other}, the tin chit never kept a {cue}, {anchor}."),
    ("open_either", "zh", "he", "{target}和{other}都在锡笺上，{cue}两条都写过，{anchor}。"),
    ("open_either", "zh", "yu", "{target}与{other}并列在册，{cue}各自留了一条，{anchor}。"),
    ("open_miss", "zh", "he", "{target}和{other}都在锡笺上，{cue}一条都没写，{anchor}。"),
    ("dismiss_hit", "en", "skip", "Skip {other} outright. The birch tally keeps the {cue} under {target}, {anchor}."),
    ("dismiss_miss", "en", "skip", "Skip {other} outright. The birch tally keeps the {cue} under {target}, {anchor}."),
    ("dismiss_hit", "zh", "bieti", "别提{other}。桦笺把{cue}留给{target}，{anchor}。"),
    ("dismiss_miss", "zh", "bieti", "别提{other}。桦笺把{cue}留给{target}，{anchor}。"),
    ("coord_either", "en", "coord", "The clerk cites only about {target} and {other}: the tin chit holds a {cue} for the pair, {anchor}."),
    ("coord_either", "zh", "coord", "只问{target}和{other}，锡笺上的{cue}记了两条，{anchor}。"),
    ("not_only_either", "en", "not_only", "not only {target} but {other} as well: the tin chit holds a {cue} for the pair, {anchor}."),
    ("not_only_either", "zh", "not_only", "不只问{target}，{other}也在册，锡笺上的{cue}记了两条，{anchor}。"),
    ("one_hit", "en", "only_about", "The clerk cites only about {target}: the birch tally holds the {cue}, {anchor}."),
    ("one_miss", "en", "only_about", "The clerk cites only about {target}: the birch tally holds the {cue}, {anchor}."),
    ("one_hit", "zh", "zhiwen", "只问{target}，桦笺把{cue}留在哪一条，{anchor}。"),
    ("one_miss", "zh", "zhiwen", "只问{target}，桦笺把{cue}留在哪一条，{anchor}。"),
)


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class FocusRow:
    kind: str
    lang: str
    pattern: str
    rel: str
    target: str
    other: str
    question: str
    expect: str
    golds: Tuple[str, ...]
    slots: Tuple[Slot, ...]


def _value(i: int, rel: str) -> str:
    if rel == "phone":
        return f"+882-640-{4000 + i:04d}"
    if rel == "birthday":
        year = 1633 + (i % 70)
        return f"{year:04d}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
    return f"quill{i}-mord{i}-rest{i}"


def _anchor(i: int, lang: str) -> str:
    if lang == "en":
        return f"quillon{i}"
    return f"黿鼉{i}"


def _echo(lang: str, cue: str, other: str) -> str:
    if lang == "en":
        return f"The tin chit parks the {cue} beside {other}"
    return f"锡笺把{cue}记在{other}旁侧"


def _expect(kind: str) -> str:
    if kind.endswith("_miss"):
        return "missing_rel"
    if kind in ("open_either", "coord_either", "not_only_either"):
        return "either"
    return "answer"


def _people(kind: str, lang: str, rel_i: int) -> Tuple[str, str]:
    if kind == "one_hit":
        return (NAMES[lang][rel_i % 4], "")
    if kind == "one_miss":
        return (NAMES[lang][(rel_i + 1) % 4], "")
    miss = kind.endswith("_miss")
    target, other = _PAIRS[lang][0 if miss else 1]
    return target, other


def _say(lang: str, name: str, cue: str) -> str:
    return _FAR[lang].format(name=name, cue=cue)


def _slots(
    kind: str, lang: str, rel: str, target: str, other: str, n: int, echo: str,
) -> Tuple[Slot, ...]:
    def one(name: str, slot_rel: str, text: str) -> Slot:
        nonlocal n
        n += 1
        return Slot(name, slot_rel, _value(n, slot_rel), text)

    others = [lab for lab in _RELS if lab != rel]
    cue = _CUE[lang][rel]
    if kind == "focus_hit":
        return (
            one(target, rel, _say(lang, target, cue)),
            one(other, rel, echo + _ECHO_TAIL[lang]),
        )
    if kind == "focus_miss":
        return (
            one(target, others[0], _say(lang, target, _CUE[lang][others[0]])),
            one(target, others[1], _say(lang, target, _CUE[lang][others[1]])),
            one(other, rel, echo + _ECHO_TAIL[lang]),
        )
    if kind in ("dismiss_hit", "open_either", "coord_either", "not_only_either"):
        return (
            one(target, rel, _say(lang, target, cue)),
            one(other, rel, _say(lang, other, cue)),
        )
    if kind == "dismiss_miss":
        # The dismissed person stores the asked relation. The focus does not.
        return (
            one(target, others[0], _say(lang, target, _CUE[lang][others[0]])),
            one(target, others[1], _say(lang, target, _CUE[lang][others[1]])),
            one(other, rel, _say(lang, other, cue)),
        )
    if kind == "open_miss":
        return (
            one(target, others[0], _say(lang, target, _CUE[lang][others[0]])),
            one(other, others[1], _say(lang, other, _CUE[lang][others[1]])),
        )
    if kind == "one_hit":
        return (one(target, rel, _say(lang, target, cue)),)
    if kind == "one_miss":
        return (
            one(target, others[0], _say(lang, target, _CUE[lang][others[0]])),
            one(target, others[1], _say(lang, target, _CUE[lang][others[1]])),
        )
    raise RuntimeError(kind)


def build_rows() -> Tuple[FocusRow, ...]:
    rows = []
    serial = 0
    value_i = 0
    for kind, lang, pattern, frame in _FRAMES:
        for rel_i, rel in enumerate(_RELS):
            cue = _CUE[lang][rel]
            target, other = _people(kind, lang, rel_i)
            echo = _echo(lang, cue, other) if kind.startswith("focus_") else ""
            question = frame.format(
                cue=cue,
                target=target,
                other=other,
                anchor=_anchor(serial, lang),
                echo=echo,
            )
            slots = _slots(kind, lang, rel, target, other, value_i, echo)
            value_i += 10
            golds: Tuple[str, ...] = ()
            if _expect(kind) == "answer":
                golds = tuple(slot.value for slot in slots if slot.name == target and slot.rel == rel)
            elif _expect(kind) == "either":
                golds = tuple(slot.value for slot in slots if slot.rel == rel)
            rows.append(FocusRow(
                kind, lang, pattern, rel, target, other, question, _expect(kind), golds, slots,
            ))
            serial += 1
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        dup = sorted({q for q in questions if questions.count(q) > 1})
        raise RuntimeError(f"duplicate focus question: {dup[:2]}")
    return tuple(rows)


def questions(rows: Sequence[FocusRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[FocusRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def bank_strings(rows: Sequence[FocusRow] | None = None) -> Tuple[str, ...]:
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


def kind_counts(rows: Sequence[FocusRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
