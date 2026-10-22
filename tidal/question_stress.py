"""Shared builder for abstain stress sets. CPU only. No model.

``question_adversarial_data`` is the frozen test. ``question_abstain_dev_data``
is the only split a later change may consult. Neither module is imported by
``question_surface`` or by the ranking fit. Strings are measured against the
probe and the surface corpus for literal equality and token Jaccard here;
sentence cosine is measured by ``question_adversarial`` once vectors exist.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tidal.question_heldout_data import ABSENT as PROBE_ABSENT
from tidal.question_heldout_data import PEOPLE as PROBE_PEOPLE
from tidal.question_heldout_data import _tokens, probe_strings
from tidal.question_surface import corpus_strings, lex_labels, name_in_text, names_used

RELS: Tuple[str, ...] = ("phone", "birthday", "places")
LABELS: Tuple[str, ...] = ("answerable", "missing_rel", "absent_person")


@dataclass(frozen=True)
class StressWrite:
    text: str
    value: str
    name: str
    rel: str


@dataclass(frozen=True)
class StressCase:
    split: str
    slice: str
    lang: str
    rel: str
    label: str
    subtype: str
    question: str
    golds: Tuple[str, ...]
    focus: str
    canonical: str
    other: str
    expect_lex: Tuple[str, ...]
    pool: str
    writes: Tuple[StressWrite, ...]


class _Ids:
    def __init__(self) -> None:
        self.n = 0

    def value(self, rel: str) -> str:
        i = self.n
        self.n += 1
        if rel == "phone":
            return f"+1-917-555-{1000 + i:04d}"
        if rel == "birthday":
            return f"{1930 + (i % 70):04d}-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}"
        return f"north{i}-ridge{i}-ford{i}"


def _stmt(bank: dict, lang: str, name: str, rel: str) -> str:
    return bank["stmt"][lang][rel].format(name=name)


def _pick(people: Sequence[str], i: int, avoid: Iterable[str] = ()) -> str:
    banned = set(avoid)
    n = len(people)
    for k in range(n):
        person = people[(i + k) % n]
        if person not in banned:
            return person
    raise RuntimeError("roster exhausted")


def _writes_answerable(bank, lang, canonical, rel, ids: _Ids, salt: int):
    people = bank["stored"][lang]
    gold = ids.value(rel)
    writes = [StressWrite(_stmt(bank, lang, canonical, rel), gold, canonical, rel)]
    others = [_pick(people, salt + j, (canonical,)) for j in range(3)]
    writes.append(StressWrite(_stmt(bank, lang, others[0], rel), ids.value(rel), others[0], rel))
    for j, person in enumerate(others[1:]):
        other_rel = RELS[(RELS.index(rel) + 1 + j) % 3]
        writes.append(StressWrite(
            _stmt(bank, lang, person, other_rel), ids.value(other_rel), person, other_rel,
        ))
    return (gold,), writes


def _writes_missing(bank, lang, canonical, rel, ids: _Ids, salt: int):
    people = bank["stored"][lang]
    writes = []
    for other_rel in RELS:
        if other_rel == rel:
            continue
        writes.append(StressWrite(
            _stmt(bank, lang, canonical, other_rel), ids.value(other_rel), canonical, other_rel,
        ))
    other = _pick(people, salt, (canonical,))
    writes.append(StressWrite(_stmt(bank, lang, other, rel), ids.value(rel), other, rel))
    return (), writes


def _writes_similar(bank, lang, similar, rel, ids: _Ids, salt: int):
    """Store the near person (with the asked relation) and two other people."""
    people = bank["stored"][lang]
    writes = [StressWrite(_stmt(bank, lang, similar, rel), ids.value(rel), similar, rel)]
    for j in range(2):
        person = _pick(people, salt + j, (similar,))
        other_rel = RELS[(RELS.index(rel) + j) % 3]
        writes.append(StressWrite(
            _stmt(bank, lang, person, other_rel), ids.value(other_rel), person, other_rel,
        ))
    return (), writes


def _cue(bank, lang, rel) -> str:
    return bank["cue"][lang][rel]


def _add(out: List[StressCase], bank, **kwargs) -> None:
    writes = tuple(kwargs.pop("writes"))
    out.append(StressCase(
        split=bank["split"],
        golds=tuple(kwargs.pop("golds")),
        writes=writes,
        pool=kwargs.pop("pool", ""),
        other=kwargs.pop("other", ""),
        **kwargs,
    ))


def build_cases(bank: dict) -> Tuple[StressCase, ...]:
    """Deterministic cases from a phrase bank. No draw."""
    ids = _Ids()
    out: List[StressCase] = []
    salt = 0
    ridge_n = 0

    def next_pool(slice_name: str, subtype: str) -> str:
        """Even ridge frames are fit, odd ones are select. Both labels share a frame."""
        nonlocal ridge_n
        if bank["split"] != "dev":
            return ""
        ridge = slice_name in ("relation_paraphrase", "bare_cue") or (
            slice_name == "compact_form" and subtype == "nocue"
        )
        if not ridge:
            return ""
        pool = "fit" if ridge_n % 2 == 0 else "select"
        ridge_n += 1
        return pool

    for lang in ("en", "zh"):
        people = bank["stored"][lang]
        for rel in RELS:
            for subtype, frame in bank["paraphrase"][lang][rel]:
                pool = next_pool("relation_paraphrase", subtype)
                for label_i, label in enumerate(("answerable", "missing_rel")):
                    who = _pick(people, salt)
                    salt += 1
                    question = frame.format(who=who)
                    if label == "answerable":
                        golds, writes = _writes_answerable(bank, lang, who, rel, ids, salt)
                    else:
                        golds, writes = _writes_missing(bank, lang, who, rel, ids, salt)
                    _add(
                        out, bank, slice="relation_paraphrase", lang=lang, rel=rel, label=label,
                        subtype=subtype, question=question, golds=golds, focus=who, canonical=who,
                        expect_lex=(), pool=pool, writes=writes,
                    )
            for subtype, frame in bank["compact"][lang][rel]:
                expect = (rel,) if subtype == "lexical" else ()
                pool = next_pool("compact_form", subtype)
                for label in ("answerable", "missing_rel"):
                    who = _pick(people, salt)
                    salt += 1
                    question = frame.format(who=who)
                    if label == "answerable":
                        golds, writes = _writes_answerable(bank, lang, who, rel, ids, salt)
                    else:
                        golds, writes = _writes_missing(bank, lang, who, rel, ids, salt)
                    _add(
                        out, bank, slice="compact_form", lang=lang, rel=rel, label=label,
                        subtype=subtype, question=question, golds=golds, focus=who, canonical=who,
                        expect_lex=expect, pool=pool, writes=writes,
                    )
            for frame in bank["bare"][lang][rel]:
                pool = next_pool("bare_cue", "bare")
                for label in ("answerable", "missing_rel"):
                    who = _pick(people, salt)
                    salt += 1
                    question = frame.format(who=who)
                    if label == "answerable":
                        golds, writes = _writes_answerable(bank, lang, who, rel, ids, salt)
                    else:
                        golds, writes = _writes_missing(bank, lang, who, rel, ids, salt)
                    _add(
                        out, bank, slice="bare_cue", lang=lang, rel=rel, label=label,
                        subtype="bare", question=question, golds=golds, focus=who, canonical=who,
                        expect_lex=(), pool=pool, writes=writes,
                    )
        for cue_rel, true_rel, frame in bank["mislead"][lang]:
            for label in ("answerable", "missing_rel"):
                who = _pick(people, salt)
                salt += 1
                question = frame.format(who=who)
                if label == "answerable":
                    # True relation only. A unique wrong cue then false-abstains.
                    golds, writes = _writes_answerable(bank, lang, who, true_rel, ids, salt)
                else:
                    # Every relation except the one asked. The misleading cue is one of those.
                    golds, writes = _writes_missing(bank, lang, who, true_rel, ids, salt)
                    if (who, cue_rel) not in {(row.name, row.rel) for row in writes}:
                        raise RuntimeError(f"misleading store dropped the cue {cue_rel} for {who}")
                _add(
                    out, bank, slice="misleading_cue", lang=lang, rel=true_rel, label=label,
                    subtype=f"cue_{cue_rel}", question=question, golds=golds, focus=who,
                    canonical=who, expect_lex=(cue_rel,), pool="", writes=writes,
                )
        for vi, (canonical, subtype, surface) in enumerate(bank["variants"][lang]):
            rel = RELS[vi % 3]
            cue = _cue(bank, lang, rel)
            if subtype == "possessive":
                frame = bank["possessive_q"][lang]
            else:
                frame = bank["variant_q"][lang]
            question = frame.format(who=surface, cue=cue)
            golds, writes = _writes_answerable(bank, lang, canonical, rel, ids, salt)
            salt += 1
            _add(
                out, bank, slice="name_variant", lang=lang, rel=rel, label="answerable",
                subtype=subtype, question=question, golds=golds, focus=surface,
                canonical=canonical, expect_lex=(rel,), pool="", writes=writes,
            )
        for vi, (similar, subtype, surface) in enumerate(bank["near"][lang]):
            rel = RELS[vi % 3]
            cue = _cue(bank, lang, rel)
            question = bank["near_q"][lang].format(who=surface, cue=cue)
            golds, writes = _writes_similar(bank, lang, similar, rel, ids, salt)
            salt += 1
            _add(
                out, bank, slice="near_dup_absent", lang=lang, rel=rel, label="absent_person",
                subtype=subtype, question=question, golds=golds, focus=surface,
                canonical=similar, expect_lex=(rel,), pool="", writes=writes,
            )
        absent_names = bank["absent"][lang]
        for rel in RELS:
            for fi, frame in enumerate(bank["two_both"][lang]):
                focus = _pick(people, salt)
                other = _pick(people, salt + 4, (focus,))
                salt += 1
                cue = _cue(bank, lang, rel)
                question = frame.format(focus=focus, other=other, cue=cue)
                # answerable: both have the relation. Gold is the focus.
                g_focus = ids.value(rel)
                g_other = ids.value(rel)
                writes = [
                    StressWrite(_stmt(bank, lang, focus, rel), g_focus, focus, rel),
                    StressWrite(_stmt(bank, lang, other, rel), g_other, other, rel),
                ]
                third = _pick(people, salt + 2, (focus, other))
                other_rel = RELS[(RELS.index(rel) + 1) % 3]
                writes.append(StressWrite(
                    _stmt(bank, lang, third, other_rel), ids.value(other_rel), third, other_rel,
                ))
                _add(
                    out, bank, slice="two_names", lang=lang, rel=rel, label="answerable",
                    subtype="both_have", question=question, golds=(g_focus,), focus=focus,
                    canonical=focus, other=other, expect_lex=(rel,), pool="", writes=writes,
                )
                # missing: focus lacks it, the other name has it.
                focus_m = _pick(people, salt + 1)
                other_m = _pick(people, salt + 5, (focus_m,))
                question_m = frame.format(focus=focus_m, other=other_m, cue=cue)
                writes_m = []
                for other_rel in RELS:
                    if other_rel == rel:
                        continue
                    writes_m.append(StressWrite(
                        _stmt(bank, lang, focus_m, other_rel), ids.value(other_rel), focus_m, other_rel,
                    ))
                writes_m.append(StressWrite(
                    _stmt(bank, lang, other_m, rel), ids.value(rel), other_m, rel,
                ))
                _add(
                    out, bank, slice="two_names", lang=lang, rel=rel, label="missing_rel",
                    subtype="other_has", question=question_m, golds=(), focus=focus_m,
                    canonical=focus_m, other=other_m, expect_lex=(rel,), pool="", writes=writes_m,
                )
            for fi, frame in enumerate(bank["two_absent"][lang]):
                stored = _pick(people, salt)
                absent = absent_names[(salt + fi) % len(absent_names)]
                salt += 1
                cue = _cue(bank, lang, rel)
                question = frame.format(absent=absent, stored=stored, cue=cue)
                golds, writes = _writes_similar(bank, lang, stored, rel, ids, salt)
                _add(
                    out, bank, slice="two_names", lang=lang, rel=rel, label="absent_person",
                    subtype="absent_mentioned_with_stored", question=question, golds=golds,
                    focus=absent, canonical="", other=stored, expect_lex=(rel,), pool="",
                    writes=writes,
                )
        for subtype, rels in bank["deixis"][lang].items():
            for rel, frames in rels.items():
                for frame in frames:
                    question = frame
                    if subtype == "pronoun_unique":
                        canonical, golds, writes = _deictic_unique(bank, lang, rel, ids)
                    elif subtype == "pronoun_ambiguous":
                        canonical, golds, writes = _deictic_ambiguous(bank, lang, rel, ids)
                    else:
                        canonical, golds, writes = _deictic_persona(bank, lang, rel, ids)
                    _add(
                        out, bank, slice="deixis", lang=lang, rel=rel, label="answerable",
                        subtype=subtype, question=question, golds=golds, focus="",
                        canonical=canonical, expect_lex=(rel,),
                        pool="", writes=writes,
                    )
    return tuple(out)


def _deictic_persona(bank, lang, rel, ids: _Ids):
    people = bank["stored"][lang]
    persona = people[0]
    gold = ids.value(rel)
    text = bank["deixis_stmt"][lang][rel]
    writes = [StressWrite(text, gold, persona, rel)]
    for j, person in enumerate(people[1:4]):
        other_rel = rel if j == 0 else RELS[(RELS.index(rel) + j) % 3]
        writes.append(StressWrite(
            _stmt(bank, lang, person, other_rel), ids.value(other_rel), person, other_rel,
        ))
    return persona, (gold,), writes


def _deictic_unique(bank, lang, rel, ids: _Ids):
    persona = bank["stored"][lang][0]
    writes = []
    gold = None
    for other_rel in RELS:
        val = ids.value(other_rel)
        if other_rel == rel:
            gold = val
        writes.append(StressWrite(_stmt(bank, lang, persona, other_rel), val, persona, other_rel))
    return persona, (gold,), writes


def _deictic_ambiguous(bank, lang, rel, ids: _Ids):
    people = bank["stored"][lang]
    first, second, third = people[0], people[1], people[2]
    v0 = ids.value(rel)
    v1 = ids.value(rel)
    other_rel = RELS[(RELS.index(rel) + 1) % 3]
    writes = [
        StressWrite(_stmt(bank, lang, first, rel), v0, first, rel),
        StressWrite(_stmt(bank, lang, second, rel), v1, second, rel),
        StressWrite(_stmt(bank, lang, third, other_rel), ids.value(other_rel), third, other_rel),
    ]
    # Either stored person is an acceptable decode. Abstaining is not.
    return second, (v0, v1), writes


def case_strings(cases: Sequence[StressCase]) -> Tuple[str, ...]:
    seen = []
    bag = set()
    for case in cases:
        texts = [case.question]
        texts.extend(row.text for row in case.writes)
        for text in texts:
            if text and text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def questions_sha256(cases: Sequence[StressCase]) -> str:
    blob = "\n".join(case.question for case in cases).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def reference_names() -> Tuple[str, ...]:
    names = list(names_used())
    for lang in ("en", "zh"):
        names.extend(PROBE_PEOPLE[lang])
        names.extend(PROBE_ABSENT[lang])
    return tuple(dict.fromkeys(names))


def _stored_hit(name: str, text: str) -> bool:
    return name_in_text(name, text)


def validate_cases(cases: Sequence[StressCase], *, other: Sequence[StressCase] = ()) -> Tuple[str, ...]:
    """Structural problems. Empty means the split is usable.

    Literal collisions with the probe, the surface corpus, and ``other``
    (the sibling split) are errors. Token Jaccard is reported by
    ``jaccard_report``, not failed here, because short questions share
    function tokens with ordinary English and Chinese.
    """
    errors = []
    if not cases:
        return ("empty stress split",)
    questions = [case.question for case in cases]
    if len(questions) != len(set(questions)):
        dup = sorted({q for q in questions if questions.count(q) > 1})
        errors.append(f"duplicate questions: {dup[:3]}")
    texts = case_strings(cases)
    if len(texts) != len(set(texts)):
        errors.append("duplicate surface string inside the split")
    ref = set(probe_strings())
    ref.update(corpus_strings())
    ref.update(case_strings(other))
    for text in texts:
        if text in ref:
            errors.append(f"literal collision: {text}")
            break
    banned = set(reference_names())
    if other:
        for row in other:
            if row.canonical:
                banned.add(row.canonical)
            for write in row.writes:
                banned.add(write.name)
    split_names = {write.name for case in cases for write in case.writes}
    for name in split_names:
        if name in banned:
            errors.append(f"stored name collides with probe, surface, or the other split: {name}")
    seen_focus = {}
    for case in cases:
        if case.label not in LABELS:
            errors.append(f"bad label {case.label}")
        if case.question.strip() != case.question or not case.question:
            errors.append(f"blank question {case.slice}")
        ids = {(row.name, row.rel): row.value for row in case.writes}
        names = {row.name for row in case.writes}
        if len(ids) != len(case.writes):
            errors.append(f"duplicate slot in {case.slice} {case.question}")
        for row in case.writes:
            if row.value in case.question or row.value in row.text:
                errors.append(f"value leaked: {case.question}")
                break
        if case.label == "answerable":
            if (case.canonical, case.rel) not in ids:
                errors.append(f"answerable missing gold slot: {case.question}")
            elif ids[(case.canonical, case.rel)] not in case.golds:
                errors.append(f"gold not stored: {case.question}")
            if not case.golds:
                errors.append(f"answerable without gold: {case.question}")
        elif case.label == "missing_rel":
            if case.canonical not in names or (case.canonical, case.rel) in ids:
                errors.append(f"missing_rel store is wrong: {case.question}")
            if case.golds:
                errors.append(f"missing_rel has a gold: {case.question}")
        else:
            if case.focus in names:
                errors.append(f"absent focus is stored: {case.question} {case.focus}")
            if case.golds:
                errors.append(f"absent has a gold: {case.question}")
        if case.slice in ("relation_paraphrase", "compact_form", "bare_cue", "misleading_cue"):
            if case.focus and case.focus not in case.question:
                errors.append(f"focus missing from question: {case.question}")
        if case.slice == "deixis":
            for name in names:
                if _stored_hit(name, case.question):
                    errors.append(f"deixis question names {name}: {case.question}")
                    break
        if case.slice == "two_names":
            if case.label == "absent_person":
                if case.focus not in case.question or (case.other and case.other not in case.question):
                    errors.append(f"two-name absent question dropped a name: {case.question}")
            elif case.focus not in case.question or case.other not in case.question:
                errors.append(f"two-name question dropped a name: {case.question}")
        got_lex = lex_labels(case.question)
        if got_lex != case.expect_lex:
            errors.append(
                f"lex {got_lex} != {case.expect_lex} on {case.split} {case.slice} "
                f"{case.subtype}: {case.question}"
            )
        if case.slice == "name_variant" and case.focus not in case.question:
            errors.append(f"variant missing: {case.question}")
        if case.slice == "near_dup_absent" and case.focus not in case.question:
            errors.append(f"near name missing: {case.question}")
        for name in banned:
            if _stored_hit(name, case.question):
                errors.append(f"question contains a foreign name {name}: {case.question}")
                break
        seen_focus.setdefault(case.slice, 0)
        seen_focus[case.slice] += 1
    for slice_name, n in seen_focus.items():
        if n < 8:
            errors.append(f"{cases[0].split} {slice_name} has only {n}")
    return tuple(errors)


def jaccard_report(
    cases: Sequence[StressCase],
    *,
    other: Sequence[StressCase] = (),
) -> dict:
    """Max token Jaccard of this split's strings against probe, surface, and ``other``."""
    left = case_strings(cases)
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
    }
    if other:
        bags["other_split"] = case_strings(other)
    report = {"n_strings": len(left)}
    for name, right in bags.items():
        best = 0.0
        pair = ("", "")
        over_50 = 0
        n = 0
        for a in left:
            for b in right:
                n += 1
                score = _jaccard(a, b)
                if score > 0.5:
                    over_50 += 1
                if score > best:
                    best = score
                    pair = (a, b)
        report[name] = {
            "max": best,
            "pair": pair,
            "pairs_above_0.5": over_50,
            "comparisons": n,
        }
    literal = []
    ref = set(probe_strings()) | set(corpus_strings())
    if other:
        ref |= set(case_strings(other))
    for text in left:
        if text in ref:
            literal.append(text)
    report["literal_hits"] = literal
    return report


def summary(cases: Sequence[StressCase]) -> dict:
    slices: Dict[str, int] = {}
    by_lang: Dict[str, int] = {}
    by_label: Dict[str, int] = {}
    cells: Dict[str, int] = {}
    subtypes: Dict[str, int] = {}
    for case in cases:
        slices[case.slice] = slices.get(case.slice, 0) + 1
        by_lang[case.lang] = by_lang.get(case.lang, 0) + 1
        by_label[case.label] = by_label.get(case.label, 0) + 1
        key = f"{case.slice}|{case.lang}|{case.label}"
        cells[key] = cells.get(key, 0) + 1
        sub = f"{case.slice}|{case.lang}|{case.subtype}|{case.label}"
        subtypes[sub] = subtypes.get(sub, 0) + 1
    return {
        "n": len(cases),
        "sha256": questions_sha256(cases),
        "slices": slices,
        "by_lang": by_lang,
        "by_label": by_label,
        "cells": cells,
        "subtypes": subtypes,
    }
