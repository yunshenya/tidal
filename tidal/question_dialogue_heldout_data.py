"""Held-out multi-turn episodes for dialogue-state name resolution.

A second en+zh bank, separate from the dialogue dev split. The carrier
is ``yelm stook … rickleNNN`` plus two extension-A marks. People, typos,
and sentence frames are not taken from that split.

Recent means the intended person was named in the last utterance, or in
the one before it (no filler, or one filler). That must be answered.
The same typo with an empty log, a different person, a write and no
utterance, four fillers, or two active edit-1 names must abstain.
Filler counts 2, 3, 5, 6, and 7 are recorded and are not a pass/fail
gate: they sit on the window boundary.

Stdlib only. The name matcher is not imported here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class Episode:
    kind: str
    lang: str
    rel: str
    label: str
    empty_label: str
    question: str
    golds: Tuple[str, ...]
    empty_golds: Tuple[str, ...]
    slots: Tuple[Slot, ...]
    turns: Tuple[str, ...]
    prior_asks: Tuple[str, ...]
    who: str
    target: str
    role: str
    lag: int


_EN = (
    ("Leofric", "Leofrid", "Dunstan"),
    ("Wulfgar", "Wulfger", "Kenelm"),
    ("Hereward", "Herewerd", "Tostig"),
    ("Eadbald", "Eadbold", "Guthrum"),
    ("Ceolwulf", "Ceolwelf", "Coenwulf"),
    ("Raedwald", "Raedweld", "Wihtred"),
)
_ZH = (
    ("嵇珂", "嵇柯", "芮馥"),
    ("裴琮", "裴棕", "糜瞳"),
    ("邢澜", "邢谰", "巫窈"),
    ("翁峙", "翁痔", "汲昶"),
    ("荀珀", "荀粕", "邴霓"),
    ("靳茗", "靳铭", "段麓"),
)
_NICK_EN = (
    ("Margaret", "Maggie", "Dunstan"),
    ("Charles", "Chuck", "Kenelm"),
    ("Michael", "Mike", "Tostig"),
    ("Andrew", "Drew", "Guthrum"),
)
_NICK_ZH = (
    ("嵇珂", "小珂", "芮馥"),
    ("裴琮", "阿琮", "糜瞳"),
)
_RELS = ("phone", "birthday", "places")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
_NEXT = {"phone": "birthday", "birthday": "places", "places": "phone"}


class _Bank:
    def __init__(self) -> None:
        self._n = 0
        self._v = 0
        self.rows: list = []

    def line(self, body: str) -> str:
        self._n += 1
        return f"yelm stook {body} rickle{self._n:03d} 㐂㐃"

    def val(self) -> str:
        self._v += 1
        return f"zyl{self._v:04d}"

    def fact(self, lang: str, rel: str, name: str, *, first: bool = False) -> str:
        cue = _CUE[lang][rel]
        if first and lang == "en":
            return self.line(f"I logged the {cue} on my own slip")
        if first:
            return self.line(f"我记下 {cue}")
        if lang == "en":
            return self.line(f"the dock slip stores the {cue} of {name}")
        return self.line(f"码头存条 记下 {name} {cue}")

    def mention(self, lang: str, name: str) -> str:
        if lang == "en":
            return self.line(f"a margin note names {name}")
        return self.line(f"页边标注 {name}")

    def quiet(self, lang: str) -> str:
        if lang == "en":
            return self.line("the drawer stays closed")
        return self.line("抽屉关闭")

    def ask(self, lang: str, rel: str, who: str) -> str:
        cue = _CUE[lang][rel]
        if lang == "en":
            return self.line(f"read the {cue} filed beside {who}")
        return self.line(f"读出 {who} 旁侧留存 {cue}")

    def emit(
        self,
        *,
        kind: str,
        lang: str,
        rel: str,
        label: str,
        empty_label: str,
        question: str,
        golds: Tuple[str, ...],
        empty_golds: Tuple[str, ...],
        slots: Tuple[Slot, ...],
        turns: Tuple[str, ...] = (),
        prior_asks: Tuple[str, ...] = (),
        who: str,
        target: str,
        role: str,
        lag: int,
    ) -> None:
        self.rows.append(Episode(
            kind=kind,
            lang=lang,
            rel=rel,
            label=label,
            empty_label=empty_label,
            question=question,
            golds=golds,
            empty_golds=empty_golds,
            slots=slots,
            turns=turns,
            prior_asks=prior_asks,
            who=who,
            target=target,
            role=role,
            lag=lag,
        ))

    def pair_store(self, lang: str, canon: str, other: str, rel: str):
        other_rel = _NEXT[rel]
        kept = self.val()
        decoy = self.val()
        slots = (
            Slot(canon, rel, kept, self.fact(lang, rel, canon)),
            Slot(other, other_rel, decoy, self.fact(lang, other_rel, other)),
        )
        return slots, kept

    def same_store(self, lang: str, left: str, right: str, rel: str):
        a = self.val()
        b = self.val()
        slots = (
            Slot(left, rel, a, self.fact(lang, rel, left)),
            Slot(right, rel, b, self.fact(lang, rel, right)),
        )
        return slots, a, b

    def typo(
        self,
        lang: str,
        canon: str,
        typo: str,
        other: str,
        rel: str,
        *,
        kind: str,
        role: str,
        label: str,
        lag: int,
        turns: Tuple[str, ...],
        prior: Tuple[str, ...] = (),
    ) -> None:
        slots, kept = self.pair_store(lang, canon, other, rel)
        golds: Tuple[str, ...] = (kept,) if label in ("answerable", "observe") else ()
        self.emit(
            kind=kind,
            lang=lang,
            rel=rel,
            label=label,
            empty_label="absent_person",
            question=self.ask(lang, rel, typo),
            golds=golds,
            empty_golds=(),
            slots=slots,
            turns=turns,
            prior_asks=prior,
            who=typo,
            target=canon,
            role=role,
            lag=lag,
        )

    def after(self, lang: str, canon: str, other: str, fillers: int) -> Tuple[str, ...]:
        turns = [self.mention(lang, canon)]
        turns.extend(self.mention(lang, other) for _ in range(fillers))
        return tuple(turns)


def _assemble() -> Tuple[Episode, ...]:
    bank = _Bank()
    packs = (("en", _EN), ("zh", _ZH))
    for lang, pairs in packs:
        for i, (canon, typo, other) in enumerate(pairs):
            rel = _RELS[i % 3]
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_hit", role="hit", label="answerable", lag=0,
                turns=bank.after(lang, canon, other, 0),
            )
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_hit", role="hit", label="answerable", lag=1,
                turns=bank.after(lang, canon, other, 1),
            )
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_silent", role="silent", label="absent_person", lag=-1,
                turns=(),
            )
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_other", role="other", label="absent_person", lag=-1,
                turns=(bank.mention(lang, other),),
            )
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_aged", role="aged", label="absent_person", lag=4,
                turns=bank.after(lang, canon, other, 4),
            )
            prior = (bank.ask(lang, rel, canon),)
            bank.typo(
                lang, canon, typo, other, rel,
                kind="typo_prior", role="hit", label="answerable", lag=0,
                turns=(), prior=prior,
            )
        canon, typo, other = pairs[0]
        for fillers in (2, 3, 5, 6, 7):
            rel = _RELS[fillers % 3]
            bank.typo(
                lang, canon, typo, other, rel,
                kind="lag_observe", role="lag", label="observe", lag=fillers,
                turns=bank.after(lang, canon, other, fillers),
            )

    for lang, (left, right) in (
        ("en", ("Godwune", "Godwene")),
        ("zh", ("庾珀", "庾霓")),
    ):
        surface = "Godwine" if lang == "en" else "庾茗"
        slots, _, _ = bank.same_store(lang, left, right, "phone")
        bank.emit(
            kind="typo_tie", lang="en" if lang == "en" else "zh", rel="phone",
            label="absent_person", empty_label="absent_person",
            question=bank.ask(lang, "phone", surface),
            golds=(), empty_golds=(), slots=slots,
            turns=(bank.mention(lang, left), bank.mention(lang, right)),
            who=surface, target="", role="tie", lag=1,
        )

    for lang, pairs in packs:
        for i in (0, 1):
            canon, _typo, other = pairs[i]
            rel = _RELS[i]
            slots, kept = bank.pair_store(lang, canon, other, rel)
            bank.emit(
                kind="spelling", lang=lang, rel=rel,
                label="answerable", empty_label="answerable",
                question=bank.ask(lang, rel, canon),
                golds=(kept,), empty_golds=(kept,), slots=slots,
                who=canon, target=canon, role="pos", lag=-1,
            )

    for i, (full, dimin, other) in enumerate(_NICK_EN):
        rel = _RELS[i % 3]
        slots, kept = bank.pair_store("en", full, other, rel)
        bank.emit(
            kind="nickname", lang="en", rel=rel,
            label="answerable", empty_label="answerable",
            question=bank.ask("en", rel, dimin),
            golds=(kept,), empty_golds=(kept,), slots=slots,
            who=dimin, target=full, role="pos", lag=-1,
        )
    for i, (full, dimin, other) in enumerate(_NICK_ZH):
        rel = _RELS[i % 3]
        slots, kept = bank.pair_store("zh", full, other, rel)
        bank.emit(
            kind="nickname", lang="zh", rel=rel,
            label="answerable", empty_label="answerable",
            question=bank.ask("zh", rel, dimin),
            golds=(kept,), empty_golds=(kept,), slots=slots,
            who=dimin, target=full, role="pos", lag=-1,
        )

    for canon, long, other in (
        ("Leofric", "Leofrico", "Dunstan"),
        ("Wulfgar", "Wulfgarr", "Kenelm"),
    ):
        slots, _kept = bank.pair_store("en", canon, other, "phone")
        bank.emit(
            kind="latin_extension", lang="en", rel="phone",
            label="absent_person", empty_label="absent_person",
            question=bank.ask("en", "phone", long),
            golds=(), empty_golds=(), slots=slots,
            turns=(bank.mention("en", canon),),
            who=long, target=canon, role="neg", lag=0,
        )
    for canon, long, other, kind in (
        ("嵇珂", "嵇珂然", "芮馥", "cjk_superstring"),
        ("裴琮", "裴琮子", "糜瞳", "cjk_superstring"),
        ("嵇珂", "老嵇珂", "芮馥", "cjk_prefix"),
        ("裴琮", "小裴琮", "糜瞳", "cjk_prefix"),
    ):
        slots, _kept = bank.pair_store("zh", canon, other, "birthday")
        bank.emit(
            kind=kind, lang="zh", rel="birthday",
            label="absent_person", empty_label="absent_person",
            question=bank.ask("zh", "birthday", long),
            golds=(), empty_golds=(), slots=slots,
            turns=(bank.mention("zh", canon),),
            who=long, target=canon, role="neg", lag=0,
        )

    for lang, pairs in packs:
        for i, kind, builder in (
            (0, "focus", _focus_q),
            (1, "focus", _focus_q),
            (2, "dismiss", _dismiss_q),
            (3, "dismiss", _dismiss_q),
            (4, "open_either", _open_q),
            (5, "open_either", _open_q),
        ):
            canon, _typo, other = pairs[i]
            rel = _RELS[i % 3]
            slots, left, right = bank.same_store(lang, canon, other, rel)
            opened = kind == "open_either"
            golds = (left, right) if opened else (left,)
            bank.emit(
                kind=kind, lang=lang, rel=rel,
                label="answerable", empty_label="answerable",
                question=builder(bank, lang, rel, canon, other),
                golds=golds, empty_golds=golds, slots=slots,
                who=canon, target=canon, role="pos", lag=-1,
            )

    for lang, pairs in packs:
        canon, _typo, other = pairs[0]
        quiet = (bank.quiet(lang),)
        phone = bank.val()
        decoy = bank.val()
        bank.emit(
            kind="indirect_hit", lang=lang, rel="phone",
            label="answerable", empty_label="answerable",
            question=_hear_q(bank, lang, "phone"),
            golds=(phone,), empty_golds=(phone,),
            slots=(
                Slot(canon, "phone", phone, bank.fact(lang, "phone", canon, first=True)),
                Slot(other, "phone", decoy, bank.fact(lang, "phone", other)),
            ),
            turns=quiet, who="", target=canon, role="pos", lag=-1,
        )
        lone = bank.val()
        bank.emit(
            kind="indirect_noself", lang=lang, rel="phone",
            label="absent_person", empty_label="absent_person",
            question=_hear_q(bank, lang, "phone"),
            golds=(), empty_golds=(),
            slots=(Slot(canon, "phone", lone, bank.fact(lang, "phone", canon)),),
            turns=quiet, who="", target=canon, role="neg", lag=-1,
        )
        born = bank.val()
        bank.emit(
            kind="indirect_missing", lang=lang, rel="phone",
            label="missing_rel", empty_label="missing_rel",
            question=_hear_q(bank, lang, "phone"),
            golds=(), empty_golds=(),
            slots=(Slot(canon, "birthday", born, bank.fact(lang, "birthday", canon, first=True)),),
            turns=quiet, who="", target=canon, role="neg", lag=-1,
        )

    rows = tuple(bank.rows)
    _check(rows)
    return rows


def _focus_q(bank: _Bank, lang: str, rel: str, target: str, other: str) -> str:
    cue = _CUE[lang][rel]
    if lang == "en":
        return bank.line(f"between {other} and {target} read only about {target} {cue}")
    return bank.line(f"{other} 与 {target} 之间 只问 {target} {cue}")


def _dismiss_q(bank: _Bank, lang: str, rel: str, target: str, other: str) -> str:
    cue = _CUE[lang][rel]
    if lang == "en":
        return bank.line(f"skip {other}, then the {cue} of {target}")
    return bank.line(f"跳过 {other} ， {target} {cue}")


def _open_q(bank: _Bank, lang: str, rel: str, target: str, other: str) -> str:
    cue = _CUE[lang][rel]
    if lang == "en":
        return bank.line(f"the {cue} of {target} and {other}")
    return bank.line(f"{target} 与 {other} {cue}")


def _hear_q(bank: _Bank, lang: str, rel: str) -> str:
    cue = _CUE[lang][rel]
    if lang == "en":
        return bank.line(f"the {cue} belonging to the interlocutor")
    return bank.line(f"跟前这位 {cue}")


def _check(rows: Sequence[Episode]) -> None:
    if not 80 <= len(rows) <= 120:
        raise RuntimeError(f"held-out size {len(rows)}")
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate held-out question")
    blobs = bank_strings(rows)
    if len(blobs) != len(set(blobs)):
        raise RuntimeError("duplicate held-out bank string")
    if {row.lang for row in rows} != {"en", "zh"}:
        raise RuntimeError("held-out langs")
    for row in rows:
        if row.rel not in _RELS:
            raise RuntimeError(row.rel)
        texts = [row.question, *row.turns, *row.prior_asks]
        texts.extend(slot.text for slot in row.slots)
        for slot in row.slots:
            for text in texts:
                if slot.value.lower() in text.lower():
                    raise RuntimeError(f"value {slot.value} leaked into {text}")
        if row.who and row.who not in row.question:
            raise RuntimeError(f"{row.kind} missing surface {row.who}")
    hit = {(row.lang, row.who) for row in rows if row.kind == "typo_hit"}
    for kind in ("typo_silent", "typo_other", "typo_aged"):
        got = {(row.lang, row.who) for row in rows if row.kind == kind}
        if got != hit:
            raise RuntimeError(f"{kind} surfaces {len(got)} != hits {len(hit)}")


_ROWS: Optional[Tuple[Episode, ...]] = None


def build_rows() -> Tuple[Episode, ...]:
    global _ROWS
    if _ROWS is None:
        _ROWS = _assemble()
    return _ROWS


def bank_strings(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    out = []
    seen = set()
    for row in rows:
        blobs = [row.question]
        blobs.extend(slot.text for slot in row.slots)
        blobs.extend(row.turns)
        blobs.extend(row.prior_asks)
        for text in blobs:
            if text not in seen:
                seen.add(text)
                out.append(text)
    return tuple(out)


def _sha(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def questions_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    rows = build_rows() if rows is None else rows
    return _sha([row.question for row in rows])


def bank_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    return _sha(bank_strings(rows))


def kind_counts(rows: Optional[Sequence[Episode]] = None) -> dict:
    rows = build_rows() if rows is None else rows
    counts = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
