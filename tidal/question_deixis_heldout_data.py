"""Deixis held-out bank. English and Chinese. The matcher does not import this module.

Carriers are ``sklent`` / ``mundle`` / ``criftNNN`` / ``plorq`` / ``龗龞``.
Names, fact wording, and question wording are not the 48-episode dev bank.
The gate is fixed in ``reports/deixis.md`` before any scored run. This file
is not an input to the rule.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_EN = ("Sunngifu", "Tidfrith", "Wynflaed", "Beorngyth", "Quenburh", "Aelflaed")
_ZH = ("鄢姝", "濮苓", "缑蔓", "佘芷", "卞琦", "邬岚")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
_RELS = ("phone", "birthday", "places")


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class Turn:
    text: str
    speaker: str = ""
    hearer: str = ""


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
    turns: Tuple[Turn, ...]
    prior_asks: Tuple[Tuple[str, str], ...]
    who: str


class _Build:
    def __init__(self) -> None:
        self.n = 0
        self.v = 0
        self.rows: list[Episode] = []

    def line(self, body: str) -> str:
        self.n += 1
        return f"sklent mundle {body} crift{self.n:03d} plorq 龗龞"

    def value(self) -> str:
        self.v += 1
        return f"hxv{self.v:04d}"

    def person(self, lang: str, index: int) -> str:
        return (_EN if lang == "en" else _ZH)[index]

    def fact(self, lang: str, name: str, rel: str, *, first: bool = False) -> Slot:
        value = self.value()
        cue = _CUE[lang][rel]
        if first and lang == "zh":
            body = f"归档我的{cue}"
        elif first:
            body = f"wynnote my {cue}"
        elif lang == "zh":
            body = f"归档{name}{cue}"
        else:
            body = f"wynnote {name} {cue}"
        return Slot(name, rel, value, self.line(body))

    def mention(self, lang: str, *names: str, speaker: str = "", hearer: str = "") -> Turn:
        if lang == "zh":
            body = "谈及" + "与".join(names) if names else "案头无话"
        elif len(names) == 0:
            body = "paused"
        elif len(names) == 1:
            body = f"plus {names[0]}"
        else:
            body = "plus " + " plus ".join(names)
        return Turn(self.line(body), speaker, hearer)

    def quiet(self, lang: str, speaker: str = "", hearer: str = "") -> Turn:
        body = "案头无话" if lang == "zh" else "paused"
        return Turn(self.line(body), speaker, hearer)

    def q(self, lang: str, rel: str, form: str) -> str:
        cue = _CUE[lang][rel]
        body = f"{form}{cue}" if lang == "zh" else f"{form} {cue}"
        return self.line(body)

    def add(
        self,
        kind: str,
        lang: str,
        rel: str,
        label: str,
        empty_label: str,
        question: str,
        golds: Sequence[str],
        empty_golds: Sequence[str],
        slots: Sequence[Slot],
        turns: Sequence[Turn],
        prior: Sequence[Tuple[str, str]] = (),
        who: str = "",
    ) -> None:
        self.rows.append(Episode(
            kind, lang, rel, label, empty_label, question,
            tuple(golds), tuple(empty_golds), tuple(slots), tuple(turns),
            tuple(prior), who,
        ))


def _hearer_rows(built: _Build) -> None:
    you_en = ("you", "your", "yours", "yourself")
    you_zh = ("你的", "您的", "你", "您")
    for lang, forms in (("en", you_en), ("zh", you_zh)):
        for i, form in enumerate(forms):
            rel = _RELS[i % 3]
            hearer = built.person(lang, 0)
            decoy = built.person(lang, 1)
            held = built.fact(lang, hearer, rel)
            other = built.fact(lang, decoy, rel)
            built.add(
                "hearer_set_you", lang, rel, "answerable", "answerable",
                built.q(lang, rel, form), (held.value,), (other.value,),
                (other, held), (built.quiet(lang, hearer=hearer),), who=form,
            )
    role_en = ("listener", "hearer")
    role_zh = ("面前这位", "身边那人")
    for lang, forms in (("en", role_en), ("zh", role_zh)):
        for i, form in enumerate(forms):
            rel = _RELS[i]
            self_name = built.person(lang, 1)
            hearer = built.person(lang, 0)
            self_slot = built.fact(lang, self_name, rel, first=True)
            held = built.fact(lang, hearer, rel)
            built.add(
                "hearer_set_role", lang, rel, "answerable", "answerable",
                built.q(lang, rel, form), (held.value,), (self_slot.value,),
                (self_slot, held), (built.quiet(lang, hearer=hearer),), who=form,
            )
    for lang, form in (("en", "your"), ("zh", "你的"), ("en", "you"), ("zh", "您")):
        rel = "phone" if form in ("your", "你的") else "birthday"
        named = built.person(lang, 2)
        other = built.fact(lang, built.person(lang, 1), rel)
        kept = built.fact(lang, named, rel)
        built.add(
            "hearer_unset", lang, rel, "absent_person", "answerable",
            built.q(lang, rel, form), (), (other.value,),
            (other, kept), (built.mention(lang, named),), who=form,
        )
    for lang, form in (("en", "your"), ("zh", "你的"), ("en", "yourself"), ("zh", "您的")):
        rel = "places" if lang == "en" else "birthday"
        first = built.person(lang, 0)
        second = built.person(lang, 1)
        decoy = built.fact(lang, built.person(lang, 2), rel)
        early = built.fact(lang, first, rel)
        late = built.fact(lang, second, rel)
        built.add(
            "hearer_switch", lang, rel, "answerable", "answerable",
            built.q(lang, rel, form), (late.value,), (decoy.value,),
            (decoy, early, late),
            (
                built.quiet(lang, hearer=first),
                built.quiet(lang, hearer=second),
            ),
            who=form,
        )
    mixed = {"en": ("you him", "your her"), "zh": ("你和他的", "您和她的")}
    for lang, forms in mixed.items():
        for i, form in enumerate(forms):
            rel = _RELS[i]
            hearer = built.person(lang, 0)
            other = built.fact(lang, built.person(lang, 1), rel)
            held = built.fact(lang, hearer, rel)
            built.add(
                "hearer_ambiguous_mixed", lang, rel, "absent_person", "answerable",
                built.q(lang, rel, form), (), (other.value,),
                (other, held),
                (built.mention(lang, built.person(lang, 2), speaker=built.person(lang, 3), hearer=hearer),),
                who=form,
            )
    alien = {"en": "Notonroster", "zh": "阙外"}
    for lang, form in (("en", "your"), ("zh", "你的"), ("en", "you"), ("zh", "您")):
        rel = "phone"
        other = built.fact(lang, built.person(lang, 1), rel)
        kept = built.fact(lang, built.person(lang, 0), rel)
        built.add(
            "hearer_ambiguous_roster", lang, rel, "absent_person", "answerable",
            built.q(lang, rel, form), (), (other.value,),
            (other, kept),
            (built.quiet(lang, hearer=alien[lang]),),
            who=form,
        )


def _addressee_rows(built: _Build) -> None:
    extra = {
        "en": ("other party", "this one", "that person", "our guest"),
        "zh": ("人家", "对方", "这位", "那位"),
    }
    for lang, forms in extra.items():
        for i, form in enumerate(forms):
            rel = _RELS[i % 3]
            hearer = built.person(lang, 0)
            other = built.fact(lang, built.person(lang, 1), rel)
            held = built.fact(lang, hearer, rel)
            built.add(
                "addressee_extra", lang, rel, "answerable", "absent_person",
                built.q(lang, rel, form), (held.value,), (),
                (other, held), (built.quiet(lang, hearer=hearer),), who=form,
            )
    unset = {"en": ("other party", "this one"), "zh": ("对方", "人家")}
    for lang, forms in unset.items():
        for i, form in enumerate(forms):
            rel = _RELS[i]
            other = built.fact(lang, built.person(lang, 1), rel)
            kept = built.fact(lang, built.person(lang, 0), rel)
            built.add(
                "addressee_extra_unset", lang, rel, "absent_person", "absent_person",
                built.q(lang, rel, form), (), (),
                (other, kept), (built.quiet(lang),), who=form,
            )


def _third_rows(built: _Build) -> None:
    classic = {
        "en": ("he", "she", "they", "he"),
        "zh": ("他", "她", "ta", "他们"),
    }
    for lang, forms in classic.items():
        for i, form in enumerate(forms):
            rel = _RELS[i % 3]
            target = built.person(lang, 2)
            speaker = built.person(lang, 0)
            hearer = built.person(lang, 1)
            decoy = built.fact(lang, built.person(lang, 3), rel)
            gold = built.fact(lang, target, rel)
            empty = "absent_person" if form == "ta" else "answerable"
            empty_golds = () if form == "ta" else (decoy.value,)
            built.add(
                "third_unique", lang, rel, "answerable", empty,
                built.q(lang, rel, form), (gold.value,), empty_golds,
                (decoy, gold),
                (built.mention(lang, target, speaker=speaker, hearer=hearer),),
                who=form,
            )
    oblique = {
        "en": ("him", "her", "his", "hers", "their"),
        "zh": ("他的", "她的", "他的", "她的", "他的"),
    }
    for lang, forms in oblique.items():
        for i, form in enumerate(forms):
            rel = _RELS[i % 3]
            # "his" on the third English row names Osgyth. Gender is not a feature.
            target = built.person(lang, 2)
            decoy = built.fact(lang, built.person(lang, 3), rel)
            gold = built.fact(lang, target, rel)
            built.add(
                "third_oblique", lang, rel, "answerable", "answerable",
                built.q(lang, rel, form), (gold.value,), (decoy.value,),
                (decoy, gold),
                (built.mention(
                    lang, target,
                    speaker=built.person(lang, 0),
                    hearer=built.person(lang, 1),
                ),),
                who=form if not (lang == "en" and form == "his") else "his names Wynflaed",
            )
    for lang in ("en", "zh"):
        for width in (3, 4):
            for rel in ("phone", "birthday"):
                speaker = built.person(lang, 0)
                hearer = built.person(lang, 1)
                older = built.person(lang, 2)
                newest = built.person(lang, 3)
                slots = [
                    built.fact(lang, speaker, rel),
                    built.fact(lang, hearer, rel),
                    built.fact(lang, older, rel),
                    built.fact(lang, newest, rel),
                ]
                if width == 3:
                    turns = (
                        built.mention(lang, speaker, older, speaker=speaker, hearer=hearer),
                        built.mention(lang, newest, speaker=speaker, hearer=hearer),
                    )
                else:
                    turns = (
                        built.mention(lang, speaker, older, speaker=speaker, hearer=hearer),
                        built.mention(lang, hearer, speaker=speaker, hearer=hearer),
                        built.mention(lang, newest, speaker=speaker, hearer=hearer),
                    )
                built.add(
                    "third_later", lang, rel, "answerable", "answerable",
                    built.q(lang, rel, "they" if lang == "en" else "他们"),
                    (slots[3].value,), (slots[0].value,),
                    slots, turns, who=f"newest of {width}",
                )
    ask_forms = {"en": ("him", "her", "his", "their"), "zh": ("他", "她", "他的", "她")}
    for lang, forms in ask_forms.items():
        for i, form in enumerate(forms):
            rel = _RELS[i % 3]
            target = built.person(lang, 2)
            speaker = built.person(lang, 0)
            hearer = built.person(lang, 1)
            decoy = built.fact(lang, built.person(lang, 3), rel)
            phone = built.fact(lang, target, "phone")
            gold = built.fact(lang, target, rel) if rel != "phone" else phone
            slots = (decoy, phone) if rel == "phone" else (decoy, phone, gold)
            prior = built.q(lang, "phone", target)
            built.add(
                "third_after_ask", lang, rel, "answerable", "answerable",
                built.q(lang, rel, form), (gold.value,), (decoy.value,),
                slots,
                (built.quiet(lang, speaker=speaker, hearer=hearer),),
                prior=((prior, "phone"),),
                who=form,
            )
    for lang in ("en", "zh"):
        for i, form in enumerate(("he", "they") if lang == "en" else ("他", "她们")):
            rel = _RELS[i]
            left = built.person(lang, 2)
            right = built.person(lang, 3)
            decoy = built.fact(lang, built.person(lang, 4), rel)
            a = built.fact(lang, left, rel)
            b = built.fact(lang, right, rel)
            built.add(
                "third_tie", lang, rel, "absent_person", "answerable",
                built.q(lang, rel, form), (), (decoy.value,),
                (decoy, a, b),
                (built.mention(
                    lang, left, right,
                    speaker=built.person(lang, 0),
                    hearer=built.person(lang, 1),
                ),),
                who=form,
            )
        rel = "places"
        older = built.person(lang, 2)
        left = built.person(lang, 3)
        right = built.person(lang, 4)
        decoy = built.fact(lang, built.person(lang, 5), rel)
        old = built.fact(lang, older, rel)
        a = built.fact(lang, left, rel)
        b = built.fact(lang, right, rel)
        built.add(
            "third_tie", lang, rel, "absent_person", "answerable",
            built.q(lang, rel, "she" if lang == "en" else "她"),
            (), (decoy.value,),
            (decoy, old, a, b),
            (
                built.mention(lang, older, speaker=built.person(lang, 0), hearer=built.person(lang, 1)),
                built.mention(lang, left, right, speaker=built.person(lang, 0), hearer=built.person(lang, 1)),
            ),
            who="newest turn has two",
        )


def _window_rows(built: _Build) -> None:
    plan = (
        ("window_filler_3", 3, "answerable"),
        ("window_filler_4", 4, "absent_person"),
        ("window_filler_5", 5, "absent_person"),
    )
    for kind, fillers, label in plan:
        for lang, form in (("en", "he"), ("zh", "他"), ("en", "her"), ("zh", "她的")):
            rel = "birthday" if form in ("her", "她的") else "phone"
            target = built.person(lang, 2)
            decoy = built.fact(lang, built.person(lang, 3), rel)
            gold = built.fact(lang, target, rel)
            speaker = built.person(lang, 0)
            hearer = built.person(lang, 1)
            turns = [built.mention(lang, target, speaker=speaker, hearer=hearer)]
            turns.extend(built.quiet(lang, speaker=speaker, hearer=hearer) for _ in range(fillers))
            built.add(
                kind, lang, rel, label, "answerable",
                built.q(lang, rel, form),
                (gold.value,) if label == "answerable" else (),
                (decoy.value,),
                (decoy, gold), tuple(turns), who=f"fillers {fillers}",
            )
    for lang, form in (("en", "his"), ("zh", "他"), ("en", "ta"), ("zh", "ta")):
        rel = "places"
        target = built.person(lang, 2)
        decoy = built.fact(lang, built.person(lang, 3), rel)
        gold = built.fact(lang, target, rel)
        empty = "absent_person" if form == "ta" else "answerable"
        built.add(
            "third_silent", lang, rel, "absent_person", empty,
            built.q(lang, rel, form), (),
            () if form == "ta" else (decoy.value,),
            (decoy, gold), (built.quiet(lang),), who=form,
        )
    for lang in ("en", "zh"):
        for role in ("speaker", "hearer"):
            for form in (("he", "him") if lang == "en" else ("他", "她")):
                rel = "phone" if form in ("he", "他") else "birthday"
                person = built.person(lang, 0)
                other_role = built.person(lang, 1)
                decoy = built.fact(lang, built.person(lang, 3), rel)
                gold = built.fact(lang, person, rel)
                speaker = person if role == "speaker" else other_role
                hearer = person if role == "hearer" else other_role
                built.add(
                    "third_wrong_role", lang, rel, "absent_person", "answerable",
                    built.q(lang, rel, form), (), (decoy.value,),
                    (decoy, gold),
                    (built.mention(lang, person, speaker=speaker, hearer=hearer),),
                    who=role,
                )
    for lang, form in (("en", "her"), ("zh", "她"), ("en", "his"), ("zh", "他的")):
        rel = "birthday"
        former = built.person(lang, 2)
        current = built.person(lang, 1)
        speaker = built.person(lang, 0)
        decoy = built.fact(lang, built.person(lang, 3), rel)
        gold = built.fact(lang, former, rel)
        built.add(
            "third_former_hearer", lang, rel, "answerable", "answerable",
            built.q(lang, rel, form), (gold.value,), (decoy.value,),
            (decoy, gold, built.fact(lang, current, rel)),
            (
                built.quiet(lang, speaker=speaker, hearer=former),
                built.mention(lang, former, speaker=speaker, hearer=current),
            ),
            who=form,
        )


def _ask_hearer_rows(built: _Build) -> None:
    for lang, form in (("en", "her"), ("zh", "她"), ("en", "him"), ("zh", "他的")):
        rel = "birthday"
        hearer = built.person(lang, 1)
        speaker = built.person(lang, 0)
        decoy = built.fact(lang, built.person(lang, 3), rel)
        phone = built.fact(lang, hearer, "phone")
        birth = built.fact(lang, hearer, rel)
        prior = built.q(lang, "phone", hearer)
        built.add(
            "third_ask_hearer", lang, rel, "absent_person", "answerable",
            built.q(lang, rel, form), (), (decoy.value,),
            (decoy, phone, birth),
            (built.quiet(lang, speaker=speaker, hearer=hearer),),
            prior=((prior, "phone"),),
            who=form,
        )


def _control_rows(built: _Build) -> None:
    for lang, form in (("en", "Sunngifu"), ("zh", "鄢姝"), ("en", "Wynflaed"), ("zh", "缑蔓")):
        rel = "phone" if form in ("Sunngifu", "鄢姝") else "places"
        target = built.person(lang, 0 if form in ("Sunngifu", "鄢姝") else 2)
        hearer = built.person(lang, 1)
        decoy = built.fact(lang, hearer, rel)
        gold = built.fact(lang, target, rel)
        built.add(
            "control_named", lang, rel, "answerable", "answerable",
            built.q(lang, rel, target), (gold.value,), (gold.value,),
            (decoy, gold),
            (built.mention(lang, built.person(lang, 3), speaker=built.person(lang, 4), hearer=hearer),),
            who=target,
        )
    for lang, form in (("en", "your"), ("zh", "你的"), ("en", "you"), ("zh", "您")):
        rel = "birthday"
        target = built.person(lang, 2)
        hearer = built.person(lang, 0)
        decoy = built.fact(lang, hearer, rel)
        gold = built.fact(lang, target, rel)
        prefix = f"{form} {target}" if lang == "en" else f"{form}{target}"
        built.add(
            "control_you_named", lang, rel, "answerable", "answerable",
            built.q(lang, rel, prefix), (gold.value,), (gold.value,),
            (decoy, gold),
            (built.quiet(lang, hearer=hearer),),
            who=prefix,
        )
    for lang, form in (("en", "listener"), ("zh", "面前这位"), ("en", "hearer"), ("zh", "身边那人")):
        rel = "phone" if form in ("listener", "面前这位") else "places"
        self_name = built.person(lang, 0)
        other = built.fact(lang, built.person(lang, 1), rel)
        gold = built.fact(lang, self_name, rel, first=True)
        built.add(
            "control_indirect", lang, rel, "answerable", "answerable",
            built.q(lang, rel, form), (gold.value,), (gold.value,),
            (other, gold), (built.quiet(lang),), who=form,
        )
    for lang, form in (("en", "listener"), ("zh", "面前这位"), ("en", "hearer"), ("zh", "身边那人")):
        rel = "birthday"
        a = built.fact(lang, built.person(lang, 0), rel)
        b = built.fact(lang, built.person(lang, 1), rel)
        built.add(
            "control_indirect_noself", lang, rel, "absent_person", "absent_person",
            built.q(lang, rel, form), (), (),
            (a, b), (built.quiet(lang),), who=form,
        )
    for lang, form in (("en", "your"), ("zh", "你的"), ("en", "you"), ("zh", "您")):
        rel = "places"
        hearer = built.person(lang, 0)
        other = built.fact(lang, built.person(lang, 1), rel)
        phone = built.fact(lang, hearer, "phone")
        built.add(
            "control_missing", lang, rel, "missing_rel", "answerable",
            built.q(lang, rel, form), (), (other.value,),
            (other, phone), (built.quiet(lang, hearer=hearer),), who=form,
        )
    for lang, form in (("en", "them"), ("zh", "你们的"), ("en", "them"), ("zh", "你们")):
        rel = "phone" if form == "them" else "birthday"
        hearer = built.person(lang, 0)
        target = built.person(lang, 2)
        decoy = built.fact(lang, built.person(lang, 3), rel)
        mentioned = built.fact(lang, target, rel)
        held = built.fact(lang, hearer, rel)
        if form == "them":
            built.add(
                "control_not_pronoun", lang, rel, "absent_person", "absent_person",
                built.q(lang, rel, form), (), (),
                (decoy, mentioned, held),
                (built.mention(lang, target, speaker=built.person(lang, 1), hearer=hearer),),
                who=form,
            )
        else:
            built.add(
                "control_plural", lang, rel, "answerable", "answerable",
                built.q(lang, rel, form), (decoy.value,), (decoy.value,),
                (decoy, mentioned, held),
                (built.quiet(lang, hearer=hearer),),
                who=form,
            )


def build_rows() -> Tuple[Episode, ...]:
    built = _Build()
    _hearer_rows(built)
    _addressee_rows(built)
    _third_rows(built)
    _window_rows(built)
    _ask_hearer_rows(built)
    _control_rows(built)
    rows = tuple(built.rows)
    _check(rows)
    return rows


def _check(rows: Sequence[Episode]) -> None:
    if len(rows) < 100:
        raise RuntimeError(f"deixis held-out has {len(rows)} episodes")
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate deixis held-out question")
    strings = bank_strings(rows)
    if len(strings) != len(set(strings)):
        raise RuntimeError("duplicate deixis held-out string")
    if {row.lang for row in rows} != {"en", "zh"}:
        raise RuntimeError("deixis held-out langs")
    if sum(row.lang == "en" for row in rows) != sum(row.lang == "zh" for row in rows):
        raise RuntimeError("deixis held-out is not balanced")
    fillers = {"window_filler_3": 4, "window_filler_4": 5, "window_filler_5": 6}
    for row in rows:
        if row.label == "answerable" and len(row.golds) != 1:
            raise RuntimeError(f"{row.kind} gold")
        if row.label != "answerable" and row.golds:
            raise RuntimeError(f"{row.kind} abstains with a gold")
        if row.empty_label == "answerable" and len(row.empty_golds) != 1:
            raise RuntimeError(f"{row.kind} empty gold")
        if row.empty_label != "answerable" and row.empty_golds:
            raise RuntimeError(f"{row.kind} empty abstain has a gold")
        if row.kind in fillers and len(row.turns) != fillers[row.kind]:
            raise RuntimeError(f"{row.kind} turn count {len(row.turns)}")
        texts = [row.question, *(turn.text for turn in row.turns)]
        texts.extend(text for text, _rel in row.prior_asks)
        texts.extend(slot.text for slot in row.slots)
        for slot in row.slots:
            for text in texts:
                if slot.value.lower() in text.lower():
                    raise RuntimeError(f"value {slot.value} leaked into {text}")


def questions(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def bank_strings(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        texts = [row.question, *(turn.text for turn in row.turns)]
        texts.extend(text for text, _rel in row.prior_asks)
        texts.extend(slot.text for slot in row.slots)
        for text in texts:
            if text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def _sha(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def questions_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    return _sha(questions(rows))


def bank_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    return _sha(bank_strings(rows))


def kind_counts(rows: Optional[Sequence[Episode]] = None) -> dict:
    rows = build_rows() if rows is None else rows
    counts: dict = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
