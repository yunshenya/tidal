"""Deixis dev bank. English and Chinese. The matcher does not import this module.

Anchors are ``kibble`` / ``rowen`` / ``swarfNNN`` / ``㐆㐇``. People are not on
the earlier rosters. The gate is fixed in ``reports/deixis.md`` and is not
fit on the frozen file.

A stored name in the question keeps the name path. ``you`` / ``你`` / ``您``
and a hearer-role cue name the hearer ``note_turn`` stored. ``he`` / ``she``
/ ``they`` / ``他`` / ``她`` / ``ta`` name the one person on the newest
conversational turn who is neither speaker nor hearer. The empty log, which
has no utterance and no hearer, is the current ask path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

_EN = ("Godgifu", "Waerburh", "Eanswith", "Osthryth", "Cyneswith", "Leofwaru")
_ZH = ("郗泠", "逄葭", "晁蘅", "蒯茗", "宓岫", "隗岑")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}


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
    who: str


class _Build:
    def __init__(self) -> None:
        self.n = 0
        self.v = 0
        self.rows: list[Episode] = []

    def line(self, body: str) -> str:
        self.n += 1
        return f"kibble rowen {body} swarf{self.n:03d} 㐆㐇"

    def value(self) -> str:
        self.v += 1
        return f"dxv{self.v:04d}"

    def who(self, lang: str, index: int) -> str:
        return (_EN if lang == "en" else _ZH)[index]

    def fact(self, lang: str, name: str, rel: str, *, first: bool = False) -> Slot:
        value = self.value()
        cue = _CUE[lang][rel]
        if first and lang == "zh":
            body = f"我登记{cue}"
        elif first:
            body = f"I filed the {cue}"
        elif lang == "zh":
            body = f"册页登记{name}{cue}"
        else:
            body = f"ledger notes the {cue} of {name}"
        return Slot(name, rel, value, self.line(body))

    def mention(self, lang: str, *names: str, speaker: str = "", hearer: str = "") -> Turn:
        if lang == "zh":
            body = "提到" + "和".join(names) if names else "窑口歇着"
        elif len(names) == 0:
            body = "kiln sat shut"
        elif len(names) == 1:
            body = f"named {names[0]} aloud"
        else:
            body = "named " + " and ".join(names) + " aloud"
        return Turn(self.line(body), speaker, hearer)

    def quiet(self, speaker: str = "", hearer: str = "") -> Turn:
        return Turn(self.line("kiln sat shut"), speaker, hearer)

    def ask(self, lang: str, rel: str, body: str) -> str:
        cue = _CUE[lang][rel]
        if lang == "zh":
            text = f"{body}{cue}" if body else cue
        else:
            text = f"{body} {cue}".strip()
        return self.line(text)

    def add(
        self,
        kind: str,
        lang: str,
        rel: str,
        label: str,
        empty_label: str,
        question: str,
        golds: Tuple[str, ...],
        empty_golds: Tuple[str, ...],
        slots: Sequence[Slot],
        turns: Sequence[Turn],
        who: str = "",
    ) -> None:
        self.rows.append(Episode(
            kind, lang, rel, label, empty_label, question,
            tuple(golds), tuple(empty_golds), tuple(slots), tuple(turns), who,
        ))


def _pair(built: _Build, lang: str, rel: str, target: str, decoy: str) -> Tuple[Slot, Slot]:
    """Decoy first, so an unscoped ranker with tied scores returns the decoy."""
    return (
        built.fact(lang, decoy, rel),
        built.fact(lang, target, rel),
    )


def build_rows() -> Tuple[Episode, ...]:
    built = _Build()
    for lang in ("en", "zh"):
        target, other, decoy, speaker, hearer_name, extra = (
            _EN if lang == "en" else _ZH
        )
        you = {"en": ("your", "yours", "the addressee"), "zh": ("你的", "您的", "对面那人的")}[lang]
        third = {
            "en": ("his", "her", "their", "ta"),
            "zh": ("她的", "他的", "他们的", "ta的"),
        }[lang]

        # Hearer set. Empty log ranks the decoy, or the first-person self for a role noun.
        for rel, cue_i, override in (
            ("phone", 0, False),
            ("birthday", 1, False),
            ("phone", 2, True),
        ):
            if override:
                self_slot = built.fact(lang, other, rel, first=True)
                hearer_slot = built.fact(lang, target, rel)
                slots = (self_slot, hearer_slot)
                empty = (self_slot.value,)
            else:
                decoy_slot, hearer_slot = _pair(built, lang, rel, target, decoy)
                slots = (decoy_slot, hearer_slot)
                empty = (decoy_slot.value,)
            built.add(
                "hearer_set", lang, rel, "answerable", "answerable",
                built.ask(lang, rel, you[cue_i]),
                (hearer_slot.value,), empty, slots,
                (built.mention(lang, target, hearer=target),),
                who=target,
            )

        # Hearer unset after an utterance. Empty log still ranks.
        for rel, word in (("phone", you[0]), ("places", you[1])):
            decoy_slot, named = _pair(built, lang, rel, target, decoy)
            built.add(
                "hearer_unset", lang, rel, "absent_person", "answerable",
                built.ask(lang, rel, word),
                (), (decoy_slot.value,), (decoy_slot, named),
                (built.mention(lang, target),),
                who=target,
            )

        # Mixed pronouns, and a hearer string that is not a roster name.
        decoy_slot, named = _pair(built, lang, "phone", target, decoy)
        mix = "your telephone or her birthday" if lang == "en" else "你的电话还是她的生日"
        built.add(
            "hearer_ambiguous", lang, "phone", "absent_person", "answerable",
            built.line(mix),
            (), (decoy_slot.value,), (decoy_slot, named),
            (built.mention(lang, target, hearer=target),),
            who="",
        )
        decoy_slot, named = _pair(built, lang, "phone", target, decoy)
        built.add(
            "hearer_ambiguous", lang, "phone", "absent_person", "answerable",
            built.ask(lang, "phone", you[0]),
            (), (decoy_slot.value,), (decoy_slot, named),
            (built.quiet(hearer="unlisted"),),
            who="",
        )

        # One eligible person, then a later person with an older one still in the window.
        decoy_slot, named = _pair(built, lang, "phone", target, decoy)
        speaker_slot = built.fact(lang, speaker, "phone")
        built.add(
            "third_unique", lang, "phone", "answerable", "answerable",
            built.ask(lang, "phone", third[0]),
            (named.value,), (decoy_slot.value,),
            (decoy_slot, named, speaker_slot),
            (built.mention(lang, target, speaker=speaker, hearer=hearer_name),),
            who=target,
        )
        decoy_slot = built.fact(lang, decoy, "birthday")
        older = built.fact(lang, other, "birthday")
        newer = built.fact(lang, target, "birthday")
        built.add(
            "third_unique", lang, "birthday", "answerable", "answerable",
            built.ask(lang, "birthday", third[1]),
            (newer.value,), (decoy_slot.value,),
            (decoy_slot, older, newer),
            (
                built.mention(lang, other, speaker=speaker, hearer=hearer_name),
                built.mention(lang, target, speaker=speaker, hearer=hearer_name),
            ),
            who=target,
        )
        decoy_slot, named = _pair(built, lang, "places", extra, decoy)
        built.add(
            "third_unique", lang, "places", "answerable", "answerable",
            built.ask(lang, "places", third[2]),
            (named.value,), (decoy_slot.value,), (decoy_slot, named),
            (built.mention(lang, extra, speaker=speaker, hearer=hearer_name),),
            who=extra,
        )
        decoy_slot, named = _pair(built, lang, "phone", target, decoy)
        # ``ta`` is not in ``is_deictic``, so an empty log abstains.
        built.add(
            "third_unique", lang, "phone", "answerable", "absent_person",
            built.ask(lang, "phone", third[3]),
            (named.value,), (), (decoy_slot, named),
            (built.mention(lang, target, speaker=speaker, hearer=hearer_name),),
            who=target,
        )

        # Two names on the newest turn.
        for rel, word, empty in (
            ("phone", third[0], "answerable"),
            ("birthday", third[3], "absent_person"),
        ):
            decoy_slot = built.fact(lang, decoy, rel)
            left = built.fact(lang, target, rel)
            right = built.fact(lang, other, rel)
            empty_golds = (decoy_slot.value,) if empty == "answerable" else ()
            built.add(
                "third_tie", lang, rel, "absent_person", empty,
                built.ask(lang, rel, word),
                (), empty_golds, (decoy_slot, left, right),
                (built.mention(lang, target, other, speaker=speaker, hearer=hearer_name),),
                who="",
            )

        # Mention, then four later utterances.
        for rel, word in (("phone", third[0]), ("places", third[1])):
            decoy_slot, named = _pair(built, lang, rel, target, decoy)
            fillers = tuple(built.quiet(speaker=speaker, hearer=hearer_name) for _ in range(4))
            built.add(
                "third_aged", lang, rel, "absent_person", "answerable",
                built.ask(lang, rel, word),
                (), (decoy_slot.value,), (decoy_slot, named),
                (built.mention(lang, target, speaker=speaker, hearer=hearer_name),) + fillers,
                who=target,
            )

        # A write plus a turn that names nobody.
        for rel, word, empty in (
            ("phone", third[0], "answerable"),
            ("birthday", third[3], "absent_person"),
        ):
            decoy_slot, named = _pair(built, lang, rel, target, decoy)
            empty_golds = (decoy_slot.value,) if empty == "answerable" else ()
            built.add(
                "third_silent", lang, rel, "absent_person", empty,
                built.ask(lang, rel, word),
                (), empty_golds, (decoy_slot, named),
                (built.quiet(speaker=speaker, hearer=hearer_name),),
                who=target,
            )

        # The only person said is the speaker, then the hearer.
        decoy_slot, named = _pair(built, lang, "phone", target, decoy)
        built.add(
            "third_wrong_role", lang, "phone", "absent_person", "answerable",
            built.ask(lang, "phone", third[0]),
            (), (decoy_slot.value,), (decoy_slot, named),
            (built.mention(lang, target, speaker=target, hearer=hearer_name),),
            who=target,
        )
        decoy_slot, named = _pair(built, lang, "birthday", other, decoy)
        built.add(
            "third_wrong_role", lang, "birthday", "absent_person", "answerable",
            built.ask(lang, "birthday", third[1]),
            (), (decoy_slot.value,), (decoy_slot, named),
            (built.mention(lang, other, speaker=speaker, hearer=other),),
            who=other,
        )

        # A name in the question wins, including when the question also says you.
        decoy_slot, named = _pair(built, lang, "phone", target, other)
        q_named = (
            built.line(f"the telephone under {target}")
            if lang == "en" else built.line(f"{target}的电话")
        )
        built.add(
            "control_named", lang, "phone", "answerable", "answerable",
            q_named, (named.value,), (named.value,), (decoy_slot, named),
            (built.mention(lang, other, hearer=other),),
            who=target,
        )
        q_you = (
            built.line(f"can you read the telephone under {target}")
            if lang == "en" else built.line(f"你读一下{target}的电话")
        )
        decoy_slot, named = _pair(built, lang, "phone", target, other)
        built.add(
            "control_you_named", lang, "phone", "answerable", "answerable",
            q_you, (named.value,), (named.value,), (decoy_slot, named),
            (built.mention(lang, other, hearer=other),),
            who=target,
        )

        # Hearer unset: the indirect path still resolves the unique first person.
        self_slot = built.fact(lang, target, "phone", first=True)
        other_slot = built.fact(lang, other, "phone")
        cue = "the addressee" if lang == "en" else "对面那人的"
        built.add(
            "control_indirect", lang, "phone", "answerable", "answerable",
            built.ask(lang, "phone", cue),
            (self_slot.value,), (self_slot.value,), (other_slot, self_slot),
            (built.quiet(),),
            who=target,
        )
        lone = built.fact(lang, target, "phone")
        built.add(
            "control_indirect_noself", lang, "phone", "absent_person", "absent_person",
            built.ask(lang, "phone", cue),
            (), (), (lone,),
            (built.quiet(),),
            who="",
        )

        # Hearer is set and does not store the asked relation. The other person does.
        other_phone = built.fact(lang, other, "phone")
        hearer_day = built.fact(lang, target, "birthday")
        built.add(
            "control_missing", lang, "phone", "missing_rel", "answerable",
            built.ask(lang, "phone", you[0]),
            (), (other_phone.value,), (other_phone, hearer_day),
            (built.mention(lang, target, hearer=target),),
            who=target,
        )

    rows = tuple(built.rows)
    _check(rows)
    return rows


def _check(rows: Sequence[Episode]) -> None:
    if not 40 <= len(rows) <= 80:
        raise RuntimeError(f"deixis dev bank has {len(rows)} episodes")
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate deixis question")
    strings = bank_strings(rows)
    if len(strings) != len(set(strings)):
        raise RuntimeError("duplicate deixis bank string")
    if {row.lang for row in rows} != {"en", "zh"}:
        raise RuntimeError("deixis langs")
    for row in rows:
        if row.label == "answerable" and not row.golds:
            raise RuntimeError(f"{row.kind} has no gold")
        if row.label != "answerable" and row.golds:
            raise RuntimeError(f"{row.kind} abstains with a gold")
        if row.empty_label == "answerable" and not row.empty_golds:
            raise RuntimeError(f"{row.kind} empty pass has no gold")
        texts = [row.question, *(turn.text for turn in row.turns)]
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
