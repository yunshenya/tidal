"""Expanded held-out bank for pronoun keep-alive and leading vocatives.

Categories and listener labels were fixed in
``reports/dialogue_keepalive_expand_prereg.md`` before this file was scored.
The 42-ask bank is not rewritten. ``build_rows`` replays the frozen rule and
keeps a listener label that disagrees. It does not relabel the row. The rule
modules are not imported in order to edit them. ``LLMMemory`` is not imported.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_dialogue_real_data import _human_outcome, _predict
from tidal.question_keepalive_heldout_data import Ask, Episode, Slot, Turn
from tidal.question_surface import (
    _ADDRESS_PUNCT,
    _second_person,
    _third_person,
    apply_utterance,
    ask_people,
    is_deictic,
    lex_labels,
    name_spans,
    refresh_resolved,
)

_REL = ("phone", "birthday", "places")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
# Decoy, three third persons, speaker, hearer. Decoy is stored first.
_CAST = {
    "en": ("Osbern", "Phelan", "Idony", "Hawise", "Thurstan", "Rohese"),
    "zh": ("傅苔", "庞荞", "倪蒲", "滕岚", "邹蒲", "雷桐"),
}
_EN_PROPS = (
    ("reed card", "marl peg"),
    ("kiln chit", "grout notch"),
    ("flax tag", "loom pin"),
    ("peat token", "bog stake"),
    ("quartz chip", "vein tray"),
    ("willow chit", "osier hook"),
    ("umber wafer", "clay nib"),
    ("nettle slip", "dock rivet"),
)
_ZH_PROPS = (
    ("藤条牌", "篱钉"),
    ("炭片", "窑槽"),
    ("麻缕卡", "织钉"),
    ("泥炭筹", "沼桩"),
    ("石英片", "矿匣"),
    ("苇秆签", "垛钉"),
    ("葛藤牌", "桩钉"),
    ("桦皮条", "榫钉"),
)
_KINDS = {
    "neg_subject": 12,
    "neg_object": 12,
    "neg_quote": 12,
    "neg_ambig": 12,
    "neg_unknown": 8,
    "neg_explicit": 10,
    "neg_age": 12,
    "pos_keep": 8,
    "pos_voc": 8,
}


def _people(lang: str) -> Tuple[str, str, str, str, str, str]:
    return _CAST[lang]  # type: ignore[return-value]


def _props(lang: str) -> Sequence[Tuple[str, str]]:
    return _EN_PROPS if lang == "en" else _ZH_PROPS


def _cue(lang: str, rel: str) -> str:
    return _CUE[lang][rel]


def _you(lang: str, rel: str, a: str, b: str, form: int) -> str:
    cue = _cue(lang, rel)
    if lang == "en":
        if form == 0:
            return f"Which {cue} is filed for you on the {a} beside the {b}?"
        return f"The counter wants your {cue} from the {a} by the {b}."
    if form == 0:
        return f"你的{cue}写在{a}靠{b}的哪一行？"
    return f"柜上要回你的{cue}，{a}还在{b}旁。"


def _third_q(lang: str, rel: str, a: str, b: str, pronoun: str) -> str:
    cue = _cue(lang, rel)
    if lang == "en":
        return f"Which {cue} did {pronoun} copy onto the {a} beside the {b}?"
    return f"{pronoun}的{cue}写在{a}靠{b}的哪一行？"


class _Props:
    def __init__(self, lang: str) -> None:
        self.pairs = _props(lang)
        self.i = 0

    def take(self) -> Tuple[str, str]:
        pair = self.pairs[self.i % len(self.pairs)]
        self.i += 1
        return pair


class _Scene:
    def __init__(self, bank: "Bank", eid: str, lang: str, kind: str) -> None:
        self.bank = bank
        self.eid = eid
        self.lang = lang
        self.kind = kind
        self.people: List[str] = []
        self.steps: list = []

    def cast(self, *names: str) -> "_Scene":
        self.people = list(names)
        return self

    def turn(self, text: str, mentions: Sequence[str], hearer: str = "") -> "_Scene":
        self.steps.append(("t", text, tuple(mentions), hearer))
        return self

    def ask(self, rel: str, question: str, human: str) -> "_Scene":
        self.steps.append(("a", rel, question, human))
        return self

    def _fact(self, name: str, rel: str) -> Slot:
        cue = _cue(self.lang, rel)
        n = self.bank.stamp
        self.bank.stamp += 1
        if self.lang == "zh":
            text = f"页边抄件{n:03d}把{name}写进{cue}那一栏 marl docket {n:03d}"
        else:
            text = (
                f"Margin folio {n:03d} files {name} under the {cue} "
                f"row marl docket {n:03d}"
            )
        return Slot(name, rel, f"mqd{n:04d}", text)

    def finish(self) -> None:
        decoy = _people(self.lang)[0]
        if not self.people or self.people[0] != decoy:
            raise RuntimeError(f"{self.eid} decoy is not first")
        slots = [self._fact(name, rel) for rel in _REL for name in self.people]
        names = [slot.name for slot in slots]
        surfaces = [slot.text for slot in slots]
        roster = {slot.name for slot in slots}
        speaker = _people(self.lang)[4]
        state = DialogueState()
        events: List[object] = []
        for step in self.steps:
            if step[0] == "t":
                _, text, mentions, hearer = step
                stamped = self.bank.line(text)
                if hearer and hearer not in roster and self.kind != "neg_explicit":
                    raise RuntimeError(f"{self.eid} unexpected off-roster hearer {hearer}")
                got = apply_utterance(
                    state, stamped, names, surfaces,
                    speaker=speaker, hearer=hearer or None,
                )
                if set(got) != set(mentions):
                    raise RuntimeError(
                        f"{self.eid} mentions {got} != {mentions} in {stamped!r}"
                    )
                events.append(Turn(stamped, speaker, hearer))
                continue
            _, rel, question, human = step
            stamped = self.bank.line(question)
            if lex_labels(stamped) != (rel,):
                raise RuntimeError(f"{self.eid} cues {lex_labels(stamped)} for {stamped!r}")
            if not is_deictic(stamped):
                raise RuntimeError(f"{self.eid} question is not deictic: {stamped!r}")
            if human not in roster and human != "ABSENT":
                raise RuntimeError(f"{self.eid} human {human} is not stored")
            rule, rule_golds, who = _predict(state, slots, stamped, rel)
            empty_rule, empty_golds, _empty_who = _predict(DialogueState(), slots, stamped, rel)
            human_label, human_golds, who_human = _human_outcome(human, slots, rel)
            events.append(Ask(
                self.kind, self.lang, rel, stamped, rule, human_label,
                rule_golds, human_golds, empty_rule, empty_golds, who, who_human,
            ))
            mentioned = ask_people(
                stamped, names, surfaces, state.conversational(ignore_surface=stamped),
            )
            state.note_ask(mentioned, stamped)
            refresh_resolved(state, stamped, names, surfaces)
        self.bank.episodes.append(Episode(
            self.eid, self.lang, tuple(slots), tuple(events),
        ))


class Bank:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []
        self.stamp = 1
        self._line = 0

    def line(self, text: str) -> str:
        self._line += 1
        return f"{text} marl docket {self._line:03d}"

    def scene(self, eid: str, lang: str, kind: str) -> _Scene:
        return _Scene(self, eid, lang, kind)


def _authored_vocatives(text: str, names: Sequence[str]) -> List[str]:
    """Exact roster names at position 0 whose next character is address punct."""
    raw = text.lstrip()
    found = []
    for name in names:
        for start, end in name_spans(name, raw):
            if start == 0 and end < len(raw) and raw[end] in _ADDRESS_PUNCT:
                found.append(name)
    return found


def _has_name(text: str, name: str) -> bool:
    return bool(name_spans(name, text))


def _pronoun_free(text: str) -> bool:
    return not _second_person(text) and not _third_person(text)


def _asks_you(lang: str, rels: Sequence[str], a: str, b: str, human: str, form: int = 0):
    return tuple((rel, _you(lang, rel, a, b, form), human) for rel in rels)


def _load(bank: Bank) -> None:
    for lang in ("en", "zh"):
        _decoy, a, b, c, speaker, hearer = _people(lang)
        del _decoy
        props = _Props(lang)
        roster = _people(lang)
        third = ("he", "she") if lang == "en" else ("他", "她")

        def add(eid: str, kind: str, turns, asks) -> None:
            _author(eid, kind, turns, asks, roster)
            scene = bank.scene(eid, lang, kind).cast(*roster)
            for text, mentions, who in turns:
                scene.turn(text, mentions, who)
            for rel, question, human in asks:
                scene.ask(rel, question, human)
            scene.finish()

        # neg_subject: name is the subject, no address punct after it.
        sub = (
            lambda n, x, y: f"{n} left the {x} beside the {y} before the stack cooled.",
            lambda n, x, y: f"Once the stack settled, {n} shifted the {x} toward the {y}.",
            lambda n, x, y: f"{n} has the {x} sitting beside the {y} after the stack stopped.",
        ) if lang == "en" else (
            lambda n, x, y: f"{n}把{x}靠在{y}旁边，窑火还没停。",
            lambda n, x, y: f"泥缝干了以后，{n}把{x}挪到{y}这边。",
            lambda n, x, y: f"{n}把{x}留在{y}旁边，织机已经停了。",
        )
        for i, name in enumerate((a, b, c)):
            x, y = props.take()
            add(
                f"{lang}-neg-subject-{i}", "neg_subject",
                ((sub[i](name, x, y), (name,), ""),),
                _asks_you(lang, ("phone", "birthday"), x, y, "ABSENT", i % 2),
            )

        obj = (
            lambda n, x, y: f"The clerk set the {x} beside {n} before the stack cooled.",
            lambda n, x, y: f"A runner handed the {x} to {n} next to the {y}.",
            lambda n, x, y: f"The desk copy kept the {x} near {n} until the {y} cooled.",
        ) if lang == "en" else (
            lambda n, x, y: f"柜员把{x}放在{n}旁边，{y}还在。",
            lambda n, x, y: f"跑腿的人把{x}递给{n}，{y}就在手边。",
            lambda n, x, y: f"页边抄件把{x}留在{n}近旁，直到{y}凉下来。",
        )
        for i, name in enumerate((a, b, c)):
            x, y = props.take()
            add(
                f"{lang}-neg-object-{i}", "neg_object",
                ((obj[i](name, x, y), (name,), ""),),
                _asks_you(lang, ("phone", "birthday"), x, y, "ABSENT", i % 2),
            )

        if lang == "en":
            def quote_turn(i, name, x, y):
                if i == 0:
                    return (
                        f"\"{name}, the {x} stays beside the {y},\" was the whole margin note.",
                        (name,),
                    )
                if i == 1:
                    return (
                        f"{speaker} read aloud, \"{name}, leave the {x} beside the {y}.\"",
                        (speaker, name),
                    )
                return (
                    f"The margin note quoted, \"{name} left the {x} beside the {y}.\"",
                    (name,),
                )
        else:
            def quote_turn(i, name, x, y):
                if i == 0:
                    return (f"「{name}，{x}先靠着{y}。」写在页边的条子上。", (name,))
                if i == 1:
                    return (f"{speaker}念道：「{name}，把{x}留在{y}旁边。」", (speaker, name))
                return (f"页边写着：“{name}把{x}留在{y}旁边。”", (name,))

        for i, name in enumerate((a, b, c)):
            x, y = props.take()
            text, mentions = quote_turn(i, name, x, y)
            add(
                f"{lang}-neg-quote-{i}", "neg_quote",
                ((text, mentions, ""),),
                _asks_you(lang, ("phone", "birthday"), x, y, "ABSENT", i % 2),
            )

        # Two people, pronoun does not pick a gender. Two asks, still the same tie.
        x, y = props.take()
        if lang == "en":
            tie = f"{a} and {b} both signed the {x} beside the {y}."
        else:
            tie = f"{a}和{b}都在{x}上签了字，{y}就在旁边。"
        add(
            f"{lang}-neg-ambig-tie", "neg_ambig",
            ((tie, (a, b), hearer),),
            (
                ("phone", _third_q(lang, "phone", x, y, third[0]), "ABSENT"),
                ("birthday", _third_q(lang, "birthday", x, y, third[1]), "ABSENT"),
            ),
        )

        # Ambiguous pronoun must not refresh. One ask, while a restamp would survive.
        x, y = props.take()
        if lang == "en":
            both = f"{a} and {c} both marked the {x} beside the {y}."
            pronoun_line = f"He smoothed that {x} once the {y} had cooled."
            fillers = (
                f"The {x} can keep waiting beside the {y}.",
                f"Noted, leave the {x} beside the {y}.",
                f"One pause beside the {y} and the {x}.",
            )
            pronoun = "he"
        else:
            both = f"{a}和{c}都在{x}上做了记号，{y}就在旁边。"
            pronoun_line = f"他把{x}抚平，{y}已经凉了。"
            fillers = (
                f"嗯，{x}先停在{y}旁。",
                f"记下了，{x}和{y}先不动。",
                f"{y}这边的{x}再等一下。",
            )
            pronoun = "他"
        add(
            f"{lang}-neg-ambig-refresh", "neg_ambig",
            (
                (both, (a, c), hearer),
                (pronoun_line, (), hearer),
                *[(line, (), hearer) for line in fillers],
            ),
            (("phone", _third_q(lang, "phone", x, y, pronoun), "ABSENT"),),
        )

        x, y = props.take()
        if lang == "en":
            two = f"{a}, {c}, the {x} stays beside the {y}."
        else:
            two = f"{a}，{c}，{x}先靠着{y}。"
        add(
            f"{lang}-neg-ambig-two", "neg_ambig",
            ((two, (a, c), ""),),
            _asks_you(lang, ("phone", "birthday"), x, y, "ABSENT", 1),
        )

        x, y = props.take()
        if lang == "en":
            both = f"{b} and {c} both initialed the {x} beside the {y}."
            pronoun_line = f"She smoothed that {x} once the {y} had cooled."
            pronoun = "she"
        else:
            both = f"{b}和{c}都在{x}上画了押，{y}就在旁边。"
            pronoun_line = f"她把{x}抚平，{y}已经凉了。"
            pronoun = "她"
        add(
            f"{lang}-neg-ambig-refresh-she", "neg_ambig",
            (
                (both, (b, c), hearer),
                (pronoun_line, (), hearer),
                (f"The {x} stays put beside the {y}." if lang == "en" else f"{x}先留在{y}旁边。", (), hearer),
                (f"Still waiting on the {x} and the {y}." if lang == "en" else f"{y}和{x}先停在原地。", (), hearer),
                (f"No further mark on the {x}." if lang == "en" else f"{x}上没有新的记号。", (), hearer),
            ),
            (("birthday", _third_q(lang, "birthday", x, y, pronoun), "ABSENT"),),
        )

        # Unknown and near-miss vocatives. The token is not a roster string.
        unknowns = ("Sigrid", "Phelam") if lang == "en" else ("沈禾", "牟芷")
        for i, token in enumerate(unknowns):
            x, y = props.take()
            if lang == "en":
                text = f"{token}, the {x} stays beside the {y} tonight."
            else:
                text = f"{token}，{x}先靠着{y}。"
            add(
                f"{lang}-neg-unknown-{i}", "neg_unknown",
                ((text, (), ""),),
                _asks_you(lang, ("phone", "birthday"), x, y, "ABSENT", i % 2),
            )

        # Explicit hearer overrides the leading name. Three roster, two off-roster.
        overrides = (
            (b, "，" if lang == "zh" else ",", "phone"),
            (c, "！" if lang == "zh" else "!", "birthday"),
            (b, "：" if lang == "zh" else ":", "phone"),
            ("沈禾" if lang == "zh" else "Sigrid", "，" if lang == "zh" else ",", "phone"),
            ("unlisted", "！" if lang == "zh" else "!", "birthday"),
        )
        for i, (who, mark, rel) in enumerate(overrides):
            x, y = props.take()
            if lang == "en":
                text = f"{a}{mark} the {x} stays beside the {y}."
            else:
                text = f"{a}{mark}{x}先靠着{y}。"
            human = who if who in roster else "ABSENT"
            add(
                f"{lang}-neg-explicit-{i}", "neg_explicit",
                ((text, (a,), who),),
                ((rel, _you(lang, rel, x, y, i % 2), human),),
            )

        # Four turns with no reference to the person, then the pronoun.
        def age(eid: str, name: str, lines: Sequence[str], pronoun: str) -> None:
            x, y = props.take()
            if lang == "en":
                first = f"{name} set the {x} beside the {y} and stepped aside."
            else:
                first = f"{name}把{x}靠在{y}旁边就退开了。"
            turns = [(first, (name,), hearer)]
            turns.extend((line.format(a=x, b=y), (), hearer) for line in lines)
            add(
                eid, "neg_age", turns,
                (
                    ("phone", _third_q(lang, "phone", x, y, pronoun), "ABSENT"),
                    ("birthday", _third_q(lang, "birthday", x, y, pronoun), "ABSENT"),
                ),
            )

        if lang == "en":
            quiet_lines = (
                "Mm, the {a} can sit by the {b}.",
                "The {b} side of the {a} can wait.",
                "Noted, the {a} stays by the {b}.",
                "One moment beside the {b} and the {a}.",
            )
            them_lines = (
                "The {a} can sit beside them.",
                "Leave them by the {b} with the {a}.",
                "Nothing else was written beside them.",
                "The {b} still holds them and the {a}.",
            )
            it_lines = (
                "It stays on the {a} beside the {b}.",
                "Leave it with the {a} by the {b}.",
                "The {b} still holds it next to the {a}.",
                "Nothing new was added beside it or the {a}.",
            )
            age(f"{lang}-neg-age-quiet", a, quiet_lines, "he")
            age(f"{lang}-neg-age-them", b, them_lines, "she")
            age(f"{lang}-neg-age-it", c, it_lines, "he")
        else:
            quiet_lines = (
                "嗯，{a}可以靠着{b}。",
                "{b}那边的{a}先等一下。",
                "记下了，{a}就留在{b}旁。",
                "{b}和{a}旁边先停一下。",
            )
            them_lines = (
                "你们先看着{a}和{b}。",
                "先让你们把{a}留在{b}旁。",
                "记下你们看过的{a}，{b}还在。",
                "你们那边的{a}先停在{b}旁。",
            )
            it_lines = (
                "把它留在{a}和{b}旁边。",
                "{a}先搁着，它就在{b}旁。",
                "记下它压着{a}的样子，{b}还在。",
                "它还压在{a}和{b}中间。",
            )
            age(f"{lang}-neg-age-quiet", a, quiet_lines, "他")
            age(f"{lang}-neg-age-them", b, them_lines, "她")
            age(f"{lang}-neg-age-it", c, it_lines, "他")

        def keep(eid: str, name: str, pronoun: str) -> None:
            x, y = props.take()
            if lang == "en":
                first = f"{name} set the {x} beside the {y} and left the seam open."
                lines = (
                    f"{pronoun.capitalize()} folded the {x} once the {y} had cooled.",
                    f"{pronoun.capitalize()} was sure the {x} belonged on the {y} side.",
                    f"{pronoun.capitalize()} tapped the shelf that holds the {y} and the {x}.",
                    f"{pronoun.capitalize()} kept the stub under the {x} next to the {y}.",
                )
            else:
                first = f"{name}把{x}靠在{y}上，边缝还敞着。"
                lines = (
                    f"{y}凉了以后{pronoun}把{x}折好了。",
                    f"{pronoun}对{y}旁边那份{x}很确定。",
                    f"{pronoun}朝{y}外侧的窗台扬了扬手，{x}还靠在那儿。",
                    f"{pronoun}把纸条压在{x}和{y}中间。",
                )
            turns = [(first, (name,), hearer)]
            turns.extend((line, (), hearer) for line in lines)
            add(
                eid, "pos_keep", turns,
                (
                    ("phone", _third_q(lang, "phone", x, y, pronoun.lower() if lang == "en" else pronoun), name),
                    ("birthday", _third_q(lang, "birthday", x, y, pronoun.lower() if lang == "en" else pronoun), name),
                ),
            )

        keep(f"{lang}-pos-keep-he", a, "He" if lang == "en" else "他")
        keep(f"{lang}-pos-keep-she", b, "She" if lang == "en" else "她")

        for i, (name, mark) in enumerate((
            (a, "," if lang == "en" else "，"),
            (b, "!" if lang == "en" else "！"),
        )):
            x, y = props.take()
            if lang == "en":
                text = f"{name}{mark} the {x} stays beside the {y} for the evening."
            else:
                text = f"{name}{mark}{x}今晚靠着{y}。"
            add(
                f"{lang}-pos-voc-{i}", "pos_voc",
                ((text, (name,), ""),),
                _asks_you(lang, ("phone", "birthday"), x, y, name, i % 2),
            )


def _author(eid: str, kind: str, turns, asks, roster: Sequence[str]) -> None:
    """String-shape checks. These do not require the listener label to match."""
    texts = [turn[0] for turn in turns]
    if kind in ("neg_subject", "neg_object", "neg_quote", "neg_age", "pos_keep"):
        for text, _mentions, _who in turns:
            if _authored_vocatives(text, roster):
                raise RuntimeError(f"{eid} authored a leading vocative: {text!r}")
    if kind == "neg_object":
        text, mentions, _who = turns[0]
        for name in mentions:
            if any(start == 0 for start, _end in name_spans(name, text.lstrip())):
                raise RuntimeError(f"{eid} object name is initial: {text!r}")
    if kind == "neg_quote":
        if not any(mark in texts[0] for mark in "\"“«「『"):
            raise RuntimeError(f"{eid} quote mark missing: {texts[0]!r}")
    if kind == "neg_age":
        if len(turns) != 5:
            raise RuntimeError(f"{eid} age chain is {len(turns)} turns")
        target = turns[0][1][0]
        for text, mentions, _who in turns[1:]:
            if mentions or _has_name(text, target) or not _pronoun_free(text):
                raise RuntimeError(f"{eid} filler refers to someone: {text!r}")
    if kind == "pos_keep":
        if len(turns) != 5:
            raise RuntimeError(f"{eid} keep chain is {len(turns)} turns")
        target = turns[0][1][0]
        for text, mentions, _who in turns[1:]:
            if mentions or _has_name(text, target) or not _third_person(text) or _second_person(text):
                raise RuntimeError(f"{eid} keep line is not a bare pronoun: {text!r}")
    if kind == "neg_unknown":
        text = texts[0].lstrip()
        token = text.split(",")[0].split("，")[0]
        if token in roster:
            raise RuntimeError(f"{eid} unknown token is on the roster")
        if not text.startswith(token) or text[len(token)] not in _ADDRESS_PUNCT:
            raise RuntimeError(f"{eid} unknown vocative shape: {text!r}")
    if kind == "pos_voc" or kind == "neg_explicit":
        text = texts[0]
        found = _authored_vocatives(text, roster)
        if len(found) != 1:
            raise RuntimeError(f"{eid} expected one leading name, got {found}: {text!r}")
    if kind == "neg_ambig" and eid.endswith("-two"):
        found = _authored_vocatives(turns[0][0], roster)
        if len(found) != 1 or len(turns[0][1]) != 2:
            raise RuntimeError(f"{eid} two-vocative shape {found} {turns[0][1]}")
    if kind == "neg_ambig" and "refresh" in eid:
        if len(turns) != 5 or len(turns[0][1]) != 2:
            raise RuntimeError(f"{eid} refresh probe shape")
        if not _third_person(turns[1][0]) or turns[1][1]:
            raise RuntimeError(f"{eid} refresh pronoun line")
        for text, mentions, _who in turns[2:]:
            if mentions or not _pronoun_free(text):
                raise RuntimeError(f"{eid} refresh filler: {text!r}")
    for _rel, question, _human in asks:
        if _first_person(question):
            raise RuntimeError(f"{eid} first person in {question!r}")
    for text, _mentions, _who in turns:
        if _first_person(text):
            raise RuntimeError(f"{eid} first person in {text!r}")


def _first_person(text: str) -> bool:
    if "我" in text:
        return True
    return re.search(r"\bI\b", text) is not None


def _check(rows: Sequence[Episode]) -> None:
    eids = [row.eid for row in rows]
    if len(eids) != len(set(eids)):
        raise RuntimeError("duplicate episode id")
    asks = [ask for row in rows for ask in row.events if isinstance(ask, Ask)]
    texts: List[str] = []
    for row in rows:
        for slot in row.slots:
            texts.append(slot.text)
            if slot.value.lower() in slot.text.lower():
                raise RuntimeError(f"value leaked into {slot.text}")
            if _first_person(slot.text):
                raise RuntimeError(f"first person in {slot.text}")
        for event in row.events:
            if isinstance(event, Turn):
                texts.append(event.text)
            elif isinstance(event, Ask):
                texts.append(event.question)
    if len(texts) != len(set(texts)):
        dupes = sorted({text for text in texts if texts.count(text) > 1})
        raise RuntimeError(f"duplicate bank string: {dupes[:4]}")
    counts: Dict[str, int] = {}
    langs = {"en": 0, "zh": 0}
    for ask in asks:
        counts[ask.kind] = counts.get(ask.kind, 0) + 1
        langs[ask.lang] += 1
    if counts != _KINDS:
        raise RuntimeError(f"kinds {counts}")
    if langs["en"] != langs["zh"] or langs["en"] != 47:
        raise RuntimeError(f"lang balance {langs}")
    if TURN_WINDOW != 4:
        raise RuntimeError(f"window {TURN_WINDOW}")
    absent = 0
    for ask in asks:
        if ask.human == "absent_person":
            absent += 1
    if absent != 72:
        raise RuntimeError(f"listener abstentions {absent}")


@lru_cache(maxsize=1)
def build_rows() -> Tuple[Episode, ...]:
    bank = Bank()
    _load(bank)
    rows = tuple(bank.episodes)
    _check(rows)
    return rows


def iter_asks(rows: Optional[Sequence[Episode]] = None):
    rows = build_rows() if rows is None else rows
    for row in rows:
        for event in row.events:
            if isinstance(event, Ask):
                yield row, event


def questions(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    return tuple(ask.question for _, ask in iter_asks(rows))


def bank_strings(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen: List[str] = []
    bag = set()
    for row in rows:
        texts = [slot.text for slot in row.slots]
        for event in row.events:
            if isinstance(event, Turn):
                texts.append(event.text)
            elif isinstance(event, Ask):
                texts.append(event.question)
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


def kind_counts(rows: Optional[Sequence[Episode]] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, ask in iter_asks(rows):
        counts[ask.kind] = counts.get(ask.kind, 0) + 1
    return counts


def listener_gaps(rows: Optional[Sequence[Episode]] = None) -> Tuple[dict, ...]:
    """Rows whose replayed rule label is not the pre-registered listener label."""
    gaps = []
    for row, ask in iter_asks(rows):
        if ask.rule != ask.human or ask.rule_golds != ask.human_golds:
            gaps.append({
                "eid": row.eid,
                "kind": ask.kind,
                "lang": ask.lang,
                "question": ask.question,
                "listener": ask.who_human,
                "rule": ask.who_rule,
                "listener_label": ask.human,
                "rule_label": ask.rule,
            })
    return tuple(gaps)


if __name__ == "__main__":
    rows = build_rows()
    asks = list(iter_asks(rows))
    print(f"episodes {len(rows)} asks {len(asks)}")
    print("kinds", kind_counts(rows))
    print("questions", questions_sha256(rows))
    print("bank", bank_sha256(rows), "strings", len(bank_strings(rows)))
    gaps = listener_gaps(rows)
    print(f"listener gaps {len(gaps)}")
    for gap in gaps:
        print(
            f"  {gap['eid']} {gap['kind']} listener {gap['listener']} "
            f"rule {gap['rule']} {gap['question']}"
        )
