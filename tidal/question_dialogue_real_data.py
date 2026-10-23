"""Natural multi-turn episodes for the shipped ask path.

The synthetic dialogue and deixis banks are template grids. These episodes
are short conversations: introductions, a hearer, a third person, a later
question. Rule labels are not hand-fit. ``build_rows`` replays ``DialogueState``,
``ask_people``, and the deixis row selectors, then checks the human reading
against that replay. Gated rows already agree. Gap rows disagree on purpose.
Open addressee forms from the deixis held-out stay out of the gate.

The frozen adversarial file is not an input. ``ABSTAIN_MARGIN`` and
``LEX_RIDGE_GAP`` are not read here. ``LLMMemory`` is not imported.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from tidal.dialogue_state import DialogueState
from tidal.question_name_align import diagnose
from tidal.question_surface import (
    ask_people,
    is_deictic,
    is_indirect_hearer,
    lex_labels,
    matched_fact_rows,
    selected_fact_rows,
)
from tidal.question_surface import _second_person, _third_person

_REL = ("phone", "birthday", "places")
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
class Ask:
    phenomenon: str
    scope: str
    band: str
    family: str
    rel: str
    question: str
    lang: str
    rule: str
    human: str
    rule_golds: Tuple[str, ...]
    human_golds: Tuple[str, ...]
    empty_rule: str
    empty_golds: Tuple[str, ...]
    note: str
    who_rule: str
    who_human: str


@dataclass(frozen=True)
class Episode:
    eid: str
    lang: str
    title: str
    slots: Tuple[Slot, ...]
    events: Tuple[object, ...]


def _text_lang(text: str) -> str:
    cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    lat = any(ch.isascii() and ch.isalpha() for ch in text)
    if cjk and lat:
        return "bi"
    return "zh" if cjk else "en"


def _first(slots: Sequence[Slot], rel: str) -> str:
    for slot in slots:
        if slot.rel == rel:
            return slot.value
    raise RuntimeError(f"no {rel} slot")


def _value(slots: Sequence[Slot], name: str, rel: str) -> str:
    for slot in slots:
        if slot.name == name and slot.rel == rel:
            return slot.value
    raise RuntimeError(f"{name} has no {rel}")


def _human_outcome(spec: str, slots: Sequence[Slot], rel: str) -> Tuple[str, Tuple[str, ...], str]:
    if spec == "ABSENT":
        return "absent_person", (), "ABSENT"
    if spec == "MISSING":
        return "missing_rel", (), "MISSING"
    if spec == "DECOY":
        return "answerable", (_first(slots, rel),), "DECOY"
    return "answerable", (_value(slots, spec, rel),), spec


def _predict(state: DialogueState, slots: Sequence[Slot], question: str, rel: str) -> Tuple[str, Tuple[str, ...], str]:
    """Shipped string decision. The tied encoder is assumed to agree with one lexical cue.

    A failed deixis resolution is ``absent_person`` even when the question
    also says ``I``. An empty log leaves a deictic question on the ranker,
    which returns the first stored fact of ``rel``.
    """
    names = [slot.name for slot in slots]
    surfaces = [slot.text for slot in slots]
    active = state.conversational(ignore_surface=question)
    selected = selected_fact_rows(question, names, surfaces, active, state)
    matched = matched_fact_rows(question, names, surfaces, active, state)
    if selected == []:
        return "absent_person", (), "absent_person"
    if selected is None:
        if is_deictic(question):
            return "answerable", (_first(slots, rel),), "DECOY"
        return "absent_person", (), "absent_person"
    if not any(slots[i].rel == rel for i in selected):
        return "missing_rel", (), "missing_rel"
    pool = range(len(slots)) if matched is None else matched
    if matched == []:
        return "absent_person", (), "absent_person"
    winners = [slots[i] for i in pool if slots[i].rel == rel]
    if not winners:
        return "missing_rel", (), "missing_rel"
    chosen = {slots[i].name for i in selected}
    if winners[0].name not in chosen:
        raise RuntimeError(f"ranker {winners[0].name} outside {chosen} for {question}")
    return "answerable", (winners[0].value,), winners[0].name


class _Scene:
    def __init__(self, bank: "Bank", eid: str, lang: str, title: str, prop: str, decoy: str, fact_lang: str) -> None:
        self.bank = bank
        self.eid = eid
        self.lang = lang
        self.title = title
        self.prop = prop
        self.decoy = decoy
        self.fact_lang = fact_lang
        self.people: List[Tuple[str, Tuple[str, ...], str]] = []
        self.scripts: list = []

    def person(self, name: str, rels: Sequence[str], self_rel: str = "") -> "_Scene":
        self.people.append((name, tuple(rels), self_rel))
        return self

    def turn(self, speaker: str, hearer: str, text: str, mentions: Sequence[str]) -> "_Scene":
        self.scripts.append(("t", speaker, hearer, text, tuple(mentions)))
        return self

    def ask(
        self,
        phenomenon: str,
        scope: str,
        band: str,
        family: str,
        rel: str,
        question: str,
        human: str,
        note: str = "",
    ) -> "_Scene":
        self.scripts.append(("a", phenomenon, scope, band, family, rel, question, human, note))
        return self

    def _slot(self, name: str, rel: str, self_fp: bool) -> Slot:
        cue = _CUE[self.fact_lang][rel]
        if self_fp and self.fact_lang == "zh":
            text = f"我把{cue}留在{self.prop}上。"
        elif self_fp:
            text = f"I keep my {cue} on the {self.prop}."
        elif self.fact_lang == "zh":
            text = f"{self.prop}记下了{name}的{cue}。"
        else:
            text = f"The {self.prop} lists the {cue} under {name}."
        return Slot(name, rel, self.bank.value(), text)

    def finish(self) -> None:
        slots: List[Slot] = []
        for rel in _REL:
            slots.append(self._slot(self.decoy, rel, False))
        selves = 0
        seen = {(self.decoy, rel) for rel in _REL}
        for name, rels, self_rel in self.people:
            if name == self.decoy:
                raise RuntimeError(f"{self.eid} reuses the decoy name {name}")
            if self_rel and self_rel not in rels:
                raise RuntimeError(f"{self.eid} {name} self {self_rel} not stored")
            for rel in rels:
                if rel not in _REL:
                    raise RuntimeError(f"{self.eid} bad rel {rel}")
                if (name, rel) in seen:
                    raise RuntimeError(f"{self.eid} duplicate {name} {rel}")
                seen.add((name, rel))
                fp = bool(self_rel) and rel == self_rel
                selves += int(fp)
                slots.append(self._slot(name, rel, fp))
        if selves > 1:
            raise RuntimeError(f"{self.eid} has {selves} first-person facts")
        names = [slot.name for slot in slots]
        surfaces = [slot.text for slot in slots]
        roster = set(names)
        state = DialogueState()
        events: List[object] = []
        for spec in self.scripts:
            if spec[0] == "t":
                _, speaker, hearer, text, mentions = spec
                if speaker and speaker not in roster:
                    raise RuntimeError(f"{self.eid} speaker {speaker} is not stored")
                got = ask_people(text, names, surfaces, state.conversational())
                if set(got) != set(mentions):
                    info = diagnose(text, names, state.conversational())
                    raise RuntimeError(
                        f"{self.eid} mentions {got} != {mentions} in {text!r} unknown={info['unknown']}"
                    )
                state.note_utterance(
                    got,
                    speaker=speaker or None,
                    hearer=hearer or None,
                    surface=text,
                )
                events.append(Turn(text, speaker, hearer))
                continue
            _, phenomenon, scope, band, family, rel, question, human, note = spec
            if rel not in _REL:
                raise RuntimeError(f"{self.eid} ask rel {rel}")
            active = state.conversational(ignore_surface=question)
            info = diagnose(question, names, active)
            rule, rule_golds, who = _predict(state, slots, question, rel)
            empty_rule, empty_golds, _empty_who = _predict(DialogueState(), slots, question, rel)
            human_label, human_golds, who_human = _human_outcome(human, slots, rel)
            if scope == "gated" and (rule, rule_golds) != (human_label, human_golds):
                raise RuntimeError(
                    f"{self.eid} {phenomenon} rule {rule}:{who} != human {human} "
                    f"q={question!r} unknown={info['unknown']} matched={info['matched']}"
                )
            if scope == "gap" and (rule, rule_golds) == (human_label, human_golds):
                raise RuntimeError(f"{self.eid} {phenomenon} was marked gap but agrees: {question!r}")
            if scope == "gated" and band == "ordinary" and rule != "answerable":
                raise RuntimeError(f"{self.eid} ordinary ask abstains: {question!r}")
            if scope == "gap" and not note:
                raise RuntimeError(f"{self.eid} {phenomenon} gap has no note")
            if scope == "oos":
                if _second_person(question) or _third_person(question) or is_indirect_hearer(question):
                    raise RuntimeError(f"{self.eid} oos form is inside the closed class: {question!r}")
                if not note:
                    raise RuntimeError(f"{self.eid} oos has no note")
            if scope not in ("gated", "gap", "oos", "control", "observe"):
                raise RuntimeError(f"{self.eid} scope {scope}")
            if family not in ("you", "third", "name", "edit", "abstain", "indirect", "window", "oos", "empty"):
                raise RuntimeError(f"{self.eid} family {family}")
            events.append(Ask(
                phenomenon, scope, band, family, rel, question, _text_lang(question),
                rule, human_label, rule_golds, human_golds, empty_rule, empty_golds,
                note, who, who_human,
            ))
            mentioned = ask_people(question, names, surfaces, state.conversational(ignore_surface=question))
            state.note_ask(mentioned, question)
        self.bank.episodes.append(Episode(
            self.eid, self.lang, self.title, tuple(slots), tuple(events),
        ))


class Bank:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []
        self._n = 0

    def value(self) -> str:
        self._n += 1
        return f"rld{self._n:04d}"

    def scene(
        self,
        eid: str,
        lang: str,
        title: str,
        prop: str,
        decoy: str,
        fact_lang: str = "",
    ) -> _Scene:
        return _Scene(self, eid, lang, title, prop, decoy, fact_lang or lang)


def _load(bank: Bank) -> None:
    # Ordinary conversations. Hearer is who was just addressed. A third-person
    # pronoun is the other person just named. Exact names do not need the window.
    s = bank.scene("en-tin", "en", "Thursday close", "green tin from the thursday close", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("phone", "places"))
    s.person("Torven", ("phone", "birthday"))
    s.turn(
        "Pellam", "Briony",
        "Briony, I set the green tin on the counter after Torven dropped his barley card in it.",
        ("Briony", "Torven"),
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "What telephone should the closer use for you off the green tin from the thursday close?",
        "Briony",
    )
    s.turn("Pellam", "Briony", "The barley card is his, once the queue thins.", ())
    s.ask(
        "third_unique", "gated", "ordinary", "third", "birthday",
        "Which birthday did he pencil on the barley card in the green tin from the thursday close?",
        "Torven",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "places",
        "Which places did Pellam keep on the linen tag for the thursday close?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("en-ferry", "en", "Tide window", "ferry chit under the brass clip", "Sable")
    s.person("Quillan", ("phone", "birthday", "places"), self_rel="phone")
    s.person("Odhran", ("birthday", "phone"))
    s.person("Elowen", ("phone", "birthday"))
    s.person("Warrick", ("places", "birthday"))
    s.turn(
        "Warrick", "Quillan",
        "Quillan, the ferry chit under the brass clip has Odhran on the front.",
        ("Quillan", "Odhran"),
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "birthday",
        "Which birthday should the desk read for you off the ferry chit under the brass clip?",
        "Quillan",
    )
    s.turn(
        "Warrick", "Quillan",
        "Elowen added a line on the back after the tide turned.",
        ("Elowen",),
    )
    s.ask(
        "third_later", "gated", "ordinary", "third", "phone",
        "Which telephone did she add on the ferry chit under the brass clip?",
        "Elowen",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "places",
        "Which places did Warrick keep on the ferry chit under the brass clip?",
        "Warrick",
    )
    s.finish()

    s = bank.scene("en-clinic", "en", "Rear door", "chalk slate by the rear door", "Sable")
    s.person("Tamsin", ("phone", "places"), self_rel="phone")
    s.person("Maelis", ("birthday", "places"))
    s.person("Lovell", ("phone", "birthday"))
    s.turn(
        "Maelis", "Tamsin",
        "Tamsin, the chalk slate by the rear door is out, and Lovell left his card under it.",
        ("Tamsin", "Lovell"),
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "birthday",
        "Which birthday did Maelis write on the chalk slate by the rear door?",
        "Maelis",
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "Your telephone on the chalk slate by the rear door is the one the night desk should ring?",
        "Tamsin",
    )
    s.ask(
        "third_unique", "gated", "ordinary", "third", "birthday",
        "Which birthday did he leave under the chalk slate by the rear door?",
        "Lovell",
    )
    s.finish()

    s = bank.scene("en-rehearsal", "en", "Window ledge", "moss folder on the window ledge", "Sable")
    s.person("Nessa", ("birthday", "places"), self_rel="birthday")
    s.person("Corin", ("phone", "places"))
    s.person("Ismay", ("phone", "birthday"))
    s.turn(
        "Corin", "Nessa",
        "Nessa, Ismay marked the moss folder on the window ledge before rehearsal.",
        ("Nessa", "Ismay"),
    )
    s.ask(
        "third_unique", "gated", "ordinary", "third", "phone",
        "Which telephone did she mark on the moss folder on the window ledge?",
        "Ismay",
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "places",
        "Which places should the desk keep for you on the moss folder on the window ledge?",
        "Nessa",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "phone",
        "Which telephone did Corin leave on the moss folder on the window ledge?",
        "Corin",
    )
    s.finish()

    # "Which places should I leave aside" contains I. you still resolves first.
    # Checked by the gated assert: the hearer, not the ranker.

    s = bank.scene("en-kitchen", "en", "Low shelf", "copper bowl on the low shelf", "Sable")
    s.person("Elowen", ("phone", "birthday", "places"), self_rel="phone")
    s.person("Vesper", ("birthday", "phone"))
    s.person("Quillan", ("phone", "places"))
    s.turn(
        "Vesper", "Elowen",
        "Elowen, the copper bowl on the low shelf is yours tonight.",
        ("Elowen",),
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "What telephone did you leave with the copper bowl on the low shelf?",
        "Elowen",
    )
    s.ask(
        "you_followup", "gated", "ordinary", "you", "birthday",
        "And which birthday on that same copper bowl on the low shelf is yours?",
        "Elowen",
    )
    s.turn(
        "Vesper", "Elowen",
        "Quillan set his card beside the bowl before he went out.",
        ("Quillan",),
    )
    s.ask(
        "third_unique", "gated", "ordinary", "third", "places",
        "Which places did he set beside the copper bowl on the low shelf?",
        "Quillan",
    )
    s.finish()

    s = bank.scene("en-map", "en", "Under the stair", "reed basket under the stair", "Sable")
    s.person("Briony", ("phone", "places"), self_rel="places")
    s.person("Odhran", ("phone", "birthday"))
    s.person("Warrick", ("birthday", "phone"))
    s.turn(
        "Odhran", "Briony",
        "Briony, I tucked the reed basket under the stair. Warrick's card is in the side pocket.",
        ("Briony", "Warrick"),
    )
    s.ask(
        "third_unique", "gated", "ordinary", "third", "birthday",
        "Which birthday is on the card he left in the reed basket under the stair?",
        "Warrick",
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "places",
        "Which places did you want kept on the reed basket under the stair?",
        "Briony",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "phone",
        "Which telephone did Odhran leave on the reed basket under the stair?",
        "Odhran",
    )
    s.finish()

    s = bank.scene("en-intro", "en", "Sill jar", "pebble jar on the sill", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("phone", "places"))
    s.person("Torven", ("birthday", "phone"))
    s.turn(
        "Pellam", "Briony",
        "Briony, I'm Pellam. I put my card next to yours on the pebble jar on the sill.",
        ("Briony", "Pellam"),
    )
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "What telephone should I read back for you off the pebble jar on the sill?",
        "Briony",
    )
    s.turn(
        "Pellam", "Briony",
        "Torven will come by for his card after supper.",
        ("Torven",),
    )
    s.ask(
        "third_unique", "gated", "ordinary", "third", "birthday",
        "Which birthday should we keep for him off the pebble jar on the sill?",
        "Torven",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "places",
        "Which places did Pellam leave on the pebble jar on the sill?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("en-tram", "en", "Clay cup", "tram stub in the clay cup", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("places", "phone"))
    s.person("Torven", ("birthday", "phone"))
    s.turn(
        "Pellam", "Briony",
        "Briony, the tram stub in the clay cup is from Torven. He said the closer can ring you.",
        ("Briony", "Torven"),
    )
    s.turn("Pellam", "Briony", "I will read the cup card once the queue thins.", ())
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "Which telephone should the closer use for you off the tram stub in the clay cup?",
        "Briony",
    )
    s.turn("Pellam", "Briony", "Torven also penciled a date on the back of that stub.", ("Torven",))
    s.ask(
        "third_unique", "gated", "ordinary", "third", "birthday",
        "Which birthday did he pencil on the tram stub in the clay cup?",
        "Torven",
    )
    s.ask(
        "name_exact", "gated", "ordinary", "name", "places",
        "Which places did Pellam keep on the tram stub in the clay cup?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-jar", "zh", "柜台青瓷", "青瓷罐", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("birthday", "places"))
    s.person("霍涟", ("places", "phone"))
    s.turn("阮澄", "闵昭", "闵昭，青瓷罐放在柜台边上了，霍涟把纸条压在罐盖下面。", ("闵昭", "霍涟"))
    s.ask("you_hearer", "gated", "ordinary", "you", "phone", "值夜要回读你的电话，就看青瓷罐上的那一页。", "闵昭")
    s.turn("阮澄", "闵昭", "纸条是她的，等队列短一点再对。", ())
    s.ask("third_unique", "gated", "ordinary", "third", "places", "她的地点记在青瓷罐的哪一页？", "霍涟")
    s.ask("name_exact", "gated", "ordinary", "name", "birthday", "阮澄的生日记在青瓷罐的哪一页？", "阮澄")
    s.finish()

    s = bank.scene("zh-ferry", "zh", "渡口夹板", "渡轮条", "靳晚")
    s.person("闵昭", ("phone", "birthday", "places"), self_rel="phone")
    s.person("邢渚", ("birthday", "phone"))
    s.person("喻棠", ("phone", "birthday"))
    s.person("卫聆", ("places", "birthday"))
    s.turn("卫聆", "闵昭", "闵昭，渡轮条先夹在板上，邢渚写在正面。", ("闵昭", "邢渚"))
    s.ask("you_hearer", "gated", "ordinary", "you", "birthday", "柜台上要你的生日，按渡轮条上的那一页回读。", "闵昭")
    s.turn("卫聆", "闵昭", "喻棠改口了，说背面那行才算。", ("喻棠",))
    s.ask("third_later", "gated", "ordinary", "third", "phone", "他的电话记在渡轮条的哪一页？", "喻棠")
    s.ask("name_exact", "gated", "ordinary", "name", "places", "卫聆的地点记在渡轮条的哪一页？", "卫聆")
    s.finish()

    s = bank.scene("zh-clinic", "zh", "后门粉笔", "粉笔板", "靳晚")
    s.person("霍涟", ("phone", "places"), self_rel="phone")
    s.person("岑汐", ("birthday", "places"))
    s.person("柏峤", ("phone", "birthday"))
    s.turn("岑汐", "霍涟", "霍涟，后门的粉笔板翻出来了，柏峤的卡片压在下面。", ("霍涟", "柏峤"))
    s.ask("name_exact", "gated", "ordinary", "name", "birthday", "岑汐的生日写在粉笔板的哪一页？", "岑汐")
    s.ask("you_hearer", "gated", "ordinary", "you", "phone", "你的电话在粉笔板上，夜班要打的是哪一页？", "霍涟")
    s.ask("third_unique", "gated", "ordinary", "third", "birthday", "他压在粉笔板下的生日是哪一页？", "柏峤")
    s.finish()

    s = bank.scene("zh-rehearsal", "zh", "窗台夹", "苔藓夹", "靳晚")
    s.person("邢渚", ("birthday", "places"), self_rel="birthday")
    s.person("卫聆", ("phone", "places"))
    s.person("霍涟", ("phone", "birthday"))
    s.turn("卫聆", "邢渚", "邢渚，排练前霍涟在窗台的苔藓夹上做了记号。", ("邢渚", "霍涟"))
    s.ask("third_unique", "gated", "ordinary", "third", "phone", "她在苔藓夹上留的电话是哪一页？", "霍涟")
    s.ask("you_hearer", "gated", "ordinary", "you", "places", "苔藓夹上要为你留的地点是哪一页？", "邢渚")
    s.ask("name_exact", "gated", "ordinary", "name", "phone", "卫聆的电话记在苔藓夹的哪一页？", "卫聆")
    s.finish()

    s = bank.scene("zh-kitchen", "zh", "矮架铜碗", "铜碗", "靳晚")
    s.person("闵昭", ("phone", "birthday", "places"), self_rel="phone")
    s.person("阮澄", ("birthday", "phone"))
    s.person("喻棠", ("phone", "places"))
    s.turn("阮澄", "闵昭", "闵昭，今晚矮架上的铜碗归你。", ("闵昭",))
    s.ask(
        "you_hearer", "gated", "ordinary", "you", "phone",
        "今晚矮架上那只铜碗旁边，你留下的电话要看哪一页？",
        "闵昭",
    )
    s.ask("you_followup", "gated", "ordinary", "you", "birthday", "同一只铜碗上，你的生日又是哪一页？", "闵昭")
    s.turn("阮澄", "闵昭", "喻棠出门前把卡片放在碗边。", ("喻棠",))
    s.ask("third_unique", "gated", "ordinary", "third", "places", "他放在铜碗边的地点是哪一页？", "喻棠")
    s.finish()

    s = bank.scene("zh-map", "zh", "楼梯下", "芦苇篮", "靳晚")
    s.person("霍涟", ("phone", "places"), self_rel="places")
    s.person("柏峤", ("phone", "birthday"))
    s.person("卫聆", ("birthday", "phone"))
    s.turn("柏峤", "霍涟", "霍涟，芦苇篮塞在楼梯下面了，侧袋里是卫聆的卡片。", ("霍涟", "卫聆"))
    s.ask("third_unique", "gated", "ordinary", "third", "birthday", "他留在芦苇篮里的生日是哪一页？", "卫聆")
    s.ask("you_hearer", "gated", "ordinary", "you", "places", "你要留在芦苇篮上的地点是哪一页？", "霍涟")
    s.ask("name_exact", "gated", "ordinary", "name", "phone", "柏峤的电话记在芦苇篮的哪一页？", "柏峤")
    s.finish()

    s = bank.scene("zh-intro", "zh", "窗台罐", "窗台小罐", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("phone", "places"))
    s.person("霍涟", ("birthday", "phone"))
    s.turn("阮澄", "闵昭", "闵昭，我是阮澄。卡片放在窗台小罐旁边，挨着你的那张。", ("闵昭", "阮澄"))
    s.ask("you_hearer", "gated", "ordinary", "you", "phone", "窗台小罐上要回读给你的电话是哪一页？", "闵昭")
    s.turn("阮澄", "闵昭", "霍涟晚饭后来拿她的卡片。", ("霍涟",))
    s.ask("third_unique", "gated", "ordinary", "third", "birthday", "她的生日在窗台小罐上是哪一页？", "霍涟")
    s.ask("name_exact", "gated", "ordinary", "name", "places", "阮澄的地点记在窗台小罐的哪一页？", "阮澄")
    s.finish()

    s = bank.scene("zh-tram", "zh", "蓝票夹", "蓝票夹", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("places", "phone"))
    s.person("喻棠", ("birthday", "phone"))
    s.turn("阮澄", "闵昭", "闵昭，蓝票夹里那张是喻棠留下的。他说值夜可以先打给你。", ("闵昭", "喻棠"))
    s.turn("阮澄", "闵昭", "队列短一点我再对夹子里的卡片。", ())
    s.ask("you_hearer", "gated", "ordinary", "you", "phone", "值夜该用你的哪一页电话，蓝票夹上的那页？", "闵昭")
    s.turn("阮澄", "闵昭", "喻棠在票背又补了一行日子。", ("喻棠",))
    s.ask("third_unique", "gated", "ordinary", "third", "birthday", "他补在蓝票夹上的生日是哪一页？", "喻棠")
    s.ask("name_exact", "gated", "ordinary", "name", "places", "阮澄的地点记在蓝票夹的哪一页？", "阮澄")
    s.finish()

    s = bank.scene(
        "bi-envelope", "bi", "Navy envelope", "navy envelope behind the clock", "靳晚", fact_lang="en",
    )
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("Briony", ("phone", "places"))
    s.person("阮澄", ("birthday",))
    s.turn("Briony", "闵昭", "闵昭，青瓷罐先放在你手边。", ("闵昭",))
    s.ask(
        "code_switch", "gated", "ordinary", "you", "phone",
        "Which telephone should the closer read for you off the navy envelope behind the clock?",
        "闵昭",
    )
    s.ask(
        "code_switch", "gated", "ordinary", "name", "birthday",
        "Which birthday did 闵昭 write on the navy envelope behind the clock?",
        "闵昭",
    )
    s.finish()

    s = bank.scene(
        "bi-flax", "bi", "Flax sheet", "flax sheet on the peg", "Sable", fact_lang="en",
    )
    s.person("Briony", ("phone", "birthday"))
    s.person("闵昭", ("birthday", "phone"), self_rel="phone")
    s.person("阮澄", ("places",))
    s.turn(
        "阮澄", "闵昭",
        "Briony left the flax sheet on the peg before the queue started.",
        ("Briony",),
    )
    s.ask("code_switch", "gated", "ordinary", "third", "phone", "她的电话写在蓝纸封的哪一页？", "Briony")
    s.ask("code_switch", "gated", "ordinary", "you", "birthday", "你的生日写在蓝纸封的哪一页？", "闵昭")
    s.finish()

    # Boundary. Still inside the contract: three quiet turns stay in the window,
    # an exact name survives four, an edit-1 does not survive silence or a tie.
    s = bank.scene("en-hold", "en", "Three nods", "twine card on the peg", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Torven left a folded note on the twine card on the peg.", ("Torven",))
    s.turn("Pellam", "Briony", "Right, the twine card can sit.", ())
    s.turn("Pellam", "Briony", "I have the peg.", ())
    s.turn("Pellam", "Briony", "Go on whenever the queue breaks.", ())
    s.ask(
        "third_hold", "gated", "boundary", "window", "phone",
        "Which telephone did he leave on the twine card on the peg?",
        "Torven",
    )
    s.finish()

    s = bank.scene("zh-hold", "zh", "三声嗯", "藤圈", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "喻棠把折好的条子留在藤圈上。", ("喻棠",))
    s.turn("阮澄", "闵昭", "嗯，藤圈先放着。", ())
    s.turn("阮澄", "闵昭", "我拿着挂钩。", ())
    s.turn("阮澄", "闵昭", "队列断一下你再接着说。", ())
    s.ask("third_hold", "gated", "boundary", "window", "phone", "他留在藤圈上的电话是哪一页？", "喻棠")
    s.finish()

    s = bank.scene("en-name-aged", "en", "Four nods, full name", "oat sack in the dry bin", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam stacked the oat sack in the dry bin before the closer arrived.", ("Pellam",))
    s.turn("Pellam", "Briony", "Mm, the dry bin is fine.", ())
    s.turn("Pellam", "Briony", "The oat sack can wait there.", ())
    s.turn("Pellam", "Briony", "I heard the bin.", ())
    s.turn("Pellam", "Briony", "One moment on the sack.", ())
    s.ask(
        "name_aged", "gated", "boundary", "name", "phone",
        "Which telephone did Pellam leave on the oat sack in the dry bin?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-name-aged", "zh", "四声之后的全名", "纸带圈", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone", "birthday"))
    s.turn("阮澄", "闵昭", "阮澄把纸带圈叠好放在干架上了。", ("阮澄",))
    s.turn("阮澄", "闵昭", "嗯，干架那边可以。", ())
    s.turn("阮澄", "闵昭", "这条纸带先放着。", ())
    s.turn("阮澄", "闵昭", "我听见干架了。", ())
    s.turn("阮澄", "闵昭", "纸带等一下。", ())
    s.ask("name_aged", "gated", "boundary", "name", "phone", "阮澄的电话记在纸带圈的哪一页？", "阮澄")
    s.finish()

    s = bank.scene("en-pronoun-hold", "en", "One pronoun turn", "horn cup by the tap", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Torven left the horn cup by the tap.", ("Torven",))
    s.turn("Pellam", "Briony", "He said the closer can ring after dusk.", ())
    s.ask(
        "pronoun_hold", "gated", "boundary", "third", "phone",
        "Which telephone did he leave with the horn cup by the tap?",
        "Torven",
    )
    s.finish()

    s = bank.scene("zh-pronoun-hold", "zh", "一句他", "角杯", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "喻棠把角杯留在水龙头旁边。", ("喻棠",))
    s.turn("阮澄", "闵昭", "他说黄昏以后值夜可以打。", ())
    s.ask("pronoun_hold", "gated", "boundary", "third", "phone", "他留在角杯上的电话是哪一页？", "喻棠")
    s.finish()

    s = bank.scene("en-edit-sub", "en", "Pellum", "paper band around the spare key", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn(
        "Pellam", "Briony",
        "Briony, Pellam left the paper band around the spare key on the hook.",
        ("Briony", "Pellam"),
    )
    s.ask(
        "edit1_sub", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellum leave on the paper band around the spare key?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-edit-sub", "zh", "闵照", "软木盘", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.turn("阮澄", "霍涟", "霍涟，闵昭把软木盘留在钩子上。", ("霍涟", "闵昭"))
    s.person("阮澄", ("places",))
    # person() after turn() is illegal: slots are built at finish, but the
    # turn is only replayed then, so order of person() vs turn() does not
    # matter. Keep people registered before finish. 阮澄 was referenced above
    # as speaker and must be stored. Registered next line is too late if we
    # already called turn with speaker 阮澄 — finish checks the roster, and
    # person() before finish is enough. The call above already happened.
    s.ask("edit1_sub", "gated", "boundary", "edit", "phone", "闵照的电话记在软木盘的哪一页？", "闵昭")
    s.finish()

    s = bank.scene("en-edit-del", "en", "Pelam", "slate shard in the blue box", "Sable")
    s.person("Briony", ("birthday",), self_rel="birthday")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam slid the slate shard into the blue box.", ("Pellam",))
    s.ask(
        "edit1_del", "gated", "boundary", "edit", "phone",
        "Which telephone did Pelam leave on the slate shard in the blue box?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-edit-del", "zh", "霍莲单字差", "石英条", "靳晚")
    # 石英 starts with 石, a surname. The question uses the typo, not the prop
    # as a second person, and no stored name sits in the question. Kept off
    # the focus path because the typo is the only candidate the edit can see.
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("霍涟", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "霍涟把那条薄片放进蓝盒子。", ("霍涟",))
    s.ask("edit1_del", "gated", "boundary", "edit", "birthday", "霍的生日记在蓝盒子的哪一页？", "霍涟")
    s.finish()

    s = bank.scene("en-edit-silent", "en", "Unread Pellum", "waxed card in the apron", "Sable")
    s.person("Pellam", ("phone", "birthday"))
    s.person("Briony", ("phone",))
    s.ask(
        "edit1_silent", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellum leave on the waxed card in the apron?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-edit-silent", "zh", "没人提过闵照", "灯旁盘", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("霍涟", ("phone",))
    s.ask("edit1_silent", "gated", "boundary", "edit", "phone", "闵照的电话记在灯旁盘的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-edit-other", "en", "Only Briony", "millet jar by the stove", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Briony, the millet jar by the stove is covered.", ("Briony",))
    s.ask(
        "edit1_other", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellum leave on the millet jar by the stove?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-edit-other", "zh", "只提了霍涟", "小米罐", "靳晚")
    s.person("霍涟", ("phone", "birthday"), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "霍涟，小米罐盖上了。", ("霍涟",))
    s.ask("edit1_other", "gated", "boundary", "edit", "phone", "闵照的电话记在小米罐的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-edit-tie", "en", "Pellum tie", "cork disc under the lamp", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.person("Tellum", ("phone", "birthday"))
    s.turn(
        "Pellam", "Briony",
        "Briony, Pellam and Tellum both signed the cork disc under the lamp.",
        ("Briony", "Pellam", "Tellum"),
    )
    s.ask(
        "edit1_tie", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellum leave on the cork disc under the lamp?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-edit-tie", "zh", "霍莲两可", "粉笔板边", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("霍涟", ("phone", "birthday"))
    s.person("霍琏", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "闵昭，霍涟和霍琏都把条子留在粉笔板边上。", ("闵昭", "霍涟", "霍琏"))
    s.ask("edit1_tie", "gated", "boundary", "edit", "phone", "霍莲的电话记在粉笔板边上的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-edit-ext", "en", "Pellamy", "hemp coil on the hook", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam hung the hemp coil on the hook.", ("Pellam",))
    s.ask(
        "edit_extension", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellamy leave on the hemp coil on the hook?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-edit-ext", "zh", "闵昭然", "角杯旁", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "闵昭把角杯旁的空位留下了。", ("闵昭",))
    s.ask("edit_extension", "gated", "boundary", "edit", "phone", "闵昭然的电话记在角杯旁的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-edit-far", "en", "Pellon", "quartz slip in the tray", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam put the quartz slip in the tray.", ("Pellam",))
    s.ask(
        "edit_far", "gated", "boundary", "edit", "phone",
        "Which telephone did Pellon leave on the quartz slip in the tray?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-edit-far", "zh", "闽照", "蓝纸封", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "闵昭把蓝纸封放进托盘。", ("闵昭",))
    s.ask("edit_far", "gated", "boundary", "edit", "phone", "闽照的电话记在蓝纸封的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-miss-you", "en", "No telephone for you", "river ticket in the coat", "Sable")
    s.person("Briony", ("birthday",), self_rel="birthday")
    s.person("Pellam", ("phone", "places"))
    s.turn("Pellam", "Briony", "Briony, the river ticket is in the coat pocket.", ("Briony",))
    s.ask(
        "missing_you", "gated", "boundary", "abstain", "phone",
        "What telephone did you leave on the river ticket in the coat?",
        "MISSING",
    )
    s.finish()

    s = bank.scene("zh-miss-you", "zh", "你没有电话", "河票", "靳晚")
    s.person("闵昭", ("birthday",), self_rel="birthday")
    s.person("阮澄", ("phone", "places"))
    s.turn("阮澄", "闵昭", "闵昭，河票在外套口袋里。", ("闵昭",))
    s.ask(
        "missing_you", "gated", "boundary", "abstain", "phone",
        "你留在外套口袋里那张河票上的电话，值夜要回看哪一页？",
        "MISSING",
    )
    s.finish()

    s = bank.scene("en-miss-he", "en", "No telephone for him", "violet band on the parcel", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("birthday", "places"))
    s.person("Pellam", ("phone",))
    s.turn("Pellam", "Briony", "Torven tied the violet band on the parcel.", ("Torven",))
    s.ask(
        "missing_third", "gated", "boundary", "abstain", "phone",
        "Which telephone did he leave on the violet band on the parcel?",
        "MISSING",
    )
    s.finish()

    s = bank.scene("zh-miss-he", "zh", "他没有电话", "紫带", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("birthday", "places"))
    s.person("阮澄", ("phone",))
    s.turn("阮澄", "闵昭", "喻棠把紫带系在包裹上。", ("喻棠",))
    s.ask("missing_third", "gated", "boundary", "abstain", "phone", "他留在紫带上的电话是哪一页？", "MISSING")
    s.finish()

    s = bank.scene("en-absent", "en", "Harth", "bone folder on the desk", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "The bone folder is on the desk.", ())
    s.ask(
        "absent_name", "gated", "boundary", "abstain", "phone",
        "Which telephone did Harth leave on the bone folder on the desk?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-absent", "zh", "殷浦", "骨夹", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone", "birthday"))
    s.turn("阮澄", "闵昭", "骨夹在桌上。", ())
    s.ask("absent_name", "gated", "boundary", "abstain", "phone", "殷浦的电话记在骨夹的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-unset", "en", "No hearer passed", "ash ladle by the hob", "Sable")
    s.person("Briony", ("phone", "birthday"))
    s.person("Pellam", ("phone", "places"))
    s.turn("Pellam", "", "The ash ladle by the hob is still warm, and Briony has the key.", ("Briony",))
    s.ask(
        "hearer_unset", "gated", "boundary", "you", "phone",
        "What telephone did you leave on the ash ladle by the hob?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-unset", "zh", "没交给对方", "灰勺", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("phone", "places"))
    s.turn("阮澄", "", "灰勺还温着，钥匙在闵昭那儿。", ("闵昭",))
    s.ask(
        "hearer_unset", "gated", "boundary", "you", "phone",
        "还温着的灰勺上，你留下的电话要回看哪一页？",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("en-offroster", "en", "Hearer not stored", "tin whistle case", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone",))
    s.turn("Pellam", "unlisted", "The tin whistle case is shut.", ())
    s.ask(
        "hearer_offroster", "gated", "boundary", "you", "phone",
        "What telephone did you leave on the tin whistle case?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-offroster", "zh", "听者不在名册", "哨盒", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone",))
    s.turn("阮澄", "名册外", "哨盒盖上了。", ())
    s.ask(
        "hearer_offroster", "gated", "boundary", "you", "phone",
        "盖上的哨盒里，你留下的电话要回看哪一页？",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("en-switch", "en", "Turned to Torven", "flax sheet on the second peg", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Briony, hold the flax sheet on the second peg a second.", ("Briony",))
    s.turn("Pellam", "Torven", "Torven, the closer should ring you from this sheet, not the old card.", ("Torven",))
    s.ask(
        "hearer_switch", "gated", "boundary", "you", "phone",
        "Which telephone should the closer use for you off the flax sheet on the second peg?",
        "Torven",
    )
    s.finish()

    s = bank.scene("zh-switch", "zh", "改口对喻棠", "第二枚藤圈", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "闵昭，第二枚藤圈先拿一下。", ("闵昭",))
    s.turn("阮澄", "喻棠", "喻棠，值夜该打给你，不要看旧卡片。", ("喻棠",))
    s.ask("hearer_switch", "gated", "boundary", "you", "phone", "值夜该用你的哪一页电话，第二枚藤圈上的？", "喻棠")
    s.finish()

    s = bank.scene("en-former", "en", "Briony stepped out", "green tin on the second shelf", "Sable")
    s.person("Briony", ("phone", "birthday"))
    s.person("Torven", ("phone", "places"))
    s.person("Pellam", ("birthday",))
    s.turn("Pellam", "Briony", "Briony, the green tin on the second shelf is set.", ("Briony",))
    s.turn("Pellam", "Torven", "Torven, Briony stepped out to the tram.", ("Torven", "Briony"))
    s.ask(
        "former_hearer", "gated", "boundary", "third", "phone",
        "Which telephone did she leave on the green tin on the second shelf?",
        "Briony",
    )
    s.finish()

    s = bank.scene("zh-former", "zh", "闵昭先走了", "第二层青瓷", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("喻棠", ("phone", "places"))
    s.person("阮澄", ("birthday",))
    s.turn("阮澄", "闵昭", "闵昭，第二层的青瓷放好了。", ("闵昭",))
    s.turn("阮澄", "喻棠", "喻棠，闵昭先去坐车了。", ("喻棠", "闵昭"))
    s.ask("former_hearer", "gated", "boundary", "third", "phone", "她留在第二层青瓷上的电话是哪一页？", "闵昭")
    s.finish()

    s = bank.scene("en-indirect-self", "en", "Listener, no hearer", "chalk slate in the clinic", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("phone",))
    s.turn("Pellam", "", "The chalk slate in the clinic is ready for the evening.", ())
    s.ask(
        "indirect_self", "gated", "boundary", "indirect", "phone",
        "Which telephone does the listener keep on the chalk slate in the clinic?",
        "Briony",
    )
    s.finish()

    s = bank.scene("zh-indirect-self", "zh", "跟前这位，没有听者", "诊室粉笔", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("phone",))
    s.turn("阮澄", "", "诊室里的粉笔板好了。", ())
    s.ask("indirect_self", "gated", "boundary", "indirect", "phone", "跟前这位的电话记在诊室粉笔上的哪一页？", "闵昭")
    s.finish()

    s = bank.scene("en-indirect-hearer", "en", "Across the table", "copper bowl between us", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Briony", "Pellam", "Pellam, the copper bowl between us is the one from tonight.", ("Pellam",))
    s.ask(
        "indirect_hearer", "gated", "boundary", "indirect", "phone",
        "Which telephone does the person across the table keep on the copper bowl between us?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-indirect-hearer", "zh", "对面那位", "今晚的铜碗", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone", "birthday"))
    s.turn("闵昭", "阮澄", "阮澄，今晚这只铜碗就放在这儿。", ("阮澄",))
    s.ask("indirect_hearer", "gated", "boundary", "indirect", "phone", "对面那位的电话记在今晚这只铜碗的哪一页？", "阮澄")
    s.finish()

    s = bank.scene("en-indirect-none", "en", "Listener, nobody", "moss folder with no self", "Sable")
    s.person("Pellam", ("phone",))
    s.person("Briony", ("birthday",))
    s.turn("Pellam", "", "The moss folder with no card on it is still shut.", ())
    s.ask(
        "indirect_noself", "gated", "boundary", "indirect", "phone",
        "Which telephone does the listener keep on the moss folder with no card?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-indirect-none", "zh", "跟前这位，没有本人", "空苔藓夹", "靳晚")
    s.person("阮澄", ("phone",))
    s.person("闵昭", ("birthday",))
    s.turn("阮澄", "", "空苔藓夹还关着。", ())
    s.ask("indirect_noself", "gated", "boundary", "indirect", "phone", "跟前这位的电话记在空苔藓夹的哪一页？", "ABSENT")
    s.finish()

    s = bank.scene("en-prior", "en", "Asked, then misspelled", "quartz slip after the ask", "Sable")
    s.person("Pellam", ("phone", "birthday"))
    s.person("Briony", ("phone",))
    s.ask(
        "prior_ask_name", "gated", "boundary", "name", "phone",
        "Which telephone did Pellam leave on the quartz slip after the ask?",
        "Pellam",
    )
    s.ask(
        "prior_ask_edit", "gated", "boundary", "edit", "birthday",
        "Which birthday did Pellum leave on the quartz slip after the ask?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-prior", "zh", "先问全名再问别字", "问过之后的软木", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("霍涟", ("phone",))
    s.ask("prior_ask_name", "gated", "boundary", "name", "phone", "闵昭的电话记在问过之后那块软木的哪一页？", "闵昭")
    s.ask("prior_ask_edit", "gated", "boundary", "edit", "birthday", "闵照的生日记在问过之后那块软木的哪一页？", "闵昭")
    s.finish()

    s = bank.scene("en-mixed", "en", "You or her", "navy envelope on the sill", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Elowen", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Briony, Elowen left the navy envelope on the sill.", ("Briony", "Elowen"))
    s.ask(
        "mixed_pronoun", "gated", "boundary", "third", "phone",
        "Should the closer use your telephone or her birthday off the navy envelope on the sill?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-mixed", "zh", "你还是她", "窗台蓝纸", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("霍涟", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "闵昭，霍涟把蓝纸放在窗台。", ("闵昭", "霍涟"))
    s.ask("mixed_pronoun", "gated", "boundary", "third", "phone", "值夜是回读你的电话，还是回读她的生日？蓝纸在窗台。", "ABSENT")
    s.finish()

    s = bank.scene("en-them", "en", "Them is not a pronoun", "wool ledger on the side counter", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Torven left the wool ledger on the side counter.", ("Torven",))
    s.ask(
        "not_pronoun_them", "gated", "boundary", "third", "phone",
        "Which telephone was kept for them on the wool ledger on the side counter?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-ta", "zh", "ta", "青瓷罐夜班", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("霍涟", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "霍涟把夜班条子压在青瓷罐上。", ("霍涟",))
    s.ask("ta_hit", "gated", "boundary", "third", "phone", "ta的电话写在青瓷罐夜班那一页的哪里？", "霍涟")
    s.finish()

    s = bank.scene("en-you-named", "en", "You, but the name is Pellam", "ferry chit on the second board", "Sable")
    s.person("Briony", ("phone", "birthday"), self_rel="phone")
    s.person("Pellam", ("phone", "places"))
    s.turn("Pellam", "Briony", "Briony, the ferry chit on the second board is the later one.", ("Briony",))
    s.ask(
        "you_named", "gated", "boundary", "you", "phone",
        "Can you read the telephone Pellam wrote on the ferry chit on the second board?",
        "Pellam",
    )
    s.finish()

    s = bank.scene("zh-you-named", "zh", "你，但名字是阮澄", "第二块渡轮条", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("phone", "places"))
    s.turn("阮澄", "闵昭", "闵昭，第二块渡轮条是后补的。", ("闵昭",))
    s.ask("you_named", "gated", "boundary", "you", "phone", "你读一下阮澄写在第二块渡轮条上的电话。", "阮澄")
    s.finish()

    s = bank.scene("en-vocative-q", "en", "Name, then you", "horn cup on the second hook", "Sable")
    s.person("Briony", ("phone", "birthday"))
    s.person("Pellam", ("phone", "places"))
    s.turn("Pellam", "Pellam", "The horn cup on the second hook is up.", ())
    # Hearer is Pellam. The question names Briony, so the name wins over you.
    s.ask(
        "vocative_question", "gated", "boundary", "you", "phone",
        "Briony, which telephone should the closer read for you off the horn cup on the second hook?",
        "Briony",
    )
    s.finish()

    s = bank.scene("zh-vocative-q", "zh", "先点名再你", "第二只角杯", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("phone", "places"))
    s.turn("阮澄", "阮澄", "第二只角杯挂上了。", ())
    s.ask("vocative_question", "gated", "boundary", "you", "phone", "闵昭，值夜要回读给你的电话，看第二只角杯上的哪一页？", "闵昭")
    s.finish()

    s = bank.scene("en-tie", "en", "Two on the chit", "ferry chit with two slips", "Sable")
    s.person("Quillan", ("phone",), self_rel="phone")
    s.person("Odhran", ("phone", "birthday"))
    s.person("Elowen", ("phone", "birthday"))
    s.person("Warrick", ("places",))
    s.turn(
        "Warrick", "Quillan",
        "Odhran and Elowen both left slips on the ferry chit with two slips.",
        ("Odhran", "Elowen"),
    )
    s.ask(
        "third_tie", "gated", "boundary", "third", "phone",
        "Which telephone did they leave on the ferry chit with two slips?",
        "ABSENT",
    )
    s.finish()

    s = bank.scene("zh-tie", "zh", "两张条子", "双条渡轮", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("邢渚", ("phone", "birthday"))
    s.person("喻棠", ("phone", "birthday"))
    s.person("卫聆", ("places",))
    s.turn("卫聆", "闵昭", "邢渚和喻棠都把条子留在双条渡轮上。", ("邢渚", "喻棠"))
    s.ask("third_tie", "gated", "boundary", "third", "phone", "他们的电话记在双条渡轮的哪一页？", "ABSENT")
    s.finish()

    # Gaps. The listener's reading and the shipped rule diverge. A miss against
    # the human column is the measurement. A miss against the rule is not expected.
    s = bank.scene("en-age", "en", "Four nods", "chalk slate after the nods", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Torven left the note on the chalk slate after the nods.", ("Torven",))
    s.turn("Pellam", "Briony", "Mm, noted.", ())
    s.turn("Pellam", "Briony", "That slate can sit.", ())
    s.turn("Pellam", "Briony", "I heard the slate.", ())
    s.turn("Pellam", "Briony", "One moment there.", ())
    s.ask(
        "window_backchannel", "gap", "gap", "window", "phone",
        "Which telephone did he leave on the chalk slate after the nods?",
        "Torven",
        "Four short acknowledgements open four turns. The name is still what he means, and TURN_WINDOW has dropped it.",
    )
    s.finish()

    s = bank.scene("zh-age", "zh", "四声嗯之后", "粉笔板之后", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "喻棠把条子留在后面那块粉笔板上。", ("喻棠",))
    s.turn("阮澄", "闵昭", "嗯，后面那块可以。", ())
    s.turn("阮澄", "闵昭", "那块板子先放着。", ())
    s.turn("阮澄", "闵昭", "我听见板子了。", ())
    s.turn("阮澄", "闵昭", "板子等一下。", ())
    s.ask(
        "window_backchannel", "gap", "gap", "window", "phone",
        "他留在后面那块粉笔板上的电话是哪一页？",
        "喻棠",
        "四句很短的应答各算一轮。听的人还是在问他，窗口已经把名字推出去了。",
    )
    s.finish()

    s = bank.scene("en-chain", "en", "Kept saying he", "horn cup after dusk", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Torven", ("phone", "birthday"))
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Torven left the horn cup after dusk.", ("Torven",))
    s.turn("Pellam", "Briony", "He said the closer can ring once it is quiet.", ())
    s.turn("Pellam", "Briony", "He was sure about the card.", ())
    s.turn("Pellam", "Briony", "He waved from the door.", ())
    s.turn("Pellam", "Briony", "He kept the card in his coat.", ())
    s.ask(
        "pronoun_chain_age", "gap", "gap", "window", "phone",
        "Which telephone did he leave with the horn cup after dusk?",
        "Torven",
        "Later turns only say he, and a pronoun is not written back onto the log, so the name ages out.",
    )
    s.finish()

    s = bank.scene("zh-chain", "zh", "一直说他", "黄昏后的角杯", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "喻棠把黄昏后的角杯留下了。", ("喻棠",))
    s.turn("阮澄", "闵昭", "他说安静了以后值夜可以打。", ())
    s.turn("阮澄", "闵昭", "他对那张卡片很确定。", ())
    s.turn("阮澄", "闵昭", "他在门口挥了下手。", ())
    s.turn("阮澄", "闵昭", "他把卡片放在外套里。", ())
    s.ask(
        "pronoun_chain_age", "gap", "gap", "window", "phone",
        "他留在黄昏后那只角杯上的电话是哪一页？",
        "喻棠",
        "后面几轮只说他。代词不会写回人名，名字就滑出了窗口。",
    )
    s.finish()

    s = bank.scene("en-vocative-miss", "en", "Addressed, hearer omitted", "green tin by the elbow", "Sable")
    s.person("Briony", ("phone", "birthday"))
    s.person("Pellam", ("phone", "places"))
    s.turn("Pellam", "", "Briony, I put the green tin by your elbow.", ("Briony",))
    s.ask(
        "vocative_unpassed", "gap", "gap", "you", "phone",
        "Which telephone should the closer read for you off the green tin by the elbow?",
        "Briony",
        "The line addresses Briony, but note_turn was not given a hearer. you does not recover a vocative.",
    )
    s.finish()

    s = bank.scene("zh-vocative-miss", "zh", "点了名却没传入", "手边青瓷", "靳晚")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("phone", "places"))
    s.turn("阮澄", "", "闵昭，青瓷放在你手边了。", ("闵昭",))
    s.ask(
        "vocative_unpassed", "gap", "gap", "you", "phone",
        "值夜要回读给你的电话，看手边青瓷上的哪一页？",
        "闵昭",
        "这句话点了闵昭，但 note_turn 没有传入听者。你不会从呼语里补出来。",
    )
    s.finish()

    s = bank.scene("en-gender", "en", "She, with both of them", "ferry chit for the pair", "Sable")
    s.person("Quillan", ("phone",), self_rel="phone")
    s.person("Odhran", ("phone", "birthday"))
    s.person("Elowen", ("phone", "birthday"))
    s.person("Warrick", ("places",))
    s.turn(
        "Warrick", "Quillan",
        "Odhran and Elowen both left slips on the ferry chit for the pair.",
        ("Odhran", "Elowen"),
    )
    s.ask(
        "gender_tie", "gap", "gap", "third", "phone",
        "Which telephone did she leave on the ferry chit for the pair?",
        "Elowen",
        "she would pick Elowen. Two names on the newest turn abstain, and gender is not a feature.",
    )
    s.ask(
        "gender_tie", "gap", "gap", "third", "birthday",
        "Which birthday did he leave on the ferry chit for the pair?",
        "Odhran",
        "he would pick Odhran. The same tie abstains.",
    )
    s.finish()

    s = bank.scene("zh-gender", "zh", "她，可是两个人", "成对的渡轮条", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("喻棠", ("phone", "birthday"))
    s.person("邢渚", ("phone", "birthday"))
    s.person("卫聆", ("places",))
    s.turn("卫聆", "闵昭", "喻棠和邢渚都把条子留在成对的渡轮条上。", ("喻棠", "邢渚"))
    s.ask(
        "gender_tie", "gap", "gap", "third", "phone",
        "她留在成对渡轮条上的电话是哪一页？",
        "邢渚",
        "她会指邢渚。最新一轮上有两个人就弃权，性别不是特征。",
    )
    s.ask(
        "gender_tie", "gap", "gap", "third", "birthday",
        "他留在成对渡轮条上的生日是哪一页？",
        "喻棠",
        "他会指喻棠。同样的并列仍然弃权。",
    )
    s.finish()

    s = bank.scene("en-recency", "en", "Going back to her", "ferry chit from the start", "Sable")
    s.person("Quillan", ("phone",), self_rel="phone")
    s.person("Elowen", ("birthday", "phone"))
    s.person("Odhran", ("birthday", "phone"))
    s.person("Warrick", ("places",))
    s.turn(
        "Warrick", "Quillan",
        "Elowen described the date on the ferry chit from the start.",
        ("Elowen",),
    )
    s.turn("Warrick", "Quillan", "Odhran waved from the gangway and left.", ("Odhran",))
    s.ask(
        "recency_clash", "gap", "gap", "third", "birthday",
        "Going back to the start, which birthday did she mention on the ferry chit from the start?",
        "Elowen",
        "she, going back, is Elowen. eligible_latest keeps Odhran, the newest name, and ignores gender.",
    )
    s.finish()

    s = bank.scene("zh-recency", "zh", "回到她刚才说的", "一开始的渡轮条", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("邢渚", ("birthday", "phone"))
    s.person("喻棠", ("birthday", "phone"))
    s.person("卫聆", ("places",))
    s.turn("卫聆", "闵昭", "邢渚一开始就在渡轮条上讲了那个日子。", ("邢渚",))
    s.turn("卫聆", "闵昭", "喻棠在跳板那边挥了下手就走了。", ("喻棠",))
    s.ask(
        "recency_clash", "gap", "gap", "third", "birthday",
        "回到一开始，她在渡轮条上说的生日是哪一页？",
        "邢渚",
        "回到她，指的是邢渚。最新的合格名字是喻棠，性别不参与。",
    )
    s.finish()

    s = bank.scene("en-edit-aged", "en", "Pellum after nods", "oat sack after the nods", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam stacked the oat sack after the nods.", ("Pellam",))
    s.turn("Pellam", "Briony", "Mm, after those nods.", ())
    s.turn("Pellam", "Briony", "This sack can wait.", ())
    s.turn("Pellam", "Briony", "I heard the nods.", ())
    s.turn("Pellam", "Briony", "One moment after that.", ())
    s.ask(
        "edit_aged", "gap", "gap", "edit", "phone",
        "Which telephone did Pellum leave on the oat sack after the nods?",
        "Pellam",
        "Pellum is still Pellam. Four later turns have taken the mention out of the window, so the edit is no longer licensed.",
    )
    s.finish()

    s = bank.scene("zh-edit-aged", "zh", "点头之后的闵照", "点头后的纸带", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone", "birthday"))
    s.person("霍涟", ("places",))
    s.turn("霍涟", "闵昭", "阮澄把点头后的纸带叠好了。", ("阮澄",))
    s.turn("霍涟", "闵昭", "嗯，点过头了。", ())
    s.turn("霍涟", "闵昭", "点头后的纸带先放着。", ())
    s.turn("霍涟", "闵昭", "我听见点头了。", ())
    s.turn("霍涟", "闵昭", "点头之后等一下。", ())
    s.ask(
        "edit_aged", "gap", "gap", "edit", "phone",
        "阮橙的电话记在点头后那条纸带的哪一页？",
        "阮澄",
        "阮橙离阮澄一笔。四轮之后窗口不再准许这一笔。",
    )
    s.finish()

    s = bank.scene("en-insert", "en", "Extra letter", "hemp coil with an extra loop", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam hung the hemp coil with an extra loop.", ("Pellam",))
    s.ask(
        "edit_insert", "gap", "gap", "edit", "phone",
        "Which telephone did Pellamm leave on the hemp coil with an extra loop?",
        "Pellam",
        "An extra letter still contains Pellam, and a longer span that contains the name is rejected as a different person.",
    )
    s.finish()

    s = bank.scene("en-swap", "en", "Pellma", "paper band typed fast", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam left the paper band typed fast on the hook.", ("Pellam",))
    s.ask(
        "edit_transpose", "gap", "gap", "edit", "phone",
        "Which telephone did Pellma leave on the paper band typed fast?",
        "Pellam",
        "Pellma is Pellam with the last two letters swapped. That distance is 2, so the edit is not licensed.",
    )
    s.finish()

    s = bank.scene("zh-swap", "zh", "昭闵", "写反的软木", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "闵昭把写反的软木留下了。", ("闵昭",))
    s.ask(
        "edit_transpose", "gap", "gap", "edit", "phone",
        "昭闵的电话记在写反的软木的哪一页？",
        "闵昭",
        "昭闵是闵昭两个字对调，距离是 2，编辑许可不会接。",
    )
    s.finish()

    s = bank.scene("zh-prefix", "zh", "老闵昭", "藤圈上的称呼", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "闵昭今晚在藤圈上的称呼就那样。", ("闵昭",))
    s.ask(
        "cjk_prefix", "gap", "gap", "edit", "phone",
        "老闵昭的电话记在藤圈上的哪一页？",
        "闵昭",
        "老加全名在口头上还是闵昭。前缀规则把它当成另一个人。",
    )
    s.finish()

    s = bank.scene("en-speaker", "en", "He means the man who spoke", "wool ledger on the stool", "Sable")
    s.person("Torven", ("phone", "birthday"))
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("places",))
    s.turn("Torven", "Briony", "I left the wool ledger on the stool before closing.", ())
    s.ask(
        "speaker_as_third", "gap", "gap", "third", "phone",
        "Which telephone did he leave in the wool ledger on the stool?",
        "Torven",
        "he is the man who just spoke. The turn names nobody, and a speaker is not eligible even if we knew gender.",
    )
    s.finish()

    s = bank.scene("zh-speaker", "zh", "他就是刚说话的人", "凳子上的账", "靳晚")
    s.person("喻棠", ("phone", "birthday"))
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("places",))
    s.turn("喻棠", "闵昭", "打烊前我把账放在凳子上了。", ())
    s.ask(
        "speaker_as_third", "gap", "gap", "third", "phone",
        "他放在凳子上那本账里的电话是哪一页？",
        "喻棠",
        "他就是刚说完话的喻棠。这一轮没有点名，说话人不会变成第三人。",
    )
    s.finish()

    s = bank.scene("zh-plural", "zh", "你们", "柜上的青瓷", "靳晚")
    s.person("闵昭", ("phone", "birthday"), self_rel="phone")
    s.person("阮澄", ("phone",))
    s.turn("阮澄", "闵昭", "闵昭，柜上的青瓷归今晚。", ("闵昭",))
    s.ask(
        "plural_you", "gap", "gap", "you", "phone",
        "你们的电话写在柜上那只青瓷的哪一页？",
        "ABSENT",
        "你们不是单数的你，不该落到听者，也不该落到名册里第一页电话。",
    )
    s.finish()

    # Open addressee forms. Known out of scope. Not a regression if they abstain.
    s = bank.scene("en-oos-party", "en", "Other party", "green tin for the other party", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("places",))
    s.turn("Pellam", "Briony", "Briony, the green tin for the other party is the later one.", ("Briony",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "phone",
        "Which telephone does the other party keep on the green tin for the other party?",
        "Briony",
        "other party was left out of the closed class in #34.",
    )
    s.finish()

    s = bank.scene("en-oos-one", "en", "This one", "chalk slate for this one", "Sable")
    s.person("Tamsin", ("phone",), self_rel="phone")
    s.person("Maelis", ("places",))
    s.turn("Maelis", "Tamsin", "Tamsin, this slate is the one from the rear door.", ("Tamsin",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "phone",
        "Which telephone does this one keep on the chalk slate for this one?",
        "Tamsin",
        "this one was left out of the closed class in #34.",
    )
    s.finish()

    s = bank.scene("en-oos-person", "en", "That person", "moss folder for that person", "Sable")
    s.person("Nessa", ("birthday",), self_rel="birthday")
    s.person("Corin", ("phone",))
    s.turn("Corin", "Nessa", "Nessa, the moss folder for that person is the rehearsal one.", ("Nessa",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "birthday",
        "Which birthday does that person keep on the moss folder for that person?",
        "Nessa",
        "that person was left out of the closed class in #34.",
    )
    s.finish()

    s = bank.scene("en-oos-guest", "en", "Our guest", "copper bowl for our guest", "Sable")
    s.person("Elowen", ("places",), self_rel="places")
    s.person("Vesper", ("phone",))
    s.turn("Vesper", "Elowen", "Elowen, the copper bowl for our guest is on the low shelf.", ("Elowen",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "places",
        "Which places does our guest keep on the copper bowl for our guest?",
        "Elowen",
        "our guest was left out of the closed class in #34.",
    )
    s.finish()

    s = bank.scene("zh-oos-duifang", "zh", "对方", "对方的青瓷", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("places",))
    s.turn("阮澄", "闵昭", "闵昭，这只青瓷是后放上柜台的。", ("闵昭",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "phone",
        "对方的电话记在这只后放的青瓷的哪一页？",
        "闵昭",
        "对方不在 #34 的封闭类里。",
    )
    s.finish()

    s = bank.scene("zh-oos-zhewei", "zh", "这位", "这位的粉笔", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("岑汐", ("places",))
    s.turn("岑汐", "霍涟", "霍涟，这块粉笔是后门那块。", ("霍涟",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "phone",
        "这位的电话记在后门那块粉笔的哪一页？",
        "霍涟",
        "这位不在 #34 的封闭类里。",
    )
    s.finish()

    s = bank.scene("zh-oos-nawei", "zh", "那位", "那位的苔藓", "靳晚")
    s.person("邢渚", ("birthday",), self_rel="birthday")
    s.person("卫聆", ("phone",))
    s.turn("卫聆", "邢渚", "邢渚，这本苔藓夹是排练用的。", ("邢渚",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "birthday",
        "那位的生日记在排练用的苔藓夹的哪一页？",
        "邢渚",
        "那位不在 #34 的封闭类里。",
    )
    s.finish()

    s = bank.scene("zh-oos-renjia", "zh", "人家", "人家的铜碗", "靳晚")
    s.person("闵昭", ("places",), self_rel="places")
    s.person("阮澄", ("phone",))
    s.turn("阮澄", "闵昭", "闵昭，矮架上的铜碗今晚归你。", ("闵昭",))
    s.ask(
        "oos_addressee", "oos", "oos", "oos", "places",
        "人家的地点记在矮架那只铜碗的哪一页？",
        "闵昭",
        "人家不在 #34 的封闭类里。",
    )
    s.finish()

    # Empty log. No turns. Main behavior, not a new dialogue bug.
    s = bank.scene("en-empty-you", "en", "Nobody has spoken", "pebble jar with no talk", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone",))
    s.ask(
        "empty_deictic", "control", "control", "empty", "phone",
        "What telephone should the closer use for you off the pebble jar with no talk?",
        "ABSENT",
        "No turn and no hearer. you stays on the ranker.",
    )
    s.finish()

    s = bank.scene("en-empty-he", "en", "Nobody was named", "reed basket with no talk", "Sable")
    s.person("Torven", ("phone",))
    s.person("Briony", ("phone",), self_rel="phone")
    s.ask(
        "empty_deictic", "control", "control", "empty", "phone",
        "Which telephone did he leave on the reed basket with no talk?",
        "ABSENT",
        "No turn and no hearer. he stays on the ranker.",
    )
    s.finish()

    s = bank.scene("zh-empty-you", "zh", "还没人说话", "没说过的窗台罐", "靳晚")
    s.person("闵昭", ("phone",), self_rel="phone")
    s.person("阮澄", ("phone",))
    s.ask(
        "empty_deictic", "control", "control", "empty", "phone",
        "还没人开口时，你的电话要看没说过的窗台罐上的哪一页？",
        "ABSENT",
        "没有轮次也没有听者。你留在排序器上。",
    )
    s.finish()

    s = bank.scene("zh-empty-he", "zh", "还没点名", "没说过的蓝票", "靳晚")
    s.person("喻棠", ("phone",))
    s.person("闵昭", ("phone",), self_rel="phone")
    s.ask(
        "empty_deictic", "control", "control", "empty", "phone",
        "他还没被提到，电话要看没说过的蓝票上的哪一页？",
        "ABSENT",
        "没有轮次也没有听者。他留在排序器上。",
    )
    s.finish()

    s = bank.scene("en-observe-pell", "en", "Pell", "horn cup for the short name", "Sable")
    s.person("Briony", ("phone",), self_rel="phone")
    s.person("Pellam", ("phone", "birthday"))
    s.turn("Pellam", "Briony", "Pellam left the horn cup for the short name.", ("Pellam",))
    s.ask(
        "short_name", "observe", "observe", "name", "phone",
        "Which telephone did Pell leave on the horn cup for the short name?",
        "Pellam",
        "Pell is a short form, not an edit of one. The name head may or may not take it.",
    )
    s.finish()

    s = bank.scene("zh-observe-nick", "zh", "小昭", "角杯上的小称", "靳晚")
    s.person("霍涟", ("phone",), self_rel="phone")
    s.person("闵昭", ("phone", "birthday"))
    s.person("阮澄", ("places",))
    s.turn("阮澄", "霍涟", "闵昭把角杯上的小称留下了。", ("闵昭",))
    s.ask(
        "given_nick", "observe", "observe", "name", "phone",
        "小昭的电话记在角杯上的小称那一页吗？",
        "闵昭",
        "小昭是名的小称。头接不接，不作为本门的通过条件。",
    )
    s.finish()

    from tidal.question_name_align import _pinyin_key

    folded = _pinyin_key("闵昭")
    surface = folded[:1].upper() + folded[1:] if folded else ""
    if len(surface) < 2:
        raise RuntimeError(f"pinyin of 闵昭 is {folded!r}")
    s = bank.scene("zh-observe-py", "bi", "Pinyin", "horn cup for the pinyin", "Sable", fact_lang="en")
    s.person("闵昭", ("phone", "birthday"))
    s.person("Briony", ("phone",), self_rel="phone")
    s.turn("Briony", "Briony", "The horn cup for the pinyin is on the second hook.", ())
    s.ask(
        "pinyin_name", "observe", "observe", "name", "phone",
        f"Which telephone did {surface} leave on the horn cup for the pinyin?",
        "闵昭",
        "The surface is the stored name's pinyin. The head may or may not take it. Not a gate.",
    )
    s.finish()


def _check(rows: Sequence[Episode]) -> None:
    eids = [row.eid for row in rows]
    if len(eids) != len(set(eids)):
        raise RuntimeError("duplicate episode id")
    asks = [ask for row in rows for ask in row.events if isinstance(ask, Ask)]
    questions = [ask.question for ask in asks]
    if len(questions) != len(set(questions)):
        dupes = sorted({q for q in questions if questions.count(q) > 1})
        raise RuntimeError(f"duplicate question: {dupes[:4]}")
    if len(rows) < 50:
        raise RuntimeError(f"only {len(rows)} episodes")
    if len(asks) < 100:
        raise RuntimeError(f"only {len(asks)} asks")
    langs = {ask.lang for ask in asks}
    if "en" not in langs or "zh" not in langs:
        raise RuntimeError(f"langs {langs}")
    if sum(ask.lang == "en" for ask in asks) < 40 or sum(ask.lang == "zh" for ask in asks) < 40:
        raise RuntimeError("one language is thin")
    ordinary = [ask for ask in asks if ask.scope == "gated" and ask.band == "ordinary"]
    gap = [ask for ask in asks if ask.scope == "gap"]
    oos = [ask for ask in asks if ask.scope == "oos"]
    if len(ordinary) < 40 or len(gap) < 12 or len(oos) < 8:
        raise RuntimeError(f"ordinary {len(ordinary)} gap {len(gap)} oos {len(oos)}")
    texts: List[str] = []
    for row in rows:
        for slot in row.slots:
            texts.append(slot.text)
            if slot.value.lower() in slot.text.lower():
                raise RuntimeError(f"value leaked into {slot.text}")
        for event in row.events:
            if isinstance(event, Turn):
                texts.append(event.text)
            elif isinstance(event, Ask):
                texts.append(event.question)
                if any(slot.value.lower() in event.question.lower() for slot in row.slots):
                    raise RuntimeError(f"value leaked into {event.question}")
    if len(texts) != len(set(texts)):
        dupes = sorted({text for text in texts if texts.count(text) > 1})
        raise RuntimeError(f"duplicate bank string: {dupes[:4]}")


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
        counts[ask.phenomenon] = counts.get(ask.phenomenon, 0) + 1
    return counts


def scope_counts(rows: Optional[Sequence[Episode]] = None) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, ask in iter_asks(rows):
        counts[ask.scope] = counts.get(ask.scope, 0) + 1
    return counts


if __name__ == "__main__":
    rows = build_rows()
    asks = list(iter_asks(rows))
    print(f"episodes {len(rows)} asks {len(asks)}")
    print("scope", scope_counts(rows))
    print("kinds", kind_counts(rows))
    print("questions", questions_sha256(rows))
    print("bank", bank_sha256(rows), "strings", len(bank_strings(rows)))
