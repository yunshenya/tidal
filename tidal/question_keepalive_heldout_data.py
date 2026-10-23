"""Held-out bank for pronoun keep-alive and leading vocatives.

Labels were fixed in ``reports/dialogue_keepalive_prereg.md`` before this
file existed. ``build_rows`` replays ``apply_utterance`` and the deixis
selectors. A row whose rule disagrees with that pre-registered label fails
here, instead of being relabeled. The #35 real-dialogue bank is not an input
to the wording. ``LLMMemory`` is not imported.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_dialogue_real_data import _human_outcome, _predict
from tidal.question_name_align import _ambiguous_edit, _extension, dialogue_licensed_edit
from tidal.question_surface import apply_utterance, ask_people, is_deictic, lex_labels, refresh_resolved

_REL = ("phone", "birthday", "places")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
_SPEAKER = {"en": "Solway", "zh": "裴涟"}
_HEARER = {"en": "Anwick", "zh": "栾昭"}
_DECOY = {"en": "Yarrow", "zh": "童屿"}


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
    kind: str
    lang: str
    rel: str
    question: str
    rule: str
    human: str
    rule_golds: Tuple[str, ...]
    human_golds: Tuple[str, ...]
    empty_rule: str
    empty_golds: Tuple[str, ...]
    who_rule: str
    who_human: str


@dataclass(frozen=True)
class Episode:
    eid: str
    lang: str
    slots: Tuple[Slot, ...]
    events: Tuple[object, ...]


def _check_pairs() -> None:
    """The edit surfaces are distance 1, and the head does not take them alone."""
    if not _ambiguous_edit("Caldar", "Calder") or _extension("Caldar", "Calder"):
        raise RuntimeError("Caldar is not an edit-1 of Calder")
    if dialogue_licensed_edit("Caldar", ["Calder", "Yarrow"], ()) is not None:
        raise RuntimeError("Caldar matched with an empty log")
    if dialogue_licensed_edit("Caldar", ["Calder", "Yarrow"], ["Calder"]) != "Calder":
        raise RuntimeError("Caldar was not licensed for Calder")
    if not _ambiguous_edit("苟柠", "苟宁") or _extension("苟柠", "苟宁"):
        raise RuntimeError("苟柠 is not an edit-1 of 苟宁")
    if dialogue_licensed_edit("苟柠", ["苟宁", "童屿"], ()) is not None:
        raise RuntimeError("苟柠 matched with an empty log")
    if dialogue_licensed_edit("苟柠", ["苟宁", "童屿"], ["苟宁"]) != "苟宁":
        raise RuntimeError("苟柠 was not licensed for 苟宁")


class _Scene:
    def __init__(self, bank: "Bank", eid: str, lang: str) -> None:
        self.bank = bank
        self.eid = eid
        self.lang = lang
        self.people: List[str] = []
        self.steps: list = []

    def cast(self, *names: str) -> "_Scene":
        self.people = list(names)
        return self

    def turn(
        self,
        text: str,
        mentions: Sequence[str],
        *,
        speaker: str = "",
        hearer: str = "",
    ) -> "_Scene":
        self.steps.append(("t", text, tuple(mentions), speaker, hearer))
        return self

    def ask(self, kind: str, rel: str, question: str, human: str) -> "_Scene":
        self.steps.append(("a", kind, rel, question, human))
        return self

    def _fact(self, name: str, rel: str) -> Slot:
        cue = _CUE[self.lang][rel]
        n = self.bank.stamp
        self.bank.stamp += 1
        if self.lang == "zh":
            text = f"抄件{n:03d}落在边栏，{name}对上{cue}这一格 bracken scrip {n:03d}"
        else:
            text = (
                f"Copy {n:03d} sits in the margin column for {name} "
                f"under the {cue} heading bracken scrip {n:03d}"
            )
        return Slot(name, rel, f"kpv{n:04d}", text)

    def finish(self) -> None:
        if not self.people or self.people[0] != _DECOY[self.lang]:
            raise RuntimeError(f"{self.eid} decoy is not first")
        slots: List[Slot] = []
        for rel in _REL:
            for name in self.people:
                slots.append(self._fact(name, rel))
        names = [slot.name for slot in slots]
        surfaces = [slot.text for slot in slots]
        roster = set(names)
        state = DialogueState()
        events: List[object] = []
        for step in self.steps:
            if step[0] == "t":
                _, text, mentions, speaker, hearer = step
                stamped = self.bank.line(text)
                if speaker and speaker not in roster:
                    raise RuntimeError(f"{self.eid} speaker {speaker} is not stored")
                if hearer and hearer not in roster:
                    raise RuntimeError(f"{self.eid} hearer {hearer} is not stored")
                got = apply_utterance(
                    state, stamped, names, surfaces,
                    speaker=speaker or None, hearer=hearer or None,
                )
                if set(got) != set(mentions):
                    raise RuntimeError(f"{self.eid} mentions {got} != {mentions} in {stamped!r}")
                events.append(Turn(stamped, speaker, hearer))
                continue
            _, kind, rel, question, human = step
            stamped = self.bank.line(question)
            if lex_labels(stamped) != (rel,):
                raise RuntimeError(f"{self.eid} cues {lex_labels(stamped)} for {stamped!r}")
            if kind in ("keep_edit", "age_edit"):
                bare = ask_people(stamped, names, surfaces, ())
                if bare:
                    raise RuntimeError(f"{self.eid} edit matched on an empty log: {bare}")
            rule, rule_golds, who = _predict(state, slots, stamped, rel)
            empty_rule, empty_golds, _empty_who = _predict(DialogueState(), slots, stamped, rel)
            human_label, human_golds, who_human = _human_outcome(human, slots, rel)
            if (rule, rule_golds) != (human_label, human_golds):
                raise RuntimeError(
                    f"{self.eid} {kind} rule {rule}:{who} != human {human} q={stamped!r}"
                )
            if human == "DECOY" and (empty_rule, empty_golds) != (human_label, human_golds):
                raise RuntimeError(f"{self.eid} empty decoy drifted: {empty_rule}")
            if human != "DECOY" and is_deictic(stamped) and empty_rule != "answerable":
                raise RuntimeError(f"{self.eid} empty deictic did not stay on the ranker")
            events.append(Ask(
                kind, self.lang, rel, stamped, rule, human_label,
                rule_golds, human_golds, empty_rule, empty_golds, who, who_human,
            ))
            mentioned = ask_people(
                stamped, names, surfaces, state.conversational(ignore_surface=stamped),
            )
            state.note_ask(mentioned, stamped)
            refresh_resolved(state, stamped, names, surfaces)
        self.bank.episodes.append(Episode(self.eid, self.lang, tuple(slots), tuple(events)))


class Bank:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []
        self.stamp = 1
        self._line = 0

    def line(self, text: str) -> str:
        self._line += 1
        return f"{text} bracken scrip {self._line:03d}"

    def scene(self, eid: str, lang: str) -> _Scene:
        return _Scene(self, eid, lang)


def _third(bank: Bank, eid: str, lang: str, target: str, kind: str, rel: str, lines: Sequence[str], question: str, extra: Sequence[str] = ()) -> None:
    speaker, hearer, decoy = _SPEAKER[lang], _HEARER[lang], _DECOY[lang]
    people = [decoy, target, speaker, hearer, *extra]
    scene = bank.scene(eid, lang).cast(*people)
    for i, text in enumerate(lines):
        mentions = (target,) if i == 0 else ()
        scene.turn(text, mentions, speaker=speaker, hearer=hearer)
    scene.ask(kind, rel, question, target)
    scene.finish()


def _load(bank: Bank) -> None:
    _check_pairs()
    sp, he = _SPEAKER, _HEARER

    _third(
        bank, "en-keep-he", "en", "Calder", "keep_he", "phone",
        (
            "Calder set the lichen board beside the night latch before the seam was stitched.",
            "He folded that lichen board once the night latch had cooled.",
            "He was sure the lichen board belonged on the night latch side.",
            "He waved from the door that faces the night latch and the lichen board.",
            "He kept the card under the lichen board next to the night latch.",
        ),
        "Which telephone did he set down on the lichen board beside the night latch?",
    )
    _third(
        bank, "en-keep-she", "en", "Mirren", "keep_she", "birthday",
        (
            "Mirren left the kelp ticket under the queue chalk before the tide mark dried.",
            "She folded the kelp ticket once the queue chalk had dried.",
            "She was sure the kelp ticket belonged with the queue chalk.",
            "She waved from the stall that keeps the queue chalk and the kelp ticket.",
            "She kept the card under the kelp ticket next to the queue chalk.",
        ),
        "Which birthday did she set down on the kelp ticket beside the queue chalk?",
    )
    _third(
        bank, "en-keep-ta", "en", "Berrick", "keep_ta", "phone",
        (
            "Berrick set the salt-wick spool beside the wick tray before the salt crust dried.",
            "Once the salt crust cooled ta folded the salt-wick spool flat.",
            "ta was sure the salt-wick spool belonged on the wick tray.",
            "ta waved from the shelf that holds the wick tray and the salt-wick spool.",
            "ta kept the card under the salt-wick spool next to the wick tray.",
        ),
        "Which telephone did ta set down on the salt-wick spool beside the wick tray?",
    )
    _third(
        bank, "zh-keep-ta", "zh", "卞茗", "keep_zhta", "phone",
        (
            "卞茗把青苔板靠在夜闩旁边，青苔缝还没干。",
            "夜闩凉了以后他把青苔板折好了。",
            "他对夜闩旁边那块青苔板很确定。",
            "他朝夜闩对面的窗台扬了扬手，青苔板还靠在那儿。",
            "他把卡片压在青苔板和夜闩中间。",
        ),
        "青苔板靠着夜闩，他的电话抄在哪一格？",
    )
    _third(
        bank, "zh-keep-she", "zh", "苟宁", "keep_zhta", "birthday",
        (
            "苟宁把海带票搁在琥珀轴旁边，海带边还潮着。",
            "琥珀轴凉了以后她把海带票折好了。",
            "她对琥珀轴旁边那张海带票很确定。",
            "她在对着琥珀轴的门口挥了下手。",
            "她把卡片压在海带票和琥珀轴中间。",
        ),
        "海带票靠着琥珀轴，她的生日抄在哪一格？",
    )
    _third(
        bank, "en-keep-edit", "en", "Calder", "keep_edit", "phone",
        (
            "Calder set the amber spindle beside the resin tray before the notch cooled.",
            "He folded that amber spindle once the resin tray had cooled.",
            "He was sure the amber spindle belonged in the resin tray.",
            "He waved from the shelf that faces the resin tray and the amber spindle.",
            "He kept the card under the amber spindle next to the resin tray.",
        ),
        "Which telephone did Caldar leave beside the amber spindle and the resin tray?",
    )
    _third(
        bank, "zh-keep-edit", "zh", "苟宁", "keep_edit", "phone",
        (
            "苟宁把盐芯搁在芯盘旁边，盐壳还没干。",
            "芯盘凉了以后她把盐芯收好了。",
            "她对芯盘旁边那枚盐芯很确定。",
            "她在对着芯盘的门口点了下头。",
            "她把卡片压在盐芯和芯盘中间。",
        ),
        "苟柠的电话记在盐芯靠近芯盘的哪一格？",
    )

    en_ask = bank.scene("en-keep-ask", "en").cast("Yarrow", "Calder", "Solway", "Anwick")
    en_ask.turn(
        "Calder set the lichen board beside the night latch and left the seam open.",
        ("Calder",), speaker=sp["en"], hearer=he["en"],
    )
    en_ask.ask("keep_ask", "phone", "Which telephone did he set down on the lichen board by the open seam?", "Calder")
    en_ask.ask("keep_ask", "birthday", "Which birthday did he set down on the lichen board by the open seam?", "Calder")
    en_ask.ask("keep_ask", "places", "Which places did he set down on the lichen board by the open seam?", "Calder")
    en_ask.ask("keep_ask", "phone", "Which telephone did he copy onto the lichen board after the seam was left open?", "Calder")
    en_ask.ask("keep_ask", "phone", "Which telephone did he write along the lichen board once the seam had been left open?", "Calder")
    en_ask.finish()

    zh_ask = bank.scene("zh-keep-ask", "zh").cast("童屿", "卞茗", "裴涟", "栾昭")
    zh_ask.turn(
        "卞茗把青苔板靠在夜闩上，青苔缝还敞着。",
        ("卞茗",), speaker=sp["zh"], hearer=he["zh"],
    )
    zh_ask.ask("keep_ask", "phone", "青苔缝还敞着，他的电话抄在青苔板的哪一格？", "卞茗")
    zh_ask.ask("keep_ask", "birthday", "青苔缝还敞着，他的生日抄在青苔板的哪一格？", "卞茗")
    zh_ask.ask("keep_ask", "places", "青苔缝还敞着，他的地点抄在青苔板的哪一格？", "卞茗")
    zh_ask.ask("keep_ask", "phone", "青苔缝敞着没合，他另抄的电话在青苔板哪一格？", "卞茗")
    zh_ask.ask("keep_ask", "phone", "青苔缝一直敞着，他写在青苔板边上的电话在哪一格？", "卞茗")
    zh_ask.finish()

    def quiet(scene: _Scene, lines: Sequence[str], speaker: str, hearer: str) -> None:
        for text in lines:
            scene.turn(text, (), speaker=speaker, hearer=hearer)

    en_quiet = bank.scene("en-age-quiet", "en").cast("Yarrow", "Calder", "Solway", "Anwick")
    en_quiet.turn(
        "Calder set the lichen board beside the night latch and stepped back.",
        ("Calder",), speaker=sp["en"], hearer=he["en"],
    )
    quiet(en_quiet, (
        "Mm, that lichen board can sit by the night latch.",
        "The night latch side of the lichen board can wait.",
        "Noted, the lichen board stays by the night latch.",
        "One moment beside the night latch and the lichen board.",
    ), sp["en"], he["en"])
    en_quiet.ask(
        "age_quiet", "phone",
        "Which telephone did he set down on the lichen board after those quiet nods by the night latch?",
        "ABSENT",
    )
    en_quiet.finish()

    zh_quiet = bank.scene("zh-age-quiet", "zh").cast("童屿", "卞茗", "裴涟", "栾昭")
    zh_quiet.turn(
        "卞茗把青苔板靠在夜闩旁边就退开了。",
        ("卞茗",), speaker=sp["zh"], hearer=he["zh"],
    )
    quiet(zh_quiet, (
        "嗯，那块青苔板可以靠着夜闩。",
        "夜闩那边的青苔板先等一下。",
        "记下了，青苔板就靠在夜闩旁。",
        "夜闩和青苔板旁边先停一下。",
    ), sp["zh"], he["zh"])
    zh_quiet.ask("age_quiet", "phone", "连着四声之后，他的电话在青苔板靠夜闩的哪一格？", "ABSENT")
    zh_quiet.finish()

    en_back = bank.scene("en-age-resurrect", "en").cast("Yarrow", "Mirren", "Solway", "Anwick")
    en_back.turn(
        "Mirren left the kelp ticket under the queue chalk and stepped away.",
        ("Mirren",), speaker=sp["en"], hearer=he["en"],
    )
    quiet(en_back, (
        "Mm, that kelp ticket can sit under the queue chalk.",
        "The queue chalk over the kelp ticket can wait.",
        "Noted, the kelp ticket stays under the queue chalk.",
        "One moment under the queue chalk and the kelp ticket.",
    ), sp["en"], he["en"])
    en_back.turn(
        "She waved once at the kelp ticket under the queue chalk.",
        (), speaker=sp["en"], hearer=he["en"],
    )
    en_back.ask(
        "age_resurrect", "birthday",
        "Which birthday did she set down on the kelp ticket after the quiet and that one wave?",
        "ABSENT",
    )
    en_back.finish()

    zh_back = bank.scene("zh-age-resurrect", "zh").cast("童屿", "苟宁", "裴涟", "栾昭")
    zh_back.turn(
        "苟宁把海带票搁在琥珀轴旁边就走开了。",
        ("苟宁",), speaker=sp["zh"], hearer=he["zh"],
    )
    quiet(zh_back, (
        "嗯，那张海带票可以搁着琥珀轴。",
        "琥珀轴那边的海带票先等一下。",
        "记下了，海带票就搁在琥珀轴旁。",
        "琥珀轴和海带票旁边先停一下。",
    ), sp["zh"], he["zh"])
    zh_back.turn("她朝琥珀轴旁边的海带票挥了一下。", (), speaker=sp["zh"], hearer=he["zh"])
    zh_back.ask("age_resurrect", "birthday", "安静了四轮又挥了一下之后，她的生日在海带票靠琥珀轴的哪一格？", "ABSENT")
    zh_back.finish()

    en_tie = bank.scene("en-age-tie", "en").cast("Yarrow", "Calder", "Berrick", "Solway", "Anwick")
    en_tie.turn(
        "Calder and Berrick both signed the kelp ticket beside the queue chalk.",
        ("Calder", "Berrick"), speaker=sp["en"], hearer=he["en"],
    )
    en_tie.ask(
        "age_tie", "phone",
        "Which telephone did she set down on the kelp ticket beside the queue chalk?",
        "ABSENT",
    )
    en_tie.finish()

    zh_tie = bank.scene("zh-age-tie", "zh").cast("童屿", "卞茗", "苟宁", "裴涟", "栾昭")
    zh_tie.turn(
        "卞茗和苟宁都在海带票上签了字，琥珀轴就在旁边。",
        ("卞茗", "苟宁"), speaker=sp["zh"], hearer=he["zh"],
    )
    zh_tie.ask("age_tie", "phone", "琥珀轴旁边那张海带票上，她的电话抄在哪一格？", "ABSENT")
    zh_tie.finish()

    en_aged = bank.scene("en-age-edit", "en").cast("Yarrow", "Calder", "Solway", "Anwick")
    en_aged.turn(
        "Calder set the salt-wick spool beside the wick tray and stepped away.",
        ("Calder",), speaker=sp["en"], hearer=he["en"],
    )
    quiet(en_aged, (
        "Mm, that salt-wick spool can sit by the wick tray.",
        "The wick tray side of the salt-wick spool can wait.",
        "Noted, the salt-wick spool stays by the wick tray.",
        "One moment beside the wick tray and the salt-wick spool.",
    ), sp["en"], he["en"])
    en_aged.ask(
        "age_edit", "phone",
        "Which telephone did Caldar leave on the salt-wick spool beside the wick tray?",
        "ABSENT",
    )
    en_aged.finish()

    zh_aged = bank.scene("zh-age-edit", "zh").cast("童屿", "苟宁", "裴涟", "栾昭")
    zh_aged.turn(
        "苟宁把青苔板靠在夜闩上就走开了。",
        ("苟宁",), speaker=sp["zh"], hearer=he["zh"],
    )
    quiet(zh_aged, (
        "嗯，那块青苔板可以靠着夜闩放。",
        "夜闩这边的青苔板先等一下。",
        "记下了，青苔板就留在夜闩旁。",
        "夜闩旁边的青苔板先停一下。",
    ), sp["zh"], he["zh"])
    zh_aged.ask("age_edit", "phone", "苟柠的电话记在青苔板靠近夜闩的哪一格？", "ABSENT")
    zh_aged.finish()

    def voc(
        eid: str,
        lang: str,
        target: str,
        kind: str,
        text: str,
        question: str,
        human: str,
        hearer: str = "",
        mentions: Sequence[str] = (),
        extra: Sequence[str] = (),
    ) -> None:
        decoy = _DECOY[lang]
        speaker = _SPEAKER[lang]
        cast = [decoy, target, speaker, *extra]
        if hearer and hearer not in cast:
            cast.append(hearer)
        if human not in ("ABSENT", "DECOY") and human not in cast:
            cast.append(human)
        scene = bank.scene(eid, lang).cast(*cast)
        scene.turn(text, mentions, speaker=speaker, hearer=hearer)
        scene.ask(kind, "phone", question, human)
        scene.finish()

    voc(
        "en-voc-comma", "en", "Mirren", "voc_lead",
        "Mirren, the kelp ticket stays under the amber spindle tonight.",
        "Which telephone should the clerk read back for you from the kelp ticket under the amber spindle?",
        "Mirren", mentions=("Mirren",),
    )
    voc(
        "en-voc-bang", "en", "Mirren", "voc_lead",
        "Mirren! the salt-wick spool stays beside the wick tray tonight.",
        "Which telephone should the clerk read back for you from the salt-wick spool beside the wick tray?",
        "Mirren", mentions=("Mirren",),
    )
    voc(
        "zh-voc-comma", "zh", "苟宁", "voc_lead",
        "苟宁，海带票先搁在琥珀轴旁边。",
        "给你回看的电话，在海带票靠近琥珀轴的哪一格？",
        "苟宁", mentions=("苟宁",),
    )
    voc(
        "zh-voc-colon", "zh", "苟宁", "voc_lead",
        "苟宁：盐芯先搁在芯盘旁边。",
        "给你回看的电话，在盐芯靠近芯盘的哪一格？",
        "苟宁", mentions=("苟宁",),
    )
    voc(
        "zh-voc-bang", "zh", "卞茗", "voc_lead",
        "卞茗！青苔板先靠在夜闩旁边。",
        "给你回看的电话，在青苔板靠近夜闩的哪一格？",
        "卞茗", mentions=("卞茗",),
    )
    voc(
        "en-voc-subject", "en", "Mirren", "voc_subject",
        "Mirren left the kelp ticket under the amber spindle before the queue thinned.",
        "Which telephone should the clerk read back for you from the kelp ticket under that amber spindle?",
        "ABSENT", mentions=("Mirren",),
    )
    voc(
        "zh-voc-subject", "zh", "苟宁", "voc_subject",
        "苟宁把海带票收进琥珀轴旁边的格子。",
        "给你回看的电话，在海带票收进琥珀轴旁边的哪一格？",
        "ABSENT", mentions=("苟宁",),
    )
    voc(
        "en-voc-object", "en", "Mirren", "voc_object",
        "The clerk set the kelp ticket beside Mirren before the queue thinned.",
        "Which telephone should the clerk read back for you from the kelp ticket set beside the queue?",
        "ABSENT", mentions=("Mirren",),
    )
    voc(
        "zh-voc-object", "zh", "苟宁", "voc_object",
        "店员把海带票放在苟宁旁边，琥珀轴还在。",
        "给你回看的电话，在海带票放在旁边、琥珀轴还在的哪一格？",
        "ABSENT", mentions=("苟宁",),
    )
    voc(
        "en-voc-medial", "en", "Mirren", "voc_medial",
        "After the queue thins, Mirren, the kelp ticket can wait under the amber spindle.",
        "Which telephone should the clerk read back for you once the queue has thinned over the kelp ticket?",
        "ABSENT", mentions=("Mirren",),
    )
    voc(
        "zh-voc-medial", "zh", "苟宁", "voc_medial",
        "队列散了以后，苟宁，海带票可以等在琥珀轴旁边。",
        "给你回看的电话，在队列散了、海带票等在琥珀轴旁的哪一格？",
        "ABSENT", mentions=("苟宁",),
    )
    voc(
        "en-voc-two", "en", "Mirren", "voc_two",
        "Mirren, Calder, the kelp ticket can wait under the amber spindle.",
        "Which telephone should the clerk read back for you from the kelp ticket waiting under the amber spindle?",
        "ABSENT", mentions=("Mirren", "Calder"), extra=("Calder",),
    )
    voc(
        "zh-voc-two", "zh", "苟宁", "voc_two",
        "苟宁，卞茗，海带票先放在琥珀轴旁边。",
        "给你回看的电话，在海带票先放、琥珀轴还在旁边的哪一格？",
        "ABSENT", mentions=("苟宁", "卞茗"), extra=("卞茗",),
    )
    voc(
        "en-voc-explicit", "en", "Mirren", "voc_explicit",
        "Mirren, the kelp ticket stays under the amber spindle for the evening.",
        "Which telephone should the clerk read back for you from the kelp ticket kept for the evening?",
        "Calder", hearer="Calder", mentions=("Mirren",),
    )
    voc(
        "zh-voc-explicit", "zh", "苟宁", "voc_explicit",
        "苟宁，海带票今晚搁在琥珀轴旁边。",
        "给你回看的电话，在海带票今晚搁着、琥珀轴在旁的哪一格？",
        "卞茗", hearer="卞茗", mentions=("苟宁",),
    )

    en_empty = bank.scene("en-empty", "en").cast("Yarrow", "Calder", "Solway", "Anwick")
    en_empty.ask(
        "empty", "phone",
        "Which telephone is yours on the salt-wick spool after the wick tray cooled?",
        "DECOY",
    )
    en_empty.finish()
    zh_empty = bank.scene("zh-empty", "zh").cast("童屿", "卞茗", "裴涟", "栾昭")
    zh_empty.ask("empty", "phone", "盐芯旁那只芯盘上，他的电话抄在哪一格？", "DECOY")
    zh_empty.finish()

    if TURN_WINDOW != 4:
        raise RuntimeError(f"window {TURN_WINDOW}")


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
    kinds = {ask.kind for ask in asks}
    needed = {
        "keep_he", "keep_she", "keep_ta", "keep_zhta", "keep_edit", "keep_ask",
        "age_quiet", "age_resurrect", "age_tie", "age_edit",
        "voc_lead", "voc_subject", "voc_object", "voc_medial", "voc_two", "voc_explicit",
        "empty",
    }
    if kinds != needed:
        raise RuntimeError(f"kinds {sorted(kinds ^ needed)}")
    if not any(ask.lang == "en" for ask in asks) or not any(ask.lang == "zh" for ask in asks):
        raise RuntimeError("one language is missing")


def _first_person(text: str) -> bool:
    import re

    if "我" in text:
        return True
    return re.search(r"\bI\b", text) is not None


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


if __name__ == "__main__":
    rows = build_rows()
    asks = list(iter_asks(rows))
    print(f"episodes {len(rows)} asks {len(asks)}")
    print("kinds", kind_counts(rows))
    print("questions", questions_sha256(rows))
    print("bank", bank_sha256(rows), "strings", len(bank_strings(rows)))
