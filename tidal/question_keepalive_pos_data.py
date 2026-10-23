"""Positive-heavy held-out for pronoun keep-alive and leading vocatives.

Listener labels were read off the dialogue text and the roster, and written
down in ``reports/dialogue_keepalive_pos_prereg.md`` before this bank is
scored. This module does not replay the rule. It checks that each question
has one lexical cue and is deictic. It does not decide who the pronoun is.

Telephone and birthday of the same shape are two episodes, so a second ask
does not push the gap out by one turn.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from tidal.question_surface import is_deictic, lex_labels

_REL = ("phone", "birthday", "places")
_CUE = {
    "en": {"phone": "telephone", "birthday": "birthday", "places": "places"},
    "zh": {"phone": "电话", "birthday": "生日", "places": "地点"},
}
# Decoy, three people the chains are about, speaker, hearer. Decoy is stored first.
_CAST = {
    "en": ("Goscelin", "Ranulf", "Isolde", "Petronilla", "Wymond", "Mahel"),
    "zh": ("穆宁", "祁松", "康梧", "钮芮", "费桐", "薛蔓"),
}
_SPEAKER = {"en": "Wymond", "zh": "费桐"}
_HEARER = {"en": "Mahel", "zh": "薛蔓"}
_CONTENT = {
    "en": ("Ranulf", "Isolde", "Petronilla"),
    "zh": ("祁松", "康梧", "钮芮"),
}
# A title in the anchor is that stored person. Read off the roster, not a replay.
_TITLE = (
    ("老祁", "祁松"),
    ("祁老师", "祁松"),
    ("老康", "康梧"),
    ("康老师", "康梧"),
    ("Dr. Ranulf", "Ranulf"),
    ("Dr. Isolde", "Isolde"),
    ("Dr. Petronilla", "Petronilla"),
)
_EN_PROPS = (
    ("gallnut tag", "quern pin"),
    ("woad slip", "vat stake"),
    ("madder chip", "dye tray"),
    ("teasel card", "fuller hook"),
    ("linseed cake", "press cloth"),
    ("oak gall", "ink horn"),
    ("weld scrap", "pot hook"),
    ("indigo cake", "vat rim"),
    ("cochineal bit", "bowl lip"),
    ("sumac chip", "tan pit"),
    ("walnut hull", "dye pot"),
    ("clover tow", "heckle pin"),
    ("broom corn", "besom peg"),
    ("ragwort bundle", "ditch stake"),
    ("mallow pad", "trough lip"),
    ("weld seed", "copper pan"),
)
_ZH_PROPS = (
    ("皂荚牌", "石磨钉"),
    ("蓝靛条", "染缸桩"),
    ("茜草片", "染槽"),
    ("起绒牌", "漂洗钩"),
    ("亚麻饼", "榨布"),
    ("橡瘿", "墨角"),
    ("淡黄屑", "锅钩"),
    ("靛蓝块", "缸沿"),
    ("胭脂屑", "碗沿"),
    ("五倍子", "鞣坑"),
    ("核桃壳", "染锅"),
    ("苜蓿缕", "栉钉"),
    ("扫帚草", "帚钉"),
    ("艾捆", "沟桩"),
    ("锦葵垫", "槽沿"),
    ("皂角渣", "磨槽"),
)
# code, intervening marks, person, pronoun, second-person anchor
_KEEP = {
    "en": (
        ("he2", "pa", "Ranulf", "he", False),
        ("she3", "apa", "Isolde", "she", False),
        ("he4", "papa", "Ranulf", "he", False),
        ("she5", "apaap", "Isolde", "she", False),
        ("he6m", "apapap", "Petronilla", "he", False),
        ("he6a", "aaaaap", "Ranulf", "he", False),
        ("you2", "pa", "Ranulf", "you", True),
        ("you3", "apa", "Isolde", "you", True),
        ("you4", "papa", "Petronilla", "you", True),
        ("you5", "apaap", "Ranulf", "you", True),
        ("you6m", "apapap", "Isolde", "you", True),
        ("you6a", "aaaaap", "Petronilla", "you", True),
    ),
    "zh": (
        ("ta2", "pa", "祁松", "他", False),
        ("taf3", "apa", "康梧", "她", False),
        ("ta4", "papa", "祁松", "他", False),
        ("taf5", "apaap", "康梧", "她", False),
        ("ta6m", "apapap", "钮芮", "他", False),
        ("ta6a", "aaaaap", "祁松", "他", False),
        ("ni2", "pa", "祁松", "你", True),
        ("nin3", "apa", "康梧", "您", True),
        ("ni4", "papa", "钮芮", "你", True),
        ("nin5", "apaap", "祁松", "您", True),
        ("ni6m", "apapap", "康梧", "你", True),
        ("nin6a", "aaaaap", "钮芮", "您", True),
    ),
}
# code, listener, anchor, acknowledgement count, question pronoun
_VOC = {
    "en": (
        ("comma_ranulf", "Ranulf", "Ranulf, the {a} rests near the {b}.", 0, "you"),
        ("comma_isolde", "Isolde", "Isolde, the {a} rests near the {b}.", 0, "you"),
        ("dr_ranulf", "Ranulf", "Dr. Ranulf, the {a} rests near the {b}.", 0, "you"),
        ("dr_isolde", "Isolde", "Dr. Isolde, the {a} rests near the {b}.", 0, "you"),
        ("comma_you_ranulf", "Ranulf", "Ranulf, you left the {a} near the {b}.", 0, "you"),
        ("comma_you_isolde", "Isolde", "Isolde, you left the {a} near the {b}.", 0, "you"),
        ("comma_later_ranulf", "Ranulf", "Ranulf, the {a} rests near the {b}.", 1, "you"),
        ("dr_later_ranulf", "Ranulf", "Dr. Ranulf, the {a} was left near the {b}.", 1, "you"),
        ("bare_you_later_isolde", "Isolde", "Isolde you left the {a} near the {b}.", 1, "you"),
        ("comma_petronilla", "Petronilla", "Petronilla, the {a} rests near the {b}.", 0, "you"),
        ("dr_later_petronilla", "Petronilla", "Dr. Petronilla, the {a} was left near the {b}.", 1, "you"),
        ("comma_you_later_petronilla", "Petronilla", "Petronilla, you left the {a} near the {b}.", 1, "you"),
    ),
    "zh": (
        ("comma_qi", "祁松", "祁松，{a}靠在{b}旁边。", 0, "你"),
        ("dun_kang", "康梧", "康梧、{a}靠在{b}旁边。", 0, "您"),
        ("a_qi", "祁松", "祁松啊，{a}靠在{b}旁边。", 0, "你"),
        ("ya_kang", "康梧", "康梧呀，{a}靠在{b}旁边。", 0, "您"),
        ("lao_qi", "祁松", "老祁，{a}靠在{b}旁边。", 0, "你"),
        ("teacher_qi", "祁松", "祁老师，{a}靠在{b}旁边。", 0, "您"),
        ("bare_ni_kang", "康梧", "康梧你把{a}挪到{b}旁边。", 0, "你"),
        ("comma_ni_qi", "祁松", "祁松，你把{a}挪到{b}旁边。", 0, "您"),
        ("lao_kang_later", "康梧", "老康，{a}靠在{b}旁边。", 1, "你"),
        ("a_niu_later", "钮芮", "钮芮啊，{a}靠在{b}旁边。", 1, "您"),
        ("teacher_kang_later", "康梧", "康老师，{a}靠在{b}旁边。", 1, "你"),
        ("dun_niu_later", "钮芮", "钮芮、{a}靠在{b}旁边。", 1, "您"),
    ),
}
# code, rel, pronoun, turn. Listener is ABSENT. The name is not the addressee.
_NEG = {
    "en": (
        ("subj_ranulf", "phone", "you", "Ranulf shifted the {a} beside the {b} without looking up."),
        ("subj_isolde", "birthday", "you", "Isolde carried the {a} over to the {b} during the lull."),
        ("obj_petronilla", "phone", "you", "The bench note mentions Petronilla next to the {a} and the {b}."),
        ("obj_ranulf", "birthday", "you", "A clerk listed Ranulf under the {a} rather than the {b}."),
        ("two_he", "phone", "he", "Ranulf and Isolde both shifted the {a} toward the {b}."),
        ("two_she", "birthday", "she", "Petronilla stood with Isolde while the {a} stayed by the {b}."),
        ("quote", "phone", "you", 'The margin slip reads, "Ranulf, leave the {a} near the {b}."'),
        ("two_voc", "birthday", "you", "Ranulf, Isolde, the {a} can rest near the {b}."),
        ("unknown", "phone", "you", "Swithin, the {a} can rest near the {b}."),
        ("title_subj", "birthday", "you", "Dr. Ranulf left the {a} beside the {b} and walked off."),
    ),
    "zh": (
        ("subj_qi", "phone", "你", "祁松把{a}挪到{b}旁边，目光没抬。"),
        ("subj_kang", "birthday", "你", "康梧抱着{a}走到{b}那边，趁空档放下。"),
        ("obj_niu", "phone", "你", "案上的条子写到钮芮，旁边是{a}和{b}。"),
        ("obj_qi", "birthday", "你", "抄写的人把祁松列在{a}下面，而不是{b}。"),
        ("two_ta", "phone", "他", "祁松和康梧一起把{a}推向{b}。"),
        ("two_taf", "birthday", "她", "钮芮挨着康梧站着，{a}仍在{b}边上。"),
        ("quote", "phone", "你", "条子上写着：「祁松，把{a}留在{b}旁边。」"),
        ("two_voc", "birthday", "你", "祁松，康梧，{a}可以靠在{b}旁边。"),
        ("unknown", "phone", "你", "鄂棠，{a}可以靠在{b}旁边。"),
        ("title_subj", "birthday", "你", "祁老师把{a}放在{b}旁边，说完就走开了。"),
    ),
}
_EN_P = (
    "{P} folded the {a} once the {b} cooled.",
    "{P} checked that the {a} belonged near the {b}.",
    "{P} tapped the {a} by the {b}.",
    "{P} kept one scrap under the {a}.",
    "{P} smoothed the {a} beside the {b}.",
    "{P} pointed at the {a} near the {b}.",
)
_EN_ACK = (
    "Quiet around the {a} for a moment, with the lid left shut.",
    "The {b} does not need a hand yet, so the bench stays clear.",
    "A nod over the {a}, and nothing else was moved.",
    "The {b} can stay as it is until the next check.",
    "No fresh mark on the {a} after that pause.",
    "Leave the {b} untouched and keep the aisle open.",
)
_ZH_P = (
    "{p}把{a}折好，{b}已经凉了。",
    "{p}确定{a}该挨着{b}。",
    "{p}在{b}旁敲了敲{a}。",
    "{p}把纸条压在{a}下面。",
    "{p}把{a}抚平，挨着{b}。",
    "{p}指了指{b}旁边的{a}。",
)
_ZH_ACK = (
    "先静一静，{a}放着就好，盖子别揭。",
    "{b}暂时不用动手，案边保持空着。",
    "点一下头，{a}先这样，别的都别动。",
    "{b}保持原样即可，下一回再看。",
    "{a}上没有新痕，这一停就到此。",
    "{b}先别挪，过道留出来。",
)
_EN_ANCHOR_HE = (
    "{name} set the {a} beside the {b} and stepped back.",
    "{name} laid the {a} next to the {b} and left the bench.",
    "{name} rested the {a} by the {b} after the bench was wiped.",
)
_EN_ANCHOR_YOU = (
    "{name}, the {a} now rests near the {b}.",
    "{name}, the {a} has been set by the {b}.",
    "{name}, the {a} stays next to the {b} for now.",
)
_ZH_ANCHOR_TA = (
    "{name}把{a}搁到{b}旁边，然后退开半步。",
    "{name}将{a}放在{b}一侧，随后离开案边。",
    "{name}把{a}挪到{b}边上，擦了擦手就停住。",
)
_ZH_ANCHOR_YOU = (
    "{name}，{a}这会儿靠在{b}旁边。",
    "{name}，{a}已经挪到{b}一侧。",
    "{name}，{a}先留在{b}旁边。",
)
_PRONOUN_RE = {
    "en": re.compile(r"\b(he|she|you|him|her|his|they|them|i)\b", re.I),
    "zh": re.compile("他们|你们|她们|我|你|您|他|她"),
}
_KIND_ORDER = ("pos_keep", "pos_voc", "neg_near")


@dataclass(frozen=True)
class Slot:
    name: str
    rel: str
    value: str
    text: str


@dataclass(frozen=True)
class Turn:
    text: str
    speaker: str
    hearer: str


@dataclass(frozen=True)
class Ask:
    kind: str
    lang: str
    rel: str
    question: str
    listener: str


@dataclass(frozen=True)
class Episode:
    eid: str
    lang: str
    slots: Tuple[Slot, ...]
    events: Tuple[object, ...]


def _people(lang: str) -> Tuple[str, ...]:
    return _CAST[lang]


def _fill(template: str, a: str, b: str, **extra: str) -> str:
    return template.format(a=a, b=b, **extra)


def _question(lang: str, rel: str, pronoun: str, a: str, b: str) -> str:
    cue = _CUE[lang][rel]
    if lang == "en":
        if pronoun in ("he", "she"):
            if rel == "phone":
                return f"Which {cue} did {pronoun} leave with the {a} near the {b}?"
            return f"Which {cue} did {pronoun} mark beside the {a} and the {b}?"
        if rel == "phone":
            return f"Which {cue} is kept for you with the {a} near the {b}?"
        return f"Which {cue} stays recorded for you, the {a} by the {b}?"
    if rel == "phone":
        return f"{pronoun}的{cue}落在{a}挨着{b}的哪一栏？"
    return f"{pronoun}的{cue}收在{a}同{b}一块的哪一栏？"


def _pronoun_line(lang: str, pronoun: str, a: str, b: str, i: int) -> str:
    if lang == "en":
        return _fill(_EN_P[i % len(_EN_P)], a, b, P=pronoun.capitalize())
    return _fill(_ZH_P[i % len(_ZH_P)], a, b, p=pronoun)


def _ack_line(lang: str, a: str, b: str, i: int) -> str:
    template = _EN_ACK[i % len(_EN_ACK)] if lang == "en" else _ZH_ACK[i % len(_ZH_ACK)]
    return _fill(template, a, b)


def _anchor(lang: str, name: str, a: str, b: str, second: bool, i: int) -> str:
    if lang == "en":
        pool = _EN_ANCHOR_YOU if second else _EN_ANCHOR_HE
    else:
        pool = _ZH_ANCHOR_YOU if second else _ZH_ANCHOR_TA
    return _fill(pool[i % len(pool)], a, b, name=name)


class _Bank:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []
        self.n = 1
        self._prop = {"en": 0, "zh": 0}

    def take(self, text: str) -> str:
        n = self.n
        self.n += 1
        return f"{text} gallnut tally {n:04d}"

    def props(self, lang: str) -> Tuple[str, str]:
        pool = _EN_PROPS if lang == "en" else _ZH_PROPS
        pair = pool[self._prop[lang] % len(pool)]
        self._prop[lang] += 1
        return pair

    def slots(self, lang: str) -> Tuple[Slot, ...]:
        people = _people(lang)
        out: List[Slot] = []
        for rel in _REL:
            cue = _CUE[lang][rel]
            for name in people:
                n = self.n
                if lang == "zh":
                    body = f"皂荚页{n:04d}把{name}记在{cue}一行"
                else:
                    body = f"Gallnut folio {n:04d} records {name} on the {cue} line"
                text = self.take(body)
                out.append(Slot(name, rel, f"gnq{n:04d}", text))
        return tuple(out)

    def add(
        self,
        eid: str,
        lang: str,
        kind: str,
        turns: Sequence[Tuple[str, str]],
        rel: str,
        pronoun: str,
        listener: str,
        a: str,
        b: str,
    ) -> None:
        slots = self.slots(lang)
        events: List[object] = []
        speaker = _SPEAKER[lang]
        for text, hearer in turns:
            events.append(Turn(self.take(text), speaker, hearer))
        events.append(Ask(kind, lang, rel, self.take(_question(lang, rel, pronoun, a, b)), listener))
        self.episodes.append(Episode(eid, lang, slots, tuple(events)))


def _turns_for_keep(lang: str, pattern: str, name: str, pronoun: str, second: bool, a: str, b: str, salt: int):
    hearer = "" if second else _HEARER[lang]
    turns = [(_anchor(lang, name, a, b, second, salt), hearer)]
    pi = salt
    ai = salt
    for mark in pattern:
        if mark == "p":
            turns.append((_pronoun_line(lang, pronoun, a, b, pi), hearer))
            pi += 1
        elif mark == "a":
            turns.append((_ack_line(lang, a, b, ai), hearer))
            ai += 1
        else:
            raise RuntimeError(mark)
    return turns


def _load(bank: _Bank) -> None:
    for lang in ("en", "zh"):
        for index, (code, pattern, name, pronoun, second) in enumerate(_KEEP[lang]):
            for rel_i, rel in enumerate(("phone", "birthday")):
                a, b = bank.props(lang)
                eid = f"pos_keep_{lang}_{code}_{rel}"
                turns = _turns_for_keep(lang, pattern, name, pronoun, second, a, b, index + rel_i)
                bank.add(eid, lang, "pos_keep", turns, rel, pronoun, name, a, b)
        for code, listener, template, later, pronoun in _VOC[lang]:
            for rel in ("phone", "birthday"):
                a, b = bank.props(lang)
                turns = [(_fill(template, a, b), "")]
                for i in range(later):
                    turns.append((_ack_line(lang, a, b, i + 1), ""))
                bank.add(
                    f"pos_voc_{lang}_{code}_{rel}", lang, "pos_voc", turns, rel, pronoun, listener, a, b,
                )
        for code, rel, pronoun, template in _NEG[lang]:
            a, b = bank.props(lang)
            turns = [(_fill(template, a, b), "")]
            bank.add(
                f"neg_near_{lang}_{code}_{rel}", lang, "neg_near", turns, rel, pronoun, "ABSENT", a, b,
            )


def _title_hit(text: str, listener: str) -> bool:
    return any(title in text and person == listener for title, person in _TITLE)


def _pronouns(lang: str, text: str) -> List[str]:
    return [item.lower() for item in _PRONOUN_RE[lang].findall(text)]


def _check(rows: Sequence[Episode]) -> None:
    eids = [row.eid for row in rows]
    if len(eids) != len(set(eids)):
        raise RuntimeError("episode ids are not unique")
    counts = kind_counts(rows)
    if counts != {"pos_keep": 48, "pos_voc": 48, "neg_near": 20}:
        raise RuntimeError(f"kind counts {counts}")
    seen = set()
    for row in rows:
        people = _people(row.lang)
        if row.slots[0].name != people[0]:
            raise RuntimeError(f"{row.eid} decoy is not first")
        for rel in _REL:
            first = next(slot for slot in row.slots if slot.rel == rel)
            if first.name != people[0]:
                raise RuntimeError(f"{row.eid} {rel} does not start with the decoy")
            if lex_labels(first.text) != (rel,):
                raise RuntimeError(f"{row.eid} fact cues {lex_labels(first.text)}")
        texts = []
        turns = []
        asks = []
        for event in row.events:
            if isinstance(event, Turn):
                turns.append(event)
                texts.append(event.text)
            elif isinstance(event, Ask):
                asks.append(event)
                texts.append(event.question)
            else:
                raise RuntimeError(type(event))
        if len(asks) != 1:
            raise RuntimeError(f"{row.eid} has {len(asks)} asks")
        ask = asks[0]
        if ask.lang != row.lang or ask.kind not in counts:
            raise RuntimeError(eid_kind(row, ask))
        if lex_labels(ask.question) != (ask.rel,):
            raise RuntimeError(f"{row.eid} question cues {lex_labels(ask.question)} {ask.question}")
        if not is_deictic(ask.question):
            raise RuntimeError(f"{row.eid} question is not deictic")
        if _pronouns(row.lang, ask.question) != [ask_pronoun(ask)]:
            raise RuntimeError(f"{row.eid} pronouns {_pronouns(row.lang, ask.question)}")
        for name in people:
            if name in ask.question:
                raise RuntimeError(f"{row.eid} question names {name}")
        if ask.listener == "ABSENT":
            if ask.kind != "neg_near":
                raise RuntimeError(f"{row.eid} positive labeled ABSENT")
        elif ask.listener not in _CONTENT[row.lang]:
            raise RuntimeError(f"{row.eid} listener {ask.listener}")
        anchor = turns[0].text
        if ask.listener != "ABSENT" and ask.listener not in anchor and not _title_hit(anchor, ask.listener):
            raise RuntimeError(f"{row.eid} anchor does not address {ask.listener}: {anchor}")
        meta = (_CAST[row.lang][0], _SPEAKER[row.lang], _HEARER[row.lang])
        for event in turns:
            if event.speaker != _SPEAKER[row.lang]:
                raise RuntimeError(f"{row.eid} speaker {event.speaker}")
            for name in meta:
                if name in event.text:
                    raise RuntimeError(f"{row.eid} meta name {name} in {event.text}")
            if re.search(r"\bI\b", event.text) or "我" in event.text or "他们" in event.text:
                raise RuntimeError(f"{row.eid} person leak {event.text}")
            if lex_labels(event.text) != ():
                raise RuntimeError(f"{row.eid} turn cue {lex_labels(event.text)} {event.text}")
        if "我" in ask.question or re.search(r"\bI\b", ask.question):
            raise RuntimeError(f"{row.eid} question person leak")
        if ask.kind == "pos_keep":
            want = _HEARER[row.lang] if _keep_third(row.eid) else ""
            if any(event.hearer != want for event in turns):
                raise RuntimeError(f"{row.eid} hearer field")
            if len(turns) < 3 or len(turns) > 7:
                raise RuntimeError(f"{row.eid} gap {len(turns) - 1}")
        elif ask.kind == "pos_voc":
            if any(event.hearer != "" for event in turns):
                raise RuntimeError(f"{row.eid} vocative passes a hearer")
            if len(turns) not in (1, 2):
                raise RuntimeError(f"{row.eid} vocative turns {len(turns)}")
        else:
            if any(event.hearer != "" for event in turns) or len(turns) != 1:
                raise RuntimeError(f"{row.eid} neg shape")
        for slot in row.slots:
            if slot.value.lower() in slot.text.lower():
                raise RuntimeError(f"{row.eid} value leak")
            texts.append(slot.text)
        for text in texts:
            if text in seen:
                raise RuntimeError(f"duplicate string {text}")
            seen.add(text)
            if "Swithin" in text and row.lang != "en":
                raise RuntimeError(text)
            if "鄂棠" in text and row.lang != "zh":
                raise RuntimeError(text)


def eid_kind(row: Episode, ask: Ask) -> str:
    return f"{row.eid} {ask.kind} {ask.lang}"


def ask_pronoun(ask: Ask) -> str:
    if ask.lang == "en":
        found = _PRONOUN_RE["en"].findall(ask.question)
        if len(found) != 1:
            raise RuntimeError(ask.question)
        return found[0].lower()
    found = _PRONOUN_RE["zh"].findall(ask.question)
    if found != [found[0]] or found[0] in ("他们", "你们", "她们", "我"):
        raise RuntimeError(ask.question)
    return found[0]


def _keep_third(eid: str) -> bool:
    return any(token in eid for token in ("_he", "_she", "_ta", "_taf"))


@lru_cache(maxsize=1)
def build_rows() -> Tuple[Episode, ...]:
    bank = _Bank()
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
    for row in rows:
        for slot in row.slots:
            seen.append(slot.text)
        for event in row.events:
            if isinstance(event, Turn):
                seen.append(event.text)
            elif isinstance(event, Ask):
                seen.append(event.question)
    return tuple(seen)


def _sha(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def questions_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    return _sha(questions(rows))


def bank_sha256(rows: Optional[Sequence[Episode]] = None) -> str:
    return _sha(bank_strings(rows))


def kind_counts(rows: Optional[Sequence[Episode]] = None) -> Dict[str, int]:
    counts = {name: 0 for name in _KIND_ORDER}
    for _, ask in iter_asks(rows):
        counts[ask.kind] += 1
    return counts


if __name__ == "__main__":
    built = build_rows()
    print(f"episodes {len(built)}")
    print(f"asks {sum(kind_counts(built).values())} {kind_counts(built)}")
    print(f"questions {questions_sha256(built)}")
    print(f"bank {bank_sha256(built)} strings {len(bank_strings(built))}")
