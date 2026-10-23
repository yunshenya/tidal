"""Held-out for titles, particles, and a name followed by you / 你 / 您.

Listener labels were read off the dialogue text and the roster, and written
down before this bank is scored. This module does not replay the hearer
rule. It checks that each question has one lexical cue and is deictic.

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
# Decoy first, then the people a vocative may address, then the speaker.
_CAST = {
    "en": ("Acca", "Aldred", "Osric", "Eadgifu", "Leofwyn", "Saewynn", "Wiglaf"),
    "zh": ("终萩", "容昭", "汲宁", "蒲荃", "纪荻", "阙岫"),
}
_SPEAKER = {"en": "Wiglaf", "zh": "阙岫"}
_CONTENT = {
    "en": ("Aldred", "Osric", "Eadgifu", "Leofwyn", "Saewynn"),
    "zh": ("容昭", "汲宁", "蒲荃", "纪荻"),
}
# A second stored person whose surname is 容. Only the shared-surname episodes.
_SHARED = ("容浦",)
_EN_PROPS = (
    ("hoop gauge", "bung stave"),
    ("brine rake", "salt pan"),
    ("firkin hoop", "croze iron"),
    ("cant piece", "chime hoop"),
    ("heading board", "croze groove"),
    ("bilge stave", "bung hole"),
    ("pitch brush", "seam lath"),
    ("salt skimmer", "pan rim"),
    ("ash leach", "lye trough"),
    ("beech dowel", "elm peg"),
    ("truss hoop", "flag stave"),
    ("cooper knife", "shaving horse"),
)
_ZH_PROPS = (
    ("竹箍", "桶板"),
    ("卤耙", "盐盘"),
    ("木钎", "桶箍"),
    ("底片", "榫销"),
    ("盖板", "木楔"),
    ("箍尺", "塞板"),
    ("沥青刷", "缝条"),
    ("撇沫", "盘沿"),
    ("灰沥", "碱桶"),
    ("桐油", "麻缕"),
    ("竹篾", "桶壁"),
    ("铜钉", "铁铆"),
)
# code, listener, anchor, acknowledgement count. Listener is the addressed person.
_POS = {
    "en": (
        ("dr_comma", "Aldred", "Dr. {name}, the {a} rests against the {b}.", 0),
        ("mr_comma", "Osric", "Mr. {name}, the {a} rests against the {b}.", 0),
        ("mrs_comma", "Eadgifu", "Mrs. {name}, the {a} rests against the {b}.", 0),
        ("ms_comma", "Leofwyn", "Ms. {name}, the {a} rests against the {b}.", 0),
        ("miss_comma", "Saewynn", "Miss {name}, the {a} rests against the {b}.", 0),
        ("dr_bang", "Aldred", "Dr. {name}! the {a} rests against the {b}.", 0),
        ("ms_colon", "Leofwyn", "Ms. {name}: the {a} rests against the {b}.", 0),
        ("mr_you", "Osric", "Mr. {name} you left the {a} against the {b}.", 0),
        ("miss_you", "Saewynn", "Miss {name} you left the {a} against the {b}.", 0),
        ("bare_you", "Eadgifu", "{name} you left the {a} against the {b}.", 0),
        ("mrs_you", "Eadgifu", "Mrs. {name} you left the {a} against the {b}.", 0),
        ("dr_comma_later", "Aldred", "Dr. {name}, the {a} rests against the {b}.", 1),
        ("mr_comma_later", "Osric", "Mr. {name}, the {a} rests against the {b}.", 1),
        ("miss_bang_later", "Saewynn", "Miss {name}! the {a} rests against the {b}.", 1),
        ("ms_you_later", "Leofwyn", "Ms. {name} you left the {a} against the {b}.", 1),
        ("bare_you_later", "Aldred", "{name} you left the {a} against the {b}.", 1),
        ("dr_you_later", "Osric", "Dr. {name} you left the {a} against the {b}.", 1),
    ),
    "zh": (
        ("lao_sur", "容昭", "老容，{a}靠着{b}。", 0),
        ("xiao_sur", "汲宁", "小汲，{a}靠着{b}。", 0),
        ("da_sur", "蒲荃", "大蒲，{a}靠着{b}。", 0),
        ("a_sur", "纪荻", "阿纪，{a}靠着{b}。", 0),
        ("lao_full", "容昭", "老容昭，{a}靠着{b}。", 0),
        ("sur_teacher", "汲宁", "汲老师，{a}靠着{b}。", 0),
        ("full_teacher", "蒲荃", "蒲荃老师，{a}靠着{b}。", 0),
        ("particle_a", "纪荻", "纪荻啊，{a}靠着{b}。", 0),
        ("particle_ya", "容昭", "容昭呀，{a}靠着{b}。", 0),
        ("particle_ne", "汲宁", "汲宁呢，{a}靠着{b}。", 0),
        ("bare_ni", "蒲荃", "蒲荃你把{a}留到{b}边。", 0),
        ("bare_nin", "纪荻", "纪荻您把{a}留到{b}边。", 0),
        ("teacher_a", "容昭", "容老师啊，{a}靠着{b}。", 0),
        ("lao_ya", "汲宁", "老汲呀，{a}靠着{b}。", 0),
        ("lao_later", "蒲荃", "老蒲，{a}靠着{b}。", 1),
        ("ne_later", "纪荻", "纪荻呢，{a}靠着{b}。", 1),
        ("ni_later", "容昭", "容昭你把{a}留到{b}边。", 1),
    ),
}
# code, anchor, extra roster names. Listener is ABSENT: the shape is not an address.
_NEG = {
    "en": (
        ("dr_said_you", "Dr. Aldred said you left the {a} against the {b}.", ()),
        ("title_subject", "Mrs. Eadgifu shifted the {a} against the {b} and stepped back.", ()),
        ("title_object", "The bench note names Mr. Osric next to the {a} and the {b}.", ()),
        ("unknown_title", "Dr. Tibba, the {a} can rest against the {b}.", ()),
        ("two_heads", "Dr. Aldred, Ms. Leofwyn, the {a} can rest against the {b}.", ()),
        ("quote", 'The margin reads, "Saewynn, leave the {a} against the {b}."', ()),
        ("hey", "Hey Leofwyn, the {a} rests against the {b}.", ()),
        ("period", "Dr. Osric. the {a} rests against the {b}.", ()),
        ("your", "Eadgifu your note keeps the {a} against the {b}.", ()),
        ("medial", "Once the lid is down, Miss Saewynn, the {a} rests against the {b}.", ()),
        ("two_you", "Mr. Osric, Miss Saewynn you left the {a} against the {b}.", ()),
    ),
    "zh": (
        ("teacher_said", "容老师说完，把{a}留在{b}边。", ()),
        ("bare_a", "啊，算了，{a}先留在{b}边。", ()),
        ("bare_ya", "呀，算了，{a}先留在{b}边。", ()),
        ("bare_ne", "呢，先这样，{a}留在{b}边。", ()),
        ("unknown_title", "晏老师，{a}先留在{b}边。", ()),
        ("shared_surname", "老容，{a}先留在{b}边。", _SHARED),
        ("two_heads", "老容，老汲，{a}先留在{b}边。", ()),
        ("quote", "条子上写着：「容昭啊，把{a}留在{b}边。」", ()),
        ("lao_full_subject", "老容昭把{a}留在{b}边。", ()),
        ("medial", "等一下，汲老师，{a}留在{b}边。", ()),
        ("nimen", "蒲荃你们把{a}留在{b}边。", ()),
    ),
}
_EN_ACK = (
    "Leave the {a} covered, and let the {b} sit.",
    "The {b} needs nothing further, so the bench stays clear.",
    "A nod, and the {a} was not moved again.",
    "Keep the aisle open and leave the {b} untouched.",
)
_ZH_ACK = (
    "{a}先盖上，{b}放着就好。",
    "{b}不用再碰，案边空着。",
    "点过头了，{a}没有再挪。",
    "过道留着，{b}先别动。",
)
_PRONOUN_RE = {
    "en": re.compile(r"\b(he|she|you|him|her|his|they|them|i|your)\b", re.I),
    "zh": re.compile("他们|你们|她们|我|你|您|他|她"),
}
_KIND_ORDER = ("pos", "neg")


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
    shape: str


@dataclass(frozen=True)
class Episode:
    eid: str
    lang: str
    slots: Tuple[Slot, ...]
    events: Tuple[object, ...]


def _fill(template: str, a: str, b: str, name: str = "") -> str:
    return template.format(a=a, b=b, name=name)


def _question(lang: str, rel: str, a: str, b: str) -> str:
    if lang == "en":
        cue = "handset" if rel == "phone" else "natal day"
        return f"Filed for you, the {cue} beside the {a} and the {b} is which?"
    cue = "分机" if rel == "phone" else "出生"
    return f"记给您的{cue}写在{a}与{b}之间的哪一行？"


class _Bank:
    def __init__(self) -> None:
        self.episodes: List[Episode] = []
        self.n = 1
        self._prop = {"en": 0, "zh": 0}

    def take(self, text: str, lang: str) -> str:
        n = self.n
        self.n += 1
        if lang == "zh":
            return f"{text} 桶号{n:04d}"
        return f"{text} firkin tally {n:04d}"

    def props(self, lang: str) -> Tuple[str, str]:
        pool = _EN_PROPS if lang == "en" else _ZH_PROPS
        pair = pool[self._prop[lang] % len(pool)]
        self._prop[lang] += 1
        return pair

    def slots(self, lang: str, extra: Sequence[str]) -> Tuple[Slot, ...]:
        people = tuple(_CAST[lang]) + tuple(extra)
        out: List[Slot] = []
        for rel in _REL:
            cue = _CUE[lang][rel]
            for name in people:
                n = self.n
                if lang == "zh":
                    body = f"桶笺{n:04d}将{name}写入{cue}栏"
                else:
                    body = f"Firkin sheet {n:04d} files {name} under the {cue} heading"
                text = self.take(body, lang)
                out.append(Slot(name, rel, f"fk{n:04d}", text))
        return tuple(out)

    def add(
        self,
        eid: str,
        lang: str,
        kind: str,
        shape: str,
        turns: Sequence[str],
        rel: str,
        listener: str,
        a: str,
        b: str,
        extra: Sequence[str],
    ) -> None:
        slots = self.slots(lang, extra)
        events: List[object] = []
        speaker = _SPEAKER[lang]
        for text in turns:
            events.append(Turn(self.take(text, lang), speaker, ""))
        events.append(Ask(
            kind, lang, rel, self.take(_question(lang, rel, a, b), lang), listener, shape,
        ))
        self.episodes.append(Episode(eid, lang, slots, tuple(events)))


def _load(bank: _Bank) -> None:
    for lang in ("en", "zh"):
        for code, listener, template, later in _POS[lang]:
            for rel in ("phone", "birthday"):
                a, b = bank.props(lang)
                turns = [_fill(template, a, b, listener)]
                for i in range(later):
                    ack = _EN_ACK[i % len(_EN_ACK)] if lang == "en" else _ZH_ACK[i % len(_ZH_ACK)]
                    turns.append(_fill(ack, a, b))
                bank.add(
                    f"pos_{lang}_{code}_{rel}", lang, "pos", code, turns, rel, listener, a, b, (),
                )
        for code, template, extra in _NEG[lang]:
            for rel in ("phone", "birthday"):
                a, b = bank.props(lang)
                bank.add(
                    f"neg_{lang}_{code}_{rel}", lang, "neg", code,
                    [_fill(template, a, b)], rel, "ABSENT", a, b, extra,
                )


def _surnames(names: Sequence[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for name in names:
        if not name or not ("\u4e00" <= name[0] <= "\u9fff"):
            continue
        out.setdefault(name[0], []).append(name)
    return out


def _check(rows: Sequence[Episode]) -> None:
    eids = [row.eid for row in rows]
    if len(eids) != len(set(eids)):
        raise RuntimeError("episode ids are not unique")
    counts = kind_counts(rows)
    if counts != {"pos": 68, "neg": 44}:
        raise RuntimeError(f"kind counts {counts}")
    by_lang = {"en": 0, "zh": 0}
    seen = set()
    for row in rows:
        by_lang[row.lang] += 1
        people = []
        for slot in row.slots:
            if slot.name not in people:
                people.append(slot.name)
        if people[0] != _CAST[row.lang][0]:
            raise RuntimeError(f"{row.eid} decoy is not first")
        if row.eid.endswith("shared_surname_phone") or row.eid.endswith("shared_surname_birthday"):
            if _SHARED[0] not in people:
                raise RuntimeError(f"{row.eid} missing the shared surname")
        elif _SHARED[0] in people:
            raise RuntimeError(f"{row.eid} stores {_SHARED[0]}")
        if row.lang == "zh" and "shared_surname" not in row.eid:
            for ch, owners in _surnames(people).items():
                if len(owners) != 1:
                    raise RuntimeError(f"{row.eid} surname {ch} {owners}")
        for rel in _REL:
            first = next(slot for slot in row.slots if slot.rel == rel)
            if first.name != _CAST[row.lang][0]:
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
        if ask.lang != row.lang or ask.kind not in counts or ask.shape not in row.eid:
            raise RuntimeError(f"{row.eid} ask header")
        if lex_labels(ask.question) != (ask.rel,):
            raise RuntimeError(f"{row.eid} question cues {lex_labels(ask.question)} {ask.question}")
        if not is_deictic(ask.question):
            raise RuntimeError(f"{row.eid} question is not deictic")
        pronouns = _PRONOUN_RE[row.lang].findall(ask.question)
        if row.lang == "en" and [item.lower() for item in pronouns] != ["you"]:
            raise RuntimeError(f"{row.eid} pronouns {pronouns}")
        if row.lang == "zh" and pronouns != ["您"]:
            raise RuntimeError(f"{row.eid} pronouns {pronouns}")
        roster = set(people)
        for name in roster:
            if name in ask.question:
                raise RuntimeError(f"{row.eid} question names {name}")
        if ask.kind == "pos":
            if ask.listener not in _CONTENT[row.lang]:
                raise RuntimeError(f"{row.eid} listener {ask.listener}")
            anchor = turns[0].text
            if ask.listener not in anchor and ask.listener[0] not in anchor:
                raise RuntimeError(f"{row.eid} anchor does not address {ask.listener}: {anchor}")
            if len(turns) not in (1, 2):
                raise RuntimeError(f"{row.eid} turns {len(turns)}")
        else:
            if ask.listener != "ABSENT" or len(turns) != 1:
                raise RuntimeError(f"{row.eid} neg shape")
        for event in turns:
            if event.speaker != _SPEAKER[row.lang] or event.hearer != "":
                raise RuntimeError(f"{row.eid} turn header")
            if _CAST[row.lang][0] in event.text or _SPEAKER[row.lang] in event.text:
                raise RuntimeError(f"{row.eid} meta name in {event.text}")
            if re.search(r"\bI\b", event.text) or "我" in event.text or "他们" in event.text:
                raise RuntimeError(f"{row.eid} person leak {event.text}")
            if "你们" in event.text and "nimen" not in row.eid:
                raise RuntimeError(f"{row.eid} 你们 in {event.text}")
            if lex_labels(event.text) != ():
                raise RuntimeError(f"{row.eid} turn cue {lex_labels(event.text)} {event.text}")
        for slot in row.slots:
            if slot.value.lower() in slot.text.lower():
                raise RuntimeError(f"{row.eid} value leak")
            texts.append(slot.text)
        for text in texts:
            if text in seen:
                raise RuntimeError(f"duplicate string {text}")
            seen.add(text)
    if by_lang != {"en": 56, "zh": 56}:
        raise RuntimeError(f"lang counts {by_lang}")
    pos_lang = {"en": 0, "zh": 0}
    neg_lang = {"en": 0, "zh": 0}
    shapes = {}
    for _, ask in iter_asks(rows):
        if ask.kind == "pos":
            pos_lang[ask.lang] += 1
        else:
            neg_lang[ask.lang] += 1
        shapes.setdefault((ask.kind, ask.lang, ask.shape), 0)
        shapes[(ask.kind, ask.lang, ask.shape)] += 1
    if pos_lang != {"en": 34, "zh": 34} or neg_lang != {"en": 22, "zh": 22}:
        raise RuntimeError(f"balance pos {pos_lang} neg {neg_lang}")
    if any(n != 2 for n in shapes.values()):
        raise RuntimeError(f"shape sizes {shapes}")
    if len(shapes) != 17 * 2 + 11 * 2:
        raise RuntimeError(f"{len(shapes)} shapes")


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
