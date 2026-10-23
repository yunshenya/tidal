"""Multi-turn dev bank for dialogue-state name resolution.

The matcher does not import this module. The frozen adversarial questions
are not an input. Each typo surface is asked after the stored person was
named in the dialogue, and again with that person absent from the dialogue
(no mention, or only a different person mentioned). Fact writes alone are
not a mention. Anchors ``barmkin`` / ``oxter`` / ``thridNN`` / ``㐀㐁`` keep
the strings off earlier banks.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

_CUE = {"phone": "telephone", "birthday": "birthday", "places": "locations"}
_CUE_ZH = {"phone": "电话", "birthday": "生日", "places": "地点"}

# (stored, edit-1 surface). Distance 1, not an extension.
_EN_TYPO = (
    ("Swithun", "Swithan"),
    ("Eadburg", "Eadberg"),
    ("Cuthbert", "Cuthbart"),
    ("Frideswide", "Frideswede"),
    ("Beornwyn", "Beornwan"),
    ("Mildthryth", "Mildthrith"),
    ("Osgyth", "Osgith"),
    ("Aelfgifu", "Aelfgefu"),
)
_ZH_TYPO = (
    ("储晏", "储宴"),
    ("毕芩", "毕苓"),
    ("伍蘅", "伍衡"),
    ("屠芃", "屠芫"),
    ("钮茜", "钮倩"),
    ("柯菱", "柯棱"),
    ("管菁", "管箐"),
    ("莫荃", "莫筌"),
)
_EN_OTHER = ("Wulfrun", "Cyneburg", "Eormengyth", "Saethryth")
_ZH_OTHER = ("宣蘅", "杭萱", "包菁", "左荃")
_EN_NICK = (
    ("Virginia", "Ginny"),
    ("Rebecca", "Becky"),
    ("Anthony", "Tony"),
    ("Dorothy", "Dolly"),
)
_ZH_NICK = (("储晏", "小晏"), ("毕芩", "阿芩"))
_EN_EXT = (("Swithun", "Swithuno"), ("Cuthbert", "Cuthberto"))
_ZH_SUPER = (
    ("储晏", "储晏然"),
    ("毕芩", "毕芩子"),
    ("伍蘅", "伍蘅儿"),
    ("屠芃", "屠芃然"),
)
_ZH_PREFIX = (("储晏", "老储晏"), ("毕芩", "小毕芩"))
_EN_TIE = ("Hrothgar", "Hrathgar", "Hrethgar")  # surface, stored, stored
_ZH_TIE = ("缪芸", "缪荭", "缪苡")


def _chars(text: str) -> Tuple[str, ...]:
    out = []
    for ch in text:
        if ch.isascii() and ch.isalpha():
            out.append(ch.lower())
        elif "\u4e00" <= ch <= "\u9fff":
            out.append(ch)
    return tuple(out)


def _lev(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _dist(a: str, b: str) -> int:
    return _lev(_chars(a), _chars(b))


def _extension(span: str, name: str) -> bool:
    small, big = _chars(name), _chars(span)
    if not small or not big or len(big) <= len(small):
        return False
    i = 0
    for ch in big:
        if i < len(small) and small[i] == ch:
            i += 1
    return i == len(small)


def _far(cands: Sequence[str], *avoid: str) -> str:
    for name in cands:
        if all(_dist(name, other) > 1 for other in avoid):
            return name
    raise RuntimeError(f"no distractor far from {avoid}")


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


class _Build:
    def __init__(self) -> None:
        self.mark = 0
        self.val = 0
        self.rows: list[Episode] = []

    def line(self, body: str) -> str:
        self.mark += 1
        return f"barmkin oxter {body} thrid{self.mark:02d} 㐀㐁"

    def value(self) -> str:
        self.val += 1
        return f"qxv{self.val:04d}"

    def fact(
        self,
        name: str,
        rel: str,
        *,
        lang: str,
        first: bool = False,
        kiln: bool = False,
    ) -> Slot:
        value = self.value()
        if first and lang == "zh":
            body = f"我留了{_CUE_ZH[rel]}"
        elif first:
            body = f"I keep the {_CUE[rel]}"
        elif lang == "zh":
            body = f"页上留了{name}的{_CUE_ZH[rel]}"
        else:
            prefix = "kilnnote " if kiln else ""
            body = f"{prefix}slip keeps the {_CUE[rel]} under {name}"
        return Slot(name, rel, value, self.line(body))

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
        turns: Sequence[str] = (),
        prior_asks: Sequence[str] = (),
        who: str = "",
        target: str = "",
    ) -> None:
        self.rows.append(Episode(
            kind, lang, rel, label, empty_label, question,
            tuple(golds), tuple(empty_golds), tuple(slots),
            tuple(turns), tuple(prior_asks), who, target,
        ))


def _pair_bank(lang: str) -> Tuple[Tuple[str, str], ...]:
    return _EN_TYPO if lang == "en" else _ZH_TYPO


def _others(lang: str) -> Tuple[str, ...]:
    return _EN_OTHER if lang == "en" else _ZH_OTHER


def _cue(lang: str, rel: str) -> str:
    return _CUE[rel] if lang == "en" else _CUE_ZH[rel]


def _ask(lang: str, who: str, rel: str) -> str:
    cue = _cue(lang, rel)
    if lang == "en":
        return f"{cue} for {who}"
    return f"{who} {cue}"


def build_rows() -> Tuple[Episode, ...]:
    built = _Build()
    for canon, surface in _EN_TYPO + _ZH_TYPO:
        if _dist(canon, surface) != 1 or _extension(surface, canon):
            raise RuntimeError(f"{surface} is not a non-extension edit of {canon}")

    def typo_block(lang: str) -> None:
        pairs = _pair_bank(lang)
        pool = _others(lang)
        for i, (canon, surface) in enumerate(pairs):
            other = _far(pool, canon, surface)
            rel = "phone"
            target = built.fact(canon, rel, lang=lang)
            distractor = built.fact(other, rel, lang=lang)
            slots = (target, distractor)
            mention_target = built.line(f"spoke of {canon}" if lang == "en" else f"提到{canon}")
            mention_other = built.line(f"spoke of {other}" if lang == "en" else f"提到{other}")
            question = built.line(_ask(lang, surface, rel))
            built.add(
                "typo_hit_utterance", lang, rel, "answerable", "absent_person",
                question, (target.value,), (), slots,
                turns=(mention_target, mention_other), who=surface, target=canon,
            )
            silent_q = built.line(_ask(lang, surface, rel))
            built.add(
                "typo_abstain_silent", lang, rel, "absent_person", "absent_person",
                silent_q, (), (), slots, who=surface, target=canon,
            )
            other_q = built.line(_ask(lang, surface, rel))
            built.add(
                "typo_abstain_other", lang, rel, "absent_person", "absent_person",
                other_q, (), (), slots, turns=(mention_other,), who=surface, target=canon,
            )
            if i < 4:
                prior_rel = "birthday"
                prior_slot = built.fact(canon, prior_rel, lang=lang)
                phone = built.fact(canon, rel, lang=lang)
                other_phone = built.fact(other, rel, lang=lang)
                prior = built.line(_ask(lang, canon, prior_rel))
                final = built.line(_ask(lang, surface, rel))
                built.add(
                    "typo_hit_prior_ask", lang, rel, "answerable", "absent_person",
                    final, (phone.value,), (), (prior_slot, phone, other_phone),
                    prior_asks=(prior,), who=surface, target=canon,
                )
            if i < 2:
                stale_slots = (
                    built.fact(canon, rel, lang=lang),
                    built.fact(other, rel, lang=lang),
                )
                first = built.line(f"spoke of {canon}" if lang == "en" else f"提到{canon}")
                fillers = tuple(
                    built.line(f"spoke of {other}" if lang == "en" else f"提到{other}")
                    for _ in range(4)
                )
                stale_q = built.line(_ask(lang, surface, rel))
                built.add(
                    "typo_abstain_stale", lang, rel, "absent_person", "absent_person",
                    stale_q, (), (), stale_slots, turns=(first, *fillers),
                    who=surface, target=canon,
                )

    typo_block("en")
    typo_block("zh")

    # Two stored names, both edit-1 of the surface, both mentioned.
    for lang, surface, left, right in (
        ("en", *_EN_TIE),
        ("zh", *_ZH_TIE),
    ):
        if _dist(surface, left) != 1 or _dist(surface, right) != 1:
            raise RuntimeError(f"tie {surface} {left} {right}")
        rel = "phone"
        a = built.fact(left, rel, lang=lang)
        b = built.fact(right, rel, lang=lang)
        if lang == "en":
            turn = built.line(f"spoke of {left} and {right}")
        else:
            turn = built.line(f"提到{left} 和 {right}")
        built.add(
            "typo_abstain_tie", lang, rel, "absent_person", "absent_person",
            built.line(_ask(lang, surface, rel)), (), (), (a, b),
            turns=(turn,), who=surface, target="",
        )

    # Exact spelling. The other person is the one the dialogue just named.
    spell_rels = ("phone", "birthday", "places", "phone")
    for lang, pairs in (("en", _EN_TYPO[3:7]), ("zh", _ZH_TYPO[3:7])):
        pool = _others(lang)
        for (canon, _surface), rel in zip(pairs, spell_rels):
            other = _far(pool, canon)
            target = built.fact(canon, rel, lang=lang)
            distractor = built.fact(other, rel, lang=lang)
            turn = built.line(f"spoke of {other}" if lang == "en" else f"提到{other}")
            built.add(
                "spelling_hit", lang, rel, "answerable", "answerable",
                built.line(_ask(lang, canon, rel)), (target.value,), (target.value,),
                (target, distractor), turns=(turn,), who=canon, target=canon,
            )

    for full, dimin in _EN_NICK:
        rel = "phone"
        other = _far(_EN_OTHER, full, dimin)
        target = built.fact(full, rel, lang="en")
        distractor = built.fact(other, rel, lang="en")
        built.add(
            "nickname_hit", "en", rel, "answerable", "answerable",
            built.line(f"telephone {dimin}"), (target.value,), (target.value,),
            (target, distractor), who=dimin, target=full,
        )
    for full, dimin in _ZH_NICK:
        rel = "phone"
        other = _far(_ZH_OTHER, full, dimin)
        target = built.fact(full, rel, lang="zh")
        distractor = built.fact(other, rel, lang="zh")
        built.add(
            "nickname_hit", "zh", rel, "answerable", "answerable",
            built.line(f"{dimin} 电话"), (target.value,), (target.value,),
            (target, distractor), who=dimin, target=full,
        )

    for canon, longer in _EN_EXT:
        rel = "birthday"
        other = _far(_EN_OTHER, canon, longer)
        target = built.fact(canon, rel, lang="en")
        distractor = built.fact(other, rel, lang="en")
        turn = built.line(f"spoke of {canon}")
        built.add(
            "latin_extension", "en", rel, "absent_person", "absent_person",
            built.line(f"birthday for {longer}"), (), (), (target, distractor),
            turns=(turn,), who=longer, target=canon,
        )
    for canon, longer in _ZH_SUPER:
        rel = "phone"
        other = _far(_ZH_OTHER, canon, longer)
        target = built.fact(canon, rel, lang="zh")
        distractor = built.fact(other, rel, lang="zh")
        turn = built.line(f"提到{canon}")
        built.add(
            "cjk_superstring", "zh", rel, "absent_person", "absent_person",
            built.line(f"{longer} 电话"), (), (), (target, distractor),
            turns=(turn,), who=longer, target=canon,
        )
    for canon, longer in _ZH_PREFIX:
        rel = "birthday"
        other = _far(_ZH_OTHER, canon, longer)
        target = built.fact(canon, rel, lang="zh")
        distractor = built.fact(other, rel, lang="zh")
        turn = built.line(f"提到{canon}")
        built.add(
            "cjk_prefix", "zh", rel, "absent_person", "absent_person",
            built.line(f"{longer} 生日"), (), (), (target, distractor),
            turns=(turn,), who=longer, target=canon,
        )

    def two_name(lang: str, kind: str, rel: str, focus: str, other: str) -> None:
        focus_slot = built.fact(focus, rel, lang=lang, kiln=False)
        other_slot = built.fact(other, rel, lang=lang, kiln=True)
        if kind == "focus_hit":
            if lang == "en":
                body = f"kilnnote between {other} and {focus} ask only about {focus} {_CUE[rel]}"
            else:
                body = f"只问 {focus} {_CUE_ZH[rel]} {other}"
        elif kind == "dismiss_hit":
            if lang == "en":
                body = f"kilnnote skip {other}. {_CUE[rel]} {focus}"
            else:
                body = f"跳过 {other} 。 {_CUE_ZH[rel]} {focus}"
        elif kind == "open_either":
            if lang == "en":
                body = f"{focus} and {other} {_CUE[rel]}"
            else:
                body = f"{focus} 和 {other} {_CUE_ZH[rel]}"
        else:
            raise RuntimeError(kind)
        golds: Tuple[str, ...]
        if kind == "open_either":
            golds = (focus_slot.value, other_slot.value)
        else:
            golds = (focus_slot.value,)
        mention = built.line(f"spoke of {other}" if lang == "en" else f"提到{other}")
        built.add(
            kind, lang, rel, "answerable", "answerable",
            built.line(body), golds, golds, (focus_slot, other_slot),
            turns=(mention,), who=focus, target=focus,
        )

    two_name("en", "focus_hit", "phone", "Swithun", "Wulfrun")
    two_name("en", "focus_hit", "birthday", "Eadburg", "Cyneburg")
    two_name("zh", "focus_hit", "phone", "储晏", "宣蘅")
    two_name("zh", "focus_hit", "birthday", "毕芩", "杭萱")
    two_name("en", "dismiss_hit", "phone", "Cuthbert", "Eormengyth")
    two_name("en", "dismiss_hit", "places", "Frideswide", "Saethryth")
    two_name("zh", "dismiss_hit", "phone", "伍蘅", "包菁")
    two_name("zh", "dismiss_hit", "places", "屠芃", "左荃")
    two_name("en", "open_either", "phone", "Beornwyn", "Wulfrun")
    two_name("en", "open_either", "birthday", "Mildthryth", "Cyneburg")
    two_name("zh", "open_either", "phone", "柯菱", "宣蘅")
    two_name("zh", "open_either", "birthday", "管菁", "杭萱")

    def indirect(lang: str, kind: str) -> None:
        if lang == "en":
            self_name, other_name = "Osgyth", "Saethryth"
            q_hit = "the listener telephone"
            q_miss = "the listener telephone"
        else:
            self_name, other_name = "莫荃", "杭萱"
            q_hit = "身旁这个人 电话"
            q_miss = "身旁这个人 电话"
        if kind == "indirect_hit":
            self_slot = built.fact(self_name, "phone", lang=lang, first=True)
            other_slot = built.fact(other_name, "phone", lang=lang)
            built.add(
                kind, lang, "phone", "answerable", "answerable",
                built.line(q_hit), (self_slot.value,), (self_slot.value,),
                (self_slot, other_slot), who="", target=self_name,
            )
        elif kind == "indirect_noself":
            a = built.fact(self_name, "phone", lang=lang)
            b = built.fact(other_name, "phone", lang=lang)
            built.add(
                kind, lang, "phone", "absent_person", "absent_person",
                built.line(q_hit), (), (), (a, b), who="", target="",
            )
        elif kind == "indirect_missing":
            self_slot = built.fact(self_name, "birthday", lang=lang, first=True)
            other_slot = built.fact(other_name, "phone", lang=lang)
            built.add(
                kind, lang, "phone", "missing_rel", "missing_rel",
                built.line(q_miss), (), (), (self_slot, other_slot),
                who="", target=self_name,
            )
        else:
            raise RuntimeError(kind)

    for lang in ("en", "zh"):
        indirect(lang, "indirect_hit")
        indirect(lang, "indirect_noself")
        indirect(lang, "indirect_missing")

    rows = tuple(built.rows)
    _check(rows)
    return rows


def _check(rows: Sequence[Episode]) -> None:
    if not 80 <= len(rows) <= 120:
        raise RuntimeError(f"dialogue dev bank has {len(rows)} episodes")
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate dialogue question")
    strings = bank_strings(rows)
    if len(strings) != len(set(strings)):
        raise RuntimeError("duplicate dialogue bank string")
    hit = {row.who for row in rows if row.kind == "typo_hit_utterance"}
    abstain = {
        row.who for row in rows
        if row.kind in {"typo_abstain_silent", "typo_abstain_other"}
    }
    if not hit or not hit <= abstain:
        raise RuntimeError("typo surfaces are not paired")
    for row in rows:
        if row.label == "answerable" and not row.golds:
            raise RuntimeError(f"{row.kind} has no gold")
        if row.label != "answerable" and row.golds:
            raise RuntimeError(f"{row.kind} abstains with a gold")
        for slot in row.slots:
            if slot.value in slot.text or slot.value in row.question:
                raise RuntimeError("value leaked into a string")
            if "\u6211" in slot.text and not slot.text.startswith("barmkin oxter \u6211"):
                # 我 only as the first-person template, not inside another fact.
                if "我留了" not in slot.text:
                    raise RuntimeError(f"unexpected 我 in {slot.text}")


def questions(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def bank_strings(rows: Optional[Sequence[Episode]] = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        texts = [row.question, *row.turns, *row.prior_asks]
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


def kind_counts(rows: Optional[Sequence[Episode]] = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts


def label_counts(rows: Optional[Sequence[Episode]] = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return counts
