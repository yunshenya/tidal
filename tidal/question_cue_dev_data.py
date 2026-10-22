"""Dev split for a misleading cue inside a dismissed clause.

A unique lexical cue that sits in a dismissed clause is not the ask. The
clause is a closed class (ignore / skip / forget / exclude, aside, a decoy
noun, and the Chinese marks), and it ends at the next clause break when
another clause still follows. A cue after that break is the ask. A container
or a polysemous ``call`` / ``locations file`` / ``打电话`` is not a dismissed
clause.

Names and sentences are disjoint from the probe, the surface cue corpus, the
abstain dev bank, the CJK dev bank, the nickname dev bank, the indirect dev
bank, the typo dev bank, and the frozen adversarial test. The matcher does
not import this module. The frozen test is not an input to the class.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, CJK, nickname, indirect, or
# adversarial rosters.
NAMES = {"en": ("Eudora", "Barnaby", "Winslow", "Petronella"), "zh": ("闵舟", "濮兰", "隗衡", "郏棠")}

# kind, lang, cue, question. ``set_aside`` is the misleading cue. The other
# kinds keep the cue: it is after the break, it is the ask, or it is only
# polysemous. Anchors (``quodlibet``, ``龘鬻``) are not people.
_ROWS: Tuple[Tuple[str, str, str, str], ...] = (
    ("set_aside", "en", "birthday", "Forget Eudora's candle tally and say how a voice lands after dusk, quodlibet."),
    ("set_aside", "en", "birthday", "Skip Barnaby's natal slip and keep the spoken trail of dens, narthex."),
    ("set_aside", "en", "phone", "Ignore the handset chit for Winslow and read which morning they grow older, ciborium."),
    ("set_aside", "en", "places", "Exclude the rendezvous card and say how Petronella is reached by voice, reredos."),
    ("set_aside", "en", "birthday", "Birthday slip for Eudora goes aside, then say how a voice finds her, amice."),
    ("set_aside", "en", "places", "That locations chit is a distraction, so give Barnaby's morning count, maniple."),
    ("set_aside", "en", "phone", "The telephone slip is the wrong drawer: which ordered halts did Winslow dictate, orphrey."),
    ("set_aside", "en", "birthday", "Ignoring the candle line, say how Petronella is reached after dusk, pallium."),
    ("set_aside", "en", "birthday", "Forgetting Eudora's natal card and keep the spoken dens in order, feretory."),
    ("scoped_after", "en", "phone", "Ignore the side chatter and read Eudora's telephone, ostensory."),
    ("scoped_after", "en", "birthday", "Skipping the filler notes, which birthday does Barnaby claim, pyx."),
    ("scoped_after", "en", "places", "Other scraps go aside. Winslow's places are the ones to recite, monstrance."),
    ("scoped_after", "en", "phone", "Park Barnaby aside. Eudora's telephone is the line to read, thurible."),
    ("scoped_after", "en", "phone", "Forget the feast. Which handset reaches Petronella, censer."),
    ("scoped_after", "en", "phone", "The extra scraps are a distraction: what telephone did Eudora leave, aspergillum."),
    ("scoped_after", "en", "phone", "Excluding the side chatter, which telephone reaches Eudora, sudarium."),
    ("genuine", "en", "phone", "Which telephone does the vestry book keep for Eudora, navicula."),
    ("genuine", "en", "birthday", "Barnaby's birthday on the vestry card is which date, chrismatory."),
    ("genuine", "en", "places", "Winslow's places stay in the spoken queue, piscina."),
    ("polysemy", "en", "places", "Eudora's voice path sits in a locations file, baldachin."),
    ("polysemy", "en", "phone", "At the parish bell, folks call round because Barnaby grows older, antependium."),
    ("polysemy", "en", "phone", "The lane will call Winslow's dens a circuit, vexillum."),
    ("set_aside", "zh", "birthday", "闵舟那张蜡烛页先跳过，夜里怎样用声音找人，龘鬻。"),
    ("set_aside", "zh", "phone", "先合上濮兰的号码夹，周岁落在哪一页，靐夔。"),
    ("set_aside", "zh", "places", "隗衡的地点清册一边去，声音该落在哪一条，飝鼽。"),
    ("set_aside", "zh", "places", "郏棠的交接单先放下，满岁写在哪一页，麤矍。"),
    ("set_aside", "zh", "birthday", "闵舟的蜡烛册以后再查，入夜后把声音送去哪一条，鬻纛。"),
    ("set_aside", "zh", "phone", "濮兰的手机夹先收到柜里，落脚顺序怎么念，夔龘。"),
    ("set_aside", "zh", "birthday", "别提闵舟那张蜡烛页，夜里按什么次序找人落脚，靐鬻。"),
    ("set_aside", "zh", "birthday", "濮兰的出生日先放一边，夜里声音往哪一条送，鼽麤。"),
    ("set_aside", "zh", "phone", "别管濮兰的手机纸条，满岁落在哪一页，纛龘。"),
    ("set_aside", "zh", "phone", "先不看郏棠的号码条，周岁记在哪一格，鼽夔。"),
    ("set_aside", "zh", "birthday", "别查闵舟的蜡烛页，夜里声音往哪送，靐鼽。"),
    ("scoped_after", "zh", "phone", "跳过那些闲篇，濮兰的手机留在哪一条，矍纛。"),
    ("scoped_after", "zh", "birthday", "别管闲谈。郏棠认的出生日在哪一页，龘靐。"),
    ("scoped_after", "zh", "places", "先放下别人。闵舟的地点该依口述念，夔飝。"),
    ("scoped_after", "zh", "phone", "先别看闲聊。隗衡那串电话留在柜侧，鼽鬻。"),
    ("scoped_after", "zh", "phone", "闲话以后再说，闵舟的号码怎么留，麤鬻。"),
    ("genuine", "zh", "phone", "柜侧藤匣收着闵舟的手机吗，麤龘。"),
    ("genuine", "zh", "birthday", "濮兰落地的出生页夹在柜侧，矍靐。"),
    ("genuine", "zh", "places", "隗衡把地点留在藤匣里了吗，鬻飝。"),
    ("polysemy", "zh", "places", "哪一份地点档案能用声音找到闵舟，夔鼽。"),
    ("polysemy", "zh", "phone", "邻居把上门贺岁说成是给濮兰打电话，落在哪一天，靐麤。"),
    ("polysemy", "zh", "places", "封面写着地方两个字，问的是闵舟哪天满岁，飝矍。"),
)


@dataclass(frozen=True)
class CueRow:
    kind: str
    lang: str
    cue: str
    question: str


def build_rows() -> Tuple[CueRow, ...]:
    rows = tuple(CueRow(*row) for row in _ROWS)
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate cue-dev question")
    return rows


def questions(rows: Sequence[CueRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[CueRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[CueRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
