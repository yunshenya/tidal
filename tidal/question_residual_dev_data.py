"""Dev split for residual misleading cues after a dismissed clause.

A unique lexical cue still decides abstain when the abstain ridge agrees
with it, or disagrees by less than ``LEX_RIDGE_GAP``. Three other rows
already drop that cue and then miss on the ordinary 0.1 margin, or answer
a stored relation the ridge named in place of the ask. This bank is the
measurement set for those shapes. It is not a template of the frozen
sentences.

Kinds:

- ``container_ans`` / ``container_miss``: a places cue is only a record
  compound (``locations folder``, ``地点卷宗``). The ask is phone or
  birthday, with no cue of its own. Answerable stores only the ask.
  Missing stores the other two, including the compound's relation.
- ``keep_ans`` / ``keep_miss``: the same compounds are the places ask.
- ``manner_ans`` / ``manner_miss``: a places ask that also says ``voice``
  / ``声音`` as how to read it.
- ``novoice_ans`` / ``novoice_miss``: phone ask inside a record compound,
  with no ``voice`` / ``声音``.
- ``aside_ans`` / ``aside_miss``: the unique cue sits in a dismissed
  clause. The ask is the next clause and has no cue word.
- ``genuine_ans`` / ``genuine_miss``: the unique cue is the ask.
- ``outer_ans``: a dismiss word is present and the cue is after the break.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, the CJK dev bank, the nickname dev bank, the indirect
dev bank, the typo dev bank, the misleading-cue dev bank, the bare-locative
dev bank, and the frozen adversarial test. The matcher does not import
this module. The frozen test is not an input.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, CJK, nickname, indirect, cue,
# bare-locative, or adversarial rosters.
NAMES = {"en": ("Godfrey", "Linnea", "Hadrian", "Odette"), "zh": ("宓珩", "邹岚", "庞屹", "岑书")}

# kind, lang, rel, family, question. Anchors (``futtock``, ``齮齯``) are
# not people. ``rel`` is the asked relation.
_ROWS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("container_ans", "en", "phone", "voice", "Godfrey's voice route sits inside a locations folder at the west loft, futtock carling."),
    ("container_ans", "en", "phone", "voice", "A locations ledger in the chart room is where Linnea is reached by voice, kneetimber hawsehole."),
    ("container_ans", "en", "phone", "voice", "Hadrian is reached by voice from the locations slip on the second shelf, garboard deadwood."),
    ("container_ans", "en", "phone", "voice", "The locations chit that lets a voice find Odette is the one the clerk keeps, breasthook knighthead."),
    ("container_ans", "en", "phone", "voice", "Inside the locations dossier, the voice path for Godfrey is the one to read, limberhole maststep."),
    ("container_ans", "en", "phone", "voice", "Linnea's voice path was copied onto a locations binder, kelson stemson."),
    ("container_ans", "en", "birthday", "year", "The morning Hadrian grows a year older is written in a locations folio, treenail rudderpost."),
    ("container_ans", "en", "birthday", "year", "Odette adds a year on the date a locations logbook still holds, gudgeon pintle."),
    ("container_ans", "zh", "phone", "voice", "宓珩的声音路径夹在地点卷宗里，齮齯。"),
    ("container_ans", "zh", "phone", "voice", "地点档案里记着怎么用声音找到邹岚，齰齱。"),
    ("container_ans", "zh", "phone", "voice", "庞屹入夜后的声音落在哪一本地方簿册，齲齳。"),
    ("container_ans", "zh", "phone", "voice", "地方案卷里留着岑书的声音去向，齴齵。"),
    ("container_ans", "zh", "birthday", "year", "宓珩满岁的那天写在地点档册上，齶齷。"),
    ("container_ans", "zh", "birthday", "year", "哪一本地方卷夹记着邹岚长一岁的日子，齸齹。"),
    ("container_ans", "zh", "birthday", "year", "庞屹添岁的页在地点册页里，齺齻。"),
    ("container_ans", "zh", "birthday", "year", "岑书又大一岁的日子收在地方卷宗，齼齽。"),
    ("container_miss", "en", "phone", "voice", "Which locations folder in the west loft holds the voice route for Godfrey, fashionpiece cantframe."),
    ("container_miss", "en", "phone", "voice", "Linnea's voice route, the one a locations slip still carries, is which, hawsepipe countertimber."),
    ("container_miss", "en", "phone", "voice", "A locations binder carries how Odette is reached by voice, orlop becket."),
    ("container_miss", "en", "birthday", "year", "The locations logbook that names the day Hadrian grows a year older is which, thole davit."),
    ("container_miss", "en", "birthday", "year", "Godfrey's added year is the date kept in a locations dossier, binnacle kedge."),
    ("container_miss", "en", "phone", "voice", "From a locations chit, a voice finds Linnea after dusk, marline ratline."),
    ("container_miss", "zh", "phone", "voice", "宓珩的声音去向收在地点卷宗的哪一页，齮齰。"),
    ("container_miss", "zh", "phone", "voice", "邹岚的声音去向抄在地方档案里了吗，齱齲。"),
    ("container_miss", "zh", "phone", "voice", "用声音找庞屹时，要翻的是哪本地方案卷，齳齴。"),
    ("container_miss", "zh", "birthday", "year", "岑书满岁写在哪一本地点簿册，齵齶。"),
    ("container_miss", "zh", "birthday", "year", "宓珩长一岁的日子压在地方档册下，齷齸。"),
    ("container_miss", "zh", "phone", "voice", "地点卷夹里是邹岚入夜后的声音路径，齹齺。"),
    ("keep_ans", "en", "places", "keep", "Read the spoken halts filed in Godfrey's locations folder, first halt onward, sheave whisker."),
    ("keep_ans", "en", "places", "keep", "Linnea's locations ledger keeps the halts in the order they were spoken, yardarm bobstay."),
    ("keep_ans", "en", "places", "keep", "Which locations slip lists Hadrian's halts without reordering, bumpkin cathead."),
    ("keep_ans", "en", "places", "keep", "The locations dossier for Odette is a sequence of halts, chainplate deadeye."),
    ("keep_ans", "zh", "places", "keep", "宓珩的地点卷宗按口述先后有哪些，齻齼。"),
    ("keep_ans", "zh", "places", "keep", "邹岚的地点档案里，落脚顺序怎么念，齽龕。"),
    ("keep_ans", "zh", "places", "keep", "庞屹的地方簿册从第一处到最后一处是什么，龢龣。"),
    ("keep_ans", "zh", "places", "keep", "岑书的地方案卷依口述留下了哪些落脚，龤龥。"),
    ("keep_miss", "en", "places", "keep", "Godfrey's locations folio, the spoken halts in order, are which, footrope gammoning."),
    ("keep_miss", "en", "places", "keep", "Which locations logbook holds Linnea's halts from first to last, jackstay kevel."),
    ("keep_miss", "en", "places", "keep", "The halts Hadrian dictated sit in a locations binder, manrope parrel."),
    ("keep_miss", "en", "places", "keep", "Odette's locations chit is the spoken sequence of halts, snatchblock traveller."),
    ("keep_miss", "zh", "places", "keep", "哪一本地点档册写着宓珩的落脚先后，齮齱。"),
    ("keep_miss", "zh", "places", "keep", "邹岚歇脚的站序抄进地方卷夹，口述时怎么排，齰齲。"),
    ("keep_miss", "zh", "places", "keep", "庞屹的地点册页要按口述念完，有哪些，齳齵。"),
    ("keep_miss", "zh", "places", "keep", "地方卷宗上岑书的落脚顺序是什么，齴齶。"),
    ("manner_ans", "en", "places", "manner", "Read Godfrey's locations folder aloud by voice, first halt onward, lanyard topsail."),
    ("manner_ans", "en", "places", "manner", "By voice, recite the halts Linnea's locations ledger holds, spritsail lateen."),
    ("manner_ans", "zh", "places", "manner", "用声音把宓珩的地点卷宗按先后念出来，齮齴。"),
    ("manner_ans", "zh", "places", "manner", "邹岚的地点档案，用声音依口述念落脚，齯齵。"),
    ("manner_miss", "en", "places", "manner", "The locations slip of halts for Hadrian is to be read by voice, mizzen bowsprit."),
    ("manner_miss", "en", "places", "manner", "Recite by voice the halts kept in Odette's locations dossier, jibboom martingale."),
    ("manner_miss", "zh", "places", "manner", "庞屹的地方簿册用声音按落脚先后念，齰齳。"),
    ("manner_miss", "zh", "places", "manner", "声音只用来念岑书地点案卷里的落脚顺序，齱齴。"),
    ("novoice_ans", "en", "phone", "novoice", "The way to get through to Godfrey is the entry a locations folder still holds, gammon cranse."),
    ("novoice_ans", "en", "phone", "novoice", "How Linnea is reached after dusk is written on a locations slip, catharpin futtockplate."),
    ("novoice_ans", "zh", "phone", "novoice", "地点卷宗里写着怎么联络上宓珩，齵齷。"),
    ("novoice_ans", "zh", "phone", "novoice", "哪一本地方簿册记着庞屹入夜后怎么找得着，齸齹。"),
    ("novoice_miss", "en", "phone", "novoice", "A locations ledger holds how Hadrian is reached once the loft is quiet, seizing whipping."),
    ("novoice_miss", "en", "phone", "novoice", "The entry for reaching Odette after dusk sits in a locations chit, fidspike marlinespike."),
    ("novoice_miss", "zh", "phone", "novoice", "怎么联络上邹岚，写在地点档案的哪一页，齺齻。"),
    ("novoice_miss", "zh", "phone", "novoice", "岑书入夜后怎么找得着，地方案卷里有没有写，齼齽。"),
    ("aside_ans", "en", "phone", "aside", "Ignore the birthday supper and leave it; how is Godfrey reached after dusk, servingboard twine."),
    ("aside_ans", "en", "places", "aside", "Skip the candle line for Linnea; keep the spoken halts in order, beeswax nettles."),
    ("aside_ans", "en", "birthday", "aside", "Forget the rendezvous card, then the morning Hadrian grows a year older, foxes sennit."),
    ("aside_ans", "en", "places", "aside", "Exclude the handset chit; the halts for Odette stay in spoken order, grommet cringle."),
    ("aside_ans", "zh", "phone", "aside", "别管宓珩的蜡烛页，入夜后怎么找得着，齮齹。"),
    ("aside_ans", "zh", "birthday", "aside", "先合上号码夹子。邹岚满岁记在第几格，齯齺。"),
    ("aside_ans", "zh", "phone", "aside", "地点清册一边去，庞屹入夜后该把声音送去哪一条，齰齻。"),
    ("aside_ans", "zh", "places", "aside", "先放下岑书的电话条，落脚按什么先后，齱齼。"),
    ("aside_miss", "en", "phone", "aside", "The birthday board goes aside. A voice is how Godfrey gets found after dusk, reefpoint earring."),
    ("aside_miss", "en", "places", "aside", "Ignoring the candle tally, keep Linnea's spoken halts as given, clewgarnet buntline."),
    ("aside_miss", "en", "birthday", "aside", "The rendezvous card for Hadrian is a distraction: which morning does the year get added, leechline bowline."),
    ("aside_miss", "en", "birthday", "aside", "That handset note is the wrong drawer. Read the morning Odette grows a year older, vangpeak peakhalyard."),
    ("aside_miss", "zh", "phone", "aside", "别提蜡烛页。夜里把声音送到宓珩的哪一条，齵齸。"),
    ("aside_miss", "zh", "places", "aside", "生日册先放一边，邹岚的落脚顺序怎么念，齹齽。"),
    ("aside_miss", "zh", "birthday", "aside", "庞屹的号码条先收到柜里，满岁写在哪一页，齺龕。"),
    ("aside_miss", "zh", "phone", "aside", "跳过岑书的地点页，入夜后怎么找得着，龢齮。"),
    ("genuine_miss", "en", "phone", "genuine", "Which telephone does the loft keep for Godfrey, halyard futtock."),
    ("genuine_miss", "en", "birthday", "genuine", "On the loft card, Linnea's birthday is the date to read, carling kneetimber."),
    ("genuine_miss", "en", "places", "genuine", "The spoken queue still has Hadrian's places, hawsehole garboard."),
    ("genuine_miss", "en", "phone", "genuine", "Which handset reaches Odette after dusk, deadwood breasthook."),
    ("genuine_miss", "zh", "phone", "genuine", "藤匣侧格有没有夹着宓珩的手机，龣龤。"),
    ("genuine_miss", "zh", "birthday", "genuine", "邹岚的出生日写在哪一页，龥齯。"),
    ("genuine_miss", "zh", "places", "genuine", "藤匣里还留着庞屹的地点没有，齰齱。"),
    ("genuine_miss", "zh", "phone", "genuine", "岑书的号码留在哪一条，齲齳。"),
    ("genuine_ans", "en", "phone", "genuine", "The loft telephone for Godfrey is which string, knighthead limberhole."),
    ("genuine_ans", "en", "birthday", "genuine", "Which birthday does Linnea claim on the loft card, maststep kelson."),
    ("genuine_ans", "en", "places", "genuine", "Hadrian's places on the loft card stay in spoken order, stemson treenail."),
    ("genuine_ans", "en", "phone", "genuine", "The handset that reaches Odette from the loft is which, rudderpost gudgeon."),
    ("genuine_ans", "zh", "phone", "genuine", "宓珩的手机留在柜侧哪一条，齴齵。"),
    ("genuine_ans", "zh", "birthday", "genuine", "柜侧夹着邹岚落地那天的出生页，齶齷。"),
    ("genuine_ans", "zh", "places", "genuine", "庞屹的地点留在藤匣里，先后怎么念，齸齹。"),
    ("genuine_ans", "zh", "birthday", "genuine", "岑书认的出生日在哪一页，齺齻。"),
    ("outer_ans", "en", "phone", "outer", "Ignore the side chatter. Godfrey's telephone is the line the loft should read, pintle fashionpiece."),
    ("outer_ans", "en", "birthday", "outer", "Filler notes are for skipping. Linnea's birthday is the claim to read, cantframe hawsepipe."),
    ("outer_ans", "en", "places", "outer", "Park every scrap aside. Hadrian's places want a reading, countertimber orlop."),
    ("outer_ans", "en", "phone", "outer", "Forget that feast. Odette is reached on which handset, becket thole."),
    ("outer_ans", "zh", "phone", "outer", "闲篇跳过就算。邹岚的手机夹在柜侧哪一格，齼齽。"),
    ("outer_ans", "zh", "birthday", "outer", "闲谈别管。出生日要看庞屹的那一页，龕龢。"),
    ("outer_ans", "zh", "places", "outer", "别人的条子先放下。宓珩的地点依口述有哪些，龣龤。"),
    ("outer_ans", "zh", "phone", "outer", "闲话留到以后再说。岑书的号码夹在哪一格，龥齮。"),
)


@dataclass(frozen=True)
class ResidualRow:
    kind: str
    lang: str
    rel: str
    family: str
    question: str


def build_rows() -> Tuple[ResidualRow, ...]:
    rows = tuple(ResidualRow(*row) for row in _ROWS)
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate residual-dev question")
    return rows


def questions(rows: Sequence[ResidualRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[ResidualRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[ResidualRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
