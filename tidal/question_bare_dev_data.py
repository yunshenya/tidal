"""Dev split for a bare locative that is not a places ask.

``place`` / ``where`` / ``哪里`` / ``位置`` are not lexical cues, so a
question can mention a placement slot and still have no unique cue. The
abstain ridge may read that slot as places. This bank is the only split
used to choose whether a cue-free placement slot should drop that label.

Four kinds:

- ``answerable``: cue-free placement or underspecified slot whose gold
  relation is phone or birthday. The harsh store keeps only that relation.
  A places label must not false-abstain.
- ``missing_rel``: the same class of question. The store keeps the other
  two relations. A confident label that is not stored must abstain.
- ``keep_places``: a places ask. Bare ``where`` / ``哪里``, ``where does
  X live``, ``place list``, ``位置是哪些``, and a places cue. The places
  label stays. A few rows also put the person in the placement slot
  (``place Sibyl``, ``where was Leander placed``, ``把卞珩放在哪里``).
  Those must not drop places. Wider verbs (``put``, ``file``, ``deposit``,
  ``搁在``, ``摆在``) are in the bank so a broader class can be measured.
  They are not the shipped pattern: ``file`` is a noun in ``on file``.
- ``guarded``: the placement slot is real and so is one cue (``place a
  call``, ``交接放在哪里``). The cue stays the ask.

Families wider than ``place`` / ``放在哪里`` (``put``, ``搁在``, a drawer
with no verb) are in the bank on purpose. They are not a promise that the
shipped pattern matches them.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, the CJK dev bank, the nickname dev bank, the indirect
dev bank, the typo dev bank, the misleading-cue dev bank, and the frozen
adversarial test. The matcher does not import this module. The frozen test
is not an input to the class.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Not people on the probe, surface, abstain, CJK, nickname, indirect, cue,
# or adversarial rosters.
NAMES = {"en": ("Sibyl", "Leander", "Quince", "Romilda"), "zh": ("卞珩", "汲棠", "郗宁", "晏书")}

# kind, lang, rel, family, question. Anchors (``quillon``, ``鼉鼐``) are
# not people. ``answerable`` and ``missing_rel`` have no lexical cue.
_ROWS: Tuple[Tuple[str, str, str, str, str], ...] = (
    ("answerable", "en", "birthday", "placed", "Sibyl's added-year tick stays where the clerk has it placed, quillon bascinet."),
    ("answerable", "en", "birthday", "placing", "Placing Leander's added-year tick is a matter of which square, gambeson bevor."),
    ("answerable", "en", "birthday", "place_verb", "I place Quince's added-year tick on which square of the slate, pauldron vambrace."),
    ("answerable", "en", "birthday", "put", "Romilda's added-year tick, where ought the clerk put it, rerebrace cuisse."),
    ("answerable", "en", "birthday", "lodged", "The added-year tick for Sibyl is lodged in which square, poleyn sabaton."),
    ("answerable", "en", "birthday", "tucked", "Leander's added-year tick gets tucked into which square, tasset fauld."),
    ("answerable", "en", "birthday", "slots", "Which square slots Quince's added-year tick, gorget ailette."),
    ("answerable", "en", "birthday", "pinned", "Romilda's added-year tick is pinned to which square, couter besagew."),
    ("answerable", "en", "phone", "placed", "A voice finds Sibyl once the chit has been placed: name the pigeonhole, spaulder plackart."),
    ("answerable", "en", "phone", "placed", "The chit, placed so a voice finds Leander, sits in which pigeonhole, rondel ventail."),
    ("answerable", "en", "phone", "putting", "Putting the chit where a voice finds Quince leaves which pigeonhole, camail gardbrace."),
    ("answerable", "en", "phone", "file", "Which pigeonhole should hold the chit I file so a voice finds Romilda, lames pizaine."),
    ("answerable", "en", "phone", "stow", "Stow the chit so a voice finds Sibyl, and say which pigeonhole, brassart cowter."),
    ("answerable", "en", "phone", "deposit", "Deposit the chit for a voice to find Leander in which pigeonhole, demigauntlet gauntlet."),
    ("answerable", "en", "phone", "jot", "Jot the chit a voice uses for Quince into which pigeonhole, morion burgonet."),
    ("answerable", "en", "birthday", "entered", "Enter Romilda's added-year tick in which square, sallet armet."),
    ("missing_rel", "en", "birthday", "put", "Where is Sibyl's added-year seal put on the slate, barbute cabasset."),
    ("missing_rel", "en", "birthday", "lay", "Lay Leander's added-year seal on which slate square, cervelliere kettlehat."),
    ("missing_rel", "en", "birthday", "situate", "Situate Quince's added-year seal in which square, spangenhelm nasal."),
    ("missing_rel", "en", "birthday", "deposited", "The added-year seal for Romilda is deposited where, coif haubergeon."),
    ("missing_rel", "en", "birthday", "no_verb", "Which square receives the inscribed added-year seal for Sibyl, byrnie chausses."),
    ("missing_rel", "en", "phone", "lodge", "Lodge the voice-chit for Leander in which pigeonhole, solleret poleaxe."),
    ("missing_rel", "en", "phone", "tuck", "Tuck Quince's voice-chit into which pigeonhole, bardiche glaive."),
    ("missing_rel", "en", "phone", "park", "Park Romilda's voice-chit in which pigeonhole, voulge guisarme."),
    ("missing_rel", "en", "phone", "slide", "Slide Sibyl's voice-chit into which pigeonhole, halberd ranseur."),
    ("missing_rel", "en", "phone", "stick", "Stick Leander's voice-chit on which pigeonhole, spontoon partizan."),
    ("missing_rel", "en", "birthday", "be_placed", "The added-year seal for Quince should be placed on which slate square, corseque brandistock."),
    ("missing_rel", "en", "phone", "place_chit", "Which pigeonhole receives Romilda's chit after I place it for a voice, estoc baselard."),
    ("missing_rel", "en", "birthday", "drawer", "Which drawer takes Sibyl's added-year seal, cinquedea falchion."),
    ("missing_rel", "en", "phone", "folio", "Which folio takes the voice-chit for Leander, messer seax."),
    ("keep_places", "en", "places", "dwell", "Where does Sibyl dwell when the parish asks, scramasax langseax."),
    ("keep_places", "en", "places", "stay", "Where does Leander stay once dusk settles, anelace misericorde."),
    ("keep_places", "en", "places", "runner", "Where ought Quince's runner to wait, first halt then the next, quokka bandicoot."),
    ("keep_places", "en", "places", "place_list", "Leander left a place list inside the tin, numbat bilby."),
    ("keep_places", "en", "places", "remind_where", "Remind the clerk where Romilda said the gathering sat, potoroo quoll."),
    ("keep_places", "en", "places", "idle", "Where does Sibyl pass the idle hour, thylacine echidna."),
    ("keep_places", "en", "places", "quarter", "In which quarter of the town is Leander usually found, platypus cassowary."),
    ("keep_places", "en", "places", "places_cue", "Which places does Quince keep on the tin card, takahe weka."),
    ("keep_places", "en", "places", "locations_cue", "Romilda's locations on the tin card stay in spoken order, kakapo huia."),
    ("keep_places", "en", "places", "person_place", "Where does the parish place Sibyl for the evening halt, kereru pukeko."),
    ("keep_places", "en", "places", "person_placed", "Where was Leander placed for the evening halt, kokako kea."),
    ("keep_places", "en", "places", "spots_cue", "The spots Quince named stay in the spoken order, whio moa."),
    ("keep_places", "en", "places", "bare_where", "Where, for Romilda, is the usual halt, tui kaka."),
    ("keep_places", "en", "places", "waypoints_cue", "Waypoints left for Sibyl stay unsorted on the tin card, okapi addax."),
    ("guarded", "en", "phone", "place_a_call", "Place a call for Sibyl and read the filed string aloud, oryx gerenuk."),
    ("guarded", "en", "phone", "voice_call", "After the quay grows quiet we place a voice call so Leander can be raised, and the filed string is which, kudu bongo."),
    ("guarded", "en", "places", "places_then_place", "Read the places Quince set down, then place the card back, eland sitatunga."),
    ("guarded", "en", "birthday", "candle_place", "Place the candle notice for Romilda and say which dawn it names, duiker dikdik."),
    ("guarded", "en", "phone", "placing_call", "Placing a call through to Sibyl, which filed string leaves the booth, nyala impala."),
    ("guarded", "en", "places", "handoff_placed", "Leander's handoffs stay listed, and where the card is placed after reading, springbok gazelle."),
    ("answerable", "zh", "birthday", "fang", "卞珩那枚岁印，册吏该把它放在哪里，鼉鼐鼒。"),
    ("answerable", "zh", "phone", "wang", "汲棠要靠嗓音找人，该往哪里把条子送去，鼖鼛鼜。"),
    ("answerable", "zh", "phone", "chao", "朝哪里把嗓音递给郗宁，青册记着哪一条，鼝鼞鼟。"),
    ("answerable", "zh", "phone", "weizhi_xian", "夜里先翻晏书的嗓音条，位置先搁下，柜侧看哪一页，鼣鼤鼥。"),
    ("answerable", "zh", "birthday", "ge", "卞珩的岁印该搁在哪一格，鼦鼧鼨。"),
    ("answerable", "zh", "birthday", "bai", "汲棠又添一岁，那方印该摆在哪一格，鼪鼫鼭。"),
    ("answerable", "zh", "phone", "tian", "郗宁的嗓音条该填在哪一格，鼮鼰鼱。"),
    ("answerable", "zh", "birthday", "ji", "晏书的岁印记在哪一页，鼲鼳鼵。"),
    ("answerable", "zh", "phone", "xie", "卞珩的嗓音条写在哪一格，鼶鼸鼺。"),
    ("answerable", "zh", "birthday", "sai", "汲棠的岁印塞到哪一格，鼉鼖鼝。"),
    ("answerable", "zh", "phone", "luo", "郗宁的嗓音条落到哪一格，鼐鼛鼞。"),
    ("answerable", "zh", "birthday", "anfang", "晏书的岁印安放在哪一格，鼒鼜鼟。"),
    ("answerable", "zh", "phone", "zhiyu", "卞珩的嗓音条置于何处，鼖鼣鼦。"),
    ("answerable", "zh", "birthday", "ding", "汲棠的岁印钉在哪一页，鼛鼤鼧。"),
    ("answerable", "zh", "phone", "jia", "郗宁的嗓音条夹进哪一页，鼜鼥鼨。"),
    ("answerable", "zh", "phone", "cun", "晏书的嗓音条存进哪一格，鼝鼪鼫。"),
    ("missing_rel", "zh", "birthday", "fang", "柜侧问卞珩那道岁记该放在哪里，鼭鼮鼰。"),
    ("missing_rel", "zh", "phone", "wang", "嗓音联络汲棠，条子该往哪里送，鼱鼲鼳。"),
    ("missing_rel", "zh", "phone", "chao", "朝哪里递郗宁的嗓音条，看哪一页，鼵鼶鼸。"),
    ("missing_rel", "zh", "birthday", "ge", "晏书的岁记先搁在哪一格，鼺鼉鼐。"),
    ("missing_rel", "zh", "birthday", "bai", "卞珩添岁的印要摆在哪一页，鼒鼖鼛。"),
    ("missing_rel", "zh", "phone", "tian", "汲棠的嗓音条填进哪一个抽屉，鼜鼝鼞。"),
    ("missing_rel", "zh", "birthday", "xie", "郗宁的岁印写进哪一页，鼟鼣鼤。"),
    ("missing_rel", "zh", "phone", "ji", "晏书的嗓音条记入哪一格，鼥鼦鼧。"),
    ("missing_rel", "zh", "birthday", "sai", "卞珩的岁印该塞进哪一页，鼨鼪鼫。"),
    ("missing_rel", "zh", "phone", "luo", "汲棠的嗓音条该落进哪一格，鼭鼮鼱。"),
    ("missing_rel", "zh", "birthday", "no_verb", "哪一格收下郗宁的岁印，鼲鼳鼵。"),
    ("missing_rel", "zh", "phone", "no_verb", "哪一页登记晏书的嗓音条，鼶鼸鼺。"),
    ("missing_rel", "zh", "birthday", "fang", "柜吏问，卞珩多一岁的朱印该放在哪里，鼉鼜鼥。"),
    ("missing_rel", "zh", "phone", "fang", "要把嗓音条子放在哪里才能找到汲棠，鼖鼝鼦。"),
    ("keep_places", "zh", "places", "dwell", "卞珩平日住在哪一段巷子，鼛鼞鼧。"),
    ("keep_places", "zh", "places", "stay", "汲棠入夜后歇在哪一条巷，鼜鼟鼨。"),
    ("keep_places", "zh", "places", "zai_nali", "替郗宁出面的人该在哪里停，先来后到不要换，鼝鼣鼪。"),
    ("keep_places", "zh", "places", "weizhi_naxie", "汲棠，册页里点过的位置是哪些，口述次序照旧，鼞鼤鼫。"),
    ("keep_places", "zh", "places", "na_yidai", "巷子深处，晏书偏向哪一带，鼟鼥鼭。"),
    ("keep_places", "zh", "places", "shi_nali", "青册问的是卞珩落脚是哪里，鼣鼦鼮。"),
    ("keep_places", "zh", "places", "didian", "郗宁的地点写在青册哪一栏，鼤鼧鼰。"),
    ("keep_places", "zh", "places", "ludian", "汲棠的路点按口述留在夹层，鼥鼨鼱。"),
    ("keep_places", "zh", "places", "pengmian", "晏书的碰面次序不要改，鼦鼪鼲。"),
    ("keep_places", "zh", "places", "person_fang", "夜巡要把卞珩放在哪里，鼧鼫鼳。"),
    ("keep_places", "zh", "places", "person_wang", "汲棠这个人该往哪里去歇，鼨鼭鼵。"),
    ("keep_places", "zh", "places", "bare_nali", "郗宁常歇的巷子在哪里，鼪鼮鼶。"),
    ("keep_places", "zh", "places", "difang", "晏书常去的地方照青册念，鼫鼰鼸。"),
    ("keep_places", "zh", "places", "zhandian", "卞珩的站点按先来后到留在夹层，鼭鼱鼺。"),
    ("guarded", "zh", "places", "jiaojie_fang", "卞珩的交接要放在哪里，册吏照着念，鼮鼲鼉。"),
    ("guarded", "zh", "phone", "call_wang", "柜侧要给汲棠打电话，往哪里递都可以，鼰鼳鼒。"),
    ("guarded", "zh", "birthday", "candle_fang", "郗宁的蜡烛页该放在哪里，鼱鼵鼖。"),
    ("guarded", "zh", "places", "didian_weizhi", "晏书的地点清册在柜侧，位置先搁着，问落脚次序，鼲鼶鼛。"),
    ("guarded", "zh", "phone", "shouji_chao", "朝哪里走，柜侧留的是卞珩的手机，鼳鼸鼜。"),
    ("guarded", "zh", "places", "tujing_fang", "汲棠的途经要放在哪里，照青册念，鼵鼺鼝。"),
)


@dataclass(frozen=True)
class BareRow:
    kind: str
    lang: str
    rel: str
    family: str
    question: str


def build_rows() -> Tuple[BareRow, ...]:
    rows = tuple(BareRow(*row) for row in _ROWS)
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate bare-locative dev question")
    return rows


def questions(rows: Sequence[BareRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[BareRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[BareRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts
