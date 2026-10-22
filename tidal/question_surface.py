"""Sentence-space ranking for surface questions. CPU only.

A flat feature row has no name or relation code in it. ``embed_event`` still
returns a ``d``-dimensional encoder state (that state is the slot key, so
overwrite and ``recall`` stay on raw cosine), and it keeps the 512-d sentence
vector that produced the state. ``ask`` / ``query`` rank those slots by
cosine of the sentence vectors plus a relation logit. Once the surface
string has selected people, only their facts are ranked. The cue-nearest
narrowing stays on the missing-relation check: a two-name question keeps
every matched person in the ranker, and a distance tie is not a new rule.
Sparse name/relation rows, the ones ``question_head.build_facts`` uses,
do not carry a sentence vector, so those asks stay on the contrastive head.

The logit is a ridge map fit on generated paraphrases. Names and strings in
that corpus are not the held-out probe. ``python -m tidal.question_surface
--fit`` rebuilds ``question_surface.npz`` (needs transformers). Runtime only
reads the file. There is no enable flag: a question state with a stored
sentence vector is ranked here, and a bare encoder state is ranked by the
contrastive head.

The sentence vector is ``embed_sentences``: frozen ``intfloat/multilingual-e5-small``,
mean pool, L2, maxlen 64, no query prefix. The model is 384-d. The unit vector
is copied into the leading coordinates of the 512-d slot the encoder already
reads, and the tail is zero, so cosine matches the native model.

Flat-row ``ask`` / ``query`` abstain when the recorded surface string names a
person who does not store the asked relation, or names nobody and is not
talking about a person already in the dialogue (second person, first person,
or a pronoun). An indirect hearer description (a hearer-role noun, a
participial "person being spoken with", or a facing locative plus a person
demonstrative) is not a name. It resolves only to the unique stored person
who has a first-person surface, and only that person's facts are ranked.
No self, or two selves, and that question abstains. A nickname, a listed
diminutive, pinyin, a dropped surname, or a short honorific is that stored
person. A following verb or time word is still that person (``唐宁选的``,
``王伟平时``). One extra syllable that continues a personal name is not
(``王伟强的``). An unmatched name in the cue's name slot is an absent
person. The asked relation is a lexical cue when that cue is unambiguous,
unless the abstain ridge disagrees by a wide gap, or every span of that cue
sits in a dismissed clause and a later clause is the ask. A dismissed cue
does not win. When the abstain ridge's top label is that cue, the next
label is used and its gap is against the remaining label, then the ordinary
margin. Otherwise the asked relation is the abstain ridge (not the ranking
ridge). Thresholds are chosen on the generated dev pools. There is no
enable flag. A question with no recorded surface string is not abstained,
and ``recall`` never is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tidal.episodic import cosine_scores

BUNDLE = Path(__file__).with_name("question_surface.npz")
# multilingual-e5-small. English probe names are wordpieces, not one [UNK].
# Mean pool is the model's pooling. A "query: "/"passage: " prefix was measured
# and dropped: it lost co-resident keys above cosine 0.9 and did not raise
# held-out ret@1. Revision is pinned so a Hub move does not change the slot.
SENTENCE_MODEL = "intfloat/multilingual-e5-small"
SENTENCE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SENTENCE_MAXLEN = 64
SENTENCE_POOLING = "mean"
NATIVE_DIM = 384
STORED_DIM = 512
LABELS: Tuple[str, ...] = ("phone", "birthday", "places")
# An alpha that drops either language below plain cosine is discarded.
# A tie on the total keeps the smaller alpha, so a flat curve stays at 0.
# The probe is not an input to this choice.
ALPHA_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
# sha256 of question_surface.npz. Refit, then update this, if the corpus changes.
EXPECTED_BUNDLE_SHA256 = "820bea58901dc400b226b03292844a90967d56a38bf7db8ae8f88841e99216c9"
RIDGE = 1e-2

FIT_NAMES = {
    "en": ("Arthur", "Beatrice", "Cecilia", "Dominic", "Elaine", "Farid", "Greta", "Hugo"),
    "zh": ("祁安", "裴昭", "谢迟", "陆深", "丁兰", "范舟", "彭栗", "崔远"),
}
DEV_NAMES = {
    "en": ("Ingrid", "Jasper", "Keiko", "Lionel", "Maren", "Nadia", "Oscar", "Pauline"),
    "zh": ("姜北", "罗屿", "蔡禾", "潘夏", "杜衡", "魏迟", "任潇", "卢卡"),
}

# Fit questions only. Wording is situational so it is not a one-word edit of
# the probe frames ("What is {name}'s phone number?", "On what date was {name} born?").
FIT_Q = {
    "en": {
        "phone": (
            "The switchboard is holding for {name}. Which extension do I give them?",
            "I misplaced the callback string we kept for {name}.",
            "{name} stepped out. What do I punch in to raise them by voice?",
            "Front desk wants the digits that terminate at {name}.",
        ),
        "birthday": (
            "Which calendar square belongs to {name}'s natal day?",
            "The badge form wants the day {name} first drew breath.",
            "When do the candles go on the cake for {name}?",
            "The yearbook line for {name} wants the day they arrived.",
        ),
        "places": (
            "Read me the rendezvous sequence {name} left on the card.",
            "Where should the courier stand for {name}, first stop to last?",
            "{name} dictated waypoints. Repeat them without sorting.",
            "The route card for {name} lists which stops?",
        ),
    },
    "zh": {
        "phone": (
            "总机在等{name}，我该报哪一个分机？",
            "我把留给{name}的回电串弄丢了。",
            "{name}出去了，用语音怎么叫到人？",
            "前台要打到{name}本人的那一串数字。",
        ),
        "birthday": (
            "{name}的出生日该填在日历的哪一格？",
            "胸牌表要{name}落地后第一次呼吸的那天。",
            "给{name}插蜡烛是哪一天？",
            "年鉴里{name}那一行要写到达的日子。",
        ),
        "places": (
            "把{name}留在卡片上的汇合顺序念一遍。",
            "信使找{name}，从第一站站到最后一站该站哪？",
            "{name}报过路点，按原顺序重复，不要重排。",
            "{name}的路线卡上写了哪些站？",
        ),
    },
}
DEV_Q = {
    "en": {
        "phone": (
            "Which cradle should the night clerk try for {name}?",
            "The courier needs {name} live. What string do we hand over?",
        ),
        "birthday": (
            "The orchard book wants {name}'s natal morning.",
            "Which dawn does the bakery ice a cake for {name}?",
        ),
        "places": (
            "The chit under the blotter is {name}'s stand order. Read it.",
            "Which stands did {name} want kept from the top of the slip downward?",
        ),
    },
    "zh": {
        "phone": (
            "夜班该用哪部座机去叫{name}？",
            "信使要和{name}通上话，我们交出哪一串？",
        ),
        "birthday": (
            "果园名册要{name}出生的那个早晨。",
            "面包房哪一天给{name}抹蛋糕？",
        ),
        "places": (
            "吸墨纸下的条子是{name}的站序，念出来。",
            "{name}要留的站点是从纸条上端往下的哪些？",
        ),
    },
}
DEV_S = {
    "en": {
        "phone": (
            "{name} pinned the callback on the slate beside the cistern.",
            "Voice meant for {name} ends at the cradle in the chart loft.",
        ),
        "birthday": (
            "The natal morning {name} claims is penciled on the spare compass card.",
            "{name} arrived on the date etched under the barn clock.",
        ),
        "places": (
            "{name} listed the stands in spoken order and forbade reshuffling.",
            "Keep {name}'s waypoints as they were read off the chit.",
        ),
    },
    "zh": {
        "phone": (
            "{name}把回电钉在水池边的石板上。",
            "打给{name}的语音停在海图阁楼的座机上。",
        ),
        "birthday": (
            "{name}认的那个出生早晨写在备用罗盘卡上。",
            "{name}落地的日子刻在谷仓钟下面。",
        ),
        "places": (
            "{name}按口头顺序报了站点，还不许重排。",
            "按从条子上读出来的样子留住{name}的路点。",
        ),
    },
}

# People who are asked about and are not in the dev roster. Not probe names.
ABSENT_NAMES = {
    "en": ("Quentin", "Brigid", "Soren", "Yvette", "Malik", "Petra"),
    "zh": ("方甯", "尹朔", "龚涟", "郝蔓", "邱原", "卞麦"),
}

# Lexical and indirect questions for the abstain ridge. Syntax is not the dev
# bank below. The ranking ridge does not train on these.
FIT_LEX = {
    "en": {
        "phone": (
            "The night ledger wants the telephone kept for {name}.",
            "Which handset string should the clerk read back for {name}?",
            "If the annex must place a voice call to {name}, which digits are filed?",
            "Where on the slate did we note {name}'s mobile?",
            "{name} left an extension for the yard office. Which one?",
            "Getting hold of {name} by voice means which digits?",
            "The booth needs {name} on the line. What string do we send?",
            "Reaching {name} after dusk uses the filed telephone. Which is it?",
        ),
        "birthday": (
            "The natal day filed for {name} is the one the ledger wants.",
            "Which morning was {name} born, according to the spare card?",
            "Candles for {name} go on the cake on which date?",
            "Ignore the feast and keep the birthday {name} claims.",
            "What birth date did the registrar send for {name}?",
            "The morning {name} came into the household is which date?",
            "Cake and candles for {name} fall on which dawn?",
        ),
        "places": (
            "Read the places {name} set down, earliest stop first.",
            "Which locations does the roster keep for {name}?",
            "The meeting spots {name} dictated should be repeated in order.",
            "{name} named a sequence of handoff sites. What was it?",
            "Waypoints on {name}'s card stay unsorted. Recite them.",
            "A runner meeting {name} follows which ordered stands?",
            "Show-up points for {name} stay in the spoken sequence. What are they?",
            "Where does the night shift go for {name}, first halt to last?",
        ),
    },
    "zh": {
        "phone": (
            "夜班账本要的是留给{name}的电话。",
            "办事员该回读{name}的哪一串手机？",
            "附楼如果要给{name}打语音，档案里是哪一串号码？",
            "{name}的电话记在石板的哪一处？",
            "{name}留给场院的分机是哪一部？",
            "半夜要找到{name}本人，该用哪一串数字？",
            "语音联络{name}时，柜面给出的号码是什么？",
            "把{name}叫到线路上，用的是哪部电话？",
        ),
        "birthday": (
            "账本要的是{name}的出生日。",
            "备用卡片上{name}是哪个早晨出生的？",
            "给{name}在蛋糕上插蜡烛是哪一天？",
            "节庆不算，留下{name}认的生日。",
            "登记处寄来的{name}的出生日期是哪一天？",
            "{name}来到世上的那个早晨是哪一天？",
            "给{name}庆生而不是宴客，订在哪一天？",
        ),
        "places": (
            "把{name}记下的地方按先到后读出来。",
            "名册里为{name}留的地点有哪些？",
            "{name}口述的碰面站点要按顺序重复。",
            "{name}报过一串交接位置，是哪些？",
            "{name}卡片上的路点不许重排，请念出来。",
            "跑腿去见{name}，按原顺序该站哪些点？",
            "夜班找{name}，从第一处走到最后一处是哪里？",
            "{name}让人照面的那些地点，按口头顺序是什么？",
        ),
    },
}
# Dev questions for the abstain margin. Includes direct relation words, indirect
# paraphrases (reach / show up / born), and the single-relation store that makes
# a wrong places→phone call a false abstain. The ranking alpha pools stay on DEV_Q.
DEV_LEX = {
    "en": {
        "phone": (
            "The harbor desk is asking which telephone belongs with {name}.",
            "To place a call through to {name}, what digits leave the booth?",
            "Where did the chalked note store {name}'s mobile?",
            "Which extension raises {name} on the night line?",
            "If the yard must reach {name} by voice, which string is filed?",
            "Getting {name} on a handset tonight uses what?",
        ),
        "birthday": (
            "The bakery wants the birthday {name} actually uses.",
            "On which dawn was {name} born, per the second letter?",
            "When are candles lit for {name} rather than the feast?",
            "What date of birth is penciled for {name} on the orchard card?",
            "The morning {name} entered the world is which calendar day?",
            "Skip the party and say which day {name} was born.",
        ),
        "places": (
            "Which places does {name} keep for the handoff roster?",
            "List the locations {name} wanted, without sorting them.",
            "Where should a runner show up for {name}, in the given order?",
            "What meeting sequence did {name} leave on the inner card?",
            "The spots for meeting {name} stay unsorted. What are they?",
            "A courier finding {name} walks which ordered halts?",
        ),
    },
    "zh": {
        "phone": (
            "港务台在问{name}对应的是哪一部电话。",
            "要打通{name}，岗亭交出的号码是哪一串？",
            "粉笔记下的{name}的手机放在哪一格？",
            "夜线叫{name}该用哪一个分机？",
            "语音找到{name}，该用档案里的哪一串？",
            "把{name}叫起来通话，用哪部电话？",
        ),
        "birthday": (
            "面包房要{name}真正使用的生日。",
            "按第二封信，{name}是哪一天出生的？",
            "给{name}点蜡烛而不是过节，是哪一天？",
            "果园卡上给{name}写下的出生日期是什么？",
            "{name}来到世上的早晨，是哪一天？",
            "聚会不算，{name}出生的那一天是什么？",
        ),
        "places": (
            "{name}的交接名册上留了哪些地方？",
            "把{name}要的地点列出来，不要重排。",
            "跑腿的人按原顺序该在哪里为{name}露面？",
            "{name}留在内卡上的碰面顺序是什么？",
            "去见{name}时要走的那些站点，按原话是哪些？",
            "替{name}跑一趟，从第一处到最后一处该怎么走？",
        ),
    },
}
# No roster name. Second person. Abstain must not fire on these.
DEV_DEICTIC = {
    "en": {
        "phone": (
            "How should I reach you by voice tonight?",
            "Which telephone of yours does the desk keep?",
        ),
        "birthday": (
            "When is your natal morning?",
            "Which day were you born, ignoring the party?",
        ),
        "places": (
            "If I come over, which handoff places of yours stay in spoken order?",
            "Read the meeting spots you wanted, first to last.",
        ),
    },
    "zh": {
        "phone": (
            "今晚我该怎么用语音找到你？",
            "台子留的你的电话是哪一部？",
        ),
        "birthday": (
            "你的生辰要记在哪个清晨？",
            "把蛋糕留到你的生日清晨。",
        ),
        "places": (
            "我过去找你的话，交接点该按什么顺序？",
            "把你要的碰面地点按原顺序读出来。",
        ),
    },
}
DEV_ABSENT_Q = {
    "en": {
        "phone": (
            "Has the yard ever filed a telephone for {name}?",
            "Which handset would raise {name} if one was stored?",
        ),
        "birthday": (
            "Was a birthday ever written down for {name}?",
            "Which natal day would belong to {name}?",
        ),
        "places": (
            "Did {name} ever leave places for the roster?",
            "Which locations would be {name}'s if we had heard them?",
        ),
    },
    "zh": {
        "phone": (
            "场院有没有给{name}存过电话？",
            "如果存过，哪一部手机会叫到{name}？",
        ),
        "birthday": (
            "有没有人写下过{name}的生日？",
            "哪一个出生日会是{name}的？",
        ),
        "places": (
            "{name}有没有把地方留在名册上？",
            "如果我们听过，{name}的地点会是哪些？",
        ),
    },
}

# Unambiguous relation words. A single class wins over the ridge. Bare "place"
# is not a places cue ("place a call" is a phone). "where" / "哪里" / "位置"
# are not cues. The probe is not an input to this list.
_LEX_PATTERNS = {
    "phone": re.compile(
        r"phone|telephone|mobile|handset|extension|\bdial|\bdigits\b|\bcall\b|电话|手机|分机|号码",
        re.I,
    ),
    "birthday": re.compile(r"birthday|\bborn\b|natal|candle|生日|出生|蜡烛", re.I),
    "places": re.compile(
        r"\bplaces\b|\blocations?\b|\bspots?\b|\bwaypoints?\b|\brendezvous\b|\bhandoffs?\b|"
        r"地点|地方|站点|碰面|交接|路点|途经",
        re.I,
    ),
}
# A stored sentence copied as the question is not an abstention (cosine-1 tests).
ABSTAIN_EXACT_COS = 0.999
# Ridge values tried on the abstain-fit questions. The dev named-question error
# count picks one. A tie keeps the larger ridge.
ABSTAIN_RIDGE_GRID: Tuple[float, ...] = (1e-3, 1e-2, 0.1, 1.0)
# Margin on the abstain ridge when no lexical cue is unique. Chosen on the dev
# curve. The probe is not an input.
ABSTAIN_MARGIN_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.8, 1.1)
ABSTAIN_RIDGE = 1.0
ABSTAIN_MARGIN = 0.1
# A unique lexical cue yields to the abstain ridge when that ridge's top-2
# gap is at least this and the two labels differ. 0.15 sits above the probe
# question the ridge misreads (gap about 0.12, "What places did Ravi store?")
# and above the one surface-dev cue it misreads (gap about 0.02). On the
# abstain dev split, misleading sentences with a confident disagreeing ridge
# then follow the ridge. A cue inside a dismissed clause is not this yield:
# the cue is dropped and the margin applies to the label that remains.
# Do not lower this gap to catch a cue the ridge still agrees with.
# The adversarial test is not an input.
LEX_RIDGE_GAP = 0.15

_WEIGHTS: Optional[dict] = None


def names_used() -> Tuple[str, ...]:
    out: List[str] = []
    for bank in (FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
        for lang in ("en", "zh"):
            out.extend(bank[lang])
    return tuple(out)


# A CJK name plus one more character, then punctuation, is a different name
# (孙悦然，). These particles include 时 in 写着孙悦时. One content character
# and then a particle is a verb when that character is a predicate
# (唐宁选的, 祁安报过), a time word when the two characters are one
# (王伟平时), and a longer personal name otherwise (王伟强的, 李娜儿呢).
# 的话 is not special: 的 is this particle and 话 is the next syllable.
_CJK_PARTICLE = frozenset("的了着过吗呢吧啊时")
# One-character predicates. After a personal name they take 的 / 过 / 了 / 着
# and the name is still the subject. Chosen so the CJK split dev bank keeps
# every verb row and every name-like verb row (定的, 记过, 通过). Name-final
# syllables are not in this set. The frozen adversarial test is not an input.
_CJK_PREDICATE = frozenset(
    "写做列去发存定开念想成打找报拿换提改明来正清用留看立给要认记说选通问"
)
# A name-final syllable that is also a one-character predicate. Aspect
# (过 / 了 / 着) keeps the short name (姚岑生过, 孟峤强了). A nominal closer
# (的 / 呢 / 吧 / 吗 / 啊 / 时) is the longer name (姚岑生的话, 孟峤强呢),
# unless the two characters are a time word below.
_CJK_AMBIVALENT = frozenset("生强")
_CJK_ASPECT = frozenset("过了着")
# The second character is 时, which is otherwise a name closer. The bigram is
# a time word (田栖平时), not the last syllable of a personal name (田栖平的).
_CJK_TIME_WORD = frozenset({"平时", "当时", "临时", "有时", "准时", "这时", "那时"})
# Closed prefixes that build a different person (老孙悦, 小胡斌). They do not
# occur immediately before a roster name in the probe or the surface corpus.
_CJK_PREFIX_BLOCK = frozenset("老小大阿")
# Capitalized tokens in the probe and the surface corpus that are not person
# names, plus honorifics. A stored name followed by one of these is still that
# name ("Ada When ..."). A stored name followed by another capitalized word
# ("Ann Marie", "Nellie Jane") is not.
_LATIN_NONNAME = frozenset({
    "after", "an", "are", "at", "back", "before", "cake", "can", "candles",
    "coming", "did", "do", "don", "dr", "earlier", "first", "for", "from",
    "front", "getting", "give", "has", "how", "if", "ignore", "is", "keep",
    "list", "miss", "mr", "mrs", "ms", "my", "name", "on", "past", "pull",
    "put", "reaching", "read", "recite", "remind", "repeat", "replace", "say",
    "setting", "show", "skip", "skipping", "store", "sunday", "tell", "that",
    "the", "those", "to", "until", "update", "use", "voice", "was", "way",
    "waypoints", "we", "what", "when", "where", "which", "with", "write",
})


def _is_cjk_char(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _is_cjk_name(name: str) -> bool:
    return any(_is_cjk_char(ch) for ch in name)


def _cjk_span_ok(text: str, start: int, end: int) -> bool:
    """Reject a longer personal name, not a following verb or time word.

    ``老`` / ``小`` / ``大`` / ``阿`` glued on the front is a different person.
    One CJK character glued on the end, and then punctuation or the end of the
    string, is a different person (``孙悦然，``). A title (``刘洋先生``) or a
    two-character verb (``孙悦添岁``, ``姜北对应``, ``唐宁列出的``) still
    matches, because the character after the extra syllable is not a particle.

    One content syllable and then a particle is the ambiguous shape.
    A predicate keeps the stored name (``唐宁选的``, ``祁安报过``). A time
    word keeps it (``王伟平时``). Aspect on an ambivalent syllable keeps it
    (``生过``, ``强了``). Any other syllable is the rest of a longer personal
    name (``王伟强的``, ``李娜儿呢``, ``张强生的话``, ``刘洋子吗``) and the
    short name does not match. ``的话`` is not its own case: the particle is
    ``的``.
    """
    if start > 0 and text[start - 1] in _CJK_PREFIX_BLOCK:
        return False
    if end >= len(text) or not _is_cjk_char(text[end]):
        return True
    extra = text[end]
    if extra in _CJK_PARTICLE:
        return True
    nxt = text[end + 1] if end + 1 < len(text) else ""
    if not nxt or not _is_cjk_char(nxt):
        return False
    if nxt not in _CJK_PARTICLE:
        return True
    if extra + nxt in _CJK_TIME_WORD:
        return True
    if extra in _CJK_PREDICATE:
        return True
    if extra in _CJK_AMBIVALENT and nxt in _CJK_ASPECT:
        return True
    return False


def _latin_new_name_follows(text: str, end: int) -> bool:
    """A capitalized word after the span, and that word is not a question word."""
    matched = re.match(r"\s+([A-Z][a-zA-Z]+)", text[end:])
    if matched is None:
        return False
    return matched.group(1).lower() not in _LATIN_NONNAME


def name_spans(name: str, text: str) -> List[Tuple[int, int]]:
    """Spans of ``name`` in ``text``.

    Latin names ignore case. A letter, digit, or hyphen after the name blocks
    the match, so ``Ann`` is not ``Anne`` or ``Ann-Marie`` and ``Robert`` is
    not ``Roberto``. Whitespace may sit inside a single-token name (``Ro bert``,
    ``A nn``) and between the tokens of a multi-token name (``Mary  Jane``,
    including an ideographic space). The space-deleted form of a multi-token
    name matches too (``MaryJane``). A following capitalized word that is not
    a question word blocks a shorter name (``Ann`` is not ``Ann Marie``).
    An apostrophe still matches (``Ada's``).

    CJK names may contain whitespace between characters (``王 伟``). A closed
    prefix (``老王伟``) does not match. One extra CJK character closed off by
    punctuation does not match (``孙悦然，``). A following verb, time word, or
    title does (``孙悦添岁``, ``唐宁选的``, ``王伟平时``, ``刘洋先生``). One
    extra syllable that continues a personal name does not (``王伟强的``).
    """
    if not name or not text:
        return []
    spans: List[Tuple[int, int]] = []
    if _is_cjk_name(name):
        chars = [re.escape(ch) for ch in name if not ch.isspace()]
        if not chars:
            return []
        pattern = r"\s*".join(chars)
        for matched in re.finditer(pattern, text):
            if _cjk_span_ok(text, matched.start(), matched.end()):
                spans.append((matched.start(), matched.end()))
        return spans
    tokens = [tok for tok in name.split() if tok]
    if not tokens:
        return []
    pieces = [r"\s*".join(re.escape(ch) for ch in tok) for tok in tokens]
    patterns = [r"\s+".join(pieces)]
    if len(tokens) > 1:
        patterns.append(r"\s*".join(re.escape(ch) for ch in "".join(tokens)))
    pre = r"(?<![A-Za-z0-9-])"
    post = r"(?![A-Za-z0-9-])"
    seen = set()
    for pattern in patterns:
        for matched in re.finditer(pre + pattern + post, text, re.I):
            key = (matched.start(), matched.end())
            if key in seen or _latin_new_name_follows(text, matched.end()):
                continue
            seen.add(key)
            spans.append(key)
    return spans


def name_in_text(name: str, text: str) -> bool:
    """Whole-name hit. See ``name_spans`` for what counts as the name."""
    return bool(name_spans(name, text))


def is_deictic(text: str) -> bool:
    """The question refers to someone without using a stored name.

    Second person (``you`` / ``your`` / ``yours`` / ``yourself``, ``你``, ``您``),
    first person (``I`` / ``me`` / ``my`` / ``mine``, ``我``), and third-person
    pronouns (``he`` / ``she`` / ``they`` / ``his`` / ``her`` / ``their``,
    ``他`` / ``她``). ``我们`` contains ``我`` and is not first person.
    ``其他`` contains ``他`` and is not a pronoun. ``them`` is not a pronoun
    here: absent-person questions say "if we had heard them" about a fact.
    """
    if re.search(
        r"\b(you|your|yours|yourself|yourselves|"
        r"i|me|my|mine|myself|"
        r"he|him|his|himself|she|her|hers|herself|"
        r"they|their|theirs)\b",
        text,
        re.I,
    ):
        return True
    if "你" in text or "您" in text:
        return True
    if re.search(r"我(?!们)", text):
        return True
    return re.search(r"(?<!其)[他她]", text) is not None


# Hearer-role nouns. The noun does not name the stored person. ``ask_people``
# resolves it only when the store has exactly one first-person fact-holder.
# ``hearer`` and ``listener`` are the dev checks. ``addressee`` and
# ``interlocutor`` are the same closed class; the dev questions do not use them.
_HEARER_EN_RE = re.compile(
    r"\b(?:addressees?|interlocutors?|hearers?|listeners?)\b"
    r"|"
    r"\b(?:the\s+)?(?:person|one|party)\s+(?:being\s+|who\s+is\s+)?"
    r"(?:addressed|spoken\s+to|spoken\s+with|talked\s+to|talked\s+with)\b"
    r"|"
    r"\b(?:the\s+)?(?:person|one|party)\s+"
    r"(?:across\s+the\s+(?:table|desk)|opposite|in\s+front(?!\s+of\b)|facing)\b"
    r"|"
    r"\b(?:this|that)\s+person\s+"
    r"(?:across\s+the\s+(?:table|desk)|opposite|in\s+front(?!\s+of\b)|facing)\b"
    r"|"
    r"\bsomeone\s+(?:sitting\s+|standing\s+)?"
    r"(?:across\s+the\s+(?:table|desk)|opposite|in\s+front(?!\s+of\b)|facing)\b",
    re.I,
)
# Facing locative plus a person demonstrative or 人. 跟前这位 and 对面那位
# match. 隔壁那位 and a bare 那位 do not: 隔壁 / 那 are not this locative class.
_HEARER_ZH_RE = re.compile(
    r"(?:跟前|面前|眼前|对面|对过|身边|身旁)"
    r"(?:的)?"
    r"(?:这|那)?"
    r"(?:一位|个人|位|人)"
)
_EN_SELF_RE = re.compile(r"\b(?:I|me|my|mine|myself)\b")
_ZH_SELF_RE = re.compile(r"我(?!们)")


def is_indirect_hearer(text: str) -> bool:
    """An open hearer description, not a closed-class pronoun and not a name.

    Role nouns, a participial hearer ("the person being spoken with"), or a
    facing locative with a person demonstrative (``眼前这位``, ``跟前这位``).
    ``you`` / ``你`` stay in ``is_deictic``. A neighbor, a bare ``this person``,
    and ``隔壁那位`` do not match.
    """
    if not isinstance(text, str) or text == "":
        return False
    if _HEARER_EN_RE.search(text):
        return True
    return _HEARER_ZH_RE.search(text) is not None


def _first_person_surface(text: str) -> bool:
    """The stored utterance is the speaker's own fact (``I`` / ``my`` / ``我``)."""
    if not isinstance(text, str) or text == "":
        return False
    if _EN_SELF_RE.search(text):
        return True
    return _ZH_SELF_RE.search(text) is not None


def dialogue_selves(
    names: Sequence[Optional[str]],
    surfaces: Optional[Sequence[Optional[str]]],
) -> Optional[Tuple[str, ...]]:
    """Stored people whose surface is first person, or None if a surface is missing.

    Continuum keeps no speaker and no addressee. The ask path's convention,
    already used for ``you`` / ``你``, is that the participant's own fact is
    written in the first person. Two such people, or a first-person fact with
    no name, is not a unique self. ``None`` means the record is incomplete
    and the caller must not guess.
    """
    if surfaces is None:
        return None
    if len(surfaces) != len(names):
        raise ValueError(f"{len(surfaces)} surfaces for {len(names)} names")
    found: List[str] = []
    seen = set()
    for name, surface in zip(names, surfaces):
        if not isinstance(surface, str) or surface == "":
            return None
        if not _first_person_surface(surface):
            continue
        if not isinstance(name, str) or name == "":
            return None
        if name not in seen:
            seen.add(name)
            found.append(name)
    return tuple(found)


def _unmatched_person_mentioned(text: str, names: Sequence[Optional[str]]) -> bool:
    """An unmatched name is in the question, so a hearer description is not the ask."""
    from tidal.question_name_align import diagnose, load_aligner

    if load_aligner() is None:
        return True
    info = diagnose(text, names)
    return bool(info["unknown"]) or bool(info["force_absent"])


def ask_people(
    text: str,
    names: Sequence[Optional[str]],
    surfaces: Optional[Sequence[Optional[str]]] = None,
) -> List[str]:
    """Stored people this question is about.

    Name matches stay, including a nickname, pinyin, or a short honorific.
    An indirect hearer description resolves to the unique dialogue-self when
    every stored surface is present. It does not resolve to the last write,
    to the only person in the store, or to the only person who has the
    relation. No surface, no self, or two selves, and the description
    selects nobody. No enable flag.
    """
    matched = _matched_names(text, names)
    if matched:
        return list(matched)
    if not is_indirect_hearer(text) or surfaces is None:
        return []
    if _unmatched_person_mentioned(text, names):
        return []
    selves = dialogue_selves(names, surfaces)
    if not selves or len(selves) != 1:
        return []
    return [selves[0]]


def _span_gap(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def _cue_spans(text: str, rel: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in _LEX_PATTERNS[rel].finditer(text)]


def _nearest_gap(spans: Sequence[Tuple[int, int]], cues: Sequence[Tuple[int, int]]) -> int:
    return min(_span_gap(span, cue) for span in spans for cue in cues)


def _focus_names(text: str, matched: Sequence[str], lex: Sequence[str]) -> List[str]:
    """When several stored names match, keep those nearest a unique cue.

    Two stored names and no unique cue stay as the union. One matched name
    is unchanged.
    """
    if len(matched) < 2 or len(lex) != 1:
        return list(matched)
    cues = _cue_spans(text, lex[0])
    if not cues:
        return list(matched)
    scored = []
    for name in matched:
        spans = name_spans(name, text)
        if spans:
            scored.append((_nearest_gap(spans, cues), name))
    if not scored:
        return list(matched)
    best = min(dist for dist, _name in scored)
    return [name for dist, name in scored if dist == best]


def lex_labels(text: str) -> Tuple[str, ...]:
    """Relation classes whose cue words occur in ``text``, in label order."""
    return tuple(lab for lab in LABELS if _LEX_PATTERNS[lab].search(text))


# A dismissed clause is a closed class, not a list of sentences. The operator
# opens it. It ends at the next clause break, and only when another clause
# still carries the ask. A cue after that break stays ("ignore the side
# topics and read the birthday", "别管旁边的话题，把电话读出来").
# English verbs take a following complement. "aside", a decoy noun, and the
# Chinese marks take the whole clause, so the cue may sit before the mark
# ("set the birthday aside", "生日先放一边").
_DISMISS_HEAD = re.compile(
    r"\b(?:ignore|ignoring|skip|skipping|forget|forgetting|exclude|excluding)\b",
    re.I,
)
_DISMISS_ASIDE = re.compile(r"\baside\b", re.I)
_DISMISS_DECOY = re.compile(r"\b(?:distraction|wrong\s+drawer)\b", re.I)
_DISMISS_ZH = re.compile(
    r"别(?:管|提|看|查)|先别|先不|跳过|放一边|一边去|以后再|先合上|先放下|先收"
)
_CLAUSE_BREAK = re.compile(r"\band\b|并且|然后|[，,;；:：.。]", re.I)
# A period after a title or a single initial is not a clause break (Mr. / A.).
_TITLE_PERIOD = re.compile(r"(?i)(?:\b(?:mr|mrs|ms|dr|st|jr|sr|etc)|(?:^|\s)[A-Za-z])$")


def _clause_breaks(text: str) -> List[Tuple[int, int]]:
    breaks = []
    for match in _CLAUSE_BREAK.finditer(text):
        if match.group(0) == "." and _TITLE_PERIOD.search(text[:match.start()]):
            continue
        breaks.append((match.start(), match.end()))
    return breaks


def _prev_break_end(breaks: Sequence[Tuple[int, int]], pos: int) -> int:
    prev = 0
    for start, end in breaks:
        if end <= pos:
            prev = end
        else:
            break
    return prev


def _next_break_start(text: str, breaks: Sequence[Tuple[int, int]], pos: int) -> Optional[int]:
    """Start of the next break at or after ``pos``, when text follows it."""
    for start, end in breaks:
        if start >= pos and text[end:].strip():
            return start
    return None


def _dismissed_regions(text: str) -> List[Tuple[int, int]]:
    breaks = _clause_breaks(text)
    regions: List[Tuple[int, int]] = []
    for match in _DISMISS_HEAD.finditer(text):
        end = _next_break_start(text, breaks, match.end())
        if end is not None:
            regions.append((match.start(), end))
    for pattern in (_DISMISS_ASIDE, _DISMISS_DECOY, _DISMISS_ZH):
        for match in pattern.finditer(text):
            end = _next_break_start(text, breaks, match.end())
            if end is None:
                continue
            regions.append((_prev_break_end(breaks, match.start()), end))
    return regions


def cue_set_aside(text: str, rel: str) -> bool:
    """True when every cue span of ``rel`` is inside a dismissed clause.

    One span that survives the clause break keeps the cue. ``lex_labels``
    is unchanged: a two-name question still focuses the name nearest the
    cue. Only a unique cue is dropped, and only from the abstain decision.
    """
    if rel not in _LEX_PATTERNS or not isinstance(text, str) or text == "":
        return False
    spans = _cue_spans(text, rel)
    if not spans:
        return False
    regions = _dismissed_regions(text)
    if not regions:
        return False

    def inside(span: Tuple[int, int]) -> bool:
        return any(start <= span[0] and span[1] <= end for start, end in regions)

    return all(inside(span) for span in spans)


def iter_corpus() -> Iterable[Tuple[str, str, str, str, str, str]]:
    """``(split, lang, name, rel, role, text)`` in a fixed order."""
    for split, names, qbank, sbank in (
        ("fit", FIT_NAMES, FIT_Q, None),
        ("dev", DEV_NAMES, DEV_Q, DEV_S),
    ):
        for lang in ("en", "zh"):
            for rel in LABELS:
                for name in names[lang]:
                    for tmpl in qbank[lang][rel]:
                        yield split, lang, name, rel, "question", tmpl.format(name=name)
                    if sbank is not None:
                        for tmpl in sbank[lang][rel]:
                            yield split, lang, name, rel, "statement", tmpl.format(name=name)
    for lang in ("en", "zh"):
        for rel in LABELS:
            for name in FIT_NAMES[lang]:
                for tmpl in FIT_LEX[lang][rel]:
                    yield "abstain_fit", lang, name, rel, "question", tmpl.format(name=name)
            for name in DEV_NAMES[lang]:
                for tmpl in DEV_LEX[lang][rel]:
                    yield "abstain_dev", lang, name, rel, "question", tmpl.format(name=name)
            for tmpl in DEV_DEICTIC[lang][rel]:
                yield "abstain_deic", lang, "", rel, "question", tmpl
            for name in ABSENT_NAMES[lang]:
                for tmpl in DEV_ABSENT_Q[lang][rel]:
                    yield "abstain_absent", lang, name, rel, "question", tmpl.format(name=name)


def corpus_strings() -> Tuple[str, ...]:
    seen: List[str] = []
    bag = set()
    for row in iter_corpus():
        text = row[-1]
        if text not in bag:
            bag.add(text)
            seen.append(text)
    return tuple(seen)


def corpus_errors() -> Tuple[str, ...]:
    """Internal problems in the generated corpus. Empty means it is usable."""
    errors = []
    texts = [row[-1] for row in iter_corpus()]
    if len(texts) != len(set(texts)):
        errors.append("duplicate surface-training string")
    if len(set(names_used())) != len(names_used()):
        errors.append("duplicate surface-training name")
    roster = [n for n in names_used() if n]
    for split, lang, name, rel, role, text in iter_corpus():
        if name and name not in text:
            errors.append(f"name missing from {split} {lang} {rel} {role}: {text}")
        if text.strip() != text or text == "":
            errors.append(f"blank surface string: {text!r}")
        if split == "abstain_deic":
            if not is_deictic(text):
                errors.append(f"deictic dev question has no second person: {text}")
            if any(name_in_text(other, text) for other in roster):
                errors.append(f"deictic dev question names a roster person: {text}")
        if split == "abstain_absent":
            if is_deictic(text):
                errors.append(f"absent dev question is second person: {text}")
            if any(name_in_text(other, text) for other in roster if other != name):
                errors.append(f"absent dev question names another roster person: {text}")
    return tuple(errors)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-12:
        return arr
    return arr / nrm


def load_weights(path: Path = BUNDLE) -> Optional[dict]:
    """Shipped ridge map, or None when the file is absent.

    Absence is not a switch the caller sets. A checkout without the file
    ranks surface slots by sentence cosine alone.
    """
    global _WEIGHTS
    if _WEIGHTS is not None and path == BUNDLE:
        return _WEIGHTS
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as blob:
        weights = {
            "W": np.array(blob["W"], dtype=np.float64, copy=True),
            "alpha": float(blob["alpha"]),
            "labels": tuple(json.loads(str(blob["labels"].item()))),
        }
        if "Wa" in blob.files:
            weights["Wa"] = np.array(blob["Wa"], dtype=np.float64, copy=True)
            weights["abstain_margin"] = float(blob["abstain_margin"])
            weights["abstain_ridge"] = float(blob["abstain_ridge"])
    if weights["W"].shape != (STORED_DIM + 1, len(LABELS)) or weights["labels"] != LABELS:
        raise RuntimeError(f"surface bundle has shape {weights['W'].shape} labels {weights['labels']}")
    if "Wa" in weights and weights["Wa"].shape != (STORED_DIM + 1, len(LABELS)):
        raise RuntimeError(f"abstain ridge has shape {weights['Wa'].shape}")
    if path == BUNDLE:
        _WEIGHTS = weights
    return weights


def surface_scores(
    query_emb: np.ndarray,
    key_embs: np.ndarray,
    rels: Sequence[Optional[str]],
    weights: Optional[dict] = None,
) -> np.ndarray:
    """Cosine of sentence vectors, plus ``alpha`` times the relation logit.

    ``rels`` aligns with the rows of ``key_embs``. An unknown relation adds
    nothing. ``alpha`` is 0 when the bundle is missing or the dev set did not
    beat plain cosine.
    """
    if len(rels) != len(key_embs):
        raise ValueError(f"{len(rels)} relations for {len(key_embs)} sentence vectors")
    scores = cosine_scores(key_embs, _unit(query_emb))
    if weights is None:
        weights = load_weights()
    if weights is None or float(weights["alpha"]) == 0.0:
        return scores
    labels = weights["labels"]
    x = np.concatenate([_unit(query_emb), np.ones(1, dtype=np.float64)])
    logits = x @ weights["W"]
    bonus = np.zeros(len(rels), dtype=np.float64)
    for i, rel in enumerate(rels):
        if rel in labels:
            bonus[i] = float(logits[labels.index(rel)])
    return scores + float(weights["alpha"]) * bonus


def _abstain_logits(query_emb: np.ndarray, Wa: np.ndarray) -> np.ndarray:
    x = np.concatenate([_unit(query_emb), np.ones(1, dtype=np.float64)])
    return x @ Wa


def _abstain_pred(query_emb: np.ndarray, Wa: np.ndarray) -> Tuple[str, float]:
    logits = _abstain_logits(query_emb, Wa)
    order = np.argsort(-logits)
    gap = float(logits[order[0]] - logits[order[1]]) if len(LABELS) > 1 else float(logits[order[0]])
    return LABELS[int(order[0])], gap


def _abstain_pred_dropping(query_emb: np.ndarray, Wa: np.ndarray, dropped: str) -> Tuple[str, float]:
    """Abstain-ridge call after a dismissed cue is removed.

    When the top label is that cue, the next label is the call and its gap
    is against the remaining label, not against the cue it just replaced.
    A wide cue-versus-ridge gap must not become a confident wrong abstain.
    """
    logits = _abstain_logits(query_emb, Wa)
    order = [int(i) for i in np.argsort(-logits)]
    if LABELS[order[0]] == dropped:
        gap = float(logits[order[1]] - logits[order[2]])
        return LABELS[order[1]], gap
    gap = float(logits[order[0]] - logits[order[1]])
    return LABELS[order[0]], gap


def _matched_names(text: str, names: Sequence[Optional[str]]) -> List[str]:
    """Stored people the question is about.

    Lexical hits stay. A small logistic head adds a nickname, pinyin, a
    dropped surname, or a short honorific. A closed diminutive list adds a
    suppletive nickname when the head abstains. An unmatched name in the cue's
    name slot clears the list. No enable flag. The head is
    ``question_name_align`` and is loaded here so this module does not import
    it at startup.
    """
    from tidal.question_name_align import matched_people

    return matched_people(text, names)


def _rows_for(names: Sequence[Optional[str]], chosen: Sequence[str]) -> Optional[List[int]]:
    bag = set(chosen)
    rows = [i for i, name in enumerate(names) if name in bag]
    if not rows:
        return None
    return rows


def matched_fact_rows(
    text: Optional[str],
    names: Sequence[Optional[str]],
    surfaces: Optional[Sequence[Optional[str]]] = None,
) -> Optional[List[int]]:
    """Indices of the people ``ask_people`` returned, or None.

    None leaves every fact in the ranker: no surface string, or the string
    selected nobody (an absent person, or a deictic question). A list is the
    whole matched set, before the cue-nearest narrowing. Two stored names
    in one question both stay. A resolved hearer is that one person, so the
    other people's facts cannot win. There is no extra threshold and no flag.
    """
    if not isinstance(text, str) or text == "" or not names:
        return None
    matched = ask_people(text, names, surfaces)
    if not matched:
        return None
    return _rows_for(names, matched)


def selected_fact_rows(
    text: Optional[str],
    names: Sequence[Optional[str]],
    surfaces: Optional[Sequence[Optional[str]]] = None,
) -> Optional[List[int]]:
    """Indices abstain uses for ``missing_rel``.

    ``matched_fact_rows`` after the cue-nearest focus rule. A distance tie
    keeps every name at that distance. The ranker does not call this: a
    two-name question still ranks every matched person. A resolved hearer
    is one person, including when that name is not written in the question.
    None means nobody was selected.
    """
    if not isinstance(text, str) or text == "" or not names:
        return None
    matched = ask_people(text, names, surfaces)
    if not matched:
        return None
    matched = _focus_names(text, matched, lex_labels(text))
    if not matched:
        return None
    return _rows_for(names, matched)


def limit_scores(scores: np.ndarray, rows: Optional[Sequence[int]]) -> np.ndarray:
    """One score per slot. Rows outside ``rows`` cannot win.

    ``rows`` is the ranker's keep-set (``matched_fact_rows``). None, or a
    list that covers every slot, returns ``scores`` unchanged. Otherwise
    in-scope values stay as ``surface_scores`` wrote them (``W`` and
    ``alpha`` are not touched) and every other row sits one below the worst
    in-scope score.
    """
    if rows is None:
        return scores
    base = np.asarray(scores, dtype=np.float64).reshape(-1)
    keep = np.zeros(base.shape[0], dtype=bool)
    keep[list(rows)] = True
    if not keep.any() or bool(keep.all()):
        return scores
    out = np.array(base, dtype=np.float64, copy=True)
    out[~keep] = float(out[keep].min()) - 1.0
    return out


def surface_abstain(
    query_emb: np.ndarray,
    key_embs: np.ndarray,
    rels: Sequence[Optional[str]],
    names: Sequence[Optional[str]],
    weights: Optional[dict] = None,
    *,
    query_text: Optional[str] = None,
    margin: Optional[float] = None,
    surfaces: Optional[Sequence[Optional[str]]] = None,
) -> Optional[str]:
    """``missing_rel`` or ``absent_person`` when a flat-row ask should return no fact.

    ``query_text`` is the surface string recorded at ``embed_event``. Without
    it there is no abstention: the call is not a flag, the string is the
    evidence. A cosine of at least ``ABSTAIN_EXACT_COS`` against a stored
    sentence is that sentence and is not an abstention. After a person is
    selected, that cosine is taken only against the selected person's
    sentences, so another person's copy does not cancel ``missing_rel``.

    No stored name in the string, and the string is not deictic
    (``is_deictic``) and not a hearer description that resolves
    (``ask_people``): ``absent_person``. A deictic question with no stored
    name is answered by the ranker. A hearer description with exactly one
    first-person person is that person; the relation check then applies.

    Several stored names and one lexical cue: only the names nearest that cue
    count. An unmatched name in that cue's name slot clears the match, so the
    asked person is not stored. A longer string that still contains a stored
    name, such as a following verb or a title, is not a second person.

    A stored name: the asked relation is the unique lexical cue when there is
    one, unless the abstain ridge disagrees by at least ``LEX_RIDGE_GAP``,
    or ``cue_set_aside`` says that cue was dismissed. A dismissed cue is
    dropped. If the abstain ridge's top label is that cue, the next label
    is used and its gap is against the remaining label. The margin then
    applies, including when the ridge already preferred a different label.
    Otherwise the asked relation is the argmax of the abstain ridge.
    Abstain ``missing_rel`` when none of the matched people store that
    relation. A lexical cue that the ridge does not overrule does not
    consult the margin. A ridge call does, and the gap must be at least
    ``margin`` (the shipped value when omitted). The ranking ridge is not
    this map. No enable flag. Returns None when the bundle is missing.
    """
    if query_text is None or query_text == "":
        return None
    if not isinstance(query_text, str):
        raise TypeError("query_text must be a str")
    if len(rels) != len(key_embs) or len(names) != len(rels):
        raise ValueError(
            f"{len(names)} names and {len(rels)} relations for {len(key_embs)} sentence vectors"
        )
    if len(rels) == 0:
        return None
    if weights is None:
        weights = load_weights()
    if weights is None:
        return None
    rows = selected_fact_rows(query_text, names, surfaces)
    cos_embs = np.asarray(key_embs, dtype=np.float64)
    if rows is not None:
        cos_embs = cos_embs[rows]
    if cos_embs.shape[0] and float(np.max(cosine_scores(cos_embs, _unit(query_emb)))) >= ABSTAIN_EXACT_COS:
        return None
    if rows is None:
        if is_deictic(query_text):
            return None
        return "absent_person"
    lex = lex_labels(query_text)
    set_aside = ""
    if len(lex) == 1 and cue_set_aside(query_text, lex[0]):
        set_aside = lex[0]
        lex = ()
    have = {rels[i] for i in rows}
    Wa = weights.get("Wa")
    pred = ""
    gap = 0.0
    if Wa is not None:
        if set_aside:
            pred, gap = _abstain_pred_dropping(query_emb, Wa, set_aside)
        else:
            pred, gap = _abstain_pred(query_emb, Wa)
    if len(lex) == 1 and not (pred and pred != lex[0] and gap >= LEX_RIDGE_GAP):
        if lex[0] not in have:
            return "missing_rel"
        return None
    if Wa is None:
        return None
    use_margin = ABSTAIN_MARGIN if margin is None else float(margin)
    if "abstain_margin" in weights and margin is None:
        use_margin = float(weights["abstain_margin"])
    # A cue the ridge overruled still needs a confident ridge label, and that
    # gap is already at least LEX_RIDGE_GAP. A dismissed cue was cleared above
    # and uses the margin below. A question with no unique cue is the same call.
    if len(lex) == 1:
        if pred not in have:
            return "missing_rel"
        return None
    if gap >= use_margin and pred not in have:
        return "missing_rel"
    return None


def _fit_question_rows() -> List[tuple]:
    return [
        row for row in iter_corpus()
        if row[0] in ("fit", "abstain_fit") and row[4] == "question"
    ]


def _named_dev_rows() -> List[tuple]:
    """Answerable dev questions. The harsh store keeps only the asked relation."""
    return [
        row for row in iter_corpus()
        if row[0] in ("dev", "abstain_dev") and row[4] == "question"
    ]


def _fit_matrix(emb: np.ndarray, rels: Sequence[str], ridge: float = RIDGE) -> np.ndarray:
    y = np.stack([(np.asarray(rels) == lab).astype(np.float64) for lab in LABELS], 1)
    x = np.concatenate([emb, np.ones((len(emb), 1), dtype=np.float64)], 1)
    gram = x.T @ x + float(ridge) * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def abstain_ridge_errors(table: Dict[str, np.ndarray], Wa: np.ndarray) -> int:
    """Named dev questions the abstain ridge labels wrong. Not the probe."""
    errors = 0
    for row in _named_dev_rows():
        pred, _gap = _abstain_pred(table[row[-1]], Wa)
        if pred != row[3]:
            errors += 1
    return errors


def choose_abstain_ridge(table: Dict[str, np.ndarray]) -> Tuple[float, np.ndarray, int]:
    """Fit ``Wa`` on fit questions. Fewest named-dev errors; a tie keeps the larger ridge."""
    rows = _fit_question_rows()
    emb = np.stack([_unit(table[row[-1]]) for row in rows])
    rels = [row[3] for row in rows]
    best = None
    for ridge in ABSTAIN_RIDGE_GRID:
        Wa = _fit_matrix(emb, rels, ridge)
        errors = abstain_ridge_errors(table, Wa)
        key = (errors, -float(ridge))
        if best is None or key < best[0]:
            best = (key, float(ridge), Wa, errors)
    _key, ridge, Wa, errors = best
    return ridge, Wa, errors


def score_abstain_margin(table: Dict[str, np.ndarray], weights: dict, margin: float) -> dict:
    """False abstains, missing-relation catches, and absent-person catches at one margin.

    Answerable questions are scored on a one-relation store: the asked person
    keeps only the asked relation. That is the places→phone failure mode. A
    wrong confident label then abstains a question that had an answer.
    """
    fa = {"en": 0, "zh": 0}
    fa_n = {"en": 0, "zh": 0}
    miss = {"en": 0, "zh": 0}
    miss_n = {"en": 0, "zh": 0}
    absent = {"en": 0, "zh": 0}
    absent_n = {"en": 0, "zh": 0}
    deictic_false = {"en": 0, "zh": 0}
    deictic_n = {"en": 0, "zh": 0}
    fa_rel = {lang: {lab: 0 for lab in LABELS} for lang in ("en", "zh")}
    miss_rel = {lang: {lab: 0 for lab in LABELS} for lang in ("en", "zh")}

    def call(text: str, emb: np.ndarray, names: Sequence[str], rels: Sequence[str]) -> Optional[str]:
        keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
        return surface_abstain(
            emb, keys, rels, names, weights, query_text=text, margin=float(margin),
        )

    for _split, lang, name, rel, _role, text in _named_dev_rows():
        emb = np.asarray(table[text], dtype=np.float64)
        fa_n[lang] += 1
        if call(text, emb, [name], [rel]) is not None:
            fa[lang] += 1
            fa_rel[lang][rel] += 1
        miss_n[lang] += 1
        others = [lab for lab in LABELS if lab != rel]
        if call(text, emb, [name, name], others) == "missing_rel":
            miss[lang] += 1
            miss_rel[lang][rel] += 1
    for split, lang, name, rel, _role, text in iter_corpus():
        emb = np.asarray(table[text], dtype=np.float64)
        if split == "abstain_deic":
            deictic_n[lang] += 1
            who = DEV_NAMES[lang][0]
            if call(text, emb, [who, who, who], list(LABELS)) is not None:
                deictic_false[lang] += 1
        elif split == "abstain_absent":
            absent_n[lang] += 1
            roster = list(DEV_NAMES[lang])
            if call(text, emb, roster, [LABELS[0]] * len(roster)) == "absent_person":
                absent[lang] += 1
    return {
        "margin": float(margin),
        "false_abstain": fa,
        "false_n": fa_n,
        "missing": miss,
        "missing_n": miss_n,
        "absent": absent,
        "absent_n": absent_n,
        "deictic_false": deictic_false,
        "deictic_n": deictic_n,
        "false_by_rel": fa_rel,
        "missing_by_rel": miss_rel,
        "missing_caught": int(miss["en"] + miss["zh"]),
        "absent_caught": int(absent["en"] + absent["zh"]),
    }


def abstain_dev_curve(
    weights: Optional[dict] = None,
    table: Optional[Dict[str, np.ndarray]] = None,
) -> List[dict]:
    """Every margin on the dev pools. Does not read the probe."""
    if weights is None:
        weights = load_weights()
    if weights is None or weights.get("Wa") is None:
        raise RuntimeError("abstain ridge is missing")
    if table is None:
        table = load_corpus_table()
    return [score_abstain_margin(table, weights, margin) for margin in ABSTAIN_MARGIN_GRID]


def choose_abstain_margin(curve: Optional[Sequence[dict]] = None) -> float:
    """Largest catch among margins with no false abstain on either language.

    A point is kept only when neither language false-abstains an answerable
    dev question and neither language abstains a second-person question.
    Among those, missing-relation catches are maximized, then absent-person
    catches, then the larger margin.
    """
    if curve is None:
        curve = abstain_dev_curve()
    allowed = [
        row for row in curve
        if row["false_abstain"]["en"] == 0 and row["false_abstain"]["zh"] == 0
        and row["deictic_false"]["en"] == 0 and row["deictic_false"]["zh"] == 0
    ]
    if not allowed:
        raise RuntimeError("no abstain margin keeps both languages free of false abstains")
    best = max(allowed, key=lambda row: (row["missing_caught"], row["absent_caught"], row["margin"]))
    return float(best["margin"])


def _dev_pools(table: Dict[str, np.ndarray]) -> List[dict]:
    """Fixed pools. Gold index varies, but the draw is seeded."""
    pools = []
    rng = np.random.default_rng(0)
    for lang in ("en", "zh"):
        names = DEV_NAMES[lang]
        statements = {}
        for rel in LABELS:
            bank = DEV_S[lang][rel]
            for i, name in enumerate(names):
                text = bank[i % len(bank)].format(name=name)
                statements[(name, rel)] = np.asarray(table[text], dtype=np.float64)
        for rel in LABELS:
            for name in names:
                for tmpl in DEV_Q[lang][rel]:
                    others = [n for n in names if n != name]
                    picks = [(name, other) for other in LABELS if other != rel]
                    picks += [(n, rel) for n in others[:4]]
                    rest = [(n, r) for n in others for r in LABELS if (n, r) not in picks]
                    rng.shuffle(rest)
                    while len(picks) < 6:
                        picks.append(rest.pop())
                    picks = picks[:6]
                    gold_at = int(rng.integers(0, 7))
                    ordered = picks[:gold_at] + [(name, rel)] + picks[gold_at:]
                    pools.append({
                        "lang": lang,
                        "q": np.asarray(table[tmpl.format(name=name)], dtype=np.float64),
                        "K": np.stack([statements[pair] for pair in ordered]),
                        "rels": [pair[1] for pair in ordered],
                        "gi": gold_at,
                    })
    return pools


def _hits(pools: Sequence[dict], weights: dict, alpha: float) -> dict:
    use = dict(weights)
    use["alpha"] = float(alpha)
    total = {"all": 0, "en": 0, "zh": 0}
    count = {"all": 0, "en": 0, "zh": 0}
    for pool in pools:
        j = int(np.argmax(surface_scores(pool["q"], pool["K"], pool["rels"], use)))
        ok = int(j == pool["gi"])
        total["all"] += ok
        count["all"] += 1
        total[pool["lang"]] += ok
        count[pool["lang"]] += 1
    return {"hits": total, "n": count}


def _choose_alpha(curve: dict) -> float:
    """Largest total that does not drop either language below plain cosine."""
    base = curve[0.0]
    allowed = []
    for alpha in ALPHA_GRID:
        row = curve[alpha]
        if row["hits"]["en"] < base["hits"]["en"] or row["hits"]["zh"] < base["hits"]["zh"]:
            continue
        allowed.append(alpha)
    best = max(allowed, key=lambda alpha: (curve[alpha]["hits"]["all"], -alpha))
    return float(best)


def embed_sentences(texts: Sequence[str]) -> np.ndarray:
    """Unit sentence vectors in the stored 512-d slot. Not used by tests.

    Mean-pool ``multilingual-e5-small`` (attention mask, special tokens included),
    L2-normalize, then copy the 384-d unit vector into the leading coordinates.
    The tail is zero. Cosine matches the native model. ``tproj`` still reads 512.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    if not texts:
        return np.zeros((0, STORED_DIM), np.float32)
    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        tok = AutoTokenizer.from_pretrained(SENTENCE_MODEL, revision=SENTENCE_REVISION)
        model = AutoModel.from_pretrained(SENTENCE_MODEL, revision=SENTENCE_REVISION)
        model.float()
        model.eval()
        rows = []
        for start in range(0, len(texts), 32):
            chunk = list(texts[start:start + 32])
            batch = tok(
                chunk, padding=True, truncation=True,
                max_length=SENTENCE_MAXLEN, return_tensors="pt",
            )
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state
                if int(hidden.shape[-1]) != NATIVE_DIM:
                    raise RuntimeError(
                        f"{SENTENCE_MODEL} hidden {hidden.shape[-1]} != {NATIVE_DIM}"
                    )
                if SENTENCE_POOLING == "mean":
                    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                elif SENTENCE_POOLING == "cls":
                    vec = hidden[:, 0]
                else:
                    raise RuntimeError(f"unknown pooling {SENTENCE_POOLING}")
                vec = F.normalize(vec, p=2, dim=-1)
            native = vec.detach().cpu().numpy().astype(np.float32, copy=True)
            stored = np.zeros((native.shape[0], STORED_DIM), np.float32)
            stored[:, :native.shape[1]] = native
            rows.append(stored)
        return np.concatenate(rows, 0)
    finally:
        torch.backends.mkldnn.enabled = prev


def _embed(texts: Sequence[str]) -> np.ndarray:
    return embed_sentences(texts)


def fit_bundle(path: Path = BUNDLE) -> dict:
    """Embed new surface strings, fit the abstain ridge, keep the ranking ridge.

    The ranking map ``W`` and ``alpha`` are copied from the bundle already on
    disk. Re-embedding those old strings would move ``ask`` ret@1. New strings
    are embedded and appended. Not used by tests.
    """
    errors = corpus_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    if not path.is_file():
        raise RuntimeError(f"ranking bundle missing at {path}")
    texts = list(corpus_strings())
    with np.load(path, allow_pickle=False) as blob:
        old_texts = json.loads(str(blob["texts"].item()))
        old_emb = np.array(blob["emb"], dtype=np.float32, copy=True)
        W = np.array(blob["W"], dtype=np.float64, copy=True)
        alpha = np.float64(blob["alpha"])
        dev_alpha = np.array(blob["dev"])
        if W.shape != (STORED_DIM + 1, len(LABELS)):
            raise RuntimeError(f"ranking ridge shape {W.shape}")
    old_table = {text: old_emb[i] for i, text in enumerate(old_texts)}
    missing = [text for text in texts if text not in old_table]
    if missing:
        fresh = _embed(missing).astype(np.float32)
        for i, text in enumerate(missing):
            old_table[text] = fresh[i]
    absent_old = [text for text in old_texts if text not in texts]
    if absent_old:
        raise RuntimeError(f"surface corpus dropped {absent_old[:2]}")
    emb = np.stack([old_table[text] for text in texts]).astype(np.float32)
    table = {text: np.asarray(emb[i], dtype=np.float64) for i, text in enumerate(texts)}
    ridge, Wa, n_errors = choose_abstain_ridge(table)
    weights = {"W": W, "Wa": Wa, "alpha": float(alpha), "labels": LABELS}
    abstain_curve = [score_abstain_margin(table, weights, margin) for margin in ABSTAIN_MARGIN_GRID]
    margin = choose_abstain_margin(abstain_curve)
    payload = {
        "W": W.astype(np.float64),
        "alpha": alpha,
        "labels": np.asarray(json.dumps(list(LABELS))),
        "texts": np.asarray(json.dumps(texts)),
        "emb": emb.astype(np.float32),
        "model": np.asarray(SENTENCE_MODEL),
        "revision": np.asarray(SENTENCE_REVISION),
        "pooling": np.asarray(SENTENCE_POOLING),
        "native_dim": np.int64(NATIVE_DIM),
        "dev": dev_alpha,
        "Wa": Wa.astype(np.float64),
        "abstain_ridge": np.float64(ridge),
        "abstain_margin": np.float64(margin),
        "abstain_dev": np.asarray(json.dumps(abstain_curve)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    global _WEIGHTS
    _WEIGHTS = None
    return {
        "alpha": float(alpha),
        "abstain_ridge": ridge,
        "abstain_margin": margin,
        "abstain_errors": n_errors,
        "n_fit": len(_fit_question_rows()),
        "n_texts": len(texts),
        "n_new": len(missing),
        "curve": abstain_curve,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }


def bundle_sha256(path: Path = BUNDLE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus_table(path: Path = BUNDLE) -> Dict[str, np.ndarray]:
    """Sentence vectors stored with the fit, keyed by training string."""
    with np.load(path, allow_pickle=False) as blob:
        texts = json.loads(str(blob["texts"].item()))
        emb = np.array(blob["emb"], dtype=np.float64, copy=True)
    if emb.shape != (len(texts), STORED_DIM):
        raise RuntimeError(f"surface corpus shape {emb.shape} for {len(texts)} strings")
    return {text: emb[i] for i, text in enumerate(texts)}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the surface abstain ridge")
    parser.add_argument("--fit", action="store_true", help="rebuild question_surface.npz")
    args = parser.parse_args(argv)
    if not args.fit:
        weights = load_weights()
        if weights is None:
            print("missing bundle")
            return
        margin = weights.get("abstain_margin")
        print(f"alpha {weights['alpha']} abstain_margin {margin}")
        return
    info = fit_bundle()
    print(
        f"alpha {info['alpha']} abstain_ridge {info['abstain_ridge']} "
        f"margin {info['abstain_margin']} errors {info['abstain_errors']} "
        f"texts {info['n_texts']} new {info['n_new']} sha256 {info['sha256']}"
    )
    for row in info["curve"]:
        fa, miss, absent = row["false_abstain"], row["missing"], row["absent"]
        print(
            f"  m={row['margin']:.2f} fa {fa['en']}+{fa['zh']} "
            f"miss {miss['en']}+{miss['zh']}={row['missing_caught']} "
            f"abs {absent['en']}+{absent['zh']} "
            f"deic {row['deictic_false']['en']}+{row['deictic_false']['zh']}"
        )


if __name__ == "__main__":
    main()
