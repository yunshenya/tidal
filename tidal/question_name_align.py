"""Name alignment for flat-row ask. CPU only.

The lexical matcher in ``question_surface.name_in_text`` is whole-string: it
misses a nickname, a dropped surname, a short honorific, and pinyin, and it
cannot tell a one-character typo from a one-character absent name. The dev
bank uses the same surface (``Harols``, ``Juduth``, ``Marcas``, ``Anna Louize``,
``吴笛``, ``郑恺``, ``徐锋``) on both sides. Edit distance has no label there,
so a one-edit pair is not a match.

This module scores a surface span against the stored names with a small
logistic head. Features are orthographic (prefix, substring, token stem,
Chinese given-name shape, surname plus title) plus cross-script pinyin.
A span with none of those is not a match.

Content-token cosine from ``multilingual-e5-small`` is not a feature. On the
dev bank it sits in one band for a real two-letter nickname (``Immy`` /
``Imogen`` about 0.86), a two-letter stub of the wrong name, and an absent
name next to a stored one (``Daphne`` / ``Lorraine`` about 0.81). It cannot
cast a separate vote, and scoring it would load the encoder on ordinary asks.

A one-edit same-script pair is not a match: the dev bank labels that edit
both ways. A longer span that still contains the stored name in order is not
a new match either (``Harrold``, ``老孙悦``, ``孙悦然``).

An unmatched name in the cue's name slot is ``absent_person`` even when
another stored name is in the string. The slot is the nearest person after
the cue when there is one, otherwise the nearest person. A person here is a
title-case Latin token or a two-character Chinese span whose first character
is a surname. Indirect descriptions are not people. No enable flag.
Thresholds are fit on the abstain dev bank only. The adversarial test is not
an input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BUNDLE = Path(__file__).with_name("question_name_align.npz")
FEATURES: Tuple[str, ...] = (
    "prefix_ge3",
    "prefix_eq2",
    "substr_len",
    "token_frac",
    "pinyin_equal",
    "given_nick",
    "given_only",
    "surname_title",
    "exact_fold",
    "honorific",
)
# Single-character surnames from the Baijiaxing, simplified, plus surnames of
# the abstain dev roster that the classic list does not carry (楚). Not a list
# of people, and not fitted on the adversarial names.
_SURNAMES = frozenset(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄"
    "和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁"
    "杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍"
    "虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚"
    "程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓"
    "牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙"
    "叶幸司韶郜黎蓟薄印宿白怀蒲邰从鄂索咸籍赖卓蔺屠蒙池乔阴胥能苍双闻"
    "莘党翟谭贡劳逄姬申扶堵冉宰郦雍却璩桑桂濮牛寿通边扈燕冀郏浦尚农温"
    "别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡"
    "国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾"
    "毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓楚"
)
# Grammatical characters that also occur as rare surnames. 那 is the
# demonstrative in 那天 / 那个; treating it as a surname clears a real person
# whenever a question says "that day".
_SURNAME_SKIP = frozenset("那和于从向")
# Closed class. A span of these is not a person mention. Indirect descriptions
# (对面那位, the addressee) are not in here: those stay out of scope.
_CJK_SPLIT = frozenset("的了着过吗呢吧啊时是不没把给让与跟和就被从到往对为以而或及也很都还又才再就在向把将")
_CJK_FUNCTION = frozenset({
    "什么", "怎么", "如何", "哪里", "哪些", "哪个", "哪天", "那儿", "那里", "这里",
    "没有", "有没", "不是", "不会", "不能", "不要", "是不是", "有没有", "是否", "能否",
    "我们", "他们", "你们", "自己", "别人", "其他", "以及", "怎么", "怎样", "为何",
    "以后", "以前", "之后", "之前", "时候", "现在", "今天", "明天", "出来", "起来",
    "可以", "应该", "已经", "还是", "或者", "因为", "所以", "但是", "如果", "虽然", "然后",
    "一个", "一条", "这一", "那一", "这部", "那部", "这个", "那个", "别的",
    "名册", "值班", "写下", "写着", "回读", "该念", "哪一", "一条", "什么",
    "电话", "生日", "地方", "地点", "手机", "号码",
})
_HONOR = frozenset({"mr", "mrs", "ms", "miss", "dr"})
_NICK_PREFIX = frozenset("小阿大")
_TITLES = ("先生", "女士", "小姐")
_LATIN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_STOP_EXTRA = frozenset({
    "under", "between", "leave", "aside", "set", "not", "but", "only", "about",
    "ask", "of", "in", "into", "onto", "over", "than", "then", "this", "these",
    "his", "her", "their", "your", "you", "ours", "belongs", "belong", "filed",
    "written", "porter", "book", "distinct", "should", "would", "could", "did",
    "does", "been", "being", "were", "are", "was", "be", "a", "an", "or", "and",
    "by", "as", "at", "on", "it", "its", "if", "so", "no", "yes", "just", "also",
    "which", "what", "when", "where", "who", "whom", "whose", "how", "there",
    "here", "from", "with", "without", "within", "across", "after", "before",
    "during", "while", "because", "once", "name", "names", "file", "files",
    "read", "reads", "reading", "keep", "keeps", "held", "hold", "holds",
    "one",
})

_WEIGHTS: Optional[dict] = None
_PY_CACHE: Dict[str, str] = {}


def _cjk(ch: str) -> bool:
    return "\u4e00" <= ch <= "\u9fff"


def _has_cjk(text: str) -> bool:
    return any(_cjk(ch) for ch in text)


def _chars(text: str) -> List[str]:
    out = []
    for ch in text:
        if ch.isascii() and ch.isalpha():
            out.append(ch.lower())
        elif _cjk(ch):
            out.append(ch)
    return out


def _script(chars: Sequence[str]) -> str:
    cjk = any(_cjk(ch) for ch in chars)
    lat = any(ch.isascii() and ch.isalpha() for ch in chars)
    if cjk and not lat:
        return "cjk"
    if lat and not cjk:
        return "lat"
    if cjk and lat:
        return "mix"
    return ""


def _subseq(small: Sequence[str], big: Sequence[str]) -> bool:
    i = 0
    for ch in big:
        if i < len(small) and small[i] == ch:
            i += 1
    return i == len(small) and len(small) > 0


def _extension(span: str, name: str) -> bool:
    """The span is a longer string that still contains the stored name in order."""
    span_c, name_c = _chars(span), _chars(name)
    if not span_c or not name_c or len(span_c) <= len(name_c):
        return False
    return _subseq(name_c, span_c)


def _lev(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _prefix_len(a: Sequence[str], b: Sequence[str]) -> int:
    n = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        n += 1
    return n


def _core(span: str) -> str:
    matched = re.match(r"(?i)\s*(?:mr|mrs|ms|miss|dr)\.?\s+(.*)$", span.strip())
    if matched is None:
        return span.strip()
    return matched.group(1).strip()


def _honorific(span: str) -> bool:
    return _core(span) != span.strip()


def _given_nick(span: str, name: str) -> bool:
    name_c = [ch for ch in name if _cjk(ch)]
    span_c = [ch for ch in span if _cjk(ch)]
    if len(name_c) < 2 or len(span_c) != 2:
        return False
    if any(ch.isascii() and ch.isalpha() for ch in span):
        return False
    given = name_c[-1]
    if span_c == [given, given]:
        return True
    return span_c[0] in _NICK_PREFIX and span_c[1] == given


def _given_only(span: str, name: str) -> bool:
    name_c = [ch for ch in name if _cjk(ch)]
    span_c = [ch for ch in span if _cjk(ch)]
    if len(name_c) < 2 or span_c != [name_c[-1]]:
        return False
    return not any(ch.isascii() and ch.isalpha() for ch in span)


def _surname_title(span: str, name: str) -> bool:
    name_c = [ch for ch in name if _cjk(ch)]
    span_c = "".join(ch for ch in span if _cjk(ch))
    if not name_c or any(ch.isascii() and ch.isalpha() for ch in span):
        return False
    return any(span_c == name_c[0] + title for title in _TITLES)


def _pinyin_key(name: str) -> str:
    cached = _PY_CACHE.get(name)
    if cached is not None:
        return cached
    from pypinyin import lazy_pinyin

    key = "".join(lazy_pinyin(name, errors="ignore")).lower()
    key = "".join(ch for ch in key if "a" <= ch <= "z")
    _PY_CACHE[name] = key
    return key


def _latin_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isascii() and ch.isalpha())


def _pinyin_equal(span: str, name: str) -> bool:
    if not _has_cjk(name) or _has_cjk(span):
        return False
    latin = _latin_key(span)
    if len(latin) < 2:
        return False
    return latin == _pinyin_key(name)


def _token_frac(span: str, name: str) -> float:
    tokens = [tok.lower() for tok in re.findall(r"[A-Za-z]+", name)]
    words = [tok.lower() for tok in re.findall(r"[A-Za-z]+", _core(span))]
    if not tokens or not words:
        return 0.0

    def hits(word: str, token: str) -> bool:
        if word == token:
            return True
        return len(word) >= 3 and token.startswith(word) and len(token) > len(word)

    if not all(any(hits(word, token) for token in tokens) for word in words):
        return 0.0
    matched = sum(1 for token in tokens if any(hits(word, token) for word in words))
    return matched / len(tokens)


def _ambiguous_edit(span: str, name: str) -> bool:
    """One edit on a similar-length same-script pair. Both labels exist on dev."""
    if _given_nick(span, name) or _given_only(span, name) or _surname_title(span, name):
        return False
    if _pinyin_equal(span, name):
        return False
    span_c, name_c = _chars(_core(span)), _chars(name)
    if _script(span_c) == "" or _script(span_c) != _script(name_c):
        return False
    if abs(len(span_c) - len(name_c)) > 1:
        return False
    dist = _lev(span_c, name_c)
    return dist == 1


def _substr_len(span: str, name: str) -> int:
    span_c, name_c = _chars(_core(span)), _chars(name)
    if len(span_c) < 3 or len(span_c) > len(name_c):
        return 0
    blob = "".join(name_c)
    needle = "".join(span_c)
    if needle and needle in blob and needle != blob:
        return len(needle)
    if needle == blob:
        return len(needle)
    return 0


def _cheap(span: str, name: str) -> Dict[str, float]:
    core = _core(span)
    span_c, name_c = _chars(core), _chars(name)
    prefix = _prefix_len(span_c, name_c)
    substr = _substr_len(span, name)
    token = _token_frac(span, name)
    # A lowercase Latin span may be the folded full name (exact_fold) or
    # pinyin. It may not use prefix or substring: "one" sits inside "Lionel".
    latin_span = any(ch.isascii() and ch.isalpha() for ch in core)
    latin_name = any(ch.isascii() and ch.isalpha() for ch in name)
    if latin_span and latin_name and not any(ch.isupper() for ch in core):
        prefix = 0
        substr = 0
        token = 0.0
    # A prefix of three or more is a nickname (Harry / Harold). A prefix of
    # two is the same feature for Immy / Imogen and for a two-letter stub, so
    # it is its own bit and the dev stubs keep it from being a match.
    return {
        "prefix_ge3": 1.0 if prefix >= 3 else 0.0,
        "prefix_eq2": 1.0 if prefix == 2 else 0.0,
        "substr_len": float(substr),
        "token_frac": float(token),
        "pinyin_equal": 1.0 if _pinyin_equal(span, name) else 0.0,
        "given_nick": 1.0 if _given_nick(span, name) else 0.0,
        "given_only": 1.0 if _given_only(span, name) else 0.0,
        "surname_title": 1.0 if _surname_title(span, name) else 0.0,
        "exact_fold": 1.0 if span_c and span_c == name_c else 0.0,
        "honorific": 1.0 if _honorific(span) else 0.0,
    }


def _cheap_active(cheap: Dict[str, float]) -> bool:
    return any(cheap[key] > 0 for key in (
        "prefix_ge3", "prefix_eq2", "substr_len", "token_frac", "pinyin_equal",
        "given_nick", "given_only", "surname_title", "exact_fold",
    ))


def _stopwords() -> frozenset:
    from tidal.question_surface import _LATIN_NONNAME

    return _LATIN_NONNAME | _STOP_EXTRA | _HONOR


def _latin_spans(text: str) -> List[Tuple[int, int, str]]:
    stop = _stopwords()
    tokens = [(m.start(), m.end(), m.group()) for m in _LATIN_RE.finditer(text)]
    spans: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(tokens):
        start, end, tok = tokens[i]
        stem = tok.split("'", 1)[0].lower()
        if stem in _HONOR:
            if i + 1 < len(tokens) and re.fullmatch(r"[\s.]*", text[end:tokens[i + 1][0]]):
                spans.append((start, tokens[i + 1][1], text[start:tokens[i + 1][1]]))
                i += 2
                continue
            i += 1
            continue
        if stem in stop:
            i += 1
            continue
        j = i + 1
        while j < len(tokens):
            nxt_stem = tokens[j][2].split("'", 1)[0].lower()
            gap = text[tokens[j - 1][1]:tokens[j][0]]
            if nxt_stem in stop or nxt_stem in _HONOR or not re.fullmatch(r"[\s\-]*", gap):
                break
            j += 1
        spans.append((start, tokens[j - 1][1], text[start:tokens[j - 1][1]]))
        i = j
    return spans


def _cjk_spans(text: str) -> List[Tuple[int, int, str]]:
    spans: List[Tuple[int, int, str]] = []
    buf: List[str] = []
    start = 0
    skip = False

    def flush() -> None:
        nonlocal buf, skip
        if buf and not skip and 1 <= len(buf) <= 4:
            spans.append((start, start + len(buf), "".join(buf)))
        buf = []
        skip = False

    for i, ch in enumerate(text):
        if _cjk(ch) and ch not in _CJK_SPLIT and not skip:
            if not buf:
                start = i
            buf.append(ch)
            if len(buf) > 4:
                buf = []
                skip = True
        else:
            flush()
    flush()
    return spans


def propose_spans(text: str) -> List[Tuple[int, int, str]]:
    """Candidate mentions. Not a stored-name lookup."""
    return _latin_spans(text) + _cjk_spans(text)


def load_aligner(path: Path = BUNDLE) -> Optional[dict]:
    """Shipped head, or None when the file is absent.

    Absence is not a switch the caller sets. A checkout without the file keeps
    the lexical name list.
    """
    global _WEIGHTS
    if _WEIGHTS is not None and path == BUNDLE:
        return _WEIGHTS
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as blob:
        names = tuple(json.loads(str(blob["features"].item())))
        weights = {
            "coef": np.array(blob["coef"], dtype=np.float64, copy=True),
            "intercept": float(blob["intercept"]),
            "threshold": float(blob["threshold"]),
            "margin": float(blob["margin"]),
            "features": names,
        }
    if weights["coef"].shape != (len(FEATURES),) or names != FEATURES:
        raise RuntimeError(f"name aligner features {names} coef {weights['coef'].shape}")
    if path == BUNDLE:
        _WEIGHTS = weights
    return weights


def _featurize(span: str, name: str) -> Optional[np.ndarray]:
    """Feature row, or None when the span cannot be this name.

    The hard rejects (a longer string that still contains the name, or one
    same-script edit) are not features. Both labels exist for the edit on the
    dev bank, so the row is dropped instead of being asked to vote.
    """
    if _extension(_core(span), name) or _ambiguous_edit(span, name):
        return None
    cheap = _cheap(span, name)
    if not _cheap_active(cheap):
        return None
    return np.asarray([float(cheap[key]) for key in FEATURES], dtype=np.float64)


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        z = np.exp(-logit)
        return float(1.0 / (1.0 + z))
    z = np.exp(logit)
    return float(z / (1.0 + z))


def rank_span(span: str, names: Sequence[str], weights: dict) -> List[Tuple[str, float, float]]:
    """``(stored name, probability, logit)`` best first. Empty if nothing scores."""
    scored: List[Tuple[str, float, float]] = []
    for name in names:
        vec = _featurize(span, name)
        if vec is None:
            continue
        logit = float(vec @ weights["coef"] + weights["intercept"])
        scored.append((name, _sigmoid(logit), logit))
    scored.sort(key=lambda item: (-item[2], item[0]))
    return scored


def best_name(span: str, names: Sequence[str], weights: Optional[dict] = None) -> Optional[str]:
    """The stored name ``span`` aligns to, or None.

    The probability has to clear the threshold. When two stored names both
    score, the logit gap has to clear the margin: a sigmoid saturates, and
    the gap between a real nickname and a prefix of the wrong name lives in
    the logit.
    """
    if weights is None:
        weights = load_aligner()
    if weights is None:
        return None
    ranked = rank_span(span, [n for n in names if n], weights)
    if not ranked:
        return None
    best_p = ranked[0][1]
    if best_p < float(weights["threshold"]):
        return None
    if len(ranked) > 1 and ranked[0][2] - ranked[1][2] < float(weights["margin"]):
        return None
    return ranked[0][0]


def _lexical(text: str, names: Sequence[str]) -> List[str]:
    from tidal.question_surface import name_in_text

    found = []
    seen = set()
    for name in sorted(names, key=len, reverse=True):
        if name not in seen and name_in_text(name, text):
            seen.add(name)
            found.append(name)
    return found


def _covers_stored(span: str, names: Sequence[str]) -> bool:
    return any(_extension(span, name) or _chars(name) == _chars(span) for name in names if name)


def _cue_boxes(text: str) -> Optional[List[Tuple[int, int]]]:
    from tidal.question_surface import _cue_spans, lex_labels

    labels = lex_labels(text)
    if len(labels) != 1:
        return None
    return _cue_spans(text, labels[0])


def _gap(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def _overlaps(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _unknown_person(span: str, roster: Sequence[str]) -> bool:
    """A span that names someone and is not a stored person.

    Latin: one or two title-case tokens (``Bianca``, ``Mary Jane``). A
    lowercase word glued on (``Green-diary birthday``) is not a name, and
    neither is a closed-class word. Chinese: two characters whose first is a
    surname. ``添岁`` and ``哪一条`` are not. A span that still contains a
    stored name (``王伟强``, ``Nellie Jane``) is that shape, not a second person.
    """
    if span in _CJK_FUNCTION or _covers_stored(span, roster):
        return False
    cjk = [ch for ch in span if _cjk(ch)]
    latin = re.findall(r"[A-Za-z]+", span)
    if cjk and latin:
        return False
    if cjk:
        return len(cjk) == 2 and cjk[0] in _SURNAMES and cjk[0] not in _SURNAME_SKIP
    if not latin or len(latin) > 2:
        return False
    if not all(re.fullmatch(r"[A-Z][a-z]{2,14}", tok) for tok in latin):
        return False
    stop = _stopwords()
    return not any(tok.lower() in stop for tok in latin)


def _unknown_is_focus(
    known: Sequence[Tuple[int, int]],
    unknown: Sequence[Tuple[int, int, bool]],
    cues: Sequence[Tuple[int, int]],
) -> bool:
    """The cue's name slot is an unmatched person.

    Nearest person wins. A title-case name after the cue also wins when no
    stored name is after that cue (``Which telephone was filed for Bianca``
    puts the stored name in the previous sentence). A Chinese word after the
    cue does not, so ``记在石板`` stays a place.
    """
    if not known or not unknown or not cues:
        return False
    persons = [("k", box, False) for box in known] + [
        ("u", (start, end), latin) for start, end, latin in unknown
    ]
    for cue in cues:
        after_known = [item for item in persons if item[0] == "k" and item[1][0] >= cue[1]]
        after_latin = [
            item for item in persons if item[0] == "u" and item[2] and item[1][0] >= cue[1]
        ]
        if after_latin and not after_known:
            return True
    best = None
    for cue in cues:
        for kind, box, _latin in persons:
            gap = _gap(box, cue)
            if best is None or gap < best[0]:
                best = (gap, kind)
    return best is not None and best[1] == "u"


def diagnose(text: str, names: Sequence[Optional[str]]) -> dict:
    """Why a question matched the names it did. Used by the fit report."""
    roster = [n for n in names if isinstance(n, str) and n]
    weights = load_aligner()
    lexical = _lexical(text, roster)
    added: List[str] = []
    aligned_at: Dict[str, List[Tuple[int, int]]] = {name: [] for name in roster}
    from tidal.question_surface import name_spans

    for name in lexical:
        aligned_at.setdefault(name, []).extend(name_spans(name, text))
    unknown: List[Tuple[int, int, str]] = []
    if weights is not None:
        for start, end, span in propose_spans(text):
            if span in _CJK_FUNCTION:
                continue
            choice = best_name(span, roster, weights)
            if choice is not None:
                aligned_at.setdefault(choice, []).append((start, end))
                if choice not in lexical and choice not in added:
                    added.append(choice)
                continue
            if not _unknown_person(span, roster):
                continue
            if any(_overlaps((start, end), box) for name in lexical for box in name_spans(name, text)):
                continue
            cues = _cue_boxes(text) or []
            if any(_overlaps((start, end), cue) for cue in cues):
                continue
            unknown.append((start, end, span))
    matched = list(lexical)
    for name in added:
        if name not in matched:
            matched.append(name)
    force = False
    cues = _cue_boxes(text)
    if cues and matched and unknown:
        known_boxes = [box for name in matched for box in aligned_at.get(name, [])]
        unk_boxes = [
            (start, end, not _has_cjk(span)) for start, end, span in unknown
        ]
        if known_boxes and _unknown_is_focus(known_boxes, unk_boxes, cues):
            force = True
    return {
        "lexical": lexical,
        "added": added,
        "unknown": [(start, end, span) for start, end, span in unknown],
        "matched": [] if force else matched,
        "force_absent": force,
    }


def matched_people(text: str, names: Sequence[Optional[str]]) -> List[str]:
    """Stored people the question is about.

    Lexical hits stay. The head adds a nickname, a short honorific, a dropped
    surname, or pinyin. An unmatched name in the cue's name slot clears the
    list: the asked person is not stored.
    """
    if not text or not any(isinstance(n, str) and n for n in names):
        return []
    if not BUNDLE.is_file() and _WEIGHTS is None:
        return _lexical(text, [n for n in names if isinstance(n, str) and n])
    return diagnose(text, names)["matched"]


def _pairs_from_dev() -> List[Tuple[str, str, int]]:
    """(span, stored name, label) from the abstain dev bank. Not the test."""
    from tidal.question_abstain_dev_data import BANK as DEV

    rows: List[Tuple[str, str, int]] = []
    positive_kinds = {
        "nickname", "honorific_short", "honorific_full", "no_surname", "pinyin",
        "lowercase", "extra_space", "missing_space", "possessive",
    }
    for lang in ("en", "zh"):
        roster = list(DEV["stored"][lang])
        for name in roster:
            rows.append((name, name, 1))
        for canon, kind, surface in DEV["variants"][lang]:
            if kind not in positive_kinds:
                continue
            rows.append((surface, canon, 1))
            for other in roster:
                if other != canon:
                    rows.append((surface, other, 0))
        for similar, _kind, surface in DEV["near"][lang]:
            rows.append((surface, similar, 0))
            for other in roster:
                if other != similar:
                    rows.append((surface, other, 0))
        for absent in DEV["absent"][lang]:
            for name in roster:
                rows.append((absent, name, 0))
        if lang == "en":
            # Two-letter stubs of a stored name. A real nickname that is only
            # two letters (Immy) has to beat these; a stub is not a nickname.
            for name in roster:
                token = name.split()[0]
                if len(token) < 4:
                    continue
                stub = token[0] + token[1].lower()
                rows.append((stub, name, 0))
    # The same pair is a typo and a near-duplicate. Drop both.
    votes: Dict[Tuple[str, str], set] = {}
    for span, name, label in rows:
        votes.setdefault((span, name), set()).add(label)
    kept = []
    for span, name, label in rows:
        if votes[(span, name)] == {0, 1}:
            continue
        kept.append((span, name, label))
    return kept


def _dev_questions():
    from tidal.question_abstain_dev_data import BANK as DEV
    from tidal.question_stress import build_cases

    return build_cases(DEV)


def _score_questions(cases, weights: dict) -> dict:
    global _WEIGHTS
    _WEIGHTS = weights
    name_points = 0
    name_n = 0
    guard = 0
    guard_n = 0
    guard_rows: List[str] = []
    by_sub: Dict[str, List[int]] = {}
    name_slices = {"name_variant", "near_dup_absent", "two_names"}
    for case in cases:
        roster = [row.name for row in case.writes]
        info = diagnose(case.question, roster)
        lexical = info["lexical"]
        matched = info["matched"]
        force = info["force_absent"]
        if case.slice in name_slices:
            name_n += 1
            if case.slice == "name_variant":
                ok = int(case.canonical in matched and not force)
            elif case.slice == "near_dup_absent" or case.label == "absent_person":
                ok = int(force or not matched)
            else:
                ok = int((not force) and case.focus in matched)
            name_points += ok
            key = f"{case.slice}|{case.lang}|{case.subtype}|{case.label}"
            slot = by_sub.setdefault(key, [0, 0])
            slot[0] += ok
            slot[1] += 1
        else:
            guard_n += 1
            baseline_empty = not lexical
            reason = ""
            if baseline_empty and matched:
                reason = "added"
            elif (not baseline_empty) and (force or not matched):
                reason = "cleared"
            elif set(matched) != set(lexical):
                if case.canonical and case.canonical not in matched:
                    reason = "lost-canonical"
                elif any(name not in lexical and name != case.canonical for name in matched):
                    reason = "extra"
            if reason:
                guard += 1
                if len(guard_rows) < 12:
                    guard_rows.append(
                        f"{reason} {case.slice} {case.lang} {case.subtype} "
                        f"lex={lexical} matched={matched} unk={info['unknown']} :: {case.question}"
                    )
    return {
        "points": name_points,
        "n": name_n,
        "guard": guard,
        "guard_n": guard_n,
        "guard_rows": guard_rows,
        "by_sub": {key: (hit, n) for key, (hit, n) in by_sub.items()},
    }


def _baseline_points(cases) -> dict:
    """Lexical matcher on the same name-slice metric. No head."""
    from tidal.question_surface import name_in_text

    points = 0
    n = 0
    by_sub: Dict[str, List[int]] = {}
    for case in cases:
        if case.slice not in {"name_variant", "near_dup_absent", "two_names"}:
            continue
        roster = [row.name for row in case.writes]
        matched = [name for name in roster if name_in_text(name, case.question)]
        n += 1
        if case.slice == "name_variant":
            ok = int(case.canonical in matched)
        elif case.slice == "near_dup_absent" or case.label == "absent_person":
            ok = int(not matched)
        else:
            ok = int(case.focus in matched)
        points += ok
        key = f"{case.slice}|{case.lang}|{case.subtype}|{case.label}"
        slot = by_sub.setdefault(key, [0, 0])
        slot[0] += ok
        slot[1] += 1
    return {"points": points, "n": n, "by_sub": by_sub}


def fit_aligner(path: Path = BUNDLE) -> dict:
    """Fit the head and the threshold on the abstain dev bank. Not the test."""
    from sklearn.linear_model import LogisticRegression

    from tidal.question_stress import questions_sha256

    cases = _dev_questions()
    dev_sha = questions_sha256(cases)
    raw = _pairs_from_dev()
    xs = []
    ys = []
    used = 0
    dropped_gate = 0
    for span, name, label in raw:
        vec = _featurize(span, name)
        if vec is None:
            dropped_gate += 1
            continue
        # prefix_eq2 alone is both Immy and a two-letter stub. The stub is
        # the negative. Drop the positive so the bit stays a non-match.
        if label == 1 and float(vec[FEATURES.index("prefix_eq2")]) == 1.0:
            others = [
                float(vec[i]) for i, name_ in enumerate(FEATURES)
                if name_ not in ("prefix_eq2", "honorific")
            ]
            if not any(value > 0 for value in others):
                dropped_gate += 1
                continue
        xs.append(vec)
        ys.append(label)
        used += 1
    x = np.stack(xs)
    y = np.asarray(ys, dtype=np.int64)
    if len(set(y.tolist())) < 2:
        raise RuntimeError("name aligner fit has one class")
    clf = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500, random_state=0,
    )
    clf.fit(x, y)
    coef = np.asarray(clf.coef_[0], dtype=np.float64)
    intercept = float(clf.intercept_[0])
    base = _baseline_points(cases)
    best = None
    grid = []
    for threshold in (0.35, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9):
        for margin in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
            weights = {
                "coef": coef,
                "intercept": intercept,
                "threshold": float(threshold),
                "margin": float(margin),
                "features": FEATURES,
            }
            scored = _score_questions(cases, weights)
            row = {
                "threshold": threshold,
                "margin": margin,
                "points": scored["points"],
                "n": scored["n"],
                "guard": scored["guard"],
            }
            grid.append(row)
            if scored["guard"] != 0:
                continue
            key = (scored["points"], threshold, margin)
            if best is None or key > best[0]:
                best = (key, weights, scored)
    if best is None:
        raise RuntimeError("no name threshold keeps the non-name dev questions")
    _key, weights, scored = best
    payload = {
        "coef": weights["coef"].astype(np.float64),
        "intercept": np.float64(weights["intercept"]),
        "threshold": np.float64(weights["threshold"]),
        "margin": np.float64(weights["margin"]),
        "features": np.asarray(json.dumps(list(FEATURES))),
        "dev_sha256": np.asarray(dev_sha),
        "dev_points": np.int64(scored["points"]),
        "dev_n": np.int64(scored["n"]),
        "baseline_points": np.int64(base["points"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    global _WEIGHTS
    _WEIGHTS = None
    load_aligner(path)
    return {
        "dev_sha256": dev_sha,
        "pairs": len(raw),
        "fit_rows": used,
        "dropped_gate": dropped_gate,
        "pos": int(y.sum()),
        "neg": int((y == 0).sum()),
        "coef": {name: float(coef[i]) for i, name in enumerate(FEATURES)},
        "intercept": intercept,
        "threshold": float(weights["threshold"]),
        "margin": float(weights["margin"]),
        "points": scored["points"],
        "n": scored["n"],
        "baseline_points": base["points"],
        "by_sub": scored["by_sub"],
        "baseline_by_sub": base["by_sub"],
        "grid_top": sorted(grid, key=lambda row: (row["guard"] == 0, row["points"], row["threshold"]), reverse=True)[:8],
        "path": str(path),
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fit the name alignment head on the abstain dev bank")
    parser.add_argument("--fit", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.fit:
        weights = load_aligner()
        if weights is None:
            print("missing bundle")
            return
        print(f"threshold {weights['threshold']} margin {weights['margin']}")
        return
    info = fit_aligner()
    print(
        f"dev {info['dev_sha256']} pairs {info['pairs']} fit {info['fit_rows']} "
        f"dropped_gate {info['dropped_gate']} pos {info['pos']} neg {info['neg']}"
    )
    print(f"threshold {info['threshold']} margin {info['margin']}")
    print(f"dev name points {info['points']}/{info['n']} baseline {info['baseline_points']}/{info['n']}")
    for name, value in info["coef"].items():
        print(f"  {name} {value:.4f}")
    print(f"  intercept {info['intercept']:.4f}")
    print("subtype head vs baseline")
    for key in sorted(set(info["by_sub"]) | set(info["baseline_by_sub"])):
        hit, n = info["by_sub"].get(key, (0, 0))
        base = info["baseline_by_sub"].get(key, [0, 0])
        print(f"  {key} {hit}/{n} baseline {base[0]}/{base[1]}")


if __name__ == "__main__":
    main()
