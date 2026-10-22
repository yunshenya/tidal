"""Dev split for a CJK name that grows by one syllable versus a verb glued on.

The abstain dev bank (sha256 26889048…) has no row of the shape
"stored name + one content character + particle". Superstrings there are
closed by ``，`` (``值班簿若写的是孙悦然，``). This bank is the contrast those
rows do not provide.

A longer personal name (``孟峤强的``, ``姚岑生的话``, ``汪柘儿呢``) must not
match the stored two-character name. A verb or time word in the same slot
(``孟峤选的``, ``邵岚报过``, ``孟峤列出的``, ``田栖平时``, ``姚岑生过``) must.

Closers are ``的`` / ``的话`` / ``呢`` / ``吧`` / ``吗`` / ``啊`` / ``时``
plus the aspect particles ``过`` / ``了`` / ``着``. ``的话`` is one closer
among those, not a template. The frozen adversarial wording
``名下是{who}的话`` is not in this file.

Names and sentences are disjoint from the probe, the surface cue corpus,
the abstain dev bank, and the frozen adversarial test. The matcher does not
import this module.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence, Tuple

# Two-character roster. Surnames are not the probe, surface, dev, or test
# surnames, so a stored name cannot collide with those banks.
NAMES: Tuple[str, ...] = (
    "孟峤", "邵岚", "姚岑", "汪柘", "戴嵊", "夏芮", "田栖", "董馥",
)

# Characters absent from the probe, the surface corpus, and both abstain
# banks. Two of them are appended to every question so token Jaccard is not
# carried by shared function characters. Not a list of people.
_MARK = "矾硼镍钴钛钨铬汞硫碱釉砂坩钼篦铰辘樟藤驿漆栲氘氚氙氡镓锗砷硒溴氪铷锶锆铌锑碲碘铯钡镧铈镨钕钐铕钆铽镝钬铒铥镱镥铪钽铼锇铱铂铊铋钋砹"

Row = Tuple[str, str, str, str, str]  # cls, name, tail, kind, question


def _mark(i: int) -> str:
    n = len(_MARK)
    return _MARK[i % n] + _MARK[(i + 17) % n]


def _render(kind: str, span: str, mark: str) -> str:
    if kind == "poss":
        return f"矾硼匣侧记着{span}呼号，{mark}页没抄。"
    if kind == "dehua":
        return f"若册页写的是{span}，先翻{mark}再开口。"
    if kind == "ne":
        return f"{span}，{mark}罐里的呼号还空着。"
    if kind == "ba":
        return f"门口条子就写{span}，年庚格仍空着，参见{mark}。"
    if kind == "ma":
        return f"你点名的是{span}，站序还没登进{mark}。"
    if kind == "a":
        return f"广播点到{span}，值日去{mark}旁取那一串。"
    if kind == "shi":
        return f"轮到{span}，只查{mark}，别的先放下。"
    if kind == "vde":
        return f"把{span}站序从{mark}侧取出来。"
    if kind == "vguo":
        return f"{span}一次呼号，夹在{mark}页下。"
    if kind == "vle":
        return f"{span}呼号就封存，见{mark}。"
    if kind == "vne":
        return f"{span}，{mark}页还没抄。"
    if kind == "vba":
        return f"门口条子写{span}，参见{mark}。"
    if kind == "vma":
        return f"你问的是{span}，站序在{mark}。"
    if kind == "va":
        return f"广播里说{span}，值日去{mark}旁看。"
    if kind == "aspect":
        return f"{span}之后才把年庚写进{mark}。"
    if kind == "adv":
        return f"{span}把呼号留在{mark}侧。"
    raise RuntimeError(f"unknown frame {kind}")


def _add(rows: List[Row], cls: str, name: str, tail: str, kind: str) -> None:
    span = name + tail
    question = _render(kind, span, _mark(len(rows)))
    rows.append((cls, name, tail, kind, question))


def build_rows() -> Tuple[Row, ...]:
    """Deterministic rows. No draw. The frozen test is not an input."""
    rows: List[Row] = []
    names = NAMES
    # Name-final syllables with no one-character verb reading. Two closers
    # each, rotating through 的 / 的话 / 呢 / 吧 / 吗 / 啊.
    finals = (
        "然", "子", "儿", "伟", "芳", "杰", "宇", "婷", "博", "轩",
        "玲", "娜", "洋", "涛", "凯", "燕", "斌", "迪", "悦", "宁",
        "峰", "磊", "敏", "静", "丽", "文", "华", "军", "安",
    )
    nominal = ("的", "的话", "呢", "吧", "吗", "啊")
    nominal_kind = {"的": "poss", "的话": "dehua", "呢": "ne", "吧": "ba", "吗": "ma", "啊": "a"}
    for i, final in enumerate(finals):
        for k in range(2):
            closer = nominal[(i + k) % len(nominal)]
            _add(rows, "longer_name", names[(i + k) % len(names)], final + closer, nominal_kind[closer])
    # 强 and 生 are also name-final. Every nominal closer is a longer name.
    # Aspect on these two is a verb and is added below, not here.
    for i, final in enumerate(("强", "生")):
        for j, closer in enumerate(nominal):
            _add(
                rows, "longer_name", names[(i + j) % len(names)],
                final + closer, nominal_kind[closer],
            )
    # 平的 / 平呢 / 平吗 are a longer name on 田栖. 田栖平时 is the time word
    # below, so the same stored name has both labels and the next character
    # is what separates them.
    for closer, kind in (("的", "poss"), ("呢", "ne"), ("吗", "ma")):
        _add(rows, "longer_name", names[6], "平" + closer, kind)
    # 时 after a name-final syllable that does not make a time word.
    for i, final in enumerate(("然", "强", "生", "伟", "芳", "杰", "宇", "婷")):
        _add(rows, "longer_name", names[i % len(names)], final + "时", "shi")

    # One-character predicates. 的 / 过 / 了 keep the stored name.
    verbs = "选报说列提认换给改存要留写看找问用做打拿来去想"
    verb_kind = {"的": "vde", "过": "vguo", "了": "vle", "呢": "vne", "吧": "vba", "吗": "vma", "啊": "va"}
    core = "选报说列提认换给改存要留"
    extra = "写看找问用做打拿来去想"
    for i, verb in enumerate(core):
        for j, closer in enumerate(("的", "过", "了")):
            _add(rows, "verb", names[(i + j) % len(names)], verb + closer, verb_kind[closer])
    for i, verb in enumerate(extra):
        for j, closer in enumerate(("的", "过")):
            _add(rows, "verb", names[(i + j) % len(names)], verb + closer, verb_kind[closer])
    for verb, closer in (("选", "呢"), ("报", "吧"), ("要", "吗"), ("留", "啊")):
        _add(rows, "verb", names[0], verb + closer, verb_kind[closer])
    # Two content syllables, then 的. The character after the name is not a
    # particle (列出, 写下, 说出, 找出, 存下). 给过的 is a predicate plus 过.
    for i, tail in enumerate(("列出的", "写下的", "说出的", "找出的", "存下的", "给过的")):
        _add(rows, "verb", names[i % len(names)], tail, "vde")

    # Near-misses: the extra syllable is common at the end of a personal name,
    # and the attachment is still a verb or a time word.
    for i, (final, closer) in enumerate(
        (
            ("生", "过"), ("生", "了"), ("生", "着"),
            ("强", "过"), ("强", "了"), ("强", "着"),
        )
    ):
        for j in range(2):
            _add(rows, "near_miss", names[(i + j) % len(names)], final + closer, "aspect")
    for i, word in enumerate(("平时", "当时", "临时", "有时", "准时", "这时", "那时")):
        for j in range(2):
            _add(rows, "near_miss", names[(i + j) % len(names)], word, "adv")
    # Same person as 田栖平的. 平时 is a time word; 平的 is a longer name.
    _add(rows, "near_miss", names[6], "平时", "adv")
    namelike = (
        ("定", "的"), ("定", "过"), ("定", "了"), ("定", "呢"),
        ("立", "的"), ("立", "了"), ("立", "过"),
        ("成", "的"), ("成", "了"),
        ("明", "的"), ("明", "了"),
        ("记", "的"), ("记", "过"),
        ("念", "的"), ("念", "过"),
        ("开", "的"), ("开", "过"),
        ("正", "的"), ("正", "了"),
        ("清", "的"), ("清", "了"),
        ("发", "的"), ("发", "了"),
        ("通", "的"), ("通", "过"),
    )
    for i, (verb, closer) in enumerate(namelike):
        kind = {"的": "vde", "过": "vguo", "了": "vle", "呢": "vne"}[closer]
        _add(rows, "near_miss", names[i % len(names)], verb + closer, kind)

    spans = [name + tail for _cls, name, tail, _kind, _q in rows]
    if len(spans) != len(set(spans)):
        raise RuntimeError("duplicate name+tail in the CJK split")
    questions = [row[-1] for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate CJK-split question")
    if len(verbs) != len(set(verbs)):
        raise RuntimeError("duplicate verb")
    return tuple(rows)


def questions(rows: Sequence[Row] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row[-1] for row in rows)


def questions_sha256(rows: Sequence[Row] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def class_counts(rows: Sequence[Row] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for cls, _name, _tail, _kind, _q in rows:
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def closer_counts(rows: Sequence[Row] | None = None) -> Dict[str, Dict[str, int]]:
    """How many rows of each class use each closer.

    The closer is the particle the extra syllable is judged by: ``的话`` stays
    one closer, and a compound such as ``列出的`` is reported as ``出…``.
    """
    rows = build_rows() if rows is None else rows
    out: Dict[str, Dict[str, int]] = {}
    for cls, _name, tail, _kind, _q in rows:
        if tail.endswith("的话"):
            closer = "的话"
        elif len(tail) == 2:
            closer = tail[1]
        else:
            closer = tail[1] + "…"
        slot = out.setdefault(cls, {})
        slot[closer] = slot.get(closer, 0) + 1
    return out


KEEP = frozenset({"verb", "near_miss"})
DROP = frozenset({"longer_name"})


def label_of(cls: str) -> str:
    if cls in KEEP:
        return "keep"
    if cls in DROP:
        return "drop"
    raise RuntimeError(f"unknown class {cls}")


def iter_names() -> Iterable[str]:
    return NAMES
