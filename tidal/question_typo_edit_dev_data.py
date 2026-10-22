"""Dev split for a one-character typo versus an absent near-duplicate.

The abstain dev bank (sha256 26889048…) uses one surface as both an
answerable typo and an absent person: ``Harols``, ``Juduth``, ``Marcas``,
``Anna Louize``, ``吴笛``, ``郑恺``, ``徐锋``. The frozen adversarial test
does the same for ``黄蕾``, ``周滔``, and ``陈靖``. A name rule that sees
only the surface and the stored roster cannot return both labels.

This bank rebuilds that collision with new people. It is not the closed
PR #24 bank (sha256 68260c01…, n=68). Each collision kind is one edit
class that would clear some frozen typo, emitted twice: once as a typo
that should hit, once as a different person that must abstain. The
surface and the roster are the same on both labels. A longer span, and a
surface at distance 1 from two stored names, are abstain-only controls
the shipped matcher already refuses.

English anchors are ``spikenard`` / ``cimbalom``. Chinese anchors are
``蠃`` / ``鼏`` / ``爟``. The matcher does not import this module. No rule
is fit here. The frozen test is not an input.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

# Collision kinds are emitted twice, once as match and once as abstain,
# with the same surface and the same roster. The closer is not a feature
# of the name decision.
_COLLISION = frozenset({
    "collision_adj_sd",
    "collision_adj_mn",
    "collision_adj_as",
    "collision_vowel_ae",
    "collision_vowel_oa",
    "collision_del",
    "collision_homo",
    "collision_nose",
    "collision_diffpy",
    "collision_contain",
})

_Spec = Tuple[str, str, str, Tuple[str, ...], Tuple[str, ...]]

_EN = ("docket", "folio", "quire")
_ZH = ("deng", "ruo", "zhi", "ye")

# kind, lang, surface, roster, closers.
# The first roster name is the stored person the surface is one edit from,
# except ``tie_en`` / ``tie_zh``, where the first two names are both one edit.
_SPECS: Tuple[_Spec, ...] = (
    # s↔d is the abstain-dev pair (Harols / Harold). Adjacent, and on the
    # denylist that would spare that pair. Both labels still exist.
    ("collision_adj_sd", "en", "Tressis", ("Tressid", "Quenby", "Islwyn"), _EN),
    # m↔n is Willian / Anm / Mary Jame, and it is not on that denylist.
    ("collision_adj_mn", "en", "Femwick", ("Fenwick", "Ostara", "Bryony"), _EN),
    # a↔s is Patricis. Adjacent, not on the denylist.
    ("collision_adj_as", "en", "Danils", ("Danila", "Torquil", "Morwen"), _EN),
    # a↔e is Robart / Catharine, and also the absent names Williem / Catherina.
    ("collision_vowel_ae", "en", "Baltram", ("Beltram", "Pryderi", "Elspeth"), _EN),
    # a↔o is Margoret, and also the absent name Mary Jone.
    ("collision_vowel_oa", "en", "Rasmund", ("Rosmund", "Idris", "Lowri"), _EN),
    # A deletion is Patrici. Accepting it also accepts this absent name.
    ("collision_del", "en", "Hartly", ("Hartley", "Ciaran", "Bronwen"), _EN),
    # Same pinyin and a shared IDS operand (璐 / 潞 share 路). 王玮 / 刘样
    # are this shape, and so are 周滔 / 陈靖, which the frozen file labels absent.
    ("collision_homo", "zh", "郗潞", ("郗璐", "盖宏", "蒯锦"), _ZH),
    # Same pinyin, no shared operand (虹 / 宏). 黄蕾 / 赵悯 are this shape,
    # and 黄蕾 is also an absent near-duplicate.
    ("collision_nose", "zh", "盖虹", ("盖宏", "郗璐", "逯媛"), _ZH),
    # Pinyin differs, shared operand (绵 / 锦 share 帛). 张疆 / 张强 is this shape.
    ("collision_diffpy", "zh", "蒯绵", ("蒯锦", "盖宏", "郗璐"), _ZH),
    # The surface character is an IDS operand of the stored character
    # (爰 inside 媛). 李那 / 李娜 is this shape.
    ("collision_contain", "zh", "逯爰", ("逯媛", "蒯锦", "盖虹"), _ZH),
    # Two stored names at distance 1. A uniqueness rule abstains. The
    # shipped matcher already abstains.
    ("tie_en", "en", "Rasmund", ("Rosmund", "Resmund", "Tanwen"), _EN),
    ("tie_zh", "zh", "郗潞", ("郗璐", "郗露", "盖宏"), _ZH),
    ("hyphen", "en", "Fenwick-Bryn", ("Fenwick", "Ostara", "Bryony"), _EN),
    ("space", "en", "Fenwick Bryn", ("Fenwick", "Ostara", "Bryony"), _EN),
    ("extension", "en", "Fenwicko", ("Fenwick", "Ostara", "Bryony"), _EN),
    ("prefix", "zh", "老郗璐", ("郗璐", "盖宏", "蒯锦"), _ZH),
    ("middle", "zh", "郗小璐", ("郗璐", "盖宏", "蒯锦"), _ZH),
    ("superstring", "zh", "郗璐然", ("郗璐", "盖宏", "蒯锦"), _ZH),
)

_EN_FRAME = {
    "docket": "spikenard cimbalom docket names {who} under glyph{mark}.",
    "folio": "The spikenard cimbalom folio keeps {who} at glyph{mark}.",
    "quire": "spikenard quire for {who} sits on the cimbalom glyph{mark}.",
}
_ZH_FRAME = {
    "deng": "蠃簿登记{who}于鼏页爟{mark}。",
    "ruo": "若蠃簿载{who}，鼏页未誊，见爟{mark}。",
    "zhi": "{who}置蠃鼏页，见爟{mark}。",
    "ye": "{who}耶，蠃鼏先检，见爟{mark}。",
}


@dataclass(frozen=True)
class TypoEditRow:
    kind: str
    lang: str
    label: str
    closer: str
    surface: str
    roster: Tuple[str, ...]
    question: str


def _frame(lang: str, closer: str, who: str, mark: str) -> str:
    template = _EN_FRAME[closer] if lang == "en" else _ZH_FRAME[closer]
    return template.format(who=who, mark=mark)


def build_rows() -> Tuple[TypoEditRow, ...]:
    rows = []
    n = 0
    for kind, lang, surface, roster, closers in _SPECS:
        labels = ("match", "abstain") if kind in _COLLISION else ("abstain",)
        for label in labels:
            for closer in closers:
                n += 1
                question = _frame(lang, closer, surface, f"{n:02d}")
                rows.append(TypoEditRow(
                    kind, lang, label, closer, surface, roster, question,
                ))
    questions = [row.question for row in rows]
    if len(questions) != len(set(questions)):
        raise RuntimeError("duplicate typo-edit question")
    if not 80 <= len(rows) <= 100:
        raise RuntimeError(f"typo-edit dev bank has {len(rows)} rows")
    return tuple(rows)


def questions(rows: Sequence[TypoEditRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    return tuple(row.question for row in rows)


def questions_sha256(rows: Sequence[TypoEditRow] | None = None) -> str:
    blob = "\n".join(questions(rows)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def kind_counts(rows: Sequence[TypoEditRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.kind] = counts.get(row.kind, 0) + 1
    return counts


def label_counts(rows: Sequence[TypoEditRow] | None = None) -> Dict[str, int]:
    rows = build_rows() if rows is None else rows
    counts: Dict[str, int] = {}
    for row in rows:
        counts[row.label] = counts.get(row.label, 0) + 1
    return counts


def names_used(rows: Sequence[TypoEditRow] | None = None) -> Tuple[str, ...]:
    rows = build_rows() if rows is None else rows
    seen = []
    bag = set()
    for row in rows:
        for name in (row.surface, *row.roster):
            if name not in bag:
                bag.add(name)
                seen.append(name)
    return tuple(seen)
