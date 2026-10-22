"""A bare locative is not a places ask. The cue list does not grow.

``place`` / ``placed`` / ``placing`` (not ``place list``), ``放在哪里``,
``往哪里``, ``朝哪里``, and ``位置先`` drop a places label when the question
has no unique cue. ``where`` alone, ``在哪里``, ``位置是哪些``, and
``place a call`` do not take that path. The margin stays 0.1. The frozen
adversarial scores are not a gate.
"""
import ast
from pathlib import Path

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_bare_dev_data import NAMES, build_rows, kind_counts, questions_sha256
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import questions as cjk_questions
from tidal.question_cue_dev_data import NAMES as CUE_NAMES
from tidal.question_cue_dev_data import questions as cue_questions
from tidal.question_heldout import load_text_table
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, build_cases, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD, bank_strings, questions as indirect_questions
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_nickname_dev_data import questions as nick_questions
from tidal.question_stress import build_cases as build_stress
from tidal.question_stress import case_strings
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    LABELS,
    LEX_RIDGE_GAP,
    STORED_DIM,
    _abstain_pred,
    bare_locative,
    corpus_strings,
    cue_set_aside,
    lex_labels,
    load_corpus_table,
    load_weights,
    score_abstain_margin,
    surface_abstain,
)

# sha256 of the question list only. A wording change moves this.
BARE_DEV_SHA = "e24bda9007a41912ce6bdcd5213baf088447437568b7ffa6b7b6a291bff4bfd6"


def _typo_questions():
    src = Path("tests/test_question_indirect_dev.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == "_TYPO_DEV_QUESTIONS":
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError("typo dev questions missing")


def test_bare_dev_sha_and_labels():
    rows = build_rows()
    assert questions_sha256(rows) == BARE_DEV_SHA
    assert len(rows) == 25
    assert kind_counts(rows) == {"locative": 10, "guarded": 5, "keep": 10}
    for row in rows:
        assert row.lang in ("en", "zh")
        assert row.rel in LABELS
        lex = lex_labels(row.question)
        bare = bare_locative(row.question)
        if row.kind == "locative":
            assert bare is True, row.question
            assert lex == (), row.question
        elif row.kind == "guarded":
            assert bare is True, row.question
            assert lex == (row.rel,), row.question
            assert cue_set_aside(row.question, row.rel) is False, row.question
        else:
            assert row.kind == "keep"
            assert bare is False, row.question


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_bare_dev_is_disjoint():
    rows = build_rows()
    qs = [row.question for row in rows]
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "abstain_dev": case_strings(build_stress(DEV_BANK)),
        "test": case_strings(build_stress(TEST_BANK)),
        "cjk": cjk_questions(),
        "nick": nick_questions(),
        "indirect": bank_strings(),
        "indirect_q": indirect_questions(),
        "typo": _typo_questions(),
        "cue": cue_questions(),
    }
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(question in bag for question in qs), name
        best = max(_jaccard(a, b) for a in qs for b in texts)
        assert best < 0.50, (name, best)
    banned = set(CJK_NAMES)
    banned.update(CUE_NAMES["en"])
    banned.update(CUE_NAMES["zh"])
    for lang in ("en", "zh"):
        for bank in (PEOPLE, ABSENT, FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
            banned.update(bank[lang])
        banned.update(DEV_BANK["stored"][lang])
        banned.update(DEV_BANK["absent"][lang])
        banned.update(TEST_BANK["stored"][lang])
        banned.update(TEST_BANK["absent"][lang])
    for row in nick_rows():
        banned.update(row.roster)
    for group in (SELF, OTHER, THIRD, INDIRECT_ABSENT):
        banned.update(group.values())
    for name in NAMES["en"] + NAMES["zh"]:
        assert name not in banned


def _weights(phone: float, birthday: float, places: float) -> dict:
    wide = np.zeros((STORED_DIM + 1, 3), np.float64)
    wide[0, 0] = phone
    wide[0, 1] = birthday
    wide[0, 2] = places
    return {
        "Wa": wide,
        "abstain_margin": 0.1,
        "W": wide,
        "alpha": 0.0,
        "labels": LABELS,
    }


def _ask(text, rels, names, weights):
    query = np.zeros(STORED_DIM, np.float64)
    query[0] = 1.0
    keys = np.zeros((len(names), STORED_DIM), np.float64)
    return surface_abstain(query, keys, rels, names, weights, query_text=text)


def test_a_bare_locative_drops_places_and_keeps_the_margin():
    text = "Sibyl's year-mark sits where the vestry has it placed, brontide pschent."
    assert bare_locative(text) is True
    assert lex_labels(text) == ()
    # Places leads birthday by 0.13. Birthday then leads phone by 0.19.
    weights = _weights(0.16, 0.35, 0.48)
    assert _ask(text, ["birthday"], ["Sibyl"], weights) is None
    assert _ask(text, ["phone"], ["Sibyl"], weights) == "missing_rel"
    assert _ask(text, ["places"], ["Sibyl"], weights) == "missing_rel"
    # The replacement lead is under 0.1. Do not abstain just to force the row.
    weak = _weights(0.26, 0.30, 0.40)
    assert _ask(text, ["phone"], ["Sibyl"], weak) is None
    # The ridge already says phone. Dropping places does not change that call.
    phone = _weights(0.70, 0.05, 0.20)
    assert _ask(text, ["phone"], ["Sibyl"], phone) is None
    assert _ask(text, ["birthday", "places"], ["Sibyl", "Sibyl"], phone) == "missing_rel"

    zh = "晏书又长一岁，柜侧执事该把那道印放在哪里，鼗鼙镛钲。"
    assert bare_locative(zh) is True
    assert _ask(zh, ["birthday"], ["晏书"], weights) is None
    slot = "夜里先找郗宁的声音条，位置先搁下，柜侧翻哪一页，鼗镛。"
    # Places leads phone by 0.004. Phone then leads birthday by 0.438.
    torn = _weights(0.477, 0.039, 0.481)
    assert _ask(slot, ["birthday", "places"], ["郗宁", "郗宁"], torn) == "missing_rel"
    assert _ask(slot, ["phone"], ["郗宁"], torn) is None


def test_place_a_call_and_a_places_cue_are_not_dropped():
    assert LEX_RIDGE_GAP == 0.15
    assert ABSTAIN_MARGIN == 0.1
    call = "Place a call for Sibyl and read those digits aloud, was-sceptre scarab."
    assert bare_locative(call) is True
    assert lex_labels(call) == ("phone",)
    assert cue_set_aside(call, "phone") is False
    # Places leads by 0.12, under the yield gap. The phone cue still wins.
    weights = _weights(0.0, 0.0, 0.12)
    assert _ask(call, ["phone"], ["Sibyl"], weights) is None
    assert _ask(call, ["places"], ["Sibyl"], weights) == "missing_rel"
    voice = "To place a call through to Ingrid, what digits leave the booth?"
    assert bare_locative(voice) is True
    assert lex_labels(voice) == ("phone",)
    shipped = load_weights()
    table = load_corpus_table()
    pred, gap = _abstain_pred(table[voice], shipped["Wa"])
    assert pred == "phone" and gap > LEX_RIDGE_GAP
    assert surface_abstain(
        table[voice], np.zeros((1, STORED_DIM)), ["phone"], ["Ingrid"], shipped, query_text=voice,
    ) is None
    assert surface_abstain(
        table[voice], np.zeros((1, STORED_DIM)), ["places"], ["Ingrid"], shipped, query_text=voice,
    ) == "missing_rel"

    text = "Those other records aren't the one I want. What places did Ravi store?"
    assert bare_locative(text) is False
    assert lex_labels(text) == ("places",)
    # Phone leads by 0.12, under the yield gap. The places cue still wins.
    narrow = _weights(0.12, 0.0, 0.0)
    assert _ask(text, ["places"], ["Ravi"], narrow) is None
    assert _ask(text, ["phone"], ["Ravi"], narrow) == "missing_rel"

    handoff = "卞珩的交接要放在哪里，执事照着念，鼗鼙镛。"
    assert bare_locative(handoff) is True
    assert lex_labels(handoff) == ("places",)
    # The places cue agrees with a confident places ridge, so the slot is not dropped.
    agreed = _weights(0.0, 0.0, 1.0)
    assert _ask(handoff, ["places"], ["卞珩"], agreed) is None
    assert _ask(handoff, ["phone"], ["卞珩"], agreed) == "missing_rel"


def test_a_places_question_worded_with_where_or_position_keeps_places():
    where = "Where ought Quince's courier to stand, earliest halt before the later one, clinquant pschent."
    listed = "Leander left a place list on the inner card, uraeus sistrum."
    which = "汲棠按先后点过的位置是哪些，钟楼侧柜里那本蓝册子鼗鼙镛钲。"
    appear = "替卞珩露面的那一站，按先来后到该在哪里，鼗鼙镛钲。"
    side = "郗宁平时窝在哪一带，口述次序不要换，鼗钲。"
    for text in (where, listed, which, appear, side):
        assert bare_locative(text) is False, text
    weights = _weights(0.10, 0.00, 0.80)
    for text, name in ((where, "Quince"), (listed, "Leander"), (which, "汲棠"), (appear, "卞珩")):
        assert _ask(text, ["places"], [name], weights) is None
        assert _ask(text, ["phone"], [name], weights) == "missing_rel"


def test_live_places_asks_are_not_bare_locatives():
    """Probe and surface places questions keep the places label.

    A bare locative with no unique cue on those banks is a phone question
    the ridge already calls phone, or it does not occur.
    """
    weights = load_weights()
    assert weights is not None and weights.get("Wa") is not None
    for text in corpus_strings():
        if not bare_locative(text):
            continue
        assert lex_labels(text) == ("phone",), text
        assert cue_set_aside(text, "phone") is False
    surface_kept = (
        "Where should a runner show up",
        "Where does the night shift go",
        "Where should the courier stand",
        "是哪里",
        "在哪里",
    )
    surface_hits = {prefix: 0 for prefix in surface_kept}
    for text in corpus_strings():
        for prefix in surface_kept:
            if prefix in text:
                surface_hits[prefix] += 1
                assert bare_locative(text) is False, text
    assert all(surface_hits.values()), surface_hits
    probe_kept = ("place list", "位置是哪些", "Remind me where")
    probe_hits = {prefix: 0 for prefix in probe_kept}
    for text in probe_strings():
        for prefix in probe_kept:
            if prefix in text:
                probe_hits[prefix] += 1
                assert bare_locative(text) is False, text
    assert all(probe_hits.values()), probe_hits

    probe, _meta = load_text_table()
    free = []
    for case in build_cases():
        text = case.question
        if text not in probe or not bare_locative(text):
            continue
        lex = lex_labels(text)
        if len(lex) == 1:
            continue
        pred, _gap = _abstain_pred(probe[text], weights["Wa"])
        assert pred != "places", (text, pred)
        assert case.rel == "phone", text
        free.append(text)
    assert free


def test_frozen_slices_outside_bare_cue_do_not_take_the_drop():
    for case in build_stress(TEST_BANK):
        if case.slice == "bare_cue":
            continue
        if bare_locative(case.question):
            assert len(lex_labels(case.question)) == 1, case.question
    # The places phantom on the frozen bare rows is this pattern. A deferred
    # 位置 that is not 位置先, and that has no other slot, stays on the ridge.
    assert bare_locative("Where should the year-older mark for Mary Jane be placed?")
    assert bare_locative("陈静长一岁的标记该放在哪里？")
    assert bare_locative("想用声音找到赵敏，位置先别查，该看哪一条记录？")
    deferred = "先别标张强的位置，满岁在哪一天？"
    assert bare_locative(deferred) is False
    assert lex_labels(deferred) == ()


def test_surface_dev_curve_at_the_shipped_margin_is_unchanged():
    weights = load_weights()
    table = load_corpus_table()
    row = score_abstain_margin(table, weights, 0.1)
    assert row["false_abstain"] == {"en": 0, "zh": 0}
    assert row["missing"] == {"en": 188, "zh": 189}
    assert row["absent"] == {"en": 36, "zh": 36}
    assert row["deictic_false"] == {"en": 0, "zh": 0}
    low = score_abstain_margin(table, weights, 0.05)
    assert low["false_abstain"] == {"en": 1, "zh": 0}
    assert low["deictic_false"] == {"en": 0, "zh": 0}
    zero = score_abstain_margin(table, weights, 0.0)
    assert zero["false_abstain"] == {"en": 1, "zh": 0}


def test_surface_does_not_import_the_bare_dev_bank():
    blob = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    assert "question_bare_dev" not in blob
    assert "enable_abstain" not in blob
    assert "use_abstain" not in blob
