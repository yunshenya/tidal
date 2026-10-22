"""Residual misleading cues stay on the shipped abstain path.

A unique cue the ridge agrees with still abstains, including when the gap is
under 0.1. A dismissed cue still uses the next label and the 0.1 margin.
The new bank is disjoint from the frozen splits. The matcher does not import
it. Margins stay 0.1 and 0.15.
"""
import ast
from pathlib import Path

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial import load_table
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_bare_dev_data import NAMES as BARE_NAMES
from tidal.question_bare_dev_data import questions as bare_questions
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import questions as cjk_questions
from tidal.question_cue_dev_data import NAMES as CUE_NAMES
from tidal.question_cue_dev_data import questions as cue_questions
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD, bank_strings, questions as indirect_questions
from tidal.question_name_align import diagnose
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_nickname_dev_data import questions as nick_questions
from tidal.question_residual_dev_data import NAMES, build_rows, kind_counts, questions_sha256
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
    _abstain_pred_dropping,
    corpus_strings,
    cue_set_aside,
    lex_labels,
    load_weights,
    surface_abstain,
)

# sha256 of the question list only. A wording change moves this.
RESIDUAL_DEV_SHA = "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5"


def _typo_questions():
    src = Path("tests/test_question_indirect_dev.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == "_TYPO_DEV_QUESTIONS":
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError("typo dev questions missing")


def test_residual_dev_sha_and_labels():
    rows = build_rows()
    assert questions_sha256(rows) == RESIDUAL_DEV_SHA
    assert len(rows) == 100
    assert sum(row.lang == "en" for row in rows) == 50
    assert sum(row.lang == "zh" for row in rows) == 50
    assert kind_counts(rows) == {
        "container_ans": 16,
        "container_miss": 12,
        "keep_ans": 8,
        "keep_miss": 8,
        "manner_ans": 4,
        "manner_miss": 4,
        "novoice_ans": 4,
        "novoice_miss": 4,
        "aside_ans": 8,
        "aside_miss": 8,
        "genuine_miss": 8,
        "genuine_ans": 8,
        "outer_ans": 8,
    }
    for row in rows:
        info = diagnose(row.question, list(NAMES[row.lang]))
        who = [name for name in NAMES[row.lang] if name in row.question]
        assert who == info["matched"], (row.question, info)
        assert info["force_absent"] is False
        lex = lex_labels(row.question)
        aside = bool(len(lex) == 1 and cue_set_aside(row.question, lex[0]))
        if row.kind.startswith("aside"):
            assert len(lex) == 1 and aside and lex[0] != row.rel, row.question
        elif row.kind.startswith(("container", "keep", "manner", "novoice")):
            assert lex == ("places",) and aside is False, row.question
        else:
            assert row.kind.startswith(("genuine", "outer"))
            assert lex == (row.rel,) and aside is False, row.question


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_residual_dev_is_disjoint():
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
        "bare": bare_questions(),
    }
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(question in bag for question in qs), name
        best = max(_jaccard(a, b) for a in qs for b in texts)
        assert best < 0.50, (name, best)
    banned = set(CJK_NAMES)
    banned.update(CUE_NAMES["en"])
    banned.update(CUE_NAMES["zh"])
    banned.update(BARE_NAMES["en"])
    banned.update(BARE_NAMES["zh"])
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


def _ask(text, rels, names, weights, margin=None):
    query = np.zeros(STORED_DIM, np.float64)
    query[0] = 1.0
    keys = np.zeros((len(names), STORED_DIM), np.float64)
    return surface_abstain(
        query, keys, rels, names, weights, query_text=text, margin=margin,
    )


def test_a_tiny_agreeing_cue_still_abstains_and_the_margin_stays():
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    text = "Which locations folder holds the voice route for Godfrey?"
    assert lex_labels(text) == ("places",)
    assert cue_set_aside(text, "places") is False
    # Places leads phone by 0.027. The cue agrees, so the margin is not consulted.
    weights = _weights(0.500, 0.0, 0.527)
    assert _ask(text, ["phone"], ["Godfrey"], weights) == "missing_rel"
    assert _ask(text, ["places"], ["Godfrey"], weights) is None

    # Dismissed birthday. Places then leads phone by 0.083. Phone is stored.
    # The same lead with places absent is the missed abstain. One margin cannot
    # separate 0.070 from 0.083.
    feast = "Ignore the birthday feast and say how to get through to Godfrey."
    assert lex_labels(feast) == ("birthday",)
    assert cue_set_aside(feast, "birthday") is True
    close = _weights(0.317, 0.600, 0.400)
    assert _ask(feast, ["phone"], ["Godfrey"], close) is None
    assert _ask(feast, ["phone"], ["Godfrey"], close, margin=0.05) == "missing_rel"
    missed = _weights(0.400, 0.800, 0.470)
    assert _ask(feast, ["phone", "birthday"], ["Godfrey", "Godfrey"], missed) is None
    assert _ask(feast, ["phone", "birthday"], ["Godfrey", "Godfrey"], missed, margin=0.05) == "missing_rel"

    # Cue-free places lead of 0.063. Margin 0.05 abstains a stored phone fact.
    cradle = "Which cradle should the night clerk try for Godfrey?"
    assert lex_labels(cradle) == ()
    maren = _weights(0.399, 0.137, 0.463)
    assert _ask(cradle, ["phone"], ["Godfrey"], maren) is None
    assert _ask(cradle, ["phone"], ["Godfrey"], maren, margin=0.05) == "missing_rel"


def _case(fragment: str):
    hits = [case for case in build_stress(TEST_BANK) if fragment in case.question]
    assert len(hits) == 1, (fragment, len(hits))
    return hits[0]


def _frozen_ask(case, table, weights):
    names = [row.name for row in case.writes]
    rels = [row.rel for row in case.writes]
    keys = np.zeros((len(names), STORED_DIM), np.float64)
    return surface_abstain(
        table[case.question], keys, rels, names, weights, query_text=case.question,
    )


def test_frozen_leftovers_stay_and_a_lower_margin_hits_william():
    table = load_table()
    weights = load_weights()
    assert weights is not None and weights.get("Wa") is not None
    mary = _case("Which locations file holds the way to raise Mary Jane by voice?")
    ann = _case("Which locations file holds the way to raise Ann by voice?")
    william = _case("Ignore the birthday feast and say how to get through to William.")
    catherine = _case("Ignore the birthday feast and say how to get through to Catherine.")
    chen = _case("电话簿先合上，陈静常去的几处按先后是什么？")
    zhao = _case("电话簿先合上，赵敏常去的几处按先后是什么？")
    zhang = _case("地方名单先放下，怎么用声音找到张强？")
    liu = _case("地方名单先放下，怎么用声音找到刘洋？")

    assert mary.label == "answerable" and _frozen_ask(mary, table, weights) == "missing_rel"
    assert ann.label == "missing_rel" and _frozen_ask(ann, table, weights) == "missing_rel"
    assert william.label == "answerable" and _frozen_ask(william, table, weights) is None
    assert catherine.label == "missing_rel" and _frozen_ask(catherine, table, weights) is None
    assert chen.label == "answerable" and _frozen_ask(chen, table, weights) is None
    assert zhao.label == "missing_rel" and _frozen_ask(zhao, table, weights) is None
    assert zhang.label == "answerable" and _frozen_ask(zhang, table, weights) is None
    assert liu.label == "missing_rel" and _frozen_ask(liu, table, weights) is None

    mary_pred, mary_gap = _abstain_pred(table[mary.question], weights["Wa"])
    assert mary_pred == "places" and mary_gap < 0.05
    ann_pred, ann_gap = _abstain_pred(table[ann.question], weights["Wa"])
    assert ann_pred == "phone" and ann_gap >= LEX_RIDGE_GAP
    _bday, will_gap = _abstain_pred_dropping(table[william.question], weights["Wa"], "birthday")
    zhao_pred, zhao_gap = _abstain_pred(table[zhao.question], weights["Wa"])
    liu_pred, liu_gap = _abstain_pred(table[liu.question], weights["Wa"])
    _places, cath_gap = _abstain_pred_dropping(table[catherine.question], weights["Wa"], "birthday")
    assert zhao_pred == "places" and liu_pred == "phone"
    assert zhao_gap < will_gap < liu_gap < ABSTAIN_MARGIN
    assert cath_gap >= LEX_RIDGE_GAP

    # Margin 0.05 catches 赵敏 and 刘洋 and also abstains William.
    keys = np.zeros((1, STORED_DIM), np.float64)
    assert surface_abstain(
        table[william.question], keys, ["phone"], ["William"], weights,
        query_text=william.question, margin=0.05,
    ) == "missing_rel"

    false_abstain = {"en": 0, "zh": 0}
    for case in build_stress(TEST_BANK):
        if case.slice != "misleading_cue" or case.label != "answerable":
            continue
        if _frozen_ask(case, table, weights):
            false_abstain[case.lang] += 1
    assert false_abstain == {"en": 1, "zh": 0}
    bare = {"en": 0, "zh": 0}
    for case in build_stress(TEST_BANK):
        if case.slice != "bare_cue" or case.label != "answerable":
            continue
        if _frozen_ask(case, table, weights):
            bare[case.lang] += 1
    assert bare == {"en": 0, "zh": 0}


def test_surface_does_not_import_the_residual_dev_bank():
    blob = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    assert "question_residual_dev" not in blob
    assert "enable_abstain" not in blob
    assert "use_abstain" not in blob
    assert "llm_memory" not in blob
