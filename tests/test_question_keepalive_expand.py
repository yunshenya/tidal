"""Expanded keep-alive and vocative held-out. The gate was written down first.

The 42-ask bank stays frozen. These 94 asks were scored once against that
rule. A listener gap would stay in ``listener_gaps`` instead of being relabeled.
"""
from pathlib import Path

from tidal.question_dialogue_eval import jaccard_report
from tidal.question_keepalive_expand_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    iter_asks,
    kind_counts,
    listener_gaps,
    questions_sha256,
)
from tidal.question_keepalive_expand_eval import prior_bags, score_rows
from tidal.question_keepalive_heldout_data import bank_sha256 as frozen_bank_sha
from tidal.question_keepalive_heldout_data import questions_sha256 as frozen_questions_sha
from tidal.question_surface import ABSTAIN_MARGIN, LEX_RIDGE_GAP
from tidal.dialogue_state import TURN_WINDOW

QUESTIONS_SHA = "9af36973b625ce4ae30f516fafdc3a4aec5c37fbfcaf2a8fc31ec423025ee19c"
BANK_SHA = "30af4bb72c1fbe346dd6b5572ebb60d811b860b71ad060385654f3a7d35f1604"
FROZEN_QUESTIONS_SHA = "092252c79252651ad6cdee2c63783a445ca1a2e4e15804732c955a8407de1a03"
FROZEN_BANK_SHA = "694fed5e12b4e4dbbcd831c8394e7101cbcab908d6406dd5ac5611a4e2c85b25"
KINDS = {
    "neg_subject": 12,
    "neg_object": 12,
    "neg_quote": 12,
    "neg_ambig": 12,
    "neg_unknown": 8,
    "neg_explicit": 10,
    "neg_age": 12,
    "pos_keep": 8,
    "pos_voc": 8,
}


def test_expand_shape_is_the_preregistered_bank():
    rows = build_rows()
    assert len(rows) == 54
    assert kind_counts(rows) == KINDS
    assert sum(KINDS.values()) == 94
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    langs = {"en": 0, "zh": 0}
    absent = 0
    for _row, ask in iter_asks(rows):
        langs[ask.lang] += 1
        absent += int(ask.human == "absent_person")
    assert langs == {"en": 47, "zh": 47}
    assert absent == 72
    assert listener_gaps(rows) == ()
    assert frozen_questions_sha() == FROZEN_QUESTIONS_SHA
    assert frozen_bank_sha() == FROZEN_BANK_SHA
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15


def test_expand_is_disjoint_from_the_42_and_earlier_banks():
    report = jaccard_report(bank_strings(build_rows()), prior_bags())
    assert report["keepalive_42_questions"]["sha"] == FROZEN_QUESTIONS_SHA
    assert report["keepalive_42_bank"]["sha"] == FROZEN_BANK_SHA
    assert report["dialogue_real_questions"]["sha"].startswith("117ef589")
    assert report["dialogue_real_bank"]["sha"].startswith("0be42845")
    assert report["frozen"]["sha"].startswith("cce20b6b")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_rule_modules_do_not_import_the_expansion():
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        blob = (root / name).read_text(encoding="utf-8")
        assert "question_keepalive_expand" not in blob
        assert "enable_keepalive" not in blob


def test_expand_score_matches_the_frozen_rule():
    scored = score_rows(build_rows())
    assert scored["misses"] == []
    assert set(scored["kind"]) == set(KINDS)
    for kind, slot in scored["kind"].items():
        assert slot["n"] == KINDS[kind], kind
        assert slot["state"] == slot["n"], kind
        assert slot["empty"] == slot["n"], kind
        assert slot["human"] == slot["n"], kind
        assert slot["state_fa"] == 0 and slot["empty_fa"] == 0, kind
