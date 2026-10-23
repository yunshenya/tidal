"""The deixis held-out bank stays disjoint from every earlier split.

The ask gate is pre-registered in reports/deixis.md. This file locks the
strings and the overlap. It does not retune a miss.
"""
from pathlib import Path

from tidal.question_deixis_heldout_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_deixis_heldout_eval import prior_bags
from tidal.question_dialogue_eval import jaccard_report

QUESTIONS_SHA = "d414c44654db45eb0df85e0220045a28a44441e65b539034545ce8bade0d0dd6"
BANK_SHA = "901390f59988cf68bebfbc352a2c0f8f768262ab13a6057cb748a144e85ef770"
KINDS = {
    "hearer_set_you": 8,
    "hearer_set_role": 4,
    "hearer_unset": 4,
    "hearer_switch": 4,
    "hearer_ambiguous_mixed": 4,
    "hearer_ambiguous_roster": 4,
    "addressee_extra": 8,
    "addressee_extra_unset": 4,
    "third_unique": 8,
    "third_oblique": 10,
    "third_later": 8,
    "third_after_ask": 8,
    "third_tie": 6,
    "window_filler_3": 4,
    "window_filler_4": 4,
    "window_filler_5": 4,
    "third_silent": 4,
    "third_wrong_role": 8,
    "third_former_hearer": 4,
    "third_ask_hearer": 4,
    "control_named": 4,
    "control_you_named": 4,
    "control_indirect": 4,
    "control_indirect_noself": 4,
    "control_missing": 4,
    "control_not_pronoun": 2,
    "control_plural": 2,
}


def test_heldout_shape_and_sha():
    rows = build_rows()
    assert len(rows) == 136
    assert sum(row.lang == "en" for row in rows) == 68
    assert sum(row.lang == "zh" for row in rows) == 68
    assert kind_counts(rows) == KINDS
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert all(row.kind in KINDS for row in rows)


def test_heldout_is_disjoint_from_earlier_splits():
    rows = build_rows()
    report = jaccard_report(bank_strings(rows), prior_bags())
    assert "deixis_dev_questions" in report
    assert report["deixis_dev_questions"]["sha"].startswith("4c582b14")
    assert report["deixis_dev_bank"]["sha"].startswith("1f9674cb")
    assert report["dialogue_dev_questions"]["sha"].startswith("11bc0412")
    assert report["dialogue_heldout_questions"]["sha"].startswith("99acea85")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_rule_sources_do_not_import_the_heldout_bank():
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        blob = (root / name).read_text(encoding="utf-8")
        assert "question_deixis_heldout_data" not in blob
        assert "enable_deixis" not in blob
