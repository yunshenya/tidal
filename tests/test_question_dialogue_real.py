"""The natural multi-turn bank stays disjoint, and the ask matches the rule.

The gate was written down in reports/dialogue_real_prereg.md before the score.
A human miss on a gap, an open addressee form, an empty-log control, or an
observed name head is not a regression. A rule miss would be.
"""
from pathlib import Path

from tidal.dialogue_state import TURN_WINDOW
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_dialogue_real_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
    scope_counts,
)
from tidal.question_dialogue_real_eval import prior_bags, score_rows
from tidal.question_surface import ABSTAIN_MARGIN, LEX_RIDGE_GAP

QUESTIONS_SHA = "117ef589bb02bd883db44a4d2d5859b17c271c00a5df324574c8095aced5b704"
BANK_SHA = "0be42845836ec2a19c961d5a415ccd0d9d0dcb1f8adf5bb632335cd26871cec5"
SCOPES = {"gated": 106, "gap": 17, "dev": 4, "oos": 8, "control": 4, "observe": 3}
KINDS = {
    "you_hearer": 16,
    "third_unique": 14,
    "name_exact": 14,
    "third_later": 2,
    "you_followup": 2,
    "code_switch": 4,
    "third_hold": 2,
    "name_aged": 2,
    "pronoun_hold": 2,
    "edit1_sub": 2,
    "edit1_del": 2,
    "edit1_silent": 2,
    "edit1_other": 2,
    "edit1_tie": 2,
    "edit_extension": 2,
    "edit_far": 2,
    "missing_you": 2,
    "missing_third": 2,
    "absent_name": 2,
    "hearer_unset": 2,
    "hearer_offroster": 2,
    "hearer_switch": 2,
    "former_hearer": 2,
    "indirect_self": 2,
    "indirect_hearer": 2,
    "indirect_noself": 2,
    "prior_ask_name": 2,
    "prior_ask_edit": 2,
    "mixed_pronoun": 2,
    "not_pronoun_them": 1,
    "ta_hit": 1,
    "you_named": 2,
    "vocative_question": 2,
    "third_tie": 2,
    "window_backchannel": 2,
    "pronoun_chain_age": 2,
    "vocative_unpassed": 2,
    "gender_tie": 4,
    "recency_clash": 2,
    "edit_aged": 2,
    "edit_insert": 1,
    "edit_transpose": 2,
    "cjk_prefix": 1,
    "speaker_as_third": 2,
    "plural_you": 1,
    "oos_addressee": 8,
    "empty_deictic": 4,
    "short_name": 1,
    "given_nick": 1,
    "pinyin_name": 1,
}


def test_real_shape_and_sha():
    rows = build_rows()
    assert len(rows) == 104
    assert sum(scope_counts(rows).values()) == 142
    assert scope_counts(rows) == SCOPES
    assert kind_counts(rows) == KINDS
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15


def test_real_is_disjoint_from_earlier_splits():
    report = jaccard_report(bank_strings(), prior_bags())
    assert report["frozen"]["sha"].startswith("cce20b6b")
    assert report["dialogue_dev_questions"]["sha"].startswith("11bc0412")
    assert report["dialogue_heldout_questions"]["sha"].startswith("99acea85")
    assert report["deixis_dev_questions"]["sha"].startswith("4c582b14")
    assert report["deixis_heldout_questions"]["sha"].startswith("d414c446")
    assert report["deixis_heldout_bank"]["sha"].startswith("901390f5")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_real_score_matches_the_rule():
    """Pre-registered gate: the ask path matches the rule. Human gaps stay gaps."""
    scored = score_rows(build_rows())
    assert scored["misses"] == []
    gated = scored["scope"]["gated"]
    assert gated == {
        "n": 106, "state": 106, "empty": 106, "human": 106, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["band"]["ordinary"]["state"] == 52
    assert scored["band"]["ordinary"]["human"] == 52
    gap = scored["scope"]["gap"]
    assert gap["state"] == gap["n"] == 17
    assert gap["human"] == 0
    assert gap["state_fa"] == 0
    dev = scored["scope"]["dev"]
    assert dev == {
        "n": 4, "state": 4, "empty": 4, "human": 4, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["kind"]["pronoun_chain_age"]["human"] == 2
    assert scored["kind"]["vocative_unpassed"]["human"] == 2
    oos = scored["scope"]["oos"]
    assert oos["state"] == oos["n"] == 8
    assert oos["human"] == 0
    assert sum(slot["state_fa"] + slot["empty_fa"] for slot in scored["scope"].values()) == 0


def test_rule_sources_do_not_import_the_real_bank():
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        blob = (root / name).read_text(encoding="utf-8")
        assert "question_dialogue_real" not in blob
        assert "enable_dialogue" not in blob
        assert "enable_deixis" not in blob
