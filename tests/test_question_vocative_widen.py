"""Wider vocative held-out.

Listener labels were committed before the score. The score lock records that
one run. It does not relabel a row.
"""
from collections import Counter
from pathlib import Path

from tidal.dialogue_state import TURN_WINDOW
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_surface import ABSTAIN_MARGIN, LEX_RIDGE_GAP
from tidal.question_vocative_widen_data import (
    Ask,
    Turn,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_vocative_widen_eval import prior_bags

QUESTIONS_SHA = "6ff0bc0811dfa57a8a5d92c8c92d1b05cd02073803860df51d66993f3617ba86"
BANK_SHA = "3f518ce41813a00357788ba063f36244fb53855280b3786da31f20313374ea1c"
LISTENERS = {
    "pos": {
        "Aldred": 8,
        "Osric": 8,
        "Eadgifu": 6,
        "Leofwyn": 6,
        "Saewynn": 6,
        "容昭": 10,
        "汲宁": 8,
        "蒲荃": 8,
        "纪荻": 8,
    },
    "neg": {"ABSENT": 44},
}
_FORBIDDEN = (
    "apply_utterance",
    "resolve_deixis",
    "vocative_hearer",
    "refresh_resolved",
    "_predict",
    "DialogueState",
    "ask_people",
    "name_spans",
    "selected_fact_rows",
)


def _ask(row):
    return next(event for event in row.events if isinstance(event, Ask))


def test_widen_bank_matches_the_preregistered_labels():
    rows = build_rows()
    assert len(rows) == 112
    assert kind_counts(rows) == {"pos": 68, "neg": 44}
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert len(bank_strings(rows)) == 2432
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    listeners = {"pos": Counter(), "neg": Counter()}
    langs = Counter()
    for row in rows:
        ask = _ask(row)
        langs[(ask.kind, ask.lang)] += 1
        listeners[ask.kind][ask.listener] += 1
        turns = [event for event in row.events if isinstance(event, Turn)]
        assert len(turns) == (2 if "later" in ask.shape else 1)
        assert all(event.hearer == "" for event in turns)
    assert langs == {("pos", "en"): 34, ("pos", "zh"): 34, ("neg", "en"): 22, ("neg", "zh"): 22}
    assert listeners["pos"] == LISTENERS["pos"]
    assert listeners["neg"] == LISTENERS["neg"]


def test_widen_bank_is_disjoint():
    rows = build_rows()
    report = jaccard_report(bank_strings(rows), prior_bags())
    assert report["frozen"]["sha"].startswith("cce20b6b")
    assert report["keepalive_pos_questions"]["sha"].startswith("322afe2b")
    assert report["keepalive_pos_bank"]["sha"].startswith("b32f1392")
    assert report["dialogue_real_questions"]["sha"].startswith("117ef589")
    assert report["dialogue_real_bank"]["sha"].startswith("0be42845")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_widen_score_locks_the_measured_run():
    from tidal.question_vocative_widen_eval import replay_rows, score_rows

    scored = score_rows(replay_rows())
    assert scored["listener_misses"] == []
    assert scored["encoder_misses"] == []
    assert scored["kind"]["pos"] == {
        "n": 68, "state": 68, "empty": 68, "human": 68, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["kind"]["neg"] == {
        "n": 44, "state": 44, "empty": 44, "human": 44, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["lang"]["en"]["n"] == 56 and scored["lang"]["en"]["human"] == 56
    assert scored["lang"]["zh"]["n"] == 56 and scored["lang"]["zh"]["human"] == 56
    assert scored["lang"]["en"]["state_fa"] == 0 and scored["lang"]["zh"]["empty_fa"] == 0
    assert len(scored["shape"]) == 56
    for name, slot in scored["shape"].items():
        assert slot == {
            "n": 2, "state": 2, "empty": 2, "human": 2, "state_fa": 0, "empty_fa": 0,
        }, name


def test_widen_bank_does_not_replay_the_rule_and_production_files_do_not_import_it():
    blob = Path("tidal/question_vocative_widen_data.py").read_text(encoding="utf-8")
    for name in _FORBIDDEN:
        assert name not in blob
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "question_vocative_widen" not in text
        assert "enable_keepalive" not in text
