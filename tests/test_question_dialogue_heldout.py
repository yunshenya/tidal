"""Held-out dialogue bank: new names, disjoint from the dev split.

The name gate is checked here. Sentence embeddings are not. Window 4 is
the shipped default; filler counts other than 0, 1, and 4 are a boundary
ladder, not a reason to move that default.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np

import tidal.dialogue_state as dialogue_state
from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_dialogue_heldout_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_dialogue_heldout_eval import CACHE, heldout_corpora
from tidal.question_dialogue_eval import load_table
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_name_align import dialogue_licensed_edit
from tidal.question_surface import ask_people, ask_targets, lex_labels

QUESTIONS_SHA = "99acea85195eca1f0e680f287b040646c321853c7312168801c46d38175f8d5d"
BANK_SHA = "e09fbcc043fddd9f3da0c831f448834700bdfe654404331e52cea99914da1376"
CACHE_SHA = "fb640d22b6b5fa9636430ddc90566e9397e8b134f33e91e2fbe92e06f2078a9d"

KINDS = {
    "typo_hit": 24,
    "typo_silent": 12,
    "typo_other": 12,
    "typo_aged": 12,
    "typo_prior": 12,
    "lag_observe": 10,
    "typo_tie": 2,
    "spelling": 4,
    "nickname": 6,
    "latin_extension": 2,
    "cjk_superstring": 2,
    "cjk_prefix": 2,
    "focus": 4,
    "dismiss": 4,
    "open_either": 4,
    "indirect_hit": 2,
    "indirect_noself": 2,
    "indirect_missing": 2,
}


def _active(row, window=TURN_WINDOW):
    saved = dialogue_state.TURN_WINDOW
    dialogue_state.TURN_WINDOW = window
    try:
        state = DialogueState()
        names = [slot.name for slot in row.slots]
        surfaces = [slot.text for slot in row.slots]
        for turn in row.turns:
            mentioned = ask_people(turn, names, surfaces, state.conversational())
            state.note_utterance(mentioned, surface=turn)
        for prior in row.prior_asks:
            live = state.conversational(ignore_surface=prior)
            mentioned = ask_people(prior, names, surfaces, live)
            state.note_ask(mentioned, prior)
        return state.conversational(ignore_surface=row.question)
    finally:
        dialogue_state.TURN_WINDOW = saved


def _targets(row, active):
    names = [slot.name for slot in row.slots]
    surfaces = [slot.text for slot in row.slots]
    if row.kind.startswith("indirect"):
        return ask_people(row.question, names, surfaces, active)
    return ask_targets(row.question, names, surfaces, active)


def test_bank_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert len(rows) == 118
    assert 80 <= len(rows) <= 120
    assert kind_counts(rows) == KINDS
    assert sum(row.lang == "en" for row in rows) == 59
    assert sum(row.lang == "zh" for row in rows) == 59
    assert TURN_WINDOW == 4
    assert len(bank_strings(rows)) == 538
    hit = {(row.lang, row.who) for row in rows if row.kind == "typo_hit"}
    for kind in ("typo_silent", "typo_other", "typo_aged"):
        assert {(row.lang, row.who) for row in rows if row.kind == kind} == hit


def test_bank_is_disjoint_from_earlier_splits_and_the_dev_bank():
    rows = build_rows()
    bags = heldout_corpora()
    assert bags["dialogue_dev_questions"]["sha"].startswith("11bc04125694")
    assert bags["dialogue_dev_bank"]["sha"].startswith("69bd3ed63276")
    assert bags["frozen"]["sha"].startswith("cce20b6bf29f")
    report = jaccard_report(bank_strings(rows), bags)
    worst = 0.0
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] <= 0.50, (name, row["max_jaccard"], row["pair"])
        worst = max(worst, row["max_jaccard"])
    assert worst <= 0.50


def test_embedding_cache_matches_the_bank():
    rows = build_rows()
    table = load_table(CACHE)
    texts = bank_strings(rows)
    assert list(table) == list(texts)
    assert hashlib.sha256(CACHE.read_bytes()).hexdigest() == CACHE_SHA
    norms = np.linalg.norm(np.stack([table[text] for text in texts]), axis=1)
    assert float(np.max(np.abs(norms - 1.0))) < 1e-3


def test_data_module_does_not_import_the_matcher():
    path = Path("tidal/question_dialogue_heldout_data.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
    matcher = Path("tidal/question_name_align.py").read_text(encoding="utf-8")
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    assert "question_dialogue_heldout" not in matcher
    assert "question_dialogue_heldout" not in surface


def test_edit_license_on_these_names():
    assert dialogue_licensed_edit("Leofrid", ["Leofric"], ()) is None
    assert dialogue_licensed_edit("Leofrid", ["Leofric"], ["Leofric"]) == "Leofric"
    assert dialogue_licensed_edit("Leofrid", ["Leofric", "Dunstan"], ["Dunstan"]) is None
    assert dialogue_licensed_edit("Leofrico", ["Leofric"], ["Leofric"]) is None
    assert dialogue_licensed_edit("Godwine", ["Godwune", "Godwene"], ["Godwune", "Godwene"]) is None
    assert dialogue_licensed_edit("Godwine", ["Godwune", "Godwene"], ["Godwune"]) == "Godwune"
    assert dialogue_licensed_edit("嵇柯", ["嵇珂"], ["嵇珂"]) == "嵇珂"
    assert dialogue_licensed_edit("嵇珂然", ["嵇珂"], ["嵇珂"]) is None
    assert dialogue_licensed_edit("老嵇珂", ["嵇珂"], ["嵇珂"]) is None
    assert dialogue_licensed_edit("小裴琮", ["裴琮"], ["裴琮"]) is None
    assert dialogue_licensed_edit("庾茗", ["庾珀", "庾霓"], ["庾珀", "庾霓"]) is None


def test_name_gate_matches_the_label_at_the_shipped_window():
    rows = build_rows()
    for row in rows:
        assert lex_labels(row.question) == (row.rel,)
        live = _active(row, 4)
        got = _targets(row, live)
        bare = _targets(row, ())
        if row.role == "hit":
            assert row.target in live
            assert got == [row.target]
            assert bare == []
            assert _targets(row, _active(row, 2)) == [row.target]
        elif row.role == "aged":
            assert row.target not in live and got == [] and bare == []
            assert row.target not in _active(row, 2)
            assert _targets(row, _active(row, 6)) == [row.target]
        elif row.role in ("silent", "other", "tie"):
            assert got == [] and bare == []
            if row.role != "tie":
                assert row.target not in live
        elif row.role == "neg" and row.kind != "indirect_missing":
            assert got == [] and bare == []
            if row.kind != "indirect_noself":
                assert row.target in live
        elif row.role == "pos" or row.kind == "indirect_missing":
            assert got == bare
            if row.kind == "open_either":
                assert set(got) == {slot.name for slot in row.slots}
            elif row.kind != "indirect_noself":
                assert got == [row.target]
        elif row.role == "lag":
            for window in (2, 4, 6, 8):
                names = _active(row, window)
                kept = row.target in names
                assert kept is (window > row.lag)
                assert _targets(row, names) == ([row.target] if kept else [])
        else:
            raise AssertionError(row.role)
    assert dialogue_state.TURN_WINDOW == 4
