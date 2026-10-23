"""Pronoun keep-alive and a leading vocative. The gate was written down first.

Development rows stay on the #35 bank: gated 106/106, ordinary 52/52, and the
four rows that used to be the experience gap. The held-out bank is disjoint,
including from that #35 bank. A miss against its rule label fails.
"""
from pathlib import Path

from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_keepalive_heldout_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_keepalive_heldout_eval import prior_bags, score_rows
from tidal.question_name_align import dialogue_licensed_edit
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    LEX_RIDGE_GAP,
    apply_utterance,
    refresh_resolved,
    resolve_deixis,
    vocative_hearer,
)

QUESTIONS_SHA = "092252c79252651ad6cdee2c63783a445ca1a2e4e15804732c955a8407de1a03"
BANK_SHA = "694fed5e12b4e4dbbcd831c8394e7101cbcab908d6406dd5ac5611a4e2c85b25"
KINDS = {
    "keep_he": 1,
    "keep_she": 1,
    "keep_ta": 1,
    "keep_zhta": 2,
    "keep_edit": 2,
    "keep_ask": 10,
    "age_quiet": 2,
    "age_resurrect": 2,
    "age_tie": 2,
    "age_edit": 2,
    "voc_lead": 5,
    "voc_subject": 2,
    "voc_object": 2,
    "voc_medial": 2,
    "voc_two": 2,
    "voc_explicit": 2,
    "empty": 2,
}


def test_vocative_is_only_a_leading_exact_name():
    names = ["Mirren", "Calder", "Pellam", "苟宁", "卞茗"]
    assert vocative_hearer("Mirren, the kelp ticket", names) == "Mirren"
    assert vocative_hearer("  Mirren, the kelp ticket", names) == "Mirren"
    assert vocative_hearer("Mirren! the kelp ticket", names) == "Mirren"
    assert vocative_hearer("Mirren: the kelp ticket", names) == "Mirren"
    assert vocative_hearer("Mirren left the kelp ticket", names) is None
    assert vocative_hearer("The clerk set it beside Mirren", names) is None
    assert vocative_hearer("After the queue, Mirren, wait", names) is None
    assert vocative_hearer("Mirren, Calder, wait", names) is None
    assert vocative_hearer("Hey Mirren, wait", names) is None
    assert vocative_hearer("Mirren. the kelp ticket", names) is None
    assert vocative_hearer("Caldar, the kelp ticket", ["Calder"]) is None
    assert vocative_hearer("Pellum, the kelp ticket", ["Pellam"]) is None
    assert vocative_hearer("苟宁，海带票", names) == "苟宁"
    assert vocative_hearer("苟宁：海带票", names) == "苟宁"
    assert vocative_hearer("苟宁！海带票", names) == "苟宁"
    assert vocative_hearer("苟宁把海带票收好", names) is None
    assert vocative_hearer("苟宁，卞茗，海带票", names) is None
    assert vocative_hearer("队列散了以后，苟宁，海带票", names) is None
    assert vocative_hearer("苟柠，海带票", ["苟宁"]) is None


def test_refresh_restamps_inside_the_window_only():
    state = DialogueState()
    state.note_utterance(["Calder"], surface="named Calder once")
    assert state.refresh("Calder") is True
    kept = [item for item in state.mentions() if item.name == "Calder"]
    assert len(kept) == 1
    assert kept[0].surface == "named Calder once"
    assert kept[0].source == "utterance"
    assert kept[0].turn == state.turn
    for i in range(4):
        state.note_utterance([], surface=f"quiet {i}")
    assert state.refresh("Calder") is False
    assert state.conversational() == ()

    live = DialogueState()
    live.note_utterance(["Calder"], surface="named Calder once")
    for i in range(4):
        live.note_utterance([], surface=f"he {i}")
        assert live.refresh("Calder") is True
    assert live.conversational() == ("Calder",)
    assert live.eligible_latest() == ("Calder",)
    assert [item.surface for item in live.mentions() if item.name == "Calder"] == ["named Calder once"]

    written = DialogueState()
    written.note_write("Calder")
    written.note_utterance([], surface="quiet")
    assert written.refresh("Calder") is False


def test_apply_utterance_keeps_a_pronoun_and_a_leading_name():
    names = ["Calder", "Mirren", "Yarrow"]
    state = DialogueState()
    assert apply_utterance(
        state, "Calder set the board.", names, None, speaker="Solway", hearer="Mirren",
    ) == ("Calder",)
    for i in range(4):
        assert apply_utterance(
            state, f"He folded the board {i}.", names, None, speaker="Solway", hearer="Mirren",
        ) == ()
    assert resolve_deixis("Which telephone did he leave?", names, state) == ("Calder",)
    assert [item.surface for item in state.mentions() if item.name == "Calder"] == ["Calder set the board."]

    quiet = DialogueState()
    apply_utterance(quiet, "Calder set the board.", names, None, speaker="Solway", hearer="Mirren")
    for i in range(4):
        apply_utterance(quiet, f"Mm noted {i}.", names, None, speaker="Solway", hearer="Mirren")
    assert resolve_deixis("Which telephone did he leave?", names, quiet) == ()
    apply_utterance(quiet, "He waved once.", names, None, speaker="Solway", hearer="Mirren")
    assert resolve_deixis("Which telephone did he leave?", names, quiet) == ()
    assert "Calder" not in quiet.conversational()

    aged_you = DialogueState()
    apply_utterance(aged_you, "Calder, the board stays.", names, None, speaker="Solway")
    assert aged_you.hearer == "Calder"
    for i in range(4):
        apply_utterance(aged_you, f"quiet nod {i}", names, None, speaker="Solway")
    assert resolve_deixis("your telephone", names, aged_you) == ("Calder",)
    apply_utterance(aged_you, "your telephone is the one on the board.", names, None)
    assert "Calder" not in aged_you.conversational()
    assert dialogue_licensed_edit("Caldar", names, aged_you.conversational()) is None

    voc = DialogueState()
    apply_utterance(voc, "Mirren, the kelp ticket stays.", names, None, speaker="Solway")
    assert voc.hearer == "Mirren"
    assert resolve_deixis("Which telephone is yours?", names, voc) == ("Mirren",)

    subject = DialogueState()
    apply_utterance(subject, "Mirren left the kelp ticket.", names, None, speaker="Solway")
    assert subject.hearer is None
    assert resolve_deixis("Which telephone is yours?", names, subject) == ()

    explicit = DialogueState()
    apply_utterance(explicit, "Mirren, the kelp ticket stays.", names, None, hearer="Calder")
    assert explicit.hearer == "Calder"
    assert resolve_deixis("your telephone", names, explicit) == ("Calder",)

    off = DialogueState()
    apply_utterance(off, "Mirren, the kelp ticket stays.", names, None, hearer="unlisted")
    assert off.hearer == "unlisted"
    assert resolve_deixis("your telephone", names, off) == ()

    both = DialogueState()
    apply_utterance(both, "Mirren, Calder, the kelp ticket waits.", names, None)
    assert both.hearer is None
    tie = DialogueState()
    apply_utterance(tie, "Calder and Mirren both signed the chit.", names, None, speaker="Yarrow")
    assert resolve_deixis("her telephone", names, tie) == ()


def test_ask_chain_refreshes_without_storing_the_pronoun():
    state = DialogueState()
    state.note_utterance(["Calder"], speaker="Solway", hearer="Mirren", surface="named")
    names = ["Calder", "Mirren"]
    for i in range(4):
        text = f"Which telephone did he leave on slip {i}?"
        state.note_ask([], text)
        refresh_resolved(state, text, names, None)
    assert resolve_deixis("Which telephone did he leave on the last slip?", names, state) == ("Calder",)
    calder = [item for item in state.mentions() if item.name == "Calder"]
    assert len(calder) == 1
    assert calder[0].surface == "named"
    assert calder[0].source == "utterance"


def test_edit_follows_only_a_name_the_pronoun_kept_alive():
    names = ["Calder", "苟宁", "Yarrow"]
    kept = DialogueState()
    apply_utterance(kept, "Calder set the board.", names, None, speaker="Yarrow", hearer="苟宁")
    for i in range(4):
        apply_utterance(kept, f"He folded the board {i}.", names, None, speaker="Yarrow", hearer="苟宁")
    assert dialogue_licensed_edit("Caldar", names, kept.conversational()) == "Calder"
    aged = DialogueState()
    apply_utterance(aged, "Calder set the board.", names, None, speaker="Yarrow", hearer="苟宁")
    for i in range(4):
        apply_utterance(aged, f"Mm {i}.", names, None, speaker="Yarrow", hearer="苟宁")
    assert dialogue_licensed_edit("Caldar", names, aged.conversational()) is None
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15


def test_sources_do_not_import_the_heldout_bank():
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        blob = (root / name).read_text(encoding="utf-8")
        assert "question_keepalive" not in blob
        assert "enable_dialogue" not in blob
        assert "enable_deixis" not in blob
        assert "enable_keepalive" not in blob


def test_heldout_shape_and_disjointness():
    rows = build_rows()
    assert len(rows) == 34
    assert kind_counts(rows) == KINDS
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    report = jaccard_report(bank_strings(rows), prior_bags())
    assert report["dialogue_real_questions"]["sha"].startswith("117ef589")
    assert report["dialogue_real_bank"]["sha"].startswith("0be42845")
    assert report["frozen"]["sha"].startswith("cce20b6b")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_heldout_score_matches_the_rule():
    scored = score_rows(build_rows())
    assert scored["misses"] == []
    for kind, slot in scored["kind"].items():
        assert slot["state"] == slot["n"], kind
        assert slot["empty"] == slot["n"], kind
        assert slot["human"] == slot["n"], kind
        assert slot["state_fa"] == 0 and slot["empty_fa"] == 0
