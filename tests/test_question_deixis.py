"""Deixis resolves to the stored hearer or one recent third person.

An empty log does not. A stored name in the question keeps the name path.
The margin stays 0.1 and the ridge gap stays 0.15. Frozen questions are
not an input to the wording.
"""
import ast
import hashlib
from pathlib import Path

from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_deixis_dev_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_deixis_eval import jaccard_report, prior_bags, score_rows
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    LEX_RIDGE_GAP,
    ask_people,
    resolve_deixis,
)

QUESTIONS_SHA = "4c582b14b181fcead6b256e1bb031af1c59ffece6047c7eaebc5c814de183734"
BANK_SHA = "1f9674cb23237ba3879a8c4938be323024d3dcc590e982da55868f15cbe53338"

PRIOR_SHA = {
    "frozen": "cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5",
    "dialogue_dev_questions": "11bc041256941969c1055bb5bfc3cd491407718131689aa2f253a2d38e669f2a",
    "dialogue_heldout_questions": "99acea85195eca1f0e680f287b040646c321853c7312168801c46d38175f8d5d",
}

KINDS = {
    "hearer_set": 6,
    "hearer_unset": 4,
    "hearer_ambiguous": 4,
    "third_unique": 8,
    "third_tie": 4,
    "third_aged": 4,
    "third_silent": 4,
    "third_wrong_role": 4,
    "control_named": 2,
    "control_you_named": 2,
    "control_indirect": 2,
    "control_indirect_noself": 2,
    "control_missing": 2,
}


def test_bank_shape_and_constants():
    rows = build_rows()
    assert len(rows) == 48
    assert kind_counts(rows) == KINDS
    assert sum(row.lang == "en" for row in rows) == 24
    assert sum(row.lang == "zh" for row in rows) == 24
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    assert len(bank_strings(rows)) == len(set(bank_strings(rows)))
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA


def test_resolver_reads_the_log_and_leaves_an_empty_one():
    state = DialogueState()
    names = ["Godgifu", "Waerburh"]
    assert resolve_deixis("your telephone", names, state) is None
    assert resolve_deixis("his telephone", names, state) is None
    state.note_write("Godgifu")
    assert resolve_deixis("his telephone", names, state) is None
    state.note_utterance(["Godgifu"], speaker="Waerburh", hearer="Godgifu", surface="named")
    assert resolve_deixis("your telephone", names, state) == ("Godgifu",)
    assert resolve_deixis("the listener telephone", names, state) == ("Godgifu",)
    assert resolve_deixis("his telephone", names, state) == ()
    assert resolve_deixis("your telephone or her birthday", names, state) == ()
    assert resolve_deixis("the telephone under Waerburh", names, state) is None
    assert ask_people("his telephone", names) == []

    bare = DialogueState()
    bare.note_utterance(["Godgifu"], surface="named")
    assert resolve_deixis("your telephone", names, bare) == ()
    assert resolve_deixis("the listener telephone", names, bare) is None
    assert resolve_deixis("her telephone", names, bare) == ("Godgifu",)

    both = DialogueState()
    both.note_utterance(["Godgifu", "Waerburh"], surface="both")
    assert resolve_deixis("her telephone", names, both) == ()
    later = DialogueState()
    later.note_utterance(["Godgifu"], surface="old")
    later.note_utterance(["Waerburh"], surface="new")
    assert later.eligible_latest() == ("Waerburh",)
    assert resolve_deixis("her telephone", names, later) == ("Waerburh",)

    aged = DialogueState()
    aged.note_utterance(["Godgifu"], surface="old")
    for i in range(4):
        aged.note_utterance([], surface=f"quiet {i}")
    assert aged.eligible_latest() == ()
    assert resolve_deixis("her telephone", names, aged) == ()

    asked = DialogueState()
    asked.note_ask(["Godgifu"], "earlier")
    assert resolve_deixis("her telephone", names, asked) is None
    asked.note_utterance([], surface="quiet")
    assert resolve_deixis("her telephone", names, asked) == ("Godgifu",)


def test_bank_is_disjoint_from_earlier_splits():
    rows = build_rows()
    bags = prior_bags()
    for name, sha in PRIOR_SHA.items():
        assert bags[name]["sha"] == sha, name
    report = jaccard_report(bank_strings(rows), bags)
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_sources_do_not_import_the_bank():
    root = Path("tidal")
    tree = ast.parse((root / "question_deixis_dev_data.py").read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        blob = (root / name).read_text(encoding="utf-8")
        assert "question_deixis_dev_data" not in blob
        assert "enable_deixis" not in blob


def test_ask_meets_the_gate():
    scored = score_rows(build_rows())
    assert not scored["failures"], "\n".join(
        f"{row['kind']} {row['lang']} {row['mode']}: expect {row['expect']} "
        f"got {row['reason'] or row['answer']}"
        for row in scored["failures"][:12]
    )
    for kind, slot in scored["kinds"].items():
        assert slot["state"] == slot["n"], kind
        assert slot["empty"] == slot["n"], kind
