"""Held-out surface questions are measured. ret@1 is not a pass/fail gate."""
import ast
from pathlib import Path

from tidal.question_heldout_data import REQUIRED_QUESTIONS, build_cases, corpus_summary


def test_corpus_is_large_varied_and_not_in_the_training_source():
    summary = corpus_summary()
    assert summary["errors"] == ()
    assert summary["n"] >= 300
    for cat, n in summary["by_category"].items():
        assert n >= 24, cat
        assert summary["frames"][cat] >= 12, cat
    questions = [case.question for case in build_cases()]
    assert len(questions) == len(set(questions))
    for question in REQUIRED_QUESTIONS:
        assert question in questions
    src = Path("tidal/question_head.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    literals = [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    for question in questions:
        assert question not in src
        assert question not in literals
        assert not any(len(literal) >= 24 and (literal in question or question in literal) for literal in literals)


def test_heldout_sources_have_no_enable_flag():
    root = Path("tidal")
    blob = (root / "question_heldout.py").read_text(encoding="utf-8")
    blob += (root / "question_heldout_data.py").read_text(encoding="utf-8")
    blob += (root / "question_surface.py").read_text(encoding="utf-8")
    assert "enable_question" not in blob
    assert "use_question_head" not in blob
    assert "enable_episodic" not in blob
    assert "episodic_scale" not in blob
    assert "enable_surface" not in blob
    assert "use_surface" not in blob


def test_heldout_eval_measures_and_keeps_invariants():
    """Invariants fail the check callback. ret@1 is reported and not required."""
    from tidal.question_heldout import CATEGORIES, EXPECTED_CACHE_SHA256, cache_sha256, eval_heldout

    assert EXPECTED_CACHE_SHA256
    assert cache_sha256() == EXPECTED_CACHE_SHA256
    failures = []

    def check(section, ok, detail):
        if not ok:
            failures.append(f"{section}: {detail}")

    report = eval_heldout(check)
    assert not failures, failures
    assert report["n"] >= 300
    assert report["disjoint"]["feature_collisions"] == 0
    assert report["disjoint"]["state_collisions"] == 0
    assert report["disjoint"]["literal_hits"] == 0
    assert report["disjoint"]["max_text_cosine_vs_training"] < 0.5
    assert report["disjoint"]["surface_literal_hits"] == 0
    assert report["disjoint"]["surface_name_hits"] == 0
    assert report["disjoint"]["surface_max_bge_cosine"] < 0.98
    assert report["disjoint"]["surface_max_jaccard"] < 0.65
    assert report["disjoint"]["surface_alpha"] == 0.5
    for cat in CATEGORIES:
        row = report["categories"][cat]
        assert row["n"] >= 24
        assert "cosine_hits" in row and "head_hits" in row
    un = report["rollups"]["unanswerable"]
    assert un["false_n"] == un["n"] > 0
    assert report["confidence"]["unanswerable_head_margin"]["50"] >= 0.0
    assert report["confidence"]["head_sweep"]
    # Accuracy is intentionally not asserted. A mediocre ret@1 must stay green.
    assert report["rollups"]["answerable"]["n"] >= 300
    # Unrelated co-resident facts used to collide above the live 0.9 overwrite
    # (5267/8390 pairs, gold lost 117/370). The key construction has to keep
    # that from returning. ret@1 itself stays a measurement.
    stored = report["store"]
    assert stored["co_resident_pairs"] > 1000
    assert stored["max_pair_cosine"] < 0.9
    assert stored["pairs_cosine_above_0.9"] == 0
    assert report["rollups"]["answerable"]["lost_09"] == 0
