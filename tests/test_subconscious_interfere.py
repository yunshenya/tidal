"""Repeat-threshold bank, selection rule, and the pre-score preregistration."""
import ast
import inspect
import re
from pathlib import Path

import numpy as np

from tidal.episodic import REPEAT_JACCARD, token_jaccard
from tidal.subconscious_interfere_data import CASES, GRID, bank_sha256
from tidal.subconscious_interfere_score import (
    IMPLICIT_DZ_INTERVAL,
    expected_slots,
    held_allowed,
    score_grid,
    select_threshold,
)
from tidal.subconscious_softmix_score import MEASURE_INTERVALS

ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")
PREREG = ROOT / "reports" / "subconscious_interfere_prereg.md"


def _tokens(text: str) -> set:
    out = set()
    for match in TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def _walk_strings(obj, found):
    if isinstance(obj, str):
        found.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _walk_strings(value, found)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _walk_strings(value, found)


def test_fit_cases_have_the_overlaps_the_rule_uses():
    by_id = {case["id"]: case for case in CASES}
    assert token_jaccard(by_id["exact-case"]["writes"][0]["text"], by_id["exact-case"]["writes"][1]["text"]) == 1.0
    extra = token_jaccard(by_id["extra-token"]["writes"][0]["text"], by_id["extra-token"]["writes"][1]["text"])
    swapped = token_jaccard(by_id["one-word"]["writes"][0]["text"], by_id["one-word"]["writes"][1]["text"])
    assert extra == 6 / 7
    assert swapped == 5 / 7
    assert token_jaccard(by_id["disjoint-same-rel"]["writes"][0]["text"], by_id["disjoint-same-rel"]["writes"][1]["text"]) == 0.0
    assert by_id["extra-token"]["rule"] == "merge"
    assert by_id["one-word"]["rule"] == "stay"
    assert expected_slots(by_id["extra-token"], 0.5) == 1
    assert expected_slots(by_id["one-word"], 0.5) == 2
    assert len(bank_sha256()) == 64
    assert GRID == (0.50, 0.67, 0.75, 0.85, 0.92, 1.0)


def test_grid_selects_the_highest_threshold_that_keeps_the_different_fact():
    rows = score_grid()
    assert [row["threshold"] for row in rows] == [float(value) for value in GRID]
    passing = [row["threshold"] for row in rows if row["pass"]]
    assert passing == [0.75, 0.85]
    selected = select_threshold(rows)
    assert selected["threshold"] == 0.85
    assert select_threshold([{"threshold": 0.5, "pass": False}]) is None
    assert REPEAT_JACCARD == 0.85


def test_surfaces_stay_disjoint_from_the_other_banks():
    import importlib

    fresh = []
    for case in CASES:
        for write in case["writes"]:
            fresh.append(write["text"])
            fresh.append(write["name"])
            if isinstance(write.get("rel"), str):
                fresh.append(write["rel"])
    old = []
    for path in sorted((ROOT / "tidal").glob("*data.py")):
        if path.stem == "subconscious_interfere_data":
            continue
        mod = importlib.import_module(f"tidal.{path.stem}")
        for key, value in vars(mod).items():
            if key.startswith("__"):
                continue
            _walk_strings(value, old)
    old_sets = [(_tokens(text), text) for text in old if str(text).strip()]
    worst = 0.0
    pair = None
    for text in fresh:
        left = _tokens(text)
        if not left:
            continue
        for right, old_text in old_sets:
            if not right:
                continue
            score = len(left & right) / len(left | right)
            if score > worst:
                worst = score
                pair = (text[:80], old_text[:80])
    assert worst < 0.50, (worst, pair)


def test_held_waits_for_the_selected_threshold():
    fit = {"complete": True, "bank_sha256": bank_sha256(), "selected": {"threshold": 0.85}}
    assert held_allowed(fit, 0.85, False)[0]
    assert not held_allowed(fit, 1.0, False)[0]
    assert not held_allowed(fit, 0.85, True)[0]
    assert not held_allowed({"complete": True, "bank_sha256": bank_sha256(), "selected": None}, 0.85, False)[0]
    assert not held_allowed(None, 0.85, False)[0]
    low, high = IMPLICIT_DZ_INTERVAL
    assert low < 1.1516089064136972 < high
    for bounds in MEASURE_INTERVALS.values():
        assert bounds[0] < bounds[1]


def test_fit_scorer_does_not_read_held_items():
    from tidal import subconscious_interfere_score as score

    source = inspect.getsource(score.score_fit)
    for name in (
        "RifMeasure",
        "prefer_items",
        "avoid_items",
        "tone_items",
        "phone_items",
        "score_quartet",
        "score_direction",
    ):
        assert name not in source
    tree = ast.parse((ROOT / "scripts" / "eval_subconscious_interfere.py").read_text(encoding="utf-8"))
    fit_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit")
    calls = [node.func.id for node in ast.walk(fit_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "_held" not in calls


def test_prereg_records_the_fit_and_waits_for_held():
    text = PREREG.read_text(encoding="utf-8")
    assert "Held has not been scored." in text
    assert "6f9530eb9a2684d258e7b2cd5714c61bb69fd6eb5950f5193e91f77d0d3130ba" in text
    low, high = IMPLICIT_DZ_INTERVAL
    assert str(low) in text
    assert str(high) in text
    for bounds in MEASURE_INTERVALS.values():
        assert str(bounds[0]) in text
        assert str(bounds[1]) in text


def test_close_key_case_really_is_parallel():
    case = next(item for item in CASES if item["id"] == "close-keys-different-text")
    left = np.asarray(case["writes"][0]["key"], dtype=np.float64)
    right = np.asarray(case["writes"][1]["key"], dtype=np.float64)
    cosine = float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    assert cosine > 0.99
