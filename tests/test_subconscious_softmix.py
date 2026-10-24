"""Soft-mixture bank, selection rule, and the pre-score preregistration."""
import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from tidal.llm_memory import activation_weights, participation_ratio
from tidal.subconscious_measure_data import choice_items, preference_items, tone_items
from tidal.subconscious_softmix_data import (
    MAX_KEY_COSINE,
    N_DISTRACTORS,
    N_FIT,
    bank_sha256,
    distractors_for,
    fit_items,
    geometry,
)
from tidal.subconscious_softmix_score import (
    MEASURE_INTERVALS,
    PARTICIPATION_MIN,
    held_allowed,
    rate_inside,
    select_cell,
)

ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")
SURFACE = {"text", "prompt", "good", "bad", "answer", "other", "name", "rel"}


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


def _surfaces(obj, found):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SURFACE and isinstance(value, str):
                found.append(value)
            else:
                _surfaces(value, found)
    elif isinstance(obj, list):
        for value in obj:
            _surfaces(value, found)


def test_fit_bank_shape_and_four_slots():
    assert N_FIT == 16
    assert N_DISTRACTORS == 3
    families = {}
    for family in ("tone", "choice", "preference"):
        rows = fit_items(family)
        assert len(rows) == 16
        families[family] = rows
        for row in rows:
            group = distractors_for(row["id"])
            assert len(group) == 3
            frames = {item["frame"] for item in group}
            assert len(frames) == 3
            identities = {(row["name"], row["rel"])} | {(item["name"], item["rel"]) for item in group}
            assert len(identities) == 4
    held = tone_items() + choice_items() + preference_items()
    assert len(held) == 150
    for row in held:
        group = distractors_for(row["id"])
        assert len(group) == 3
        identities = {(row["name"], row["rel"])} | {(item["name"], item["rel"]) for item in group}
        assert len(identities) == 4
    geo = geometry()
    assert geo["n_groups"] == 198
    assert geo["target_argmax"] == 198
    assert geo["max_key_cosine"] == pytest.approx(MAX_KEY_COSINE)
    assert MAX_KEY_COSINE < 0.85
    assert len(bank_sha256()) == 64


def test_geometry_participation_is_recomputable_and_not_peaked():
    geo = geometry()
    for temp, by_k in geo["participation"].items():
        for k, summary in by_k.items():
            parts = []
            for group in geo["groups"]:
                assert group["target_argmax"] is True
                scores = group["query_cosine"]
                assert scores[0] > max(scores[1:])
                weights = activation_weights(scores, [1.0, 1.0, 1.0, 1.0], temperature=float(temp), k=int(k))
                parts.append(participation_ratio(weights))
            assert _mean(parts) == pytest.approx(summary["mean"])
            assert min(parts) == pytest.approx(summary["min"])
            assert summary["mean"] >= PARTICIPATION_MIN


def _mean(vals):
    return sum(vals) / len(vals)


def test_softmix_surfaces_stay_disjoint():
    import importlib

    fresh = []
    for family in ("tone", "choice", "preference"):
        _surfaces(fit_items(family), fresh)
    for row in tone_items() + choice_items() + preference_items():
        _surfaces(distractors_for(row["id"]), fresh)
    for family in ("tone", "choice", "preference"):
        for row in fit_items(family):
            _surfaces(distractors_for(row["id"]), fresh)
    old = []
    for path in sorted((ROOT / "tidal").glob("*data.py")):
        if path.stem == "subconscious_softmix_data":
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


def test_selection_prefers_participation_then_smaller_k_then_smaller_temperature():
    rows = [
        _row(8, 0.40, 3.0, 1.0, 1.0),
        _row(4, 0.40, 3.0, 1.2, 1.0),
        _row(4, 0.20, 2.0, 1.2, 1.0),
        _row(2, 0.40, 1.5, 0.9, 1.0),
        _row(4, 0.05, 1.04, 2.0, 1.0, slots_ok=True),
        _row(4, 0.10, 1.2, None, 1.0),
    ]
    # 0.40 at k=4 and k=8 tie on participation; smallest k wins. The 0.20 cell is less mixed.
    # The k=2 cell loses on the score. Participation 1.04 and a missing score do not pass.
    chosen = select_cell(rows)
    assert chosen["k"] == 4
    assert chosen["temperature"] == 0.40
    assert select_cell([_row(4, 0.20, 1.0, 0.5, 1.0)]) is None
    assert select_cell([]) is None


def _row(k, temperature, part, soft, top, slots_ok=True):
    return {
        "k": k,
        "temperature": temperature,
        "participation_mean": part,
        "soft_dz": soft,
        "top1_dz": top,
        "slots_ok": slots_ok,
    }


def test_held_rates_use_the_measurement_intervals():
    low, high = MEASURE_INTERVALS["prefer_delta_pos"]
    assert rate_inside(50, 50, (low, high))
    assert rate_inside(47, 50, (low, high))
    assert not rate_inside(46, 50, (low, high))
    low, high = MEASURE_INTERVALS["avoid_delta_pos"]
    assert rate_inside(33, 50, (low, high))
    assert rate_inside(43, 50, (low, high))
    assert not rate_inside(32, 50, (low, high))
    assert not rate_inside(44, 50, (low, high))
    low, high = MEASURE_INTERVALS["away_signed"]
    assert rate_inside(48, 50, (low, high))
    assert not rate_inside(50, 50, (low, high))
    assert not rate_inside(43, 50, (low, high))


def test_held_waits_for_the_fit_selection():
    fit = {"complete": True, "selected": {"k": 4, "temperature": 0.20}}
    assert held_allowed(fit, 4, 0.20, False)[0]
    assert not held_allowed(fit, 4, 0.05, False)[0]
    assert not held_allowed(fit, 2, 0.20, False)[0]
    assert not held_allowed(fit, 4, 0.20, True)[0]
    assert not held_allowed({"complete": True, "selected": None}, 4, 0.20, False)[0]
    assert not held_allowed(None, 4, 0.20, False)[0]


def test_fit_scorer_does_not_read_held_items():
    from tidal import subconscious_softmix_score as score

    source = inspect.getsource(score.score_fit)
    for name in ("prefer_items", "avoid_items", "toward_items", "away_items", "tone_items", "phone_items", "birthday_items", "score_direction", "score_signed"):
        assert name not in source
    tree = ast.parse((ROOT / "scripts" / "eval_subconscious_softmix.py").read_text(encoding="utf-8"))
    fit_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit")
    held_calls = [node.func.id for node in ast.walk(fit_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "_held" not in held_calls


def test_prereg_is_before_scoring_and_quotes_the_intervals():
    text = (ROOT / "reports" / "subconscious_softmix_prereg.md").read_text(encoding="utf-8")
    for low, high in MEASURE_INTERVALS.values():
        assert f"{low}" in text
        assert f"{high}" in text
    assert "Fit margins have not been scored." in text
    assert "The held-out bank has not been scored." in text
    assert "3ba1fab4559982dcf959121224de3d0be86e03a25a056dccd874c6534dcdd6aa" in text
    assert not (ROOT / "reports" / "subconscious_softmix_fit.json").exists()
    assert not (ROOT / "reports" / "subconscious_softmix_held.json").exists()
