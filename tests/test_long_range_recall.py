"""Long-range recall held-out harness. No SmolLM download."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from tidal.llm_memory import CUE_INTERFACE_VERSION, CUE_THETA
from tidal.long_range_recall_data import (
    CONDITIONS,
    FROZEN_CUE_INTERFACE,
    FROZEN_RECONSTRUCT_INTERFACE,
    FROZEN_RECONSTRUCT_SHA,
    H4_MIN,
    H5_MAX,
    ITEMS,
    ITEMS_N,
    N_VALUES,
    PACKET_KEYS,
    PROTOCOL,
    SEEDS,
    WEAK_COSINE,
    bank_sha256,
    item_by_id,
    mean_std,
    score_hypotheses,
    similar_turns,
)
from tidal.long_range_recall_eval import (
    Geometry,
    frozen_interface,
    play_dialogue,
    run_trial,
    schedule_ok,
    score_strong,
    score_weak,
)
from tidal.subconscious_reconstruct_data import (
    bank_sha256 as reconstruct_sha,
    canonical_surfaces,
)
from tidal.weak_cue import WEAK_CEILING, WEAK_FLOOR, context_fragment

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = (
    ROOT / "tidal" / "llm_memory.py",
    ROOT / "tidal" / "consolidation.py",
    ROOT / "tidal" / "sleep_phases.py",
    ROOT / "tidal" / "retroactive.py",
    ROOT / "tidal" / "weak_cue.py",
)
FLAG_NEEDLES = (
    "LONG_RANGE",
    "enable_long_range",
    "HARNESS_ABLATION",
    "TIDAL_LONG_RANGE",
    "os.environ",
)


def _imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    return imported


def _empty_cell(mean: float) -> dict:
    return {"mean": float(mean), "std": 0.0, "per_seed": [float(mean)] * 5, "hits": 0, "n": 50}


def _summary(full_strong_10, base_strong_10, full_strong_100, base_strong_100, full_weak_100, off83_weak_100, conflict, false_recall):
    def pack(strong, weak, newest, false_rate):
        return {
            "recall_at_1_strong": _empty_cell(strong),
            "hit_rate_weak": _empty_cell(weak),
            "conflict_newest_take": _empty_cell(newest),
            "false_recall": _empty_cell(false_rate),
        }

    return {
        "conditions": {
            "full": {
                "10": pack(full_strong_10, 0.0, conflict, false_recall),
                "30": pack(full_strong_10, 0.0, conflict, false_recall),
                "100": pack(full_strong_100, full_weak_100, conflict, false_recall),
            },
            "baseline": {
                "10": pack(base_strong_10, 0.0, 0.0, 0.0),
                "30": pack(base_strong_10, 0.0, 0.0, 0.0),
                "100": pack(base_strong_100, 0.0, 0.0, 0.0),
            },
            "off_83": {
                "10": pack(0.0, 0.0, 0.0, 0.0),
                "30": pack(0.0, 0.0, 0.0, 0.0),
                "100": pack(0.0, off83_weak_100, 0.0, 0.0),
            },
        }
    }


def test_protocol_matches_prereg_and_frozen_interface():
    prereg = (ROOT / "docs" / "long_range_recall_prereg.md").read_text(encoding="utf-8")
    assert PROTOCOL == "long-range-recall-held-v0"
    assert PROTOCOL in prereg
    assert SEEDS == (0, 1, 2, 3, 4)
    assert N_VALUES == (10, 30, 100)
    assert ITEMS_N == 10 == len(ITEMS)
    assert CUE_INTERFACE_VERSION == FROZEN_CUE_INTERFACE == 7
    surfaces = canonical_surfaces()
    assert surfaces["interface_version"] == FROZEN_RECONSTRUCT_INTERFACE == 6
    assert reconstruct_sha() == FROZEN_RECONSTRUCT_SHA
    assert FROZEN_RECONSTRUCT_SHA in prereg
    assert "The held bank was not scored." in prereg
    assert "The hypothesis gates were not scored." in prereg
    assert "do not retune" in prereg.lower() or "not be edited" in prereg
    for row in ITEMS:
        assert row["answer"] in row["text"]
        assert row["conflict_answer"] in row["conflict_text"]
        assert row["text"] in prereg
        assert row["weak_cue"] in prereg
        assert row["unrelated"] in prereg
    assert ITEMS[0]["gloss_plant"] == "那天吃了抹茶蛋糕"
    assert ITEMS[0]["gloss_weak"] == "我又吃到了"
    assert similar_turns(30) == (1, 10)
    assert similar_turns(100) == (1, 33)
    assert len(bank_sha256()) == 64
    assert CUE_THETA == 0.32
    iface = frozen_interface()
    assert iface["cue_interface_version"] == 7
    assert iface["reconstruct_interface_version"] == 6
    assert tuple(iface["packet_keys"]) == PACKET_KEYS


def test_runtime_modules_have_no_eval_flag():
    for path in RUNTIME:
        text = path.read_text(encoding="utf-8")
        for needle in FLAG_NEEDLES:
            assert needle not in text
        tree = ast.parse(text)
        imported = _imports(text)
        assert "os" not in imported or path.name == "llm_memory.py"
        if path.name != "llm_memory.py":
            assert "environ" not in text
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "def remember" in memory
    assert "long_range" not in memory


def test_harness_modules_do_not_import_transformers():
    import tidal.long_range_recall_data as data
    import tidal.long_range_recall_eval as ev

    for mod in (data, ev):
        assert "transformers" not in _imports(inspect.getsource(mod))


def test_locked_criteria_do_not_treat_ties_as_passes():
    tied = _summary(1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 0.80, 0.05)
    rows = score_hypotheses(tied)
    assert rows["H1"]["pass"] is False
    assert rows["H2"]["pass"] is False
    assert rows["H3"]["pass"] is False
    assert rows["H4"]["pass"] is True
    assert rows["H4"]["full"] == pytest.approx(0.80)
    assert rows["H5"]["full"] == pytest.approx(0.05)
    assert rows["H5"]["pass"] is True
    assert H4_MIN == 0.80
    assert H5_MAX == 0.05
    winning = _summary(0.9, 0.1, 0.8, 0.2, 0.5, 0.0, 0.9, 0.0)
    won = score_hypotheses(winning)
    assert won["H1"]["pass"] and won["H2"]["pass"] and won["H3"]["pass"]
    assert won["H4"]["pass"] and won["H5"]["pass"]
    miss_conflict = _summary(0.9, 0.1, 0.8, 0.2, 0.5, 0.0, 0.79, 0.06)
    missed = score_hypotheses(miss_conflict)
    assert missed["H4"]["pass"] is False
    assert missed["H5"]["pass"] is False
    mean, std = mean_std([0.0, 1.0])
    assert mean == 0.5
    assert std == pytest.approx(0.5 ** 0.5)


def test_geometry_lands_in_the_locked_bands():
    item = item_by_id(0)
    geom = Geometry(item)
    strong = geom.vector(item["strong_cue"])
    weak = geom.vector(item["weak_cue"])
    unrelated = geom.vector(item["unrelated"])
    target = geom.vector(item["text"])
    similar = geom.vector("Near 0-0-1 shares the same day.")
    assert np.allclose(strong, target)
    assert float(np.dot(target, weak) / (np.linalg.norm(target) * np.linalg.norm(weak))) == pytest.approx(
        WEAK_COSINE, abs=1e-9
    )
    assert float(np.dot(target, unrelated)) == pytest.approx(0.0, abs=1e-9)
    assert float(np.dot(target, similar) / (np.linalg.norm(target) * np.linalg.norm(similar))) == pytest.approx(
        0.95, abs=1e-9
    )
    fragment = context_fragment(target, WEAK_COSINE)
    assert WEAK_FLOOR <= float(np.dot(target, fragment) / (np.linalg.norm(target) * np.linalg.norm(fragment))) < WEAK_CEILING


def test_schedule_and_public_api_smoke():
    mem, _geom, trace = play_dialogue(seed=0, item_id=0, n=10, family="recall", condition="full")
    try:
        names = [row[0] for row in trace]
        assert names[0] == "remember"
        assert ("wake_for", 1.0) in trace
        assert ("offline_ticks", 8) in trace
        assert all(row[0] != "sleep_for" for row in trace)
        assert schedule_ok(trace, 10)
        strong = score_strong(mem, item_by_id(0))
        assert strong["keys"] == PACKET_KEYS or set(strong["keys"]) == set(PACKET_KEYS)
        assert CUE_INTERFACE_VERSION == 7
    finally:
        mem.close()
    mid = play_dialogue(seed=0, item_id=0, n=30, family="recall", condition="full")
    try:
        assert schedule_ok(mid[2], 30)
        assert ("sleep_for", 2.0) in mid[2]
    finally:
        mid[0].close()
    long = play_dialogue(seed=0, item_id=0, n=100, family="recall", condition="full")
    try:
        assert schedule_ok(long[2], 100)
        assert ("sleep_for", 2.0) in long[2]
        assert ("offline_ticks", 16) in long[2]
    finally:
        long[0].close()


def test_strong_hit_and_unrelated_miss_on_one_item():
    strong = run_trial(seed=0, item_id=0, n=10, family="recall", condition="full", probe="strong")
    unrelated = run_trial(seed=0, item_id=0, n=10, family="unrelated", condition="full")
    conflict = run_trial(seed=0, item_id=0, n=10, family="conflict", condition="full")
    assert strong["hit"] is True
    assert strong["source"] == "cue"
    assert strong["winner_cosine"] >= CUE_THETA
    assert unrelated["hit"] is False
    assert conflict["hit"] is True
    assert conflict["answer"] == "strawberry cake"


def test_off_83_is_a_harness_patch_and_blocks_weak_hits():
    full = run_trial(seed=0, item_id=0, n=10, family="recall", condition="full", probe="weak")
    off = run_trial(seed=0, item_id=0, n=10, family="recall", condition="off_83", probe="weak")
    assert full["hit"] is True
    assert off["hit"] is False
    source = Path(inspect.getfile(score_weak)).read_text(encoding="utf-8")
    assert "_condition_patches" in source
    assert "weak_winner" in source
    assert "os.environ" not in source
