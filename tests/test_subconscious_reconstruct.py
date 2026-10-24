"""Reconstructive phone and birthday recall. No SmolLM download."""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from tidal.llm_memory import CUE_INTERFACE_VERSION, LLMMemory
from tidal.reconstruct import (
    EXACT_AGE,
    EXACT_STRENGTH,
    EXACT_VALENCE,
    LONG_AGE,
    LOSS_LINE_OFFSET,
    LOSS_STRENGTH,
    grade,
    loss_day,
    loss_line,
    recall_surface,
    residue_from_answer,
    slip_day,
    slip_line,
)
from tidal.subconscious_reconstruct_data import (
    BIRTHDAY,
    FIT_IDS,
    HELD_IDS,
    LURE_BIRTHDAY,
    LURE_PHONE,
    PHONE,
    PROTOCOL,
    bank_sha256,
    canonical_surfaces,
)

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1017/CBO9780511759185",
    "10.1016/0010-0285(81)90008-6",
    "10.1177/1754073908100432",
)
UNSCORED = (
    "The held bank was not scored.",
    "The cue guard was not scored.",
    "The sleep guard was not scored.",
    "The arousal guard was not scored.",
    "The clock guard was not scored.",
)


def _imports(source: str) -> list:
    tree = ast.parse(source)
    found = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module.split(".")[0])
    return found


def test_grade_boundaries_and_probe_spellings():
    assert grade(1.0, 0, 0.0) == "exact"
    assert grade(EXACT_STRENGTH, EXACT_AGE, 0.0) == "exact"
    assert grade(EXACT_STRENGTH, EXACT_AGE, EXACT_VALENCE) == "slip"
    assert grade(EXACT_STRENGTH, 1, 0.0) == "slip"
    assert grade(LOSS_STRENGTH, 0, 0.0) == "slip"
    assert grade(LOSS_STRENGTH - 0.001, 0, 0.0) == "loss"
    assert grade(1.0, LONG_AGE, 0.0) == "loss"
    assert grade(1.0, 0, 1.5) == "slip"
    assert grade(0.5, 0, 0.0) == "slip"
    assert grade(float("nan"), 0, 0.0) == "loss"
    assert slip_line(4826) == 4862
    assert loss_line(4826) == 5937
    assert slip_line(1111) == 1112
    assert slip_day(21) == 12
    assert slip_day(11) == 10
    assert loss_day(21) == 1
    assert loss_day(1) == 2
    assert LOSS_LINE_OFFSET == 1111
    surfaces = canonical_surfaces()
    assert surfaces["phone_high"] == PHONE == "+1-617-555-4826"
    assert surfaces["birthday_high"] == BIRTHDAY == "April 21, 1984"
    assert surfaces["phone_low"] == surfaces["phone_long"] == "+1-617-555-5937"
    assert surfaces["birthday_low"] == surfaces["birthday_long"] == "April 1, 1984"
    assert surfaces["phone_emotion"] == surfaces["phone_slip"] == "+1-617-555-4862"
    assert surfaces["birthday_emotion"] == surfaces["birthday_slip"] == "April 12, 1984"
    assert surfaces["off_relation"] is None
    assert surfaces["interface_version"] == 5
    for surface in surfaces.values():
        if isinstance(surface, str):
            assert surface not in (LURE_PHONE, LURE_BIRTHDAY)


def test_residue_drops_the_raw_string_and_city_has_none():
    phone = residue_from_answer(PHONE)
    birthday = residue_from_answer(BIRTHDAY)
    assert phone == ("phone", 617, 555, 4826)
    assert birthday[0] == "birthday"
    assert PHONE not in repr(phone)
    assert BIRTHDAY not in repr(birthday)
    assert residue_from_answer("Oslo") is None
    assert residue_from_answer(None) is None
    assert recall_surface(None, 1.0, 0, 0.0) is None
    assert recall_surface(phone, 0.2, 0, 0.0) != PHONE


def test_reconstruct_module_has_no_switch_and_banks_are_disjoint():
    source = (ROOT / "tidal" / "reconstruct.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "time" not in imported
    assert "environ" not in source
    assert "HYPOTHESIS" in source
    assert "synonym" in source
    assert "10.1007/BF00487332" not in source
    data = (ROOT / "tidal" / "subconscious_reconstruct_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "import growbrain" not in memory
    assert "environ" not in memory
    spelling = memory.split("def _stored_spelling")[1].split("def _spell_ids")[0]
    assert "slot.answer" not in spelling
    assert "recall_surface" in spelling
    assert "PHONE_TAIL" not in spelling
    assert CUE_INTERFACE_VERSION == 5
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert PROTOCOL == "reconstruct-spell-v1"
    assert PHONE not in ("+1-449-555-3249", "+1-802-555-3145")
    assert BIRTHDAY != "July 26, 1990"
    assert len(bank_sha256()) == 64
    prereg = (ROOT / "reports" / "subconscious_reconstruct_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        for doi in DOIS:
            assert doi in text
        assert "10.1007/BF00487332" not in text
    assert "HYPOTHESIS" in prereg
    for sentence in UNSCORED:
        assert sentence in prereg
        assert sentence in readme
        assert sentence in architecture
    assert bank_sha256() in prereg
    assert bank_sha256() in readme
    assert bank_sha256() in architecture
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture


def test_one_live_item_matches_the_locked_surfaces():
    path = ROOT / "scripts" / "eval_subconscious_reconstruct.py"
    spec = importlib.util.spec_from_file_location("eval_subconscious_reconstruct", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module._patch_centers()
    try:
        row = module.score_item(FIT_IDS[0])
    finally:
        LLMMemory._key_centers = original
    assert row["pass"] is True
    assert row["intrusion_event"] is False
    assert row["canonical"] == canonical_surfaces()
    assert row["live"]["phone_low"] != PHONE
    assert row["live"]["phone_high"] == PHONE
