"""Sleep two-phase SWS gist and REM bind/emotion. No SmolLM download."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import AROUSAL_BASELINE, AROUSAL_TAU, AROUSAL_WINDOW
from tidal.backend import GROWBRAIN_D_MODEL, GROWBRAIN_MEM_DIM, GROWBRAIN_MEM_KEY_DIM
from tidal.consolidation import consolidation_gain
from tidal.llm_memory import (
    ByteTokenizer,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_THETA,
    LLMMemory,
)
from tidal.pattern_separation import match_scores, separate_keys
from tidal.reconstruct import recall_surface
from tidal.sleep import PE_THRESHOLD, REPLAY_BLEND, STRENGTH_STEP, prediction_error, reconsolidate
from tidal.sleep_phases import (
    PHASES,
    REM_BIND,
    REM_VALENCE_KEEP,
    apply_entry_phase,
    apply_exit_phase,
    gist_holds,
    gist_stronger_peripheral_weaker,
    peripheral_changed,
    rem_bind_pair,
    rem_refresh_valence,
    sleep_phase,
    sws_age,
)
from tidal.subconscious_reconstruct_data import BIRTHDAY, PHONE, canonical_surfaces as reconstruct_surfaces
from tidal.subconscious_sleep_phases_data import (
    AROUSAL_HIGH,
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    VALENCE_HIGH,
    bank_sha256,
    canonical_rem,
    canonical_surfaces,
    formula_gates,
    schedule_pressures,
)

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1038/nrn2762",
    "10.1038/s41593-019-0467-3",
)
UNSCORED = (
    "The held bank was not scored.",
    "The cue guard was not scored.",
    "The sleep guard was not scored.",
    "The arousal guard was not scored.",
)
EXPORT_KEYS = {
    "interface_version",
    "cue_dim",
    "memory_id",
    "vec",
    "activation",
    "strength_confidence",
    "valence",
    "modality",
    "circadian_phase",
    "triggered",
    "gap",
    "winner_cosine",
    "sleep_pressure",
    "arousal",
    "source",
    "fingerprint",
    "fusion_confidence",
}


class _Eye(nn.Module):
    def __init__(self):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, vocab)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.lm_head = nn.Linear(vocab, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.eye(vocab))
            self.lm_head.weight.copy_(torch.eye(vocab))
        self.config = type("Cfg", (), {"hidden_size": vocab, "vocab_size": vocab})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **_kwargs):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        return type("Out", (), {"logits": logits, "hidden_states": tuple(states) if output_hidden_states else None})()


def _memory():
    centers = LLMMemory._key_centers

    def zeros(self):
        row = np.zeros(self.hidden, dtype=np.float64)
        return row, row.copy()

    LLMMemory._key_centers = zeros
    try:
        return LLMMemory(_Eye(), ByteTokenizer())
    finally:
        LLMMemory._key_centers = centers


def _imports(source: str) -> list:
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    return imported


def _axis(mem, index: int):
    row = np.zeros(mem.hidden, dtype=np.float64)
    row[int(index)] = 1.0
    return row


def _spelling(slot) -> str:
    return recall_surface(slot.residue, float(slot.strength), int(slot.age), float(slot.valence))


def test_formula_gates_lock_phase_gist_and_rem_numbers():
    pressures = schedule_pressures()
    surfaces = canonical_surfaces()
    rem = canonical_rem()
    assert sleep_phase(pressures["floor"], 0.0) == "none"
    assert sleep_phase(pressures["after_wake"], 0.0) == "sws"
    assert sleep_phase(pressures["after_sws"], 0.0) == "sws"
    assert sleep_phase(pressures["after_rem"], 0.0) == "rem"
    assert sleep_phase(pressures["floor"], 1.0) == "rem"
    assert apply_entry_phase("sws") is True
    assert apply_entry_phase("rem") is False
    assert apply_entry_phase("none") is False
    assert apply_exit_phase("rem") is True
    assert apply_exit_phase("sws") is False
    assert apply_exit_phase("none") is False
    assert PHASES == ("sws", "rem", "none")
    assert surfaces["phone_exact"] == PHONE
    assert surfaces["birthday_exact"] == BIRTHDAY
    assert surfaces["phone_sws"] == "+1-617-555-4862"
    assert surfaces["birthday_sws"] == "April 12, 1984"
    assert rem["valence_after"] == pytest.approx(1.35)
    assert rem["bind_cosine"] == pytest.approx(0.10497237569060777)
    assert rem["gain_after"] == consolidation_gain(AROUSAL_HIGH, rem["valence_after"])
    row = formula_gates()
    assert row["identity"] and row["sws"] and row["rem_emotion"] and row["rem_bind"] and row["compose"]
    with pytest.raises(ValueError, match="sleep pressure"):
        sleep_phase(0.0, 0.0)
    with pytest.raises(TypeError):
        sleep_phase(True, 0.0)
    with pytest.raises(ValueError, match="circadian"):
        sleep_phase(0.2, float("nan"))
    with pytest.raises(RuntimeError, match="unhandled sleep phase"):
        apply_entry_phase("nrem")


def test_sws_sleep_weakens_peripheral_against_replay_only():
    surfaces = canonical_surfaces()
    replay = _memory()
    try:
        key = _axis(replay, 0)
        assert replay.store.write(
            key,
            key.copy(),
            name="ada",
            rel="phone",
            group="ada",
            answer=PHONE,
            text="raw-phone-sws",
        )
        slot = replay.store.slots[0]
        assert slot.age == 0
        assert _spelling(slot) == PHONE
        report = replay.sleep_for(1.0)
        slot = replay.store.slots[0]
        assert replay.sleep_pressure == pytest.approx(schedule_pressures()["floor"])
        assert slot.age == 0
        assert _spelling(slot) == PHONE
        assert slot.text == "raw-phone-sws" and slot.answer == PHONE
        assert report["replay"][0]["updated"] is False
        assert gist_stronger_peripheral_weaker(slot.residue, 1.0, 0, 0.0) is False
    finally:
        replay.close()
    sws = _memory()
    try:
        key = _axis(sws, 0)
        assert sws.store.write(
            key,
            key.copy(),
            name="ada",
            rel="phone",
            group="ada",
            answer=PHONE,
            text="raw-phone-sws",
        )
        birthday_key = _axis(sws, 2)
        assert sws.store.write(
            birthday_key,
            birthday_key.copy(),
            name="ada",
            rel="birthday",
            group="ada",
            answer=BIRTHDAY,
            text="raw-birthday-sws",
        )
        sws.wake_for(4.0)
        report = sws.sleep_for(1.0)
        phone, birthday = sws.store.slots
        assert phone.age == 1 and birthday.age == 1
        assert _spelling(phone) == surfaces["phone_sws"]
        assert _spelling(birthday) == surfaces["birthday_sws"]
        assert gist_holds(phone.residue, _spelling(phone))
        assert peripheral_changed(phone.residue, _spelling(phone))
        assert gist_holds(birthday.residue, _spelling(birthday))
        assert peripheral_changed(birthday.residue, _spelling(birthday))
        assert phone.text == "raw-phone-sws" and phone.answer == PHONE
        assert birthday.answer == BIRTHDAY
        assert phone.strength == pytest.approx(1.0)
        assert np.allclose(phone.v, key)
        assert "replay_phase" not in report
        blob = str(report)
        assert "raw-phone-sws" not in blob
    finally:
        sws.close()


def test_rem_pass_refreshes_emotion_and_binds_same_group_values():
    rem = canonical_rem()
    mem = _memory()
    try:
        left = _axis(mem, 0)
        right = _axis(mem, 1)
        outsider = _axis(mem, 3)
        assert mem.store.write(left, left.copy(), name="left", rel="note", group="pair", answer="L", text="raw-left")
        assert mem.store.write(right, right.copy(), name="right", rel="note", group="pair", answer="R", text="raw-right")
        assert mem.store.write(outsider, outsider.copy(), name="out", rel="note", group="other", answer="O", text="raw-out")
        mem.store.slots[0].valence = float(VALENCE_HIGH)
        mem.store.slots[0].consolidation_gain = consolidation_gain(float(AROUSAL_HIGH), float(VALENCE_HIGH))
        mem.store.slots[1].valence = -float(VALENCE_HIGH)
        mem._arousal = float(AROUSAL_HIGH)
        before_keys = [np.asarray(slot.k, dtype=np.float64).copy() for slot in mem.store.slots]
        mem.store.clock_set(math.pi / 2.0)
        assert sleep_phase(mem.sleep_pressure, mem.store.clock_phase()) == "rem"
        report = mem.sleep_for(1.0)
        slots = mem.store.slots
        assert slots[0].age == 0
        assert slots[0].valence == pytest.approx(rem["valence_after"])
        assert slots[1].valence == pytest.approx(rem["valence_neg_after"])
        assert slots[2].valence == 0.0
        assert slots[0].consolidation_gain == pytest.approx(rem["gain_after"])
        assert rem["gain_after"] < rem["gain_before"]
        bound = float(np.dot(slots[0].v, slots[1].v) / (np.linalg.norm(slots[0].v) * np.linalg.norm(slots[1].v)))
        assert bound == pytest.approx(rem["bind_cosine"])
        assert np.allclose(slots[2].v, outsider)
        for index, key in enumerate(before_keys):
            assert np.array_equal(slots[index].k, key)
        assert slots[0].text == "raw-left" and slots[0].answer == "L"
        assert "replay_phase" not in report
        assert float(mem.fusion_confidence) == 0.0
        assert consolidation_gain(AROUSAL_BASELINE, float(VALENCE_HIGH)) == 1.0
    finally:
        mem.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert "replay_phase" not in fields
        assert float(fields["fusion_confidence"]) == 0.0
        assert "consolidation_gain" not in fields
    finally:
        mem.close()
    surfaces = reconstruct_surfaces()
    assert surfaces["interface_version"] == 6
    assert GROWBRAIN_D_MODEL == 192
    assert GROWBRAIN_MEM_DIM == 64
    assert GROWBRAIN_MEM_KEY_DIM == 32
    assert schedule_pressures()["floor"] == 0.1
    assert AROUSAL_BASELINE == 0.2
    assert AROUSAL_TAU == 8.0
    assert AROUSAL_WINDOW == 8
    assert CUE_THETA == 0.32
    assert CUE_HALF_LIFE == 30.0
    assert CUE_LAMBDA == 0.3
    assert REM_VALENCE_KEEP == 0.9
    assert REM_BIND == 0.05
    key = np.zeros(4, dtype=np.float64)
    key[0] = 1.0
    parallel = reconsolidate(key, key.copy(), 1.0)
    assert parallel["updated"] is False
    orthogonal = np.zeros(4, dtype=np.float64)
    orthogonal[1] = 1.0
    changed = reconsolidate(key, orthogonal, 1.0)
    assert changed["updated"] is True
    assert changed["strength"] == pytest.approx(math.exp(-STRENGTH_STEP))
    assert prediction_error(key, orthogonal) > PE_THRESHOLD
    expected = (1.0 - REPLAY_BLEND) * orthogonal + REPLAY_BLEND * key
    assert np.allclose(changed["value"], expected)
    left = np.zeros((2, 4), dtype=np.float64)
    left[0, 0] = 1.0
    left[1, 1] = 1.0
    assert np.allclose(separate_keys(left), left)
    assert np.allclose(match_scores(left[0], left), np.array([1.0, 0.0]))
    assert consolidation_gain(AROUSAL_BASELINE, VALENCE_HIGH) == 1.0


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "sleep-two-phase-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "63775a5bc64260c1996ded7bed7628e6b698d6a0175a28746c01e298774d2caf"
    assert set(GATES) == {
        "identity",
        "sws",
        "rem_emotion",
        "rem_bind",
        "compose",
        "export",
        "replay",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["sws"], item_id
        assert row["rem_emotion"], item_id
        assert row["rem_bind"], item_id
        assert row["compose"], item_id


def test_sleep_phases_module_has_no_switch_and_docs_name_the_formula():
    source = (ROOT / "tidal" / "sleep_phases.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_sleep_phases_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "sleep_phase" not in residual
    assert "replay_phase" not in residual
    assert "replay_phase" not in memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_sleep_phases_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        for doi in DOIS:
            assert doi in text
        assert "10.1007/BF00487332" not in text
        assert "replay_phase" in text
    assert "HYPOTHESIS" in source
    assert "HYPOTHESIS" in prereg
    for sentence in UNSCORED:
        assert sentence in prereg
        assert sentence in readme
        assert sentence in architecture
    assert bank_sha256() in prereg
    assert bank_sha256() in readme
    assert bank_sha256() in architecture
    assert "slow oscillation" in source.lower() or "spindles" in source.lower()
    assert "does not simulate" in source
    assert REM_VALENCE_KEEP == 0.9
    assert REM_BIND == 0.05
    assert sws_age(0) == 1
    assert rem_refresh_valence(0.0) == 0.0
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    left, right = rem_bind_pair(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert float(np.dot(left, right)) == pytest.approx(canonical_rem()["bind_cosine"])
