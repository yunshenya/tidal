"""Encode vs consolidate mode from live scalars. No SmolLM download."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import AROUSAL_BASELINE, AROUSAL_TAU, AROUSAL_WINDOW
from tidal.backend import GROWBRAIN_D_MODEL, GROWBRAIN_MEM_DIM, GROWBRAIN_MEM_KEY_DIM
from tidal.consolidation import consolidation_gain
from tidal.fusion import score_fusion
from tidal.llm_memory import (
    ByteTokenizer,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_THETA,
    LLMMemory,
)
from tidal.memory_mode import (
    CONSOLIDATE_WRITE_KEEP,
    ENCODE_SPONTANEOUS_KEEP,
    ENCODE_WRITE_KEEP,
    MODES,
    apply_sleep_operators,
    memory_mode,
    spontaneous_pressure,
    write_keep,
)
from tidal.pattern_separation import match_scores, separate_keys
from tidal.reconsolidation import window_open
from tidal.reconstruct import recall_surface
from tidal.retroactive import neighbor_keep
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_PERIOD, reconsolidate, wake_after
from tidal.sleep_phases import sleep_phase, sws_age
from tidal.subconscious_mode_data import (
    AROUSAL_HIGH,
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    bank_sha256,
    canonical_surfaces,
    formula_gates,
    schedule_scalars,
)
from tidal.subconscious_reconstruct_data import BIRTHDAY, PHONE, canonical_surfaces as reconstruct_surfaces

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1016/S0079-6123(03)45015-2",
    "10.1038/nrn2881",
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


def _quiet_write(mem):
    mem._value_and_surprise = lambda _text, _answer: (
        np.zeros(mem.hidden, dtype=np.float32),
        1.0,
    )


def _axis(mem, index: int):
    row = np.zeros(mem.hidden, dtype=np.float64)
    row[int(index)] = 1.0
    return row


def _encode_key(mem, key):
    stored = np.asarray(key, dtype=np.float64).reshape(-1)

    def encode(_text, answer=None):
        del _text, answer
        return stored.astype(np.float32, copy=False)

    mem.encode_key = encode


def _spelling(slot) -> str:
    return recall_surface(slot.residue, float(slot.strength), int(slot.age), float(slot.valence))


def test_formula_gates_lock_encode_vs_consolidate():
    scalars = schedule_scalars()
    surfaces = canonical_surfaces()
    assert memory_mode(scalars["floor"], scalars["baseline"], 0.0) == "encode"
    assert memory_mode(scalars["after_wake"], scalars["baseline"], 0.0) == "consolidate"
    assert memory_mode(scalars["after_wake"], scalars["arousal_high"], 0.0) == "encode"
    assert memory_mode(scalars["after_wake"], scalars["baseline"], 1.0) == "encode"
    assert memory_mode(scalars["floor"], scalars["baseline"], -1.0) == "encode"
    assert write_keep("encode") == ENCODE_WRITE_KEEP == 1.0
    assert write_keep("consolidate") == CONSOLIDATE_WRITE_KEEP == 0.75
    assert ENCODE_SPONTANEOUS_KEEP == 0.5
    assert surfaces["phone_encode"] == PHONE
    assert surfaces["birthday_encode"] == BIRTHDAY
    assert surfaces["phone_consolidate"] == "+1-617-555-4862"
    assert surfaces["birthday_consolidate"] == "April 12, 1984"
    assert apply_sleep_operators("encode") is True
    assert apply_sleep_operators("consolidate") is True
    assert MODES == ("encode", "consolidate")
    row = formula_gates()
    assert row["identity"] and row["encode"] and row["consolidate"]
    assert row["write"] and row["sleep"] and row["compose"]
    with pytest.raises(ValueError, match="sleep pressure"):
        memory_mode(0.0, 0.2, 0.0)
    with pytest.raises(TypeError):
        memory_mode(True, 0.2, 0.0)
    with pytest.raises(ValueError, match="arousal"):
        memory_mode(0.2, 1.5, 0.0)
    with pytest.raises(ValueError, match="circadian"):
        memory_mode(0.2, 0.2, float("nan"))
    with pytest.raises(RuntimeError, match="unhandled memory mode"):
        write_keep("nrem")


def test_same_remember_write_differs_under_encode_vs_consolidate():
    surfaces = canonical_surfaces()
    encode = _memory()
    try:
        _quiet_write(encode)
        _encode_key(encode, _axis(encode, 0))
        assert encode.encoding_mode() == "encode"
        assert encode.remember(f"Ada phone is {PHONE}", answer=PHONE, name="ada", rel="phone")
        slot = encode.store.slots[0]
        assert slot.strength == 1.0
        assert slot.age == 0
        assert _spelling(slot) == surfaces["phone_encode"] == PHONE
        assert slot.consolidation_gain == 1.0
        assert float(encode.fusion_confidence) == 0.0
    finally:
        encode.close()
    consolidate = _memory()
    try:
        _quiet_write(consolidate)
        _encode_key(consolidate, _axis(consolidate, 0))
        consolidate.wake_for(4.0)
        assert consolidate.encoding_mode() == "consolidate"
        assert consolidate.remember(f"Ada phone is {PHONE}", answer=PHONE, name="ada", rel="phone")
        slot = consolidate.store.slots[0]
        assert slot.strength == pytest.approx(CONSOLIDATE_WRITE_KEEP)
        assert slot.age == 0
        assert _spelling(slot) == surfaces["phone_consolidate"]
        assert slot.text == f"Ada phone is {PHONE}" and slot.answer == PHONE
        assert slot.consolidation_gain == 1.0
        assert neighbor_keep(0.5) == 1.0
    finally:
        consolidate.close()


def test_same_offline_ticks_differ_and_sleep_operators_still_compose():
    scalars = schedule_scalars()
    consolidate = _memory()
    try:
        key = _axis(consolidate, 0)
        assert consolidate.store.write(key, key.copy(), name="wander", rel="note", answer="STAY", text="raw-mode-sleep")
        consolidate.wake_for(4.0)
        assert consolidate.encoding_mode() == "consolidate"
        ticks = consolidate.offline_ticks(SPONTANEOUS_PERIOD)
        sources = [tick["source"][0][0] for tick in ticks]
        assert sources.count("spontaneous") == 1
        assert sources.count("none") == SPONTANEOUS_PERIOD - 1
        assert consolidate.sleep_pressure == pytest.approx(scalars["after_wake"])
    finally:
        consolidate.close()
    encode = _memory()
    try:
        key = _axis(encode, 0)
        assert encode.store.write(key, key.copy(), name="wander", rel="note", answer="STAY", text="raw-mode-sleep")
        encode.wake_for(4.0)
        encode._arousal = float(AROUSAL_HIGH)
        assert encode.encoding_mode() == "encode"
        assert spontaneous_pressure("encode", encode.sleep_pressure) < scalars["spontaneous_min"]
        ticks = encode.offline_ticks(SPONTANEOUS_PERIOD)
        sources = [tick["source"][0][0] for tick in ticks]
        assert sources.count("spontaneous") == 0
        assert sources.count("none") == SPONTANEOUS_PERIOD
        assert encode.sleep_pressure == pytest.approx(scalars["after_wake"])
        assert encode.store.slots[0].age == 0
        report = encode.sleep_for(1.0)
        assert encode.store.slots[0].age == sws_age(0) == 1
        assert sleep_phase(scalars["after_wake"], 0.0) == "sws"
        assert "replay_phase" not in report
        assert encode.store.slots[0].text == "raw-mode-sleep"
    finally:
        encode.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert "mode" not in fields
        assert "encoding_mode" not in fields
        assert "write_keep" not in fields
        assert "consolidation_gain" not in fields
        assert score_fusion(()) == 0.0
        assert consolidation_gain(AROUSAL_BASELINE, 0.0) == 1.0
        assert not window_open(0, 0.0, 0, 0.0)
    finally:
        mem.close()
    surfaces = reconstruct_surfaces()
    assert surfaces["interface_version"] == 6
    assert GROWBRAIN_D_MODEL == 192
    assert GROWBRAIN_MEM_DIM == 64
    assert GROWBRAIN_MEM_KEY_DIM == 32
    assert SLEEP_FLOOR == 0.1
    assert AROUSAL_BASELINE == 0.2
    assert AROUSAL_TAU == 8.0
    assert AROUSAL_WINDOW == 8
    assert CUE_THETA == 0.32
    assert CUE_HALF_LIFE == 30.0
    assert CUE_LAMBDA == 0.3
    key = np.zeros(4, dtype=np.float64)
    key[0] = 1.0
    assert reconsolidate(key, key.copy(), 1.0)["updated"] is False
    left = np.zeros((2, 4), dtype=np.float64)
    left[0, 0] = 1.0
    left[1, 1] = 1.0
    assert np.allclose(separate_keys(left), left)
    assert np.allclose(match_scores(left[0], left), np.array([1.0, 0.0]))
    assert neighbor_keep(1.0) == 0.75
    assert wake_after(SLEEP_FLOOR, 4.0) == pytest.approx(schedule_scalars()["after_wake"])


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "encode-consolidate-mode-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "11a2828810dbe8cb41f1288772b959c9a573a940174f0b5ac3d44ff2a773b64a"
    assert set(GATES) == {
        "identity",
        "encode",
        "consolidate",
        "write",
        "sleep",
        "compose",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["encode"], item_id
        assert row["consolidate"], item_id
        assert row["write"], item_id
        assert row["sleep"], item_id
        assert row["compose"], item_id


def test_mode_module_has_no_flag_and_docs_name_the_papers():
    source = (ROOT / "tidal" / "memory_mode.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_mode_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "encoding_mode" not in residual
    assert "write_keep" not in residual
    packed = memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "encoding_mode" not in packed
    assert "write_keep" not in packed
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_mode_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        for doi in DOIS:
            assert doi in text
        assert "10.1007/BF00487332" not in text
    assert "HYPOTHESIS" in source
    assert "HYPOTHESIS" in prereg
    for sentence in UNSCORED:
        assert sentence in prereg
        assert sentence in readme
        assert sentence in architecture
    assert bank_sha256() in prereg
    assert bank_sha256() in readme
    assert bank_sha256() in architecture
    assert "does not reimplement" in source
    assert ENCODE_WRITE_KEEP == 1.0
    assert CONSOLIDATE_WRITE_KEEP == 0.75
    assert ENCODE_SPONTANEOUS_KEEP == 0.5
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
