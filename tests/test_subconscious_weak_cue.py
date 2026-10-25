"""Weak-overlap spontaneous recollection. No SmolLM download."""
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
from tidal.memory_mode import ENCODE_SPONTANEOUS_KEEP, apply_sleep_operators, memory_mode, spontaneous_pressure
from tidal.pattern_separation import match_scores, separate_keys
from tidal.reconsolidation import window_open
from tidal.retroactive import neighbor_keep
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_PERIOD, reconsolidate, wake_after
from tidal.sleep_phases import sleep_phase, sws_age
from tidal.subconscious_reconstruct_data import canonical_surfaces as reconstruct_surfaces
from tidal.subconscious_weak_cue_data import (
    AROUSAL_HIGH,
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    bank_sha256,
    formula_gates,
    schedule_scalars,
)
from tidal.weak_cue import (
    WEAK_CEILING,
    WEAK_FLOOR,
    WEAK_OVERLAP,
    context_fragment,
    in_weak_band,
    overlap_score,
    weak_winner,
)

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.1098/rstb.2019.0693"
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


def _plant_pair(mem):
    target = _axis(mem, 0)
    decoy = _axis(mem, 2)
    assert mem.store.write(target, target.copy(), name="target", rel="note", answer="T", text="raw-weak-target")
    assert mem.store.write(decoy, decoy.copy(), name="decoy", rel="note", answer="D", text="raw-weak-decoy")
    mem.store.slots[0].salience = 0.1
    mem.store.slots[1].salience = 0.9
    mem.store.slots[0].strength = 1.0
    mem.store.slots[1].strength = 1.0
    mem.store.slots[0].consolidation_gain = 1.0
    mem.store.slots[1].consolidation_gain = 1.0
    return target, decoy


def test_formula_gates_lock_weak_band_against_strong_cue():
    scalars = schedule_scalars()
    assert WEAK_OVERLAP == 0.25
    assert WEAK_FLOOR == 0.10
    assert WEAK_CEILING == CUE_THETA == 0.32
    assert scalars["fragment_cosine"] == 0.25
    assert scalars["encode_pressure"] == pytest.approx(0.1495396476178678)
    assert in_weak_band(0.25) is True
    assert in_weak_band(0.32) is False
    assert in_weak_band(1.0) is False
    assert in_weak_band(0.0) is False
    key = np.zeros(8, dtype=np.float64)
    key[0] = 1.0
    fragment = context_fragment(key)
    assert overlap_score(fragment, key) == pytest.approx(0.25)
    assert weak_winner([0.25, 0.0], [0.1, 0.9]) == 0
    assert weak_winner([1.0, 0.25]) == 1
    assert weak_winner([0.0, 0.0]) is None
    row = formula_gates()
    assert row["identity"] and row["weak"] and row["strong"]
    assert row["consolidate"] and row["encode"] and row["compose"]
    assert row["export"]
    with pytest.raises(ValueError, match="empty"):
        context_fragment([])
    with pytest.raises(ValueError, match="non-finite"):
        context_fragment([float("nan"), 0.0])
    with pytest.raises(TypeError):
        in_weak_band(True)
    with pytest.raises(ValueError, match="overlaps"):
        weak_winner([])


def test_weak_overlap_surfaces_under_consolidate_and_not_the_salience_default():
    scalars = schedule_scalars()
    mem = _memory()
    try:
        target, _decoy = _plant_pair(mem)
        mem._recent_fragment = context_fragment(target)
        mem.wake_for(4.0)
        assert mem.encoding_mode() == "consolidate"
        ticks = mem.offline_ticks(SPONTANEOUS_PERIOD)
        sources = [tick["source"][0][0] for tick in ticks]
        assert sources.count("spontaneous") == 1
        assert sources.count("none") == SPONTANEOUS_PERIOD - 1
        hit = next(tick for tick in ticks if tick["source"][0][0] == "spontaneous")
        assert int(hit["memory_id"][0, 0]) == 0
        assert bool(hit["triggered"][0, 0]) is False
        assert float(hit["winner_cosine"][0, 0]) == pytest.approx(WEAK_OVERLAP)
        assert float(hit["winner_cosine"][0, 0]) < CUE_THETA
        assert set(hit) == EXPORT_KEYS
        assert mem.sleep_pressure == pytest.approx(scalars["after_wake"])
    finally:
        mem.close()
    salience = _memory()
    try:
        _plant_pair(salience)
        salience.wake_for(4.0)
        ticks = salience.offline_ticks(1)
        assert ticks[0]["source"][0][0] == "spontaneous"
        assert int(ticks[0]["memory_id"][0, 0]) == 1
        assert ticks[0]["winner_cosine"][0, 0] != ticks[0]["winner_cosine"][0, 0]
    finally:
        salience.close()


def test_remember_plants_a_partial_key_and_encode_mode_stays_silent():
    scalars = schedule_scalars()
    planted = _memory()
    try:
        key = _axis(planted, 0)
        _quiet_write(planted)
        _encode_key(planted, key)
        planted.wake_for(4.0)
        assert planted.encoding_mode() == "consolidate"
        assert planted.remember("code is T", name="target", rel="note")
        assert overlap_score(planted._recent_fragment, planted.store.slots[0].k) == pytest.approx(WEAK_OVERLAP)
        ticks = planted.offline_ticks(SPONTANEOUS_PERIOD)
        sources = [tick["source"][0][0] for tick in ticks]
        assert sources.count("spontaneous") == 1
        hit = next(tick for tick in ticks if tick["source"][0][0] == "spontaneous")
        assert int(hit["memory_id"][0, 0]) == 0
        assert bool(hit["triggered"][0, 0]) is False
        assert float(hit["winner_cosine"][0, 0]) == pytest.approx(WEAK_OVERLAP)
    finally:
        planted.close()
    encode = _memory()
    try:
        target, _decoy = _plant_pair(encode)
        encode._recent_fragment = context_fragment(target)
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
        assert encode.store.slots[0].text == "raw-weak-target"
    finally:
        encode.close()


def test_strong_cue_still_fires_and_is_not_spontaneous():
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
        assert mem.store.write(key, value, name="ada", rel="note", answer="A")
        fields = mem.cue_fields(text)
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert fields["source"][0][0] == "cue"
        assert "spontaneous" not in fields["source"][0]
        assert bool(fields["triggered"][0, 0]) is True
        assert float(fields["winner_cosine"][0, 0]) >= CUE_THETA
        assert int(fields["memory_id"][0, 0]) == 0
        assert "weak_overlap" not in fields
        assert "recent_fragment" not in fields
    finally:
        mem.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert "weak_overlap" not in fields
        assert "context_fragment" not in fields
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
    assert apply_sleep_operators("encode") is True
    assert memory_mode(SLEEP_FLOOR, AROUSAL_BASELINE, 0.0) == "encode"
    assert ENCODE_SPONTANEOUS_KEEP == 0.5


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "weak-cue-recollection-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "73ecc2a57e111f2234d5c978c6117001802ae92f5eb43b237ec25654625bfc10"
    assert set(GATES) == {
        "identity",
        "weak",
        "strong",
        "consolidate",
        "encode",
        "compose",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["weak"], item_id
        assert row["strong"], item_id
        assert row["consolidate"], item_id
        assert row["encode"], item_id
        assert row["compose"], item_id
        assert row["export"], item_id


def test_weak_cue_module_has_no_flag_and_docs_name_the_paper():
    source = (ROOT / "tidal" / "weak_cue.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_weak_cue_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "weak_winner" not in residual
    assert "context_fragment" not in residual
    packed = memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "weak_overlap" not in packed
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_weak_cue_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        assert DOI in text
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
    assert "default-mode-network" in source
    assert "cannot be equated with mind-wandering" in prereg
    assert "cannot be equated with mind-wandering" in readme
    assert "cannot be equated with mind-wandering" in architecture
    assert WEAK_OVERLAP == 0.25
    assert WEAK_FLOOR == 0.10
    assert WEAK_CEILING == 0.32
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
