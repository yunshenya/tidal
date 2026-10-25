"""McGaugh arousal×|valence| consolidation gain. No SmolLM download."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import AROUSAL_BASELINE, AROUSAL_TAU, AROUSAL_WINDOW
from tidal.backend import GROWBRAIN_D_MODEL, GROWBRAIN_MEM_DIM, GROWBRAIN_MEM_KEY_DIM
from tidal.consolidation import CONSOLIDATION_SPAN, consolidation_gain, replay_priority
from tidal.llm_memory import (
    VALENCE_STEP,
    ByteTokenizer,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_THETA,
    LLMMemory,
)
from tidal.sleep import SLEEP_FLOOR, spontaneous_winner
from tidal.subconscious_consolidation_data import (
    AROUSAL_HIGH,
    AROUSAL_MID,
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    VALENCE_HIGH,
    VALENCE_LOW,
    bank_sha256,
    canonical_gains,
    formula_gates,
)
from tidal.subconscious_reconstruct_data import canonical_surfaces

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.1146/annurev.neuro.27.070203.144157"
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


def test_formula_gates_lock_identity_and_monotone_gain():
    gains = canonical_gains()
    assert gains["baseline_zero"] == 1.0
    assert gains["baseline_high"] == 1.0
    assert gains["high_zero"] == 1.0
    assert gains["mid_low"] == pytest.approx(1.043323483493126)
    assert gains["mid_high"] == pytest.approx(1.0848576487792063)
    assert gains["high_low"] == pytest.approx(1.1155292893150024)
    assert gains["high_high"] == pytest.approx(1.2262870634112166)
    assert gains["high_high_neg"] == gains["high_high"]
    assert CONSOLIDATION_SPAN == 0.25
    assert replay_priority(AROUSAL_HIGH, VALENCE_HIGH) == consolidation_gain(AROUSAL_HIGH, VALENCE_HIGH)
    row = formula_gates()
    assert row["identity"] and row["arousal_mono"] and row["valence_mono"] and row["sign"]
    assert consolidation_gain(AROUSAL_BASELINE, VALENCE_HIGH) == 1.0
    assert consolidation_gain(AROUSAL_MID, 0.0) == 1.0
    with pytest.raises(ValueError, match="arousal"):
        consolidation_gain(1.01, 1.0)
    with pytest.raises(TypeError):
        consolidation_gain(True, 1.0)
    with pytest.raises(ValueError, match="valence"):
        consolidation_gain(1.0, float("nan"))


def test_remember_scales_write_strength_only_when_arousal_and_valence_rise():
    rest = _memory()
    try:
        _quiet_write(rest)
        fact = {"text": "code is Y", "answer": "Y", "name": "ada", "rel": "note", "experience": 1}
        assert rest.remember(dict(fact))
        slot = rest.store.slots[0]
        assert slot.valence == pytest.approx(VALENCE_STEP)
        assert slot.consolidation_gain == 1.0
        assert slot.strength == 1.0
        assert rest.sleep_pressure == SLEEP_FLOOR
        assert float(rest.fusion_confidence) == 0.0
    finally:
        rest.close()
    aroused = _memory()
    try:
        _quiet_write(aroused)
        aroused._arousal = float(AROUSAL_HIGH)
        fact = {"text": "code is Y", "answer": "Y", "name": "ada", "rel": "note", "experience": 1}
        ambient = float(aroused.arousal)
        assert aroused.remember(dict(fact))
        slot = aroused.store.slots[0]
        want = consolidation_gain(ambient, float(VALENCE_STEP))
        assert ambient == AROUSAL_HIGH
        assert slot.consolidation_gain == pytest.approx(want)
        assert slot.strength == 1.0
        assert slot.clock_gain == 1.0
        assert want > 1.0
        assert aroused.sleep_pressure == SLEEP_FLOOR
        assert float(aroused.fusion_confidence) == 0.0
        fields = aroused.cue_fields("Ab")
        assert float(fields["fusion_confidence"]) == 0.0
        assert "consolidation_gain" not in fields
    finally:
        aroused.close()
    neutral = _memory()
    try:
        _quiet_write(neutral)
        neutral._arousal = float(AROUSAL_HIGH)
        assert neutral.remember("code is Q", name="bea", rel="note")
        slot = neutral.store.slots[0]
        assert slot.valence == 0.0
        assert slot.consolidation_gain == 1.0
        assert slot.strength == 1.0
    finally:
        neutral.close()


def test_replay_priority_raises_high_arousal_traces_and_keeps_the_old_tie():
    assert spontaneous_winner([0.1, 0.4, 0.4], [1.0, 0.2, 0.9]) == 2
    assert spontaneous_winner([0.5, 0.5], [1.0, 1.0], [1.0, 1.0]) == 0
    high = canonical_gains()["high_high"]
    assert spontaneous_winner([0.5, 0.5], [1.0, 1.0], [1.0, high]) == 1
    mem = _memory()
    try:
        dim = mem.hidden
        low_k = np.zeros(dim, dtype=np.float64)
        low_k[0] = 1.0
        high_k = np.zeros(dim, dtype=np.float64)
        high_k[2] = 1.0
        assert mem.store.write(low_k, low_k.copy(), name="low", rel="note", group="low")
        assert mem.store.write(high_k, high_k.copy(), name="high", rel="note", group="high")
        mem.store.slots[0].salience = 0.4
        mem.store.slots[1].salience = 0.4
        mem.store.slots[0].strength = 1.0
        mem.store.slots[1].strength = 1.0
        mem.store.slots[0].consolidation_gain = 1.0
        mem.store.slots[1].consolidation_gain = high
        mem.wake_for(4.0)
        ticks = mem.offline_ticks(1)
        assert ticks[0]["source"][0][0] == "spontaneous"
        assert int(ticks[0]["memory_id"][0, 0]) == 1
        ranked = mem.store.ranked_by_salience()
        assert ranked[0] == 1
    finally:
        mem.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert "consolidation_gain" not in fields
    finally:
        mem.close()
    surfaces = canonical_surfaces()
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


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "mcgaugh-consolidation-v1"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "9396a6a2cb463e4ed0f2f746dbaff3101ca913a06d56f9e437fa9668a666c66b"
    assert set(GATES) == {
        "identity",
        "arousal_mono",
        "valence_mono",
        "sign",
        "write",
        "replay",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["arousal_mono"], item_id
        assert row["valence_mono"], item_id
        assert row["sign"], item_id


def test_consolidation_module_has_no_switch_and_docs_name_the_formula():
    source = (ROOT / "tidal" / "consolidation.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_consolidation_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "arousal" not in residual
    assert "consolidation_gain" not in residual
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_consolidation_prereg.md").read_text(encoding="utf-8")
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
    assert "does not invent" in source
    assert CONSOLIDATION_SPAN == 0.25
    assert VALENCE_LOW == 0.5
    assert VALENCE_HIGH == 1.5
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    assert "speak" in source
