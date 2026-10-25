"""Retroactive interference from a new similar write. No SmolLM download."""
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
from tidal.pattern_separation import match_scores, separate_keys
from tidal.reconsolidation import window_open
from tidal.retroactive import (
    INTERFERENCE_SPAN,
    INTERFERENCE_TAU,
    apply_retroactive,
    neighbor_keep,
)
from tidal.sleep import SLEEP_FLOOR, reconsolidate
from tidal.subconscious_reconstruct_data import canonical_surfaces
from tidal.subconscious_retroactive_data import (
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    REPLAY_DECAY,
    bank_sha256,
    canonical_scores,
    formula_gates,
)
from tidal.subconscious_separation_data import similar_pair

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.1146/annurev.psych.55.090902.141555"
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


def _near(mem):
    row = np.zeros(mem.hidden, dtype=np.float64)
    row[0] = 1.0
    row[1] = 0.1
    return row / float(np.linalg.norm(row))


def _encode_queue(mem, keys):
    pending = [np.asarray(key, dtype=np.float64).reshape(-1) for key in keys]

    def encode(_text, answer=None):
        del _text, answer
        if not pending:
            raise AssertionError("encode_key queue is empty")
        return pending.pop(0).astype(np.float32, copy=False)

    mem.encode_key = encode


def test_formula_gates_lock_similar_vs_orthogonal():
    scores = canonical_scores()
    assert scores["tau"] == INTERFERENCE_TAU == 0.90
    assert scores["span"] == INTERFERENCE_SPAN == 0.25
    assert scores["similar_raw"] == pytest.approx(0.9950371902099893)
    assert scores["similar_keep"] == pytest.approx(0.7624070244750268)
    assert scores["exact_keep"] == 0.75
    assert scores["similar_age"] == 1
    assert scores["decay_treated"] == 0.25
    assert scores["decay_control"] == 0.5
    row = formula_gates()
    assert row["identity"] and row["similar"] and row["control"]
    assert row["orthogonal"] and row["write"] and row["compose"]
    assert neighbor_keep(0.5) == 1.0
    assert neighbor_keep(1.0) == 0.75
    left, right = similar_pair()
    assert apply_retroactive([], left, 0) == 0
    with pytest.raises(TypeError):
        neighbor_keep(True)
    with pytest.raises(TypeError):
        apply_retroactive([], left, True)


def test_store_write_ages_similar_not_orthogonal():
    mem = _memory()
    try:
        left = _axis(mem, 0)
        right = _near(mem)
        ortho = _axis(mem, 2)
        value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
        control = _memory()
        try:
            assert control.store.write(left, value, name="ada", rel="note", answer="A")
            assert control.store.slots[0].age == 0
            assert control.store.slots[0].strength == 1.0
            assert control.store.write(ortho.copy(), value.copy(), name="cy", rel="note", answer="C")
            assert control.store.slots[0].age == 0
            assert control.store.slots[0].strength == 1.0
            control.store.recompute_decay(REPLAY_DECAY)
            assert control.store.slots[0].strength == 0.5
        finally:
            control.close()
        assert mem.store.write(left, value, name="ada", rel="note", answer="A")
        assert mem.store.slots[0].age == 0
        assert mem.store.write(right, value.copy(), name="bea", rel="note", answer="B")
        assert mem.store.slots[0].age == 1
        assert mem.store.slots[0].strength == 1.0
        assert mem.store.slots[1].age == 0
        assert mem.store.slots[1].strength == 1.0
        assert mem.store.slots[0].consolidation_gain == 1.0
        assert mem.store.slots[0].salience == 0.0
        mem.store.recompute_decay(REPLAY_DECAY)
        assert mem.store.slots[0].strength == 0.25
        assert mem.store.slots[0].strength < 0.5
        assert mem.store.write(left, value.copy(), name="ada", rel="note", answer="A2")
        assert len(mem.store) == 2
        assert mem.store.slots[1].age == 1
        assert mem.store.slots[1].strength == 0.5
    finally:
        mem.close()


def test_remember_similar_is_weaker_than_time_matched_control():
    treated = _memory()
    control = _memory()
    orthogonal = _memory()
    try:
        for mem in (treated, control, orthogonal):
            _quiet_write(mem)
        left = _axis(treated, 0)
        right = _near(treated)
        ortho = _axis(treated, 2)
        _encode_queue(control, [left])
        assert control.remember("code is ada", name="ada", rel="note")
        assert control.store.slots[0].age == 0
        assert control.store.slots[0].strength == 1.0
        _encode_queue(orthogonal, [left, ortho])
        assert orthogonal.remember("code is ada", name="ada", rel="note")
        assert orthogonal.remember("code is cy", name="cy", rel="note")
        assert orthogonal.store.slots[0].age == 0
        assert orthogonal.store.slots[0].strength == 1.0
        assert orthogonal.store.slots[1].age == 0
        _encode_queue(treated, [left, right])
        assert treated.remember("code is ada", name="ada", rel="note")
        assert treated.remember("code is bea", name="bea", rel="note")
        assert treated.store.slots[0].age == 1
        assert treated.store.slots[0].age > control.store.slots[0].age
        assert treated.store.slots[0].age > orthogonal.store.slots[0].age
        assert treated.store.slots[0].strength == 1.0
        assert treated.store.slots[1].age == 0
        assert treated.store.slots[0].consolidation_gain == 1.0
        assert float(treated.fusion_confidence) == 0.0
        assert np.array_equal(treated.store.slots[0].k, left)
        treated.store.recompute_decay(REPLAY_DECAY)
        control.store.recompute_decay(REPLAY_DECAY)
        orthogonal.store.recompute_decay(REPLAY_DECAY)
        assert treated.store.slots[0].strength == 0.25
        assert control.store.slots[0].strength == 0.5
        assert orthogonal.store.slots[0].strength == 0.5
        assert treated.store.slots[0].strength < control.store.slots[0].strength
    finally:
        treated.close()
        control.close()
        orthogonal.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert "interference" not in fields
        assert "retroactive" not in fields
        assert "consolidation_gain" not in fields
        assert score_fusion(()) == 0.0
        assert consolidation_gain(AROUSAL_BASELINE, 0.0) == 1.0
        assert not window_open(0, 0.0, 0, 0.0)
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
    key = np.zeros(4, dtype=np.float64)
    key[0] = 1.0
    assert reconsolidate(key, key.copy(), 1.0)["updated"] is False
    left = np.zeros((2, 4), dtype=np.float64)
    left[0, 0] = 1.0
    left[1, 1] = 1.0
    assert np.allclose(separate_keys(left), left)
    assert np.allclose(match_scores(left[0], left), np.array([1.0, 0.0]))


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "retroactive-interference-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "6ae2a76c48853badac5ba62e21434291242c7111744387884970e2d812a251a1"
    assert set(GATES) == {
        "identity",
        "similar",
        "control",
        "orthogonal",
        "write",
        "compose",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["similar"], item_id
        assert row["control"], item_id
        assert row["orthogonal"], item_id
        assert row["write"], item_id
        assert row["compose"], item_id


def test_retroactive_module_has_no_switch_and_docs_name_the_paper():
    source = (ROOT / "tidal" / "retroactive.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_retroactive_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "apply_retroactive" not in residual
    assert "INTERFERENCE_TAU" not in residual
    packed = memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "apply_retroactive" not in packed
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_retroactive_prereg.md").read_text(encoding="utf-8")
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
    assert "do not blanket" in source.lower()
    assert INTERFERENCE_TAU == 0.90
    assert INTERFERENCE_SPAN == 0.25
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
