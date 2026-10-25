"""Fusion scorer v0. No SmolLM download. The Eye model is the cue-test stand-in."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import AROUSAL_BASELINE, AROUSAL_TAU, AROUSAL_WINDOW
from tidal.backend import GROWBRAIN_D_MODEL, GROWBRAIN_MEM_DIM, GROWBRAIN_MEM_KEY_DIM
from tidal.fusion import (
    CAUSAL_FLOOR,
    CAUSAL_TAU,
    FP_FLOOR,
    FUSION_WINDOW,
    RELIABILITY_SAT,
    TIME_PRIOR,
    FusionObservation,
    observation_reliability,
    observations_from_slots,
    pair_common_cause,
    score_fusion,
    suggested_fusion_weights,
)
from tidal.interface_v7 import (
    FINGERPRINT_DIM,
    check_fusion_confidence,
    empty_fusion_weights,
    zero_fingerprint,
)
from tidal.llm_memory import ByteTokenizer, CUE_HALF_LIFE, CUE_INTERFACE_VERSION, CUE_LAMBDA, CUE_THETA, LLMMemory
from tidal.sleep import SLEEP_FLOOR
from tidal.subconscious_fusion_data import (
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    bank_sha256,
    canonical_scores,
    conflicting_pair,
    far_pair,
    gate_row,
    matching_pair,
    time_only_pair,
)
from tidal.subconscious_reconstruct_data import canonical_surfaces

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1038/415429a",
    "10.1371/journal.pbio.0050260",
)
UNSCORED = (
    "The held bank was not scored.",
    "The cue guard was not scored.",
    "The sleep guard was not scored.",
)


class _Eye(nn.Module):
    """Byte id ``i`` is hidden axis ``i``. No download."""

    def __init__(self):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, vocab)
        eye = torch.eye(vocab)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.lm_head = nn.Linear(vocab, vocab, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(eye)
            self.lm_head.weight.copy_(eye)
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
    LLMMemory._key_centers = lambda self: (
        np.zeros(self.hidden, dtype=np.float64),
        np.zeros(self.hidden, dtype=np.float64),
    )
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


def _write_slot(mem, name, modality, fingerprint, *, text="Ab"):
    hidden = mem.cue_hidden(text)
    key = hidden[0] - mem.query_mean
    value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
    assert mem.store.write(
        key,
        value,
        name=name,
        rel="note",
        answer=name[:1],
        fingerprint=fingerprint,
        modality=modality,
    )


def test_score_fusion_gates_lock_the_formula():
    scores = canonical_scores()
    assert scores["matching"] == 1.0
    assert scores["time_only"] == 0.1375
    assert scores["weighted"] == 0.625
    assert scores["conflict"] == 0.0
    assert scores["single"] == 0.0
    assert scores["empty"] == 0.0
    assert scores["far"] == 0.0
    assert 0.0 < scores["time_only"] < scores["matching"]
    assert scores["weighted"] < scores["matching"]
    row = gate_row()
    assert row["pass"] is True
    assert set(row["gates"]) == set(GATES)
    assert all(row["gates"].values())
    assert score_fusion(matching_pair()[:1]) == 0.0
    assert score_fusion(()) == 0.0
    none_only = FusionObservation(t=1, modality="none", fingerprint=zero_fingerprint())
    fused_only = FusionObservation(t=2, modality="fused", fingerprint=zero_fingerprint())
    assert score_fusion((none_only, fused_only)) == 0.0
    assert pair_common_cause(matching_pair()[0].fingerprint, matching_pair()[1].fingerprint) == 1.0
    assert pair_common_cause(conflicting_pair()[0].fingerprint, conflicting_pair()[1].fingerprint) == 0.0
    assert pair_common_cause(zero_fingerprint(), zero_fingerprint()) == TIME_PRIOR
    mark = matching_pair()[0].fingerprint
    assert observation_reliability(FusionObservation(t=0, modality="text", fingerprint=mark)) == 1.0
    assert observation_reliability(FusionObservation(t=0, modality="text", fingerprint=zero_fingerprint())) == FP_FLOOR
    suggested = suggested_fusion_weights({"text": 1.0, "audio": 1.0})
    assert suggested.shape == (3,)
    assert suggested.dtype.name == "float32"
    assert np.allclose(suggested, [0.5, 0.5, 0.0])
    with pytest.raises(ValueError, match="fusion_confidence"):
        check_fusion_confidence(float("nan"))
    with pytest.raises(TypeError):
        score_fusion(matching_pair(), now=True)
    with pytest.raises(ValueError, match="strength"):
        FusionObservation(t=0, modality="text", fingerprint=zero_fingerprint(), strength=float("inf"))


def test_refresh_writes_cue_last_token_and_leaves_weights():
    mem = _memory()
    try:
        assert float(mem.fusion_confidence) == 0.0
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.0
        mark = np.linspace(1.0, 2.0, FINGERPRINT_DIM, dtype=np.float32)
        _write_slot(mem, "ada", "text", mark)
        assert mem.refresh_fusion_confidence() == 0.0
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.0
        _write_slot(mem, "bea", "audio", mark.copy())
        level = mem.refresh_fusion_confidence()
        assert level == 1.0
        assert 0.0 < float(mem.fusion_confidence) <= 1.0
        fields = mem.cue_fields("Ab")
        assert float(fields["fusion_confidence"]) == 1.0
        assert isinstance(fields["fusion_confidence"], float)
        assert "fusion_weights" not in fields
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        before = mem.fusion_weights.copy()
        assert np.array_equal(before, empty_fusion_weights())
        mem.set_fusion_weights(np.array([1.0, 0.25, 0.0], dtype=np.float32), pair="ab")
        stored = mem.fusion_weights.copy()
        weighted = mem.refresh_fusion_confidence()
        assert weighted == 0.625
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.625
        assert np.array_equal(mem.fusion_weights, stored)
        assert mem.fusion_pair == "ab"
        assert "fusion_weights" not in mem.cue_fields("Ab")
    finally:
        mem.close()


def test_single_modality_and_conflict_stay_zero():
    mem = _memory()
    try:
        mark = np.linspace(1.0, 2.0, FINGERPRINT_DIM, dtype=np.float32)
        _write_slot(mem, "ada", "audio", mark)
        assert mem.refresh_fusion_confidence() == 0.0
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.0
        _write_slot(mem, "bea", "image", -mark)
        assert mem.refresh_fusion_confidence() == 0.0
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.0
        obs = observations_from_slots(mem.store.slots)
        assert len(obs) == 2
        assert score_fusion(obs) == 0.0
        far = far_pair()
        assert score_fusion(far, now=far[1].t) == 0.0
        assert score_fusion(time_only_pair()) == 0.1375
    finally:
        mem.close()


def test_remember_scores_after_accepted_write():
    mem = _memory()
    try:
        mem._value_and_surprise = lambda _text, _answer: (
            np.zeros(mem.hidden, dtype=np.float32),
            1.0,
        )
        mark = np.linspace(1.0, 2.0, FINGERPRINT_DIM, dtype=np.float32)
        assert mem.remember("code is ada", name="ada", rel="note", modality="text", fingerprint=mark)
        assert float(mem.fusion_confidence) == 0.0
        assert mem.remember("code is bea", name="bea", rel="note", modality="audio", fingerprint=mark.copy())
        assert float(mem.fusion_confidence) == 1.0
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 1.0
        assert np.array_equal(mem.fusion_weights, empty_fusion_weights())
        assert mem.sleep_pressure == SLEEP_FLOOR
        assert mem.arousal == AROUSAL_BASELINE
    finally:
        mem.close()


def test_validation_and_reconstruct_bank_stay_put():
    mem = _memory()
    try:
        for bad in (-0.01, 1.01, float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError, match="fusion_confidence"):
                mem.set_fusion_confidence(bad)
        assert float(mem.fusion_confidence) == 0.0
        mem.set_fusion_confidence(1.0)
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 1.0
        mem.set_fusion_confidence(0.0)
        assert float(mem.cue_fields("Ab")["fusion_confidence"]) == 0.0
    finally:
        mem.close()
    surfaces = canonical_surfaces()
    assert surfaces["interface_version"] == 6
    assert CUE_INTERFACE_VERSION == 7
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
    assert PROTOCOL == "v7-fusion-scorer-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "98bef28c3b4f687396f3405cc36d547715e85df49eaf2bd915916efb3bfd90f1"
    for item_id in FIT_IDS:
        row = gate_row()
        assert row["pass"] is True, item_id
        assert all(row["gates"].values()), item_id


def test_fusion_module_has_no_switch_and_docs_name_the_formula():
    source = (ROOT / "tidal" / "fusion.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_fusion_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "import growbrain" not in memory
    assert "refresh_fusion_confidence" in memory
    assert "set_fusion_weights" in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "v7_fusion_scorer_prereg.md").read_text(encoding="utf-8")
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
    assert "set_fusion_weights" in source
    assert "do not write" in source.lower() or "does not write" in source
    assert FUSION_WINDOW == 4
    assert FP_FLOOR == 0.25
    assert TIME_PRIOR == 0.55
    assert CAUSAL_TAU == 0.20
    assert CAUSAL_FLOOR == 0.15
    assert RELIABILITY_SAT == 2.0
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
