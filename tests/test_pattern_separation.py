"""Pattern-separation v0. No SmolLM download. The Eye model is the cue-test stand-in."""
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
from tidal.cue_inject import select_cue_winner
from tidal.episodic import cosine_scores
from tidal.fusion import score_fusion
from tidal.interface_v7 import FINGERPRINT_DIM, zero_fingerprint
from tidal.llm_memory import (
    ByteTokenizer,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_THETA,
    LLMMemory,
)
from tidal.pattern_separation import (
    NORM_ZERO,
    SEPARATION_SPAN,
    SEPARATION_TAU,
    complement,
    key_cosine,
    match_scores,
    overlap_weight,
    separate_key,
    separate_keys,
)
from tidal.sleep import SLEEP_FLOOR
from tidal.subconscious_reconstruct_data import canonical_surfaces
from tidal.subconscious_separation_data import (
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    SEPARATION_DIM,
    bank_sha256,
    canonical_scores,
    distinct_pair,
    equal_pair,
    formula_gates,
    query_toward_left,
    similar_pair,
)

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1016/j.tins.2011.06.006",
    "10.1146/annurev.psych.55.090902.141555",
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


def _plant(mem, name, key, *, rel="note"):
    value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
    assert mem.store.write(key, value, name=name, rel=rel, answer=name[:1])


def test_formula_gates_lock_similar_vs_distinct():
    scores = canonical_scores()
    assert scores["dim"] == SEPARATION_DIM == FINGERPRINT_DIM == 32
    assert scores["tau"] == 0.90
    assert scores["span"] == 0.25
    assert scores["similar_raw"] == pytest.approx(0.9950371902099893)
    assert scores["similar_sep"] == pytest.approx(0.8506285107986749)
    assert scores["confusion_raw_gap"] == pytest.approx(0.002972141040810561)
    assert scores["confusion_sep_gap"] == pytest.approx(0.016305700273454904)
    assert scores["distinct_raw"] == 0.0
    assert scores["distinct_sep"] == 0.0
    assert scores["equal_sep"] == 1.0
    assert scores["query_winner_raw"] == 0
    assert scores["query_winner_sep"] == 0
    row = formula_gates()
    assert row["identity"] and row["similar"] and row["confusion"]
    assert row["distinct"] and row["equal"]
    assert scores["similar_sep"] < scores["similar_raw"]
    assert scores["confusion_sep_gap"] > scores["confusion_raw_gap"]
    left, right = similar_pair()
    assert np.allclose(separate_key(left, ()), left)
    assert float(np.linalg.norm(separate_key(left, [right]))) == pytest.approx(float(np.linalg.norm(left)))
    distinct_left, distinct_right = distinct_pair()
    assert np.array_equal(separate_keys(np.stack([distinct_left, distinct_right], 0)), np.stack([distinct_left, distinct_right], 0))
    equal_left, equal_right = equal_pair()
    equal_bank = separate_keys(np.stack([equal_left, equal_right], 0))
    assert key_cosine(equal_bank[0], equal_bank[1]) == 1.0
    with pytest.raises(ValueError, match="empty"):
        separate_key([])
    with pytest.raises(ValueError, match="non-finite"):
        separate_key([float("nan"), 0.0])
    with pytest.raises(TypeError):
        overlap_weight(True)
    assert overlap_weight(0.5) == 0.0
    assert overlap_weight(1.0) == SEPARATION_SPAN
    assert complement(np.array([1.0, 0.0]))[1] == pytest.approx(1.0)


def test_rank_view_lowers_confusion_and_keeps_theta_on_raw():
    left, right = similar_pair()
    query = query_toward_left(left, right)
    raw = cosine_scores(np.stack([left, right], 0), query)
    sep = match_scores(query, np.stack([left, right], 0))
    assert float(raw[0]) > float(raw[1])
    assert float(sep[0] - sep[1]) > float(raw[0] - raw[1])
    choice = select_cue_winner(raw, [1.0, 1.0], 0.32, rank_cosines=sep)
    assert choice["index"] == 0
    assert choice["n"] == 2
    assert choice["candidates"] == [0, 1]
    assert choice["gap"] == pytest.approx(float(sep[0] - sep[1]))
    raw_only = select_cue_winner(raw, [1.0, 1.0], 0.32)
    assert raw_only["n"] == 2
    assert raw_only["gap"] == pytest.approx(float(raw[0] - raw[1]))
    assert float(choice["gap"]) > float(raw_only["gap"])
    distinct_left, distinct_right = distinct_pair()
    distinct_query = distinct_left
    distinct_raw = cosine_scores(np.stack([distinct_left, distinct_right], 0), distinct_query)
    distinct_sep = match_scores(distinct_query, np.stack([distinct_left, distinct_right], 0))
    assert np.allclose(distinct_raw, distinct_sep)
    assert key_cosine(np.zeros(SEPARATION_DIM), left) == 0.0
    assert float(np.linalg.norm(separate_key(np.zeros(SEPARATION_DIM), [left]))) <= NORM_ZERO


def test_remember_keeps_raw_keys_and_cue_widens_a_similar_gap():
    mem = _memory()
    try:
        _quiet_write(mem)
        dim = int(mem.hidden)
        left = np.zeros(dim, dtype=np.float64)
        left[0] = 1.0
        right = np.zeros(dim, dtype=np.float64)
        right[0] = 1.0
        right[1] = 0.1
        right = right / float(np.linalg.norm(right))
        assert mem.remember("code is ada", name="ada", rel="note")
        first = np.asarray(mem.store.slots[0].k, dtype=np.float64).copy()
        assert mem.remember("code is bea", name="bea", rel="note")
        assert np.array_equal(mem.store.slots[0].k, first)
        mem.store.clear()
        _plant(mem, "ada", left)
        _plant(mem, "bea", right)
        for slot in mem.store._slots:
            slot.strength = 1.0
        assert np.allclose(mem.store.slots[0].k, left)
        assert np.allclose(mem.store.slots[1].k, right)
        query = left.copy()
        query[1] = 0.02
        query = query / float(np.linalg.norm(query))
        raw = cosine_scores(np.stack([left, right], 0), query)
        mem.query_mean = np.zeros(dim, dtype=np.float64)
        packed = mem.cue_fields("A")
        trace, _residual, _state = mem._trace_numpy(query.reshape(1, -1))
        assert int(trace["memory_id"][0]) == 0
        assert float(trace["gap"][0]) > float(raw[0] - raw[1])
        assert float(mem.fusion_confidence) == 0.0
        assert score_fusion(()) == 0.0
        assert consolidation_gain(AROUSAL_BASELINE, 0.0) == 1.0
        assert mem.store.slots[0].consolidation_gain == 1.0
        assert mem.sleep_pressure == SLEEP_FLOOR
        assert mem.arousal == AROUSAL_BASELINE
        assert packed["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert "pattern_separation" not in packed
        assert set(packed) == EXPORT_KEYS
    finally:
        mem.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert fields["fingerprint"].shape == (FINGERPRINT_DIM,)
        assert np.array_equal(fields["fingerprint"], zero_fingerprint())
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
    assert SEPARATION_TAU == 0.90
    assert SEPARATION_SPAN == 0.25


def test_fit_ids_pass_and_held_is_disjoint():
    assert PROTOCOL == "dg-pattern-separation-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "9a32efc8e415732cbcdfaa58d5f04138d1802c245964753fc59348768465d600"
    assert set(GATES) == {
        "identity",
        "similar",
        "confusion",
        "distinct",
        "equal",
        "write",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["similar"], item_id
        assert row["confusion"], item_id
        assert row["distinct"], item_id
        assert row["equal"], item_id


def test_separation_module_has_no_switch_and_docs_name_the_formula():
    source = (ROOT / "tidal" / "pattern_separation.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_separation_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "import growbrain" not in memory
    assert "separate_keys" in memory
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "separate_keys" not in residual
    assert "fusion_confidence" not in residual
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "pattern_separation_prereg.md").read_text(encoding="utf-8")
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
    assert "does not write" in source.lower() or "does not write" in source
    assert SEPARATION_TAU == 0.90
    assert SEPARATION_SPAN == 0.25
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    assert "reconsolidation window" in source.lower() or "not a reconsolidation" in source.lower()
