"""Post-retrieval labile window. No SmolLM download."""
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
from tidal.reconsolidation import (
    LABILE_TOKENS,
    LABILE_WALL,
    apply_writeback,
    matched_indices,
    open_window,
    reconstructive_writeback,
    window_open,
)
from tidal.reconstruct import residue_from_answer
from tidal.sleep import SLEEP_FLOOR, reconsolidate
from tidal.subconscious_reconsolidation_data import (
    FIT_IDS,
    GATES,
    HELD_IDS,
    PROTOCOL,
    bank_sha256,
    canonical_window,
    formula_gates,
)
from tidal.subconscious_reconstruct_data import LURE_PHONE, PHONE, canonical_surfaces

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.1038/35021052"
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
HIT = "A"
MISS = "B" * int(LABILE_TOKENS)


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


def _plant_phone(mem, cue: str = HIT, *, name: str = "ada"):
    key = _axis(mem, ord(cue[0]))
    assert mem.store.write(
        key,
        key.copy(),
        name=name,
        rel="phone",
        answer=PHONE,
        text="Ada phone",
    )
    return key


def test_formula_gates_lock_closed_open_and_writeback():
    locked = canonical_window()
    assert locked["tokens"] == LABILE_TOKENS == 8
    assert locked["wall"] == LABILE_WALL == 4.0
    assert locked["until_token"] == 8
    assert locked["until_wall"] == 4.0
    assert locked["exact"] == PHONE
    assert locked["lure_surface"] == LURE_PHONE
    assert locked["slip"] == "+1-617-555-4862"
    row = formula_gates()
    assert row["identity"] and row["hit"] and row["update"]
    assert row["token"] and row["wall"] and row["miss"]
    assert not window_open(0, 0.0, 0, 0.0)
    assert window_open(*open_window(0, 0.0), 0, 0.0)
    assert not window_open(*open_window(0, 0.0), LABILE_TOKENS, 0.0)
    assert not window_open(*open_window(0, 0.0), 0, LABILE_WALL)
    assert reconstructive_writeback(residue_from_answer(PHONE), 1.0, 0, 0.0)[0] == PHONE
    with pytest.raises(ValueError, match="token"):
        open_window(-1, 0.0)
    with pytest.raises(TypeError):
        window_open(True, 1.0, 0, 0.0)
    assert matched_indices([False], [-1], [[]]) == ()
    assert matched_indices([True], [2], [[2, 3]]) == (2, 3)


def test_cue_hit_opens_window_and_writeback_sticks():
    mem = _memory()
    try:
        _plant_phone(mem)
        assert mem.labile(0) is False
        assert mem.write_back(0, LURE_PHONE) is False
        assert mem.store.slots[0].answer == PHONE
        fields = mem.cue_fields(HIT)
        assert bool(fields["triggered"][0, 0])
        assert int(fields["memory_id"][0, 0]) == 0
        assert mem.labile(0) is True
        assert mem.write_back(0, LURE_PHONE) is True
        slot = mem.store.slots[0]
        assert slot.answer == LURE_PHONE
        assert slot.residue == residue_from_answer(LURE_PHONE)
        assert slot.text == "Ada phone"
        assert float(slot.strength) == 1.0
        assert slot.consolidation_gain == 1.0
        assert float(mem.fusion_confidence) == 0.0
        assert mem.sleep_pressure == SLEEP_FLOOR
        assert "labile" not in fields
        assert set(fields) == EXPORT_KEYS
    finally:
        mem.close()


def test_window_closes_on_token_or_wall_and_miss_does_not_open():
    tokens = _memory()
    try:
        _plant_phone(tokens)
        assert tokens.cue_fields(HIT)["triggered"][0, 0]
        assert tokens.write_back(0, LURE_PHONE)
        miss = tokens.cue_fields(MISS)
        assert not bool(np.any(miss["triggered"]))
        assert tokens.labile(0) is False
        assert tokens.write_back(0, PHONE) is False
        assert tokens.store.slots[0].answer == LURE_PHONE
    finally:
        tokens.close()
    wall = _memory()
    try:
        _plant_phone(wall)
        assert wall.cue_fields(HIT)["triggered"][0, 0]
        assert wall.write_back(0, LURE_PHONE)
        wall.advance_wall(float(LABILE_WALL))
        assert wall.labile(0) is False
        assert wall.write_back(0, PHONE) is False
        assert wall.store.slots[0].answer == LURE_PHONE
        assert wall.store.slots[0].strength == 1.0
    finally:
        wall.close()
    miss = _memory()
    try:
        _plant_phone(miss)
        fields = miss.cue_fields("B")
        assert not bool(np.any(fields["triggered"]))
        assert miss.labile(0) is False
        assert miss.write_back(0, LURE_PHONE) is False
        assert miss.store.slots[0].answer == PHONE
    finally:
        miss.close()
    empty = _memory()
    try:
        empty.cue_fields(HIT)
        _plant_phone(empty)
        assert empty.labile(0) is False
        assert empty.write_back(0, LURE_PHONE) is False
        assert empty.store.slots[0].answer == PHONE
    finally:
        empty.close()


def test_export_versions_and_held_pins_stay_put():
    mem = _memory()
    try:
        fields = mem.cue_fields("Ab")
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        assert set(fields) == EXPORT_KEYS
        assert float(fields["fusion_confidence"]) == 0.0
        assert "labile" not in fields
        assert "labile_until_token" not in fields
        assert "consolidation_gain" not in fields
        assert score_fusion(()) == 0.0
        assert consolidation_gain(AROUSAL_BASELINE, 0.0) == 1.0
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
    assert PROTOCOL == "reconsolidation-window-v0"
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(bank_sha256()) == 64
    assert bank_sha256() == "abb2518688c5f51bff8d022afac5cc1970642fab503d51932abcec5d6b8df83d"
    assert set(GATES) == {
        "identity",
        "hit",
        "update",
        "token",
        "wall",
        "miss",
        "export",
        "item",
    }
    for item_id in FIT_IDS:
        row = formula_gates()
        assert row["identity"], item_id
        assert row["hit"], item_id
        assert row["update"], item_id
        assert row["token"], item_id
        assert row["wall"], item_id
        assert row["miss"], item_id


def test_reconsolidation_module_has_no_switch_and_docs_name_the_paper():
    source = (ROOT / "tidal" / "reconsolidation.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_reconsolidation_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "labile" not in residual
    assert "write_back" not in residual
    packed = memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "labile" not in packed
    assert "import growbrain" not in memory
    assert CUE_INTERFACE_VERSION == 7
    prereg = (ROOT / "reports" / "subconscious_reconsolidation_prereg.md").read_text(encoding="utf-8")
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
    assert "overwrite API" in source
    assert LABILE_TOKENS == 8
    assert LABILE_WALL == 4.0
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    slot = type("S", (), {})()
    slot.answer = PHONE
    slot.residue = residue_from_answer(PHONE)
    slot.strength = 1.0
    slot.age = 0
    slot.valence = 0.0
    slot.labile_until_token = 0
    slot.labile_until_wall = 0.0
    assert apply_writeback(slot, 0, 0.0, LURE_PHONE) is False
    assert slot.answer == PHONE
