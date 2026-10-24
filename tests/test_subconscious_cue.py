"""Cue recurrence, offline fields, and the same-token add.

No SmolLM download. The schedule constants in ``llm_memory`` are read at call
time. This file does not force theta to -1.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.cue_inject import CueState, select_cue_winner, step_cue, trace_sequence
from tidal.episodic import cosine_scores
from tidal.llm_memory import (
    CUE_A_TRIG,
    CUE_FLOOR,
    CUE_INTERFACE_VERSION,
    CUE_TAU,
    CUE_THETA,
    DEFAULT_GAP_TAU,
    VALENCE_SPAN,
    ByteTokenizer,
    LLMMemory,
)

ROOT = Path(__file__).resolve().parent.parent


def test_log_strength_beats_a_higher_cosine_above_theta():
    choice = select_cue_winner([0.2, 0.9], [math.exp(2.0), 1.0], 0.15)
    assert choice["index"] == 0
    assert choice["cosine"] == pytest.approx(0.2)
    assert choice["gap"] == pytest.approx(0.2 - 0.9)
    assert choice["tied"] is False


def test_strength_below_theta_is_not_a_candidate():
    choice = select_cue_winner([0.10, 0.20], [100.0, 1.0], 0.15)
    assert choice["index"] == 1
    assert choice["gap"] == pytest.approx(0.10)


def test_cosine_tie_records_the_lower_index_and_adds_nothing():
    values = np.array([[3.0, 0.0], [9.0, 0.0]], dtype=np.float32)
    state, row = step_cue(
        CueState(),
        [0.5, 0.5],
        [1.0, 1.0],
        [0.0, 0.0],
        values,
        theta=0.1,
        tau=4.0,
        a_trig=1.0,
        floor=0.05,
        gap_tau=0.8,
        valence_span=1.0,
    )
    assert state.memory_id == 0
    assert state.confidence == 0.0
    assert row["triggered"] is True
    assert row["strength_confidence"] == 0.0
    assert row["activation"] == pytest.approx(1.0)
    assert np.allclose(row["vec"], [3.0, 0.0])


def test_decay_floor_and_a_new_trigger():
    values = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    keys = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    quiet = [0.0, 0.0, 1.0]
    cues = np.array(
        [
            [1.0, 0.0, 0.0],
            quiet,
            quiet,
            quiet,
            quiet,
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    fields, final = trace_sequence(
        cues,
        keys,
        [1.0, 1.0],
        [0.0, 0.0],
        values,
        np.zeros(3),
        theta=0.5,
        tau=1.0,
        a_trig=1.0,
        floor=0.05,
        gap_tau=0.8,
        valence_span=1.0,
    )
    assert fields["memory_id"][0] == 0
    assert fields["triggered"][0]
    assert fields["activation"][0] == pytest.approx(1.0)
    assert fields["activation"][1] == pytest.approx(math.exp(-1.0))
    assert fields["activation"][2] == pytest.approx(math.exp(-2.0))
    # exp(-3) is below the floor, so the trace clears and stays clear.
    assert fields["activation"][3] == 0.0
    assert fields["memory_id"][3] == -1
    assert fields["activation"][4] == 0.0
    assert fields["triggered"][5]
    assert fields["memory_id"][5] == 1
    assert fields["activation"][5] == pytest.approx(1.0)
    assert final.dt == 0
    assert final.memory_id == 1


def test_stepwise_advance_matches_the_full_trace():
    rng = np.random.default_rng(0)
    cues = rng.normal(size=(6, 4))
    keys = rng.normal(size=(3, 4))
    values = rng.normal(size=(3, 4)).astype(np.float32)
    strengths = np.array([0.2, 1.0, 0.5])
    valences = np.array([0.0, -0.4, 1.2])
    kwargs = dict(
        theta=0.05,
        tau=4.0,
        a_trig=0.5,
        floor=0.05,
        gap_tau=0.8,
        valence_span=1.0,
    )
    fields, final = trace_sequence(
        cues, keys, strengths, valences, values, np.zeros(4), **kwargs
    )
    state = CueState()
    for index in range(cues.shape[0]):
        state, row = step_cue(
            state,
            cosine_scores(keys, cues[index]),
            strengths,
            valences,
            values,
            **kwargs,
        )
        assert row["memory_id"] == int(fields["memory_id"][index])
        assert row["activation"] == pytest.approx(float(fields["activation"][index]))
        assert row["triggered"] == bool(fields["triggered"][index])
        assert np.allclose(row["vec"], fields["vec"][index])
    assert state.memory_id == final.memory_id
    assert state.dt == final.dt
    assert state.a == pytest.approx(final.a)


def test_cue_module_does_not_name_a_decoder_index():
    source = (ROOT / "tidal" / "cue_inject.py").read_text(encoding="utf-8")
    assert "layer 15" not in source
    assert "576" not in source
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "torch" not in imported


class _Eye(nn.Module):
    """Byte id ``i`` is hidden axis ``i``. Later tokens are orthogonal."""

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


def test_offline_fields_and_interface_version():
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
        assert mem.store.write(key, value, name="ada", rel="note", answer="A")
        before = float(mem.store.slots[0].strength)
        mem.clock_set(0.4)
        fields = mem.cue_fields(text)
        assert mem.store.slots[0].strength == before
        steps = hidden.shape[0]
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 1
        assert fields["cue_dim"] == mem.cue_dim
        assert tuple(fields["memory_id"].shape) == (1, steps)
        assert tuple(fields["vec"].shape) == (1, steps, mem.cue_dim)
        assert tuple(fields["activation"].shape) == (1, steps)
        assert tuple(fields["strength_confidence"].shape) == (1, steps)
        assert tuple(fields["valence"].shape) == (1, steps)
        assert tuple(fields["circadian_phase"].shape) == (1, steps)
        assert tuple(fields["triggered"].shape) == (1, steps)
        assert int(fields["memory_id"][0, 0]) == 0
        assert torch.allclose(fields["vec"][0, 0], torch.tensor(value))
        assert float(fields["circadian_phase"][0, 0]) == pytest.approx(float(mem.clock_phase()))
        assert float(fields["circadian_phase"][0, -1]) == pytest.approx(float(mem.clock_phase()))
        state = CueState()
        for row in hidden:
            state, _step = mem.advance_cue(state, row)
        assert state.memory_id == mem.cue_state.memory_id
        assert state.dt == mem.cue_state.dt
        assert state.a == pytest.approx(mem.cue_state.a)
    finally:
        mem.close()


def test_same_token_cue_is_the_pre_add_residual():
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.zeros(mem.cue_dim, dtype=np.float32)
        value[3] = 4.0
        assert mem.store.write(key, value, name="ada", rel="note", answer="A")
        import tidal.llm_memory as mod

        old = (mod.CUE_THETA, mod.CUE_A_TRIG, mod.CUE_TAU, mod.CUE_FLOOR)
        mod.CUE_THETA = -1.0
        mod.CUE_A_TRIG = 1.0
        mod.CUE_TAU = 1.0e6
        mod.CUE_FLOOR = 0.0
        try:
            fields = mem.cue_fields(text)
            assert float(fields["activation"][0, -1]) == pytest.approx(1.0)
            mem._cue_live = True
            mem._cue_resume = False
            enc = mem._tokenize(text)
            ids = enc["input_ids"]
            mem.model(input_ids=ids, use_cache=False)
            assert np.allclose(mem._last_cue_rows, hidden)
            assert float(np.linalg.norm(mem.last_added.numpy())) > 1.0
        finally:
            mod.CUE_THETA, mod.CUE_A_TRIG, mod.CUE_TAU, mod.CUE_FLOOR = old
            mem._cue_live = False
    finally:
        mem.close()


def test_resume_carries_decay_for_one_new_token(monkeypatch):
    import tidal.llm_memory as mod

    monkeypatch.setattr(mod, "CUE_THETA", 0.99)
    monkeypatch.setattr(mod, "CUE_TAU", 4.0)
    monkeypatch.setattr(mod, "CUE_A_TRIG", 1.0)
    monkeypatch.setattr(mod, "CUE_FLOOR", 0.05)
    mem = _memory()
    try:
        value = np.ones(mem.cue_dim, dtype=np.float32)
        key = np.zeros(mem.cue_dim, dtype=np.float64)
        key[0] = 1.0
        assert mem.store.write(key, value, name="ada", rel="note", answer="A")
        mem.cue_state = CueState(
            a=1.0,
            dt=0,
            memory_id=0,
            confidence=1.0,
            strength=1.0,
            valence=0.0,
            gap=1.0,
            winner_cosine=1.0,
            vec=value.copy(),
        )
        mem._cue_resume = True
        mem._cue_live = True
        # Token 1 is a different axis from the stored key, so cosine stays under theta.
        mem.model(input_ids=torch.tensor([[1]]), use_cache=False)
        assert mem.cue_state.memory_id == 0
        assert mem.cue_state.dt == 1
        assert mem.cue_state.a == pytest.approx(math.exp(-1.0 / 4.0))
        assert bool(mem.last_cue_fields["triggered"][0, 0]) is False
    finally:
        mem._cue_live = False
        mem.close()


def test_late_token_below_the_floor_adds_nothing(monkeypatch):
    import tidal.llm_memory as mod

    monkeypatch.setattr(mod, "CUE_THETA", 0.5)
    monkeypatch.setattr(mod, "CUE_TAU", 1.0)
    monkeypatch.setattr(mod, "CUE_A_TRIG", 1.0)
    monkeypatch.setattr(mod, "CUE_FLOOR", 0.05)
    mem = _memory()
    try:
        text = "ABCDE"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.zeros(mem.cue_dim, dtype=np.float32)
        value[0] = 2.0
        assert mem.store.write(key, value, name="ada", rel="note", answer="A")
        fields = mem.cue_fields(text)
        assert bool(fields["triggered"][0, 0]) is True
        assert float(fields["activation"][0, 0]) == pytest.approx(1.0)
        assert float(fields["activation"][0, 1]) == pytest.approx(math.exp(-1.0))
        assert float(fields["activation"][0, 2]) == pytest.approx(math.exp(-2.0))
        assert int(fields["memory_id"][0, 3]) == -1
        assert float(fields["activation"][0, 3]) == 0.0
        assert float(fields["activation"][0, 4]) == 0.0
        assert torch.count_nonzero(fields["vec"][0, 4]) == 0
        residual = mem.injection(text)
        assert np.allclose(residual, 0.0)
        assert mem.store.slots[0].strength == 1.0
    finally:
        mem.close()


def test_placeholder_schedule_is_documented_as_unselected():
    source = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "not a selected cell" in source
    assert CUE_FLOOR == 0.05
    assert 0.0 < CUE_THETA <= 1.0
    assert CUE_TAU > 0.0
    assert 0.0 < CUE_A_TRIG <= 1.0
    assert DEFAULT_GAP_TAU == 0.80
    assert VALENCE_SPAN == 1.0


def test_prereg_locks_the_schedule_and_the_citations():
    text = (ROOT / "reports" / "subconscious_cue_prereg.md").read_text(encoding="utf-8")
    for doi in (
        "10.1037/h0020071",
        "10.1037/0735-7044.100.2.147",
        "10.1126/science.aau8956",
        "10.7554/eLife.74445",
    ):
        assert doi in text
    assert "same index" in text
    assert "0.10" in text
    assert "CUE_FLOOR" in text
    assert "0.32, 0.35, 0.40, 0.43" in text or "{0.32, 0.35, 0.40, 0.43}" in text
    assert "largest mean cued margin" in text
    assert "39/50" in text and "42/50" in text and "48/50" in text
    assert "No cell has been scored." in text
    script = (ROOT / "scripts" / "eval_subconscious_cue.py").read_text(encoding="utf-8")
    assert "(0.32, 0.35, 0.40, 0.43)" in script
    assert "(2.0, 4.0, 8.0)" in script
    assert "(0.5, 1.0)" in script


def test_fit_script_keeps_held_banks_out_of_fit_and_histogram():
    path = ROOT / "scripts" / "eval_subconscious_cue.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    banned = ("held_bank", "prefer_items", "avoid_items", "toward_items", "away_items", "RifMeasure")
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in ("histogram", "fit"):
            continue
        blob = ast.get_source_segment(source, node)
        for name in banned:
            assert name not in blob
