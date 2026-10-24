"""Wall-clock decay and a drifting internal clock. No SmolLM download."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import AROUSAL_BASELINE, AROUSAL_TAU, AROUSAL_WINDOW, W_SURPRISE, W_TRIGGER, W_VALENCE
from tidal.clock_drift import (
    DRIFT_RATE,
    ENTRAIN_GAIN,
    WALL_HALF_LIFE,
    WALL_TAU,
    WALL_TICK,
    clock_activation,
    elapsed_since,
)
from tidal.cue_inject import CueState, trace_sequence
from tidal.episodic import CLOCK_SPAN, CLOCK_STEP
from tidal.llm_memory import (
    ByteTokenizer,
    CUE_A_TRIG,
    CUE_FLOOR,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_TAU,
    CUE_THETA,
    LLMMemory,
)
from tidal.sleep import SLEEP_FLOOR, TAU_SLEEP, TAU_WAKE
from tidal.subconscious_clock_data import (
    BOTH_TOKEN_STRONG,
    BOTH_WALL_STRONG,
    ENTRAIN_SECONDS,
    FIT_IDS,
    HELD_IDS,
    PROTOCOL,
    TOKEN_STEPS,
    WALL_CLEAR,
    WALL_SHORT,
    bank_sha256,
)

ROOT = Path(__file__).resolve().parent.parent
DOIS = (
    "10.1037/0278-7393.23.5.1083",
    "10.1038/nrn1764",
    "10.1146/annurev-psych-010213-115117",
)


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


def _armed(mem, **overrides) -> CueState:
    payload = dict(
        a=1.0,
        dt=0,
        memory_id=0,
        confidence=1.0,
        strength=1.0,
        valence=0.0,
        gap=1.0,
        winner_cosine=1.0,
        fan_factor=1.0,
        vec=np.ones(mem.cue_dim, dtype=np.float32),
        trigger_wall=0.0,
    )
    payload.update(overrides)
    return CueState(**payload)


def _imports(source: str) -> list:
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    return imported


def test_token_formula_is_exact_when_the_wall_clock_is_frozen():
    assert WALL_TICK == 1.0
    assert WALL_HALF_LIFE == 60.0
    assert WALL_TAU == pytest.approx(60.0 / math.log(2.0))
    assert DRIFT_RATE == 1.25
    assert ENTRAIN_GAIN == 0.5
    assert CUE_HALF_LIFE == 30.0
    assert CUE_TAU == pytest.approx(30.0 / math.log(2.0))
    assert CUE_THETA == 0.32
    assert CUE_LAMBDA == 0.3
    assert CUE_FLOOR == 0.05
    assert CUE_A_TRIG == 1.0
    for dt in TOKEN_STEPS:
        got = clock_activation(1.0, int(dt), 0.0, CUE_TAU, WALL_TAU)
        assert got == math.exp(-int(dt) / CUE_TAU)
    values = np.ones((1, 3), dtype=np.float32)
    state = CueState(
        a=1.0,
        dt=0,
        memory_id=0,
        confidence=1.0,
        strength=1.0,
        fan_factor=1.0,
        vec=values[0].copy(),
        trigger_wall=0.0,
    )
    from tidal.cue_inject import step_cue

    omitted, omitted_row = step_cue(
        state,
        [0.0],
        [1.0],
        [0.0],
        values,
        theta=0.5,
        tau=CUE_TAU,
        a_trig=1.0,
        floor=CUE_FLOOR,
        gap_tau=0.8,
        valence_span=1.0,
    )
    frozen, _row = step_cue(
        state,
        [0.0],
        [1.0],
        [0.0],
        values,
        theta=0.5,
        tau=CUE_TAU,
        a_trig=1.0,
        floor=CUE_FLOOR,
        gap_tau=0.8,
        valence_span=1.0,
        now=0.0,
    )
    assert omitted.a == math.exp(-1.0 / CUE_TAU)
    assert frozen.a == omitted.a
    assert omitted_row["activation"] == omitted.a
    assert elapsed_since(5.0, 4.0) == 0.0
    assert elapsed_since(5.0, None) == 0.0


def test_wall_decay_is_the_lower_factor_and_does_not_use_the_product():
    short, mid = WALL_SHORT
    assert short == 1.0 and mid == 4.0
    assert WALL_CLEAR == 300.0
    at_zero = clock_activation(1.0, 0, 0.0, CUE_TAU, WALL_TAU)
    at_short = clock_activation(1.0, 0, float(short), CUE_TAU, WALL_TAU)
    at_mid = clock_activation(1.0, 0, float(mid), CUE_TAU, WALL_TAU)
    at_clear = clock_activation(1.0, 0, float(WALL_CLEAR), CUE_TAU, WALL_TAU)
    assert at_zero == 1.0
    assert at_zero > at_short > at_mid > CUE_FLOOR
    assert at_clear < CUE_FLOOR
    assert at_mid == math.exp(-float(mid) / WALL_TAU)
    assert at_mid != clock_activation(1.0, 0, float(mid) * DRIFT_RATE, CUE_TAU, WALL_TAU)
    for dt, elapsed, stronger in (
        (*BOTH_WALL_STRONG, "wall"),
        (*BOTH_TOKEN_STRONG, "token"),
    ):
        token_only = clock_activation(1.0, int(dt), 0.0, CUE_TAU, WALL_TAU)
        wall_only = clock_activation(1.0, 0, float(elapsed), CUE_TAU, WALL_TAU)
        both = clock_activation(1.0, int(dt), float(elapsed), CUE_TAU, WALL_TAU)
        assert both == min(token_only, wall_only)
        assert both <= token_only
        assert both <= wall_only
        assert both > token_only * wall_only
        if stronger == "wall":
            assert wall_only < token_only
        else:
            assert token_only < wall_only
    cues = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    keys = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    values = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    fields, final = trace_sequence(
        cues,
        keys,
        [1.0],
        [0.0],
        values,
        np.zeros(3),
        theta=0.5,
        tau=CUE_TAU,
        a_trig=1.0,
        floor=CUE_FLOOR,
        gap_tau=0.8,
        valence_span=1.0,
        now=100.0,
    )
    assert fields["activation"][0] == 1.0
    assert fields["activation"][1] == math.exp(-1.0 / CUE_TAU)
    assert final.trigger_wall == 100.0
    assert final.dt == 1


def test_advance_wall_decays_a_carried_trace_and_calibration_does_not():
    mem = _memory()
    try:
        assert mem.wall_seconds == 0.0
        assert mem.internal_seconds == 0.0
        assert mem.clock_error == 0.0
        phase = mem.clock_phase()
        mem.wake_for(4.0)
        mem.observe_valence(1.5)
        pressure = mem.sleep_pressure
        arousal = mem.arousal
        assert pressure > SLEEP_FLOOR
        assert arousal > AROUSAL_BASELINE
        held = _armed(mem)
        mem.cue_state = held
        assert mem.advance_wall(0.0)["wall"] == 0.0
        assert mem.cue_state is held
        reading = mem.advance_wall(float(WALL_SHORT[1]))
        assert reading["wall"] == 4.0
        assert reading["internal"] == 5.0
        assert reading["clock_error"] == 1.0
        assert mem.cue_state.dt == 0
        assert mem.cue_state.memory_id == 0
        assert mem.cue_state.a == clock_activation(1.0, 0, 4.0, CUE_TAU, WALL_TAU)
        assert mem.cue_state.a > CUE_FLOOR
        assert mem.sleep_pressure == pressure
        assert mem.arousal == arousal
        assert mem.clock_phase() == phase
        carried = mem.cue_state
        level = float(carried.a)
        once = mem.calibrate(mem.wall_seconds)
        assert once["wall"] == 4.0
        assert once["clock_error"] == 1.0 * (1.0 - ENTRAIN_GAIN)
        assert mem.cue_state is carried
        assert mem.cue_state.a == level
        twice = mem.calibrate(mem.wall_seconds)
        assert twice["clock_error"] == 0.25
        assert mem.cue_state.a == level
        assert mem.sleep_pressure == pressure
        assert mem.arousal == arousal
        mem.store.clear()
        assert mem.wall_seconds == 4.0
        assert mem.internal_seconds == twice["internal"]
        with pytest.raises(ValueError):
            mem.advance_wall(-1.0)
    finally:
        mem.close()
    cleared = _memory()
    try:
        cleared.cue_state = _armed(cleared)
        cleared.advance_wall(float(WALL_CLEAR))
        assert cleared.cue_state.memory_id == -1
        assert cleared.cue_state.inactive()
        assert cleared.internal_seconds == float(WALL_CLEAR) * DRIFT_RATE
    finally:
        cleared.close()
    entrained = _memory()
    try:
        entrained.cue_state = _armed(entrained)
        entrained.advance_wall(float(ENTRAIN_SECONDS))
        assert entrained.clock_error == 2.0
        entrained.calibrate(entrained.wall_seconds)
        assert entrained.clock_error == 1.0
        entrained.calibrate(entrained.wall_seconds)
        assert entrained.clock_error == 0.5
        assert entrained.cue_state.a == clock_activation(1.0, 0, float(ENTRAIN_SECONDS), CUE_TAU, WALL_TAU)
    finally:
        entrained.close()


def test_fresh_export_ignores_wall_time_and_adds_no_field():
    before = _memory()
    after = _memory()
    try:
        text = "Ab"
        for mem in (before, after):
            hidden = mem.cue_hidden(text)
            key = hidden[0] - mem.query_mean
            value = np.zeros(mem.cue_dim, dtype=np.float32)
            value[3] = 4.0
            assert mem.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")
        early = before.cue_fields(text)
        after.advance_wall(float(ENTRAIN_SECONDS))
        assert after.wall_seconds == 8.0
        assert after.internal_seconds == 10.0
        late = after.cue_fields(text)
        assert int(early["interface_version"]) == int(late["interface_version"]) == CUE_INTERFACE_VERSION == 7
        assert torch.equal(early["activation"], late["activation"])
        for fields in (early, late):
            for name in ("clock_error", "wall_seconds", "internal_seconds", "internal"):
                assert name not in fields
        assert before.sleep_pressure == after.sleep_pressure == SLEEP_FLOOR
    finally:
        before.close()
        after.close()


def test_clock_module_has_no_switch_and_banks_are_disjoint():
    source = (ROOT / "tidal" / "clock_drift.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "time" not in imported
    assert "time.time" not in source
    assert "environ" not in source
    assert "HYPOTHESIS" in source
    assert "Not a circadian phase shift." in source
    assert "10.1037/0033-295X.84.3.279" not in source
    assert "10.1007/BF00487332" not in source
    data = (ROOT / "tidal" / "subconscious_clock_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "import growbrain" not in memory
    assert "environ" not in memory
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "wall" not in residual
    assert "clock" not in residual
    assert CUE_INTERFACE_VERSION == 7
    assert AROUSAL_BASELINE == 0.2
    assert AROUSAL_TAU == 8.0
    assert W_SURPRISE == 0.35
    assert W_VALENCE == 0.25
    assert W_TRIGGER == 0.20
    assert AROUSAL_WINDOW == 8
    assert SLEEP_FLOOR == 0.1
    assert TAU_WAKE == 16.0
    assert TAU_SLEEP == 4.0
    assert CLOCK_SPAN == 0.25
    assert CLOCK_STEP == pytest.approx(math.pi / 16.0)
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert PROTOCOL == "clock-drift-v1"
    assert len(bank_sha256()) == 64
    prereg = (ROOT / "reports" / "subconscious_clock_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        for doi in DOIS:
            assert doi in text
        assert "10.1007/BF00487332" not in text
    assert "HYPOTHESIS" in prereg
    for sentence in (
        "The held bank was not scored.",
        "The cue guard was not scored.",
        "The sleep guard was not scored.",
        "The arousal guard was not scored.",
    ):
        assert sentence in prereg
        assert sentence in readme
        assert sentence in architecture
    assert bank_sha256() in prereg
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "speak" in (ROOT / "tidal" / "arousal.py").read_text(encoding="utf-8")
