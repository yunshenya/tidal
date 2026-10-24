"""LC-NE arousal. No SmolLM download. The Eye model is the cue-test stand-in."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.arousal import (
    AROUSAL_BASELINE,
    AROUSAL_TAU,
    AROUSAL_WINDOW,
    W_SURPRISE,
    W_TRIGGER,
    W_VALENCE,
    after_surprise,
    after_triggers,
    after_valence,
    apply_trace,
    quiet_step,
)
from tidal.cue_inject import CueState
from tidal.llm_memory import ByteTokenizer, CUE_HALF_LIFE, CUE_INTERFACE_VERSION, CUE_LAMBDA, CUE_THETA, LLMMemory
from tidal.sleep import SLEEP_FLOOR
from tidal.subconscious_arousal_data import FIT_IDS, HELD_IDS, PROTOCOL, bank_sha256

ROOT = Path(__file__).resolve().parent.parent
DOI = "10.1146/annurev.neuro.28.061604.135709"


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


def test_quiet_decay_stays_above_baseline_and_phasic_bumps_are_monotone():
    assert quiet_step(AROUSAL_BASELINE) == AROUSAL_BASELINE
    high = after_surprise(AROUSAL_BASELINE, 4.0, (1.0, 1.0, 1.0, 1.0))
    assert AROUSAL_BASELINE < high <= 1.0
    one = quiet_step(high)
    four = high
    for _ in range(4):
        four = quiet_step(four)
    assert high > one > four > AROUSAL_BASELINE
    assert after_surprise(AROUSAL_BASELINE, 1.0, ()) == AROUSAL_BASELINE
    assert after_surprise(AROUSAL_BASELINE, 1.0, (1.0,)) == AROUSAL_BASELINE
    assert after_surprise(AROUSAL_BASELINE, 1.0, (1.0, 1.0)) == AROUSAL_BASELINE
    assert after_valence(AROUSAL_BASELINE, 0.0) == AROUSAL_BASELINE
    low = after_valence(AROUSAL_BASELINE, 0.5)
    higher = after_valence(AROUSAL_BASELINE, 1.5)
    assert AROUSAL_BASELINE < low < higher <= 1.0
    assert after_valence(AROUSAL_BASELINE, -1.5) == higher
    assert after_triggers(high, 0) == high
    one_trigger = after_triggers(AROUSAL_BASELINE, 1)
    four_triggers = after_triggers(AROUSAL_BASELINE, 4)
    assert AROUSAL_BASELINE < one_trigger < four_triggers <= 1.0
    assert apply_trace(AROUSAL_BASELINE, [False, False]) == [AROUSAL_BASELINE, AROUSAL_BASELINE]
    dense = apply_trace(AROUSAL_BASELINE, [True, True, True, True])
    assert dense[-1] > apply_trace(AROUSAL_BASELINE, [True])[-1] > AROUSAL_BASELINE
    assert AROUSAL_TAU == 8.0
    assert W_SURPRISE == 0.35
    assert W_VALENCE == 0.25
    assert W_TRIGGER == 0.20
    assert AROUSAL_WINDOW == 8
    with pytest.raises(ValueError):
        after_surprise(AROUSAL_BASELINE, float("nan"), (1.0, 1.0))
    with pytest.raises(TypeError):
        after_triggers(AROUSAL_BASELINE, True)


def test_remember_updates_arousal_without_touching_the_write_gate_or_sleep():
    mem = _memory()
    try:
        surprises = iter((1.0, 1.0, 1.0, 1.0, 4.0))

        def fake(_text, _answer):
            return np.zeros(mem.hidden, dtype=np.float32), next(surprises)

        mem._value_and_surprise = fake
        names = ("ada", "bea", "cal", "dee", "eve")
        for name in names[:4]:
            assert mem.remember(f"code is {name}", name=name, rel="note")
            assert mem.arousal == AROUSAL_BASELINE
        assert mem.remember("code is eve", name="eve", rel="note")
        assert mem.arousal > AROUSAL_BASELINE
        assert mem.surprise_window() == (1.0, 1.0, 1.0, 1.0, 4.0)
        assert mem.arousal_window() == (1.0, 1.0, 1.0, 1.0, 4.0)
        assert mem.sleep_pressure == SLEEP_FLOOR
        window = mem.surprise_window()
        mem.observe_surprise(9.0)
        assert mem.surprise_window() == window
        assert mem.sleep_pressure == SLEEP_FLOOR
        rejected = _memory()
        try:
            assert rejected.remember("note is K", name="kay", rel="note", kind="chat") is False
            assert rejected.arousal == AROUSAL_BASELINE
            assert rejected.surprise_window() == ()
            assert rejected.arousal_window() == ()
        finally:
            rejected.close()
        noisy = _memory()
        try:
            noisy._value_and_surprise = lambda _text, _answer: (
                np.zeros(noisy.hidden, dtype=np.float32),
                float("nan"),
            )
            assert noisy.remember("code is Q", name="que", rel="note") is True
            assert noisy.arousal == AROUSAL_BASELINE
            assert noisy.arousal_window() == ()
            assert math.isnan(noisy.surprise_window()[-1])
            assert noisy.sleep_pressure == SLEEP_FLOOR
        finally:
            noisy.close()
        liked = _memory()
        try:
            liked._value_and_surprise = lambda _text, _answer: (
                np.zeros(liked.hidden, dtype=np.float32),
                1.0,
            )
            fact = {"text": "code is Y", "answer": "Y", "name": "ada", "rel": "note", "experience": 1}
            assert liked.remember(fact)
            assert liked.store.slots[0].valence == pytest.approx(0.25)
            assert liked.arousal == pytest.approx(after_valence(AROUSAL_BASELINE, 0.25))
            assert liked.surprise_window() == (1.0,)
            assert liked.sleep_pressure == SLEEP_FLOOR
        finally:
            liked.close()
    finally:
        mem.close()


def test_cue_export_steps_and_the_generate_hook_does_not():
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.zeros(mem.cue_dim, dtype=np.float32)
        value[3] = 4.0
        assert mem.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")
        other = _memory()
        try:
            assert other.store.write(key.copy(), value.copy(), name="ada", rel="note", answer="A", text="awake-cue-text")
            other.observe_valence(1.5)
            before = float(mem.arousal)
            pressure = float(mem.sleep_pressure)
            fields = mem.cue_fields(text)
            moved = other.cue_fields(text)
            flags = [bool(item) for item in fields["triggered"][0].tolist()]
            expected = apply_trace(before, flags)
            exported = [float(item) for item in fields["arousal"].detach().reshape(-1).tolist()]
            assert flags
            assert any(flags)
            assert len(exported) == len(expected)
            assert all(math.isclose(got, want, abs_tol=1e-6) for got, want in zip(exported, expected))
            assert fields["interface_version"] == CUE_INTERFACE_VERSION == 6
            assert torch.equal(fields["activation"], moved["activation"])
            assert torch.equal(mem.last_added, other.last_added)
            assert mem.sleep_pressure == pressure
            assert other.sleep_pressure == pressure
            state = CueState()
            level = float(mem.arousal)
            state, step = mem.advance_cue(state, hidden[0])
            step_flag = bool(step["triggered"][0, 0])
            want = apply_trace(level, [step_flag])
            assert math.isclose(float(step["arousal"][0, 0]), want[0], abs_tol=1e-6)
        finally:
            other.close()
        hooked = _memory()
        try:
            assert hooked.store.write(key.copy(), value.copy(), name="ada", rel="note", answer="A")
            hooked.observe_valence(1.5)
            level = float(hooked.arousal)
            hooked._cue_live = True
            hooked._cue_resume = False
            enc = hooked._tokenize(text)
            hooked.model(input_ids=enc["input_ids"], use_cache=False)
            assert hooked.arousal == level
            flat = [float(item) for item in hooked.last_cue_fields["arousal"].detach().reshape(-1).tolist()]
            assert flat
            assert all(math.isclose(item, level, abs_tol=1e-6) for item in flat)
            walked = apply_trace(level, [bool(item) for item in hooked.last_cue_fields["triggered"][0].tolist()])
            assert walked[-1] != level
        finally:
            hooked._cue_live = False
            hooked.close()
        quiet = _memory()
        try:
            quiet.observe_valence(1.5)
            level = float(quiet.arousal)
            ticks = quiet.offline_ticks(1)
            assert quiet.arousal == level
            assert math.isclose(float(ticks[0]["arousal"][0, 0]), level, abs_tol=1e-6)
            woke = quiet.wake_for(4.0)
            assert quiet.arousal == level
            slept = quiet.sleep_for(1.0)
            assert quiet.arousal == level
            assert slept["pressure_after"] < woke
            assert quiet.sleep_pressure > SLEEP_FLOOR
        finally:
            quiet.close()
    finally:
        mem.close()


def test_arousal_module_has_no_switch_and_banks_are_disjoint():
    source = (ROOT / "tidal" / "arousal.py").read_text(encoding="utf-8")
    imported = _imports(source)
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "environ" not in source
    data = (ROOT / "tidal" / "subconscious_arousal_data.py").read_text(encoding="utf-8")
    assert "torch" not in _imports(data)
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "import growbrain" not in memory
    assert "environ" not in memory
    residual = memory.split("def _residual_from_fields")[1].split("def cue_hidden")[0]
    assert "arousal" not in residual
    assert "sleep_pressure" not in residual
    assert CUE_INTERFACE_VERSION == 6
    assert CUE_THETA == 0.32
    assert CUE_HALF_LIFE == 30.0
    assert CUE_LAMBDA == 0.3
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert PROTOCOL == "arousal-lcne-v1"
    assert len(bank_sha256()) == 64
    prereg = (ROOT / "reports" / "subconscious_arousal_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (source, prereg, readme, architecture):
        assert DOI in text
        assert "10.1007/BF00487332" not in text
    assert "HYPOTHESIS" in prereg
    assert "The held bank was not scored." in prereg
    assert "The cue guard was not scored." in prereg
    assert "The sleep guard was not scored." in prereg
    assert bank_sha256() in prereg
    assert "The held bank was not scored." in readme
    assert "The cue guard was not scored." in readme
    assert "The sleep guard was not scored." in readme
    assert "The held bank was not scored." in architecture
    assert "The cue guard was not scored." in architecture
    assert "The sleep guard was not scored." in architecture
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "speak" in source
    assert "exploration" in source
