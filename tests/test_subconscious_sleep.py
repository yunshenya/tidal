"""Process S, reconsolidation, and the version-4 export.

No SmolLM download. The Eye model is the same stand-in the cue tests use.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.llm_memory import ByteTokenizer, LLMMemory
from tidal.sleep import (
    PE_THRESHOLD,
    REPLAY_BLEND,
    SLEEP_FLOOR,
    SPONTANEOUS_CEILING,
    SPONTANEOUS_PERIOD,
    SPONTANEOUS_S_MIN,
    STRENGTH_STEP,
    TAU_SLEEP,
    TAU_WAKE,
    prediction_error,
    reconsolidate,
    sleep_after,
    spontaneous_due,
    spontaneous_winner,
    wake_after,
)
from tidal.subconscious_sleep_data import FIT_IDS, HELD_IDS, PROTOCOL, bank_sha256

ROOT = Path(__file__).resolve().parent.parent


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


def test_wake_rises_and_sleep_stays_above_the_floor():
    start = float(SLEEP_FLOOR)
    pressures = [wake_after(start, duration) for duration in (1.0, 2.0, 4.0)]
    assert start < pressures[0] < pressures[1] < pressures[2] <= 1.0
    assert wake_after(wake_after(start, 1.0), 1.0) == pytest.approx(wake_after(start, 2.0))
    assert wake_after(start, 0.0) == start
    before = wake_after(start, 4.0)
    after_short = sleep_after(before, 1.0)
    after_long = sleep_after(before, 3.0)
    drop_short = before - after_short
    drop_long = before - after_long
    assert drop_long > drop_short > 0.0
    assert after_long > SLEEP_FLOOR
    assert after_short > SLEEP_FLOOR
    assert sleep_after(before, 0.0) == before
    dense = [sleep_after(before, duration) for duration in (1.0, 2.0, 3.0)]
    assert dense[0] > dense[1] > dense[2] > SLEEP_FLOOR
    with pytest.raises(ValueError):
        wake_after(start, -1.0)
    with pytest.raises(ValueError):
        sleep_after(before, float("inf"))
    with pytest.raises(ValueError):
        wake_after(0.0, 1.0)


def test_prediction_error_gate_leaves_a_parallel_trace_alone():
    key = np.zeros(4, dtype=np.float64)
    key[0] = 1.0
    parallel = key.copy()
    orthogonal = np.zeros(4, dtype=np.float64)
    orthogonal[1] = 1.0
    assert prediction_error(key, parallel) == pytest.approx(0.0)
    assert prediction_error(key, orthogonal) == pytest.approx(1.0)
    assert prediction_error(key, parallel) <= PE_THRESHOLD
    kept = reconsolidate(key, parallel, 1.0)
    assert kept["updated"] is False
    assert kept["strength"] == pytest.approx(1.0)
    assert np.array_equal(kept["value"], parallel)
    changed = reconsolidate(key, orthogonal, 1.0)
    assert changed["updated"] is True
    assert changed["strength"] == pytest.approx(math.exp(-STRENGTH_STEP))
    expected = (1.0 - REPLAY_BLEND) * orthogonal + REPLAY_BLEND * key
    assert np.allclose(changed["value"], expected)
    assert changed["strength"] > 0.0
    assert "text" not in changed


def test_spontaneous_rate_is_inside_the_ceiling():
    fires = [spontaneous_due(tick, 0.3, 1) for tick in range(SPONTANEOUS_PERIOD)]
    assert fires == [True] + [False] * (SPONTANEOUS_PERIOD - 1)
    rate = sum(fires) / SPONTANEOUS_PERIOD
    assert 0.0 < rate < SPONTANEOUS_CEILING
    assert spontaneous_due(0, SPONTANEOUS_S_MIN, 1) is True
    assert spontaneous_due(0, SPONTANEOUS_S_MIN - 1e-9, 1) is False
    assert spontaneous_due(0, 1.0, 0) is False
    assert spontaneous_winner([0.1, 0.4, 0.4], [1.0, 0.2, 0.9]) == 2
    assert TAU_WAKE > 0.0 and TAU_SLEEP > 0.0


def test_memory_exports_pressure_and_does_not_replay_raw_text():
    mem = _memory()
    try:
        assert mem.sleep_pressure == pytest.approx(SLEEP_FLOOR)
        raised = mem.wake_for(4.0)
        assert raised == pytest.approx(wake_after(SLEEP_FLOOR, 4.0))
        assert raised > SPONTANEOUS_S_MIN
        other = _memory()
        try:
            other.wake_for(4.0)
            short = mem.sleep_for(1.0)
            long = other.sleep_for(3.0)
        finally:
            other.close()
        assert long["drop"] > short["drop"] > 0.0
        assert short["pressure_after"] > SLEEP_FLOOR
        assert long["pressure_after"] > 0.0
        fresh = _memory()
        try:
            dim = fresh.hidden
            low_k = np.zeros(dim, dtype=np.float64)
            low_k[0] = 1.0
            high_k = np.zeros(dim, dtype=np.float64)
            high_k[2] = 1.0
            high_v = np.zeros(dim, dtype=np.float64)
            high_v[3] = 1.0
            assert fresh.store.write(
                low_k, low_k.copy(), name="low", rel="note", group="low", answer="LOW-ANSWER", text="raw-low-sleep"
            )
            assert fresh.store.write(
                high_k, high_v, name="high", rel="note", group="high", answer="HIGH-ANSWER", text="raw-high-sleep"
            )
            report = fresh.sleep_for(1.0)
            low, high = fresh.store.slots
            assert low.text == "raw-low-sleep" and low.answer == "LOW-ANSWER"
            assert high.text == "raw-high-sleep" and high.answer == "HIGH-ANSWER"
            assert low.strength == pytest.approx(1.0)
            assert np.allclose(low.v, low_k)
            assert high.strength == pytest.approx(math.exp(-STRENGTH_STEP))
            assert high.strength > 0.0
            assert not np.allclose(high.v, high_v)
            blob = json.dumps(report["replay"])
            assert "raw-low-sleep" not in blob and "raw-high-sleep" not in blob
            assert "LOW-ANSWER" not in blob and "HIGH-ANSWER" not in blob
            for row in fresh.sleep_audit():
                assert set(row) == {"index", "error", "updated", "strength_before", "strength_after"}
            assert report["replay"][0]["updated"] is False
            assert report["replay"][1]["updated"] is True
            assert fresh.sleep_pressure == pytest.approx(SLEEP_FLOOR)
            fields = fresh.cue_fields("Zz")
            assert "sleep_audit" not in fields
            assert "raw-high-sleep" not in json.dumps(fields["source"])
            assert all(item in ("cue", "none") for item in fields["source"][0])
        finally:
            fresh.close()
        awake = _memory()
        try:
            awake.wake_for(4.0)
            dim = awake.hidden
            key = np.zeros(dim, dtype=np.float64)
            key[0] = 1.0
            assert awake.store.write(key, key.copy(), name="wander", rel="note", answer="STAY", text="raw-spontaneous-sleep")
            ticks = awake.offline_ticks(SPONTANEOUS_PERIOD)
            sources = [tick["source"][0][0] for tick in ticks]
            assert sources.count("spontaneous") == 1
            assert sources.count("none") == SPONTANEOUS_PERIOD - 1
            rate = sources.count("spontaneous") / SPONTANEOUS_PERIOD
            assert 0.0 < rate < SPONTANEOUS_CEILING
            assert awake.sleep_pressure == pytest.approx(wake_after(SLEEP_FLOOR, 4.0))
            assert awake.store.slots[0].text == "raw-spontaneous-sleep"
            assert awake.store.slots[0].strength == pytest.approx(1.0)
            audit = json.dumps(awake.sleep_audit())
            assert "raw-spontaneous-sleep" not in audit
            assert awake.sleep_audit()[0]["updated"] is False
        finally:
            awake.close()
    finally:
        mem.close()


def test_cue_source_is_never_spontaneous_and_pressure_skips_the_residual():
    left = _memory()
    right = _memory()
    try:
        text = "Ab"
        hidden = left.cue_hidden(text)
        key = hidden[0] - left.query_mean
        value = np.zeros(left.cue_dim, dtype=np.float32)
        value[3] = 4.0
        assert left.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")
        hidden_r = right.cue_hidden(text)
        key_r = hidden_r[0] - right.query_mean
        assert right.store.write(key_r, value.copy(), name="ada", rel="note", answer="A", text="awake-cue-text")
        right.wake_for(4.0)
        before = float(left.store.slots[0].strength)
        full = left.cue_fields(text)
        moved = right.cue_fields(text)
        assert left.store.slots[0].strength == before
        assert left.sleep_pressure == pytest.approx(SLEEP_FLOOR)
        assert right.sleep_pressure > left.sleep_pressure
        assert torch.equal(full["activation"], moved["activation"])
        assert torch.equal(full["vec"], moved["vec"])
        assert torch.equal(left.last_added, right.last_added)
        assert set(full["source"][0]) <= {"cue", "none"}
        assert "cue" in full["source"][0]
        assert "spontaneous" not in full["source"][0]
        assert float(full["sleep_pressure"][0, 0]) == pytest.approx(left.sleep_pressure)
        assert float(moved["sleep_pressure"][0, 0]) == pytest.approx(right.sleep_pressure)
        assert "awake-cue-text" not in json.dumps(full["source"])
        empty = _memory()
        try:
            quiet = empty.cue_fields("Ab")
            assert quiet["source"] == [["none"] * int(quiet["memory_id"].shape[1])]
            assert float(quiet["sleep_pressure"][0, 0]) == pytest.approx(SLEEP_FLOOR)
        finally:
            empty.close()
    finally:
        left.close()
        right.close()


def test_sleep_module_has_no_switch_and_banks_are_disjoint():
    source = (ROOT / "tidal" / "sleep.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "torch" not in imported
    assert "os" not in imported
    assert "growbrain" not in imported
    assert "import growbrain" not in source
    assert "environ" not in source
    memory = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "compete_retrieved" in memory
    assert "_sleep_audit" not in memory.split("def cue_fields")[1].split("def _rank")[0]
    assert "_sleep_audit" not in memory.split("def generate")[1].split("def teacher_force_loss")[0]
    data = (ROOT / "tidal" / "subconscious_sleep_data.py").read_text(encoding="utf-8")
    data_tree = ast.parse(data)
    data_imports = []
    for node in data_tree.body:
        if isinstance(node, ast.Import):
            data_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            data_imports.append(node.module.split(".")[0])
    assert "torch" not in data_imports
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert len(FIT_IDS) == 8
    assert len(HELD_IDS) == 50
    assert PROTOCOL == "sleep-pressure-v1"
    assert len(bank_sha256()) == 64
    prereg = (ROOT / "reports" / "subconscious_sleep_prereg.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")
    for text in (prereg, readme, architecture):
        for doi in (
            "10.1152/ajpregu.1984.246.2.R161",
            "10.1111/jsr.12371",
            "10.1111/jsr.13598",
            "10.1126/science.1231357",
            "10.1098/rstb.2019.0693",
        ):
            assert doi in text
        assert "10.1007/BF00487332" not in text
        assert "The held bank was not scored." in text
        assert "The cue guard was not scored." in text
        assert "cannot be equated with mind-wandering" in text
        assert "sleep_pressure" in text
    assert "HYPOTHESIS" in prereg
    assert "The backend bank was not scored." in readme
    assert "The backend bank was not scored." in architecture
    assert "`interface_version` is 4" in readme
    assert "`interface_version` is 4" in architecture
    assert "duplex" in architecture.lower()
    assert "0.15452824346787175" in prereg
    assert "50/50" in prereg
