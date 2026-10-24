"""Growbrain is the default memory backend. SmolLM2 stays an explicit constructor."""
from __future__ import annotations

import ast
import inspect
import os

import numpy as np
import pytest
import torch
from torch import nn

from tidal.backend import GrowbrainBackend, MemoryBackend, SmolLM2Backend, answer_nll, try_import_growbrain
from tidal.llm_memory import ByteTokenizer, LLMMemory, answer_surprise

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Eye(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(260, 64)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.lm_head = nn.Linear(64, 260, bias=False)
        self.config = type("Cfg", (), {"hidden_size": 64})()

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


def test_passed_in_model_is_smollm2_and_default_is_growbrain():
    signature = inspect.signature(LLMMemory.__init__)
    assert "backend" not in signature.parameters
    source = inspect.getsource(LLMMemory.__init__)
    assert "SmolLM2Backend(" in source
    assert "GrowbrainBackend(" in source
    assert "environ" not in source
    text = open(os.path.join(ROOT, "tidal", "backend.py"), encoding="utf-8").read()
    tree = ast.parse(text)
    imported = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "growbrain" not in imported
    assert "transformers" not in imported
    assert "environ" not in text
    assert "import growbrain" in text
    assert try_import_growbrain() is None
    with pytest.raises(ImportError, match="GrowbrainBackend"):
        LLMMemory()
    mem = _memory()
    try:
        assert type(mem.backend) is SmolLM2Backend
        assert not isinstance(mem.backend, GrowbrainBackend)
        assert isinstance(mem.backend, MemoryBackend)
        assert mem.model is mem.backend.model
        assert mem.layers is mem.backend.layers
        assert mem.layer == mem.backend.layer == 2
        assert "inject_residual" in inspect.getsource(LLMMemory._on_layer)
        assert "count_nonzero" in inspect.getsource(LLMMemory._on_layer)
        assert "compete_retrieved" in inspect.getsource(LLMMemory._rank)
    finally:
        mem.close()


def test_encode_and_nll_share_one_forward_and_skip_inject():
    mem = _memory()
    try:
        seen = {"n": 0}
        original = mem.model.forward

        def wrapped(*args, **kwargs):
            seen["n"] += 1
            seen["hidden_states"] = kwargs.get("output_hidden_states")
            return original(*args, **kwargs)

        mem.model.forward = wrapped
        mem.backend.model.forward = wrapped
        value, surprise = mem._value_and_surprise("ab", "b")
        assert seen["n"] == 1
        assert seen["hidden_states"] is True
        assert value.shape == (64,)
        hidden, _enc, logits = mem.backend.encode_hidden("ab")
        again = mem.backend.answer_nll(logits, _enc["input_ids"][0].tolist(), [False, True])
        assert again == answer_surprise(logits, _enc["input_ids"][0].tolist(), [False, True])
        assert again == answer_nll(logits, _enc["input_ids"][0].tolist(), [False, True])
        cue = mem.cue_hidden("ab")
        assert cue.dtype == np.float64
        assert np.array_equal(cue, hidden.detach().cpu().numpy().astype(np.float64))
        mem._train_delta = torch.ones(mem.hidden)
        mem._cue_live = True
        clean, _enc, _logits = mem.backend.encode_hidden("ab")
        mem._train_delta = None
        mem._cue_live = False
        plain, _enc, _logits = mem.backend.encode_hidden("ab")
        assert torch.equal(clean, plain)
        assert np.isfinite(surprise)
    finally:
        mem.close()


def test_inject_residual_matches_the_same_index_add():
    mem = _memory()
    try:
        backend = mem.backend
        hidden = torch.zeros(1, 3, 4)
        residual = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
        updated = backend.inject_residual(hidden, residual, broadcast=False, batched=True)
        assert torch.equal(updated, hidden + torch.tensor(residual))
        flat = torch.zeros(3, 4)
        flat_add = residual
        updated_flat = backend.inject_residual(flat, flat_add, broadcast=False, batched=False)
        assert torch.equal(updated_flat, flat + torch.tensor(residual[0]))
        delta = torch.arange(4, dtype=torch.float32)
        broadcast = backend.inject_residual(hidden, delta, broadcast=True)
        assert torch.equal(broadcast, hidden + delta.view(1, 1, 4))
        wrapped = backend.inject_residual((hidden, "tail"), delta, broadcast=True)
        assert torch.equal(wrapped[0], hidden + delta.view(1, 1, 4))
        assert wrapped[1] == "tail"
        with pytest.raises(NotImplementedError):
            MemoryBackend().encode_hidden("ab")
    finally:
        mem.close()


def test_recorded_baseline_matches_itself_and_catches_a_moved_floor():
    import copy
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location(
        "eval_cue_backend", os.path.join(ROOT, "scripts", "eval_subconscious_cue.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = json.loads(open(os.path.join(ROOT, "reports", "subconscious_cue_event_held.json"), encoding="utf-8").read())
    winner = json.loads(open(os.path.join(ROOT, "reports", "subconscious_cue_winner_event.json"), encoding="utf-8").read())
    floors = {
        "rates": copy.deepcopy(base["floor_rates"]),
        "means": copy.deepcopy(base["floor_means"]),
        "winner_events": {"k": winner["events"]["k"], "n": winner["events"]["n"]},
    }
    floors["rates"]["winner_same"]["k"] = winner["items"]["k"]
    floors["rates"]["winner_same"]["n"] = winner["items"]["n"]
    assert mod._pr57_diffs(base["direction"], floors) == []
    floors["rates"]["prefer_up"]["k"] = 45
    diffs = mod._pr57_diffs(base["direction"], floors)
    assert diffs == [{"field": "floors.prefer_up.k", "got": 45, "expected": 46}]


def test_prereg_and_docs_lock_the_interface_only_claim():
    prereg = open(os.path.join(ROOT, "reports", "subconscious_cue_prereg.md"), encoding="utf-8").read()
    for needle in (
        "This pull request is interface-only.",
        "No number is expected to change.",
        "The backend bank was not scored.",
        "encode_hidden(text) -> (hidden, encoding, logits)",
        "answer_nll(logits, token_ids, answer_mask) -> float",
        "inject_residual(output, residual, *, broadcast=False, batched=True)",
    ):
        assert needle in prereg
    for name in ("README.md", "docs/architecture.md"):
        text = open(os.path.join(ROOT, name), encoding="utf-8").read()
        assert "SmolLM2Backend" in text
        assert "GrowbrainBackend" in text
        assert "d_model" in text
        assert "192" in text
        assert "The live `interface_version` is 6." in text
        assert "encode_hidden" in text
        assert "The backend bank was not scored." in text
        assert "The held bank was not scored." in text
        assert "duplex" in text.lower()
