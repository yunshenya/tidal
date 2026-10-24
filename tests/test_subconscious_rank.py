"""The key-layer stop matches the full forward, and passed-in models stay on it."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch import nn

from tidal.llm_memory import ByteTokenizer, LLMMemory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "reports", "subconscious_rank_prereg.md")
PROMPT = "Weigh unlabeled cadence.\nTone mqtag0:"
FACT = "Tihababa recounts festival throughout mtone0."
ANSWER = "festival"


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


def test_passed_in_model_keeps_the_full_key_forward():
    model = _Eye()
    seen = {}
    original = model.forward

    def wrapped(*args, **kwargs):
        seen["output_hidden_states"] = kwargs.get("output_hidden_states")
        return original(*args, **kwargs)

    model.forward = wrapped
    centers = LLMMemory._key_centers
    LLMMemory._key_centers = lambda self: (
        np.zeros(self.hidden, dtype=np.float64),
        np.zeros(self.hidden, dtype=np.float64),
    )
    try:
        mem = LLMMemory(model, ByteTokenizer())
    finally:
        LLMMemory._key_centers = centers
    try:
        assert mem._key_layer_stop_ok() is False
        pooled = mem._raw_key_pool("ab cd")
        assert seen["output_hidden_states"] is True
        assert pooled.shape == (64,)
        assert np.isfinite(pooled).all()
    finally:
        mem.close()


def test_prereg_locks_bitwise_identity_and_the_cost_grid():
    text = open(PREREG, encoding="utf-8").read()
    assert "torch.equal" in text
    assert "np.array_equal" in text
    assert "No absolute tolerance" in text
    assert "Weigh unlabeled cadence." in text
    assert "Tone mqtag0:" in text
    assert "8, 16, 32, 128" in text
    assert "warmup 2" in text
    assert "repeats 7" in text
    assert "MIX_K" in text and "4" in text
    assert "MIX_TEMP" in text and "0.05" in text
    assert "rankcost" in text


@pytest.fixture(scope="module")
def shipped():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    mem = LLMMemory()
    try:
        yield mem
    finally:
        mem.close()


def test_shipped_key_stop_matches_the_full_hidden(shipped):
    mem = shipped
    assert mem._key_layer_stop_ok() is True
    hits = {"n": 0}
    handle = mem.layers[-1].register_forward_hook(lambda *_args: hits.__setitem__("n", hits["n"] + 1))
    try:
        fast = mem._raw_key_pool(PROMPT)
    finally:
        handle.remove()
    assert hits["n"] == 0
    slow = mem._raw_key_pool(PROMPT, full=True)
    assert np.array_equal(fast, slow)
    fast_fact = mem._raw_key_pool(FACT, ANSWER)
    slow_fact = mem._raw_key_pool(FACT, ANSWER, full=True)
    assert np.array_equal(fast_fact, slow_fact)
    again = mem._raw_key_pool(PROMPT, full=True)
    assert np.array_equal(again, slow)
