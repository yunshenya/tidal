"""The key prefix matches the full forward, and passed-in models stay on it."""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from torch import nn

from tidal.llm_memory import DEFAULT_MODEL, ByteTokenizer, LLMMemory

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
        assert mem._key_prefix_ok() is False
        pooled = mem._raw_key_pool("ab cd")
        assert seen["output_hidden_states"] is True
        assert pooled.shape == (64,)
        assert np.isfinite(pooled).all()
    finally:
        mem.close()


def test_tiny_llama_prefix_matches_the_full_hidden():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    from tidal.llm_memory import build_tiny_lm

    model, tokenizer = build_tiny_lm(seed=0)
    centers = LLMMemory._key_centers
    LLMMemory._key_centers = lambda self: (
        np.zeros(self.hidden, dtype=np.float64),
        np.zeros(self.hidden, dtype=np.float64),
    )
    try:
        mem = LLMMemory(model, tokenizer)
    finally:
        LLMMemory._key_centers = centers
    try:
        assert mem._key_prefix_ok() is True
        text = "Ada's phone is 111"
        fast_hidden, _enc = mem._run_key_hidden(text)
        slow_hidden, _enc, _logits = mem._run_hidden(text)
        assert torch.equal(fast_hidden, slow_hidden)
        assert np.array_equal(mem._raw_key_pool(text, "111"), mem._raw_key_pool(text, "111", full=True))
        hits = {"n": 0}
        handle = mem.layers[-1].register_forward_hook(lambda *_args: hits.__setitem__("n", hits["n"] + 1))
        try:
            mem._raw_key_pool(text)
        finally:
            handle.remove()
        assert hits["n"] == 0
    finally:
        mem.close()
        torch.set_num_threads(previous)


def test_prereg_locks_bitwise_identity_and_the_cost_grid():
    text = open(PREREG, encoding="utf-8").read()
    assert "torch.equal" in text
    assert "np.array_equal" in text
    assert "No absolute tolerance" in text
    assert "one CPU thread" in text
    assert "Intra-op threads and inter-op threads are both 1." in text
    assert "up to 3 draws" in text
    assert "Intra-op threads are `os.cpu_count()`." in text
    assert "noise floor" in text
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
    previous = torch.get_num_threads()
    # One intra-op thread and one inter-op thread is where this bfloat16 forward is bitwise.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_num_threads(1)
    mem = LLMMemory(model_name=DEFAULT_MODEL)
    try:
        yield mem
    finally:
        mem.close()
        torch.set_num_threads(previous)


def test_shipped_key_prefix_matches_the_full_hidden(shipped):
    mem = shipped
    assert mem._key_prefix_ok() is True
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
    assert mem.model.model.config.num_hidden_layers == mem.n_layers
