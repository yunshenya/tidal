"""Live v7 multimodal contract: shapes, cue export, and fusion write API."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from tidal.interface_v7 import (
    FINGERPRINT_DIM,
    FUSION_WEIGHT_DIM,
    FUSION_WEIGHT_ORDER,
    INTERFACE_VERSION_V7,
    MODALITY_VALUES,
    FusionWeightsStub,
    check_fingerprint,
    check_fusion_weights,
    check_modality,
    empty_fusion_weights,
    zero_fingerprint,
)
from tidal.llm_memory import ByteTokenizer, CUE_INTERFACE_VERSION, LLMMemory


def _live_interface_version() -> int:
    src = Path(__file__).resolve().parents[1] / "tidal" / "llm_memory.py"
    tree = ast.parse(src.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "CUE_INTERFACE_VERSION":
                    return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "CUE_INTERFACE_VERSION":
                return ast.literal_eval(node.value)
    raise AssertionError("CUE_INTERFACE_VERSION not found")


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


def test_live_export_is_v7():
    live = _live_interface_version()
    assert live == 7
    assert INTERFACE_VERSION_V7 == 7
    assert CUE_INTERFACE_VERSION == INTERFACE_VERSION_V7 == live


def test_stub_shapes():
    assert FINGERPRINT_DIM == 32
    assert FUSION_WEIGHT_DIM == 3
    assert FUSION_WEIGHT_ORDER == ("text", "audio", "image")
    assert MODALITY_VALUES == ("text", "audio", "image", "fused", "none")
    fp = zero_fingerprint()
    w = empty_fusion_weights()
    assert fp.shape == (32,) and fp.dtype.name == "float32"
    assert w.shape == (3,) and w.dtype.name == "float32"
    assert check_fingerprint(fp).shape == (32,)
    assert check_fusion_weights(w).shape == (3,)
    assert check_modality("fused") == "fused"
    stub = FusionWeightsStub(weights=w)
    assert stub.weights.shape == (3,)


def test_cue_fields_emits_v7_fingerprint_and_zero_fusion():
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.linspace(0.1, 1.0, mem.cue_dim, dtype=np.float32)
        mark = np.linspace(1.0, 2.0, FINGERPRINT_DIM, dtype=np.float32)
        assert mem.store.write(key, value, name="ada", rel="note", answer="A", fingerprint=mark)
        fields = mem.cue_fields(text)
        assert fields["interface_version"] == CUE_INTERFACE_VERSION == 7
        fp = np.asarray(fields["fingerprint"], dtype=np.float32)
        assert fp.shape == (32,) and fp.dtype.name == "float32"
        assert np.allclose(fp, mark)
        assert float(fields["fusion_confidence"]) == 0.0
        assert "fusion_weights" not in fields
        empty = _memory()
        try:
            quiet = empty.cue_fields(text)
            zeros = np.asarray(quiet["fingerprint"], dtype=np.float32)
            assert zeros.shape == (32,)
            assert np.array_equal(zeros, zero_fingerprint())
            assert float(quiet["fusion_confidence"]) == 0.0
        finally:
            empty.close()
    finally:
        mem.close()


def test_set_fusion_weights_rejects_wrong_shape():
    mem = _memory()
    try:
        with pytest.raises(ValueError, match="fusion_weights"):
            mem.set_fusion_weights(np.zeros((2,), dtype=np.float32))
        mem.set_fusion_weights(np.array([0.2, 0.5, 0.3], dtype=np.float32), pair="ab")
        assert mem.fusion_weights.shape == (3,)
        assert mem.fusion_weights.dtype.name == "float32"
        assert np.allclose(mem.fusion_weights, [0.2, 0.5, 0.3])
        assert mem.fusion_pair == "ab"
        fields = mem.cue_fields("Ab")
        assert "fusion_weights" not in fields
        assert float(fields["fusion_confidence"]) == 0.0
    finally:
        mem.close()


def test_modality_check_rejects_unknown():
    with pytest.raises(ValueError, match="unknown modality"):
        check_modality("voice")
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.ones(mem.cue_dim, dtype=np.float32)
        with pytest.raises(ValueError, match="unknown modality"):
            mem.store.write(key, value, name="ada", rel="note", modality="voice")
        assert mem.store.write(key, value, name="ada", rel="note", modality="audio")
        assert mem.store.slots[0].modality == "audio"
        fields = mem.cue_fields(text)
        assert fields["modality"][0][0] == "audio"
        assert float(fields["fusion_confidence"]) == 0.0
    finally:
        mem.close()
