"""Stage0 growbrain backend. d_model 192 is the packet width. mem_dim 64 is not."""
from __future__ import annotations

import ast
import os
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from tidal.backend import (
    GROWBRAIN_CHECKPOINT,
    GROWBRAIN_D_MODEL,
    GROWBRAIN_HEAD_DIM,
    GROWBRAIN_INJECT_LAYER,
    GROWBRAIN_MEM_DIM,
    GROWBRAIN_MEM_KEY_DIM,
    GROWBRAIN_N_LAYERS,
    GROWBRAIN_STAGE0_SHA256,
    GROWBRAIN_VOCAB_SIZE,
    GrowbrainBackend,
    load_growbrain_stage0,
    resolve_d_model,
    stage0_checkpoint_files,
    try_import_growbrain,
)
from tidal.growbrain_stage0_data import FIT, FIT_IDS, HELD, HELD_IDS, PROTOCOL, bank_sha256
from tidal.llm_memory import CUE_INTERFACE_VERSION, CUE_LAMBDA, CUE_THETA, LLMMemory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class _Char(nn.Module):
    """Four blocks. Token i is an embedding of width d_model. model(idx) returns logits."""

    def __init__(self, hidden=192, n_layers=4, vocab=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.head = nn.Linear(hidden, vocab, bias=False)
        self.config = type(
            "Cfg",
            (),
            {
                "d_model": hidden,
                "mem_dim": 64,
                "mem_key_dim": 32,
                "inject_layer": 2,
                "n_layers": n_layers,
                "head_dim": 48,
            },
        )()

    def forward(self, idx):
        if idx.dim() == 1:
            idx = idx.unsqueeze(0)
        hidden = self.embed(idx)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)


class _Tok:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(ch) % 128 for ch in text]


def _zeros(self):
    row = np.zeros(self.hidden, dtype=np.float64)
    return row, row.copy()


def test_stage0_pins_and_interface_version():
    assert GROWBRAIN_D_MODEL == 192
    assert GROWBRAIN_MEM_DIM == 64
    assert GROWBRAIN_MEM_KEY_DIM == 32
    assert GROWBRAIN_INJECT_LAYER == 2
    assert GROWBRAIN_N_LAYERS == 4
    assert GROWBRAIN_HEAD_DIM == 48
    assert GROWBRAIN_CHECKPOINT == "e1_base_rebalance"
    assert CUE_INTERFACE_VERSION == 6
    assert CUE_THETA == 0.32
    assert CUE_LAMBDA == 0.3
    assert resolve_d_model(None, None) == 192
    assert resolve_d_model({"mem_dim": 64, "mem_key_dim": 32}, None) == 192
    assert resolve_d_model({"d_model": 320, "mem_dim": 64}, None) == 320
    with pytest.raises(ValueError, match="d_model"):
        resolve_d_model({"d_model": 320}, type("M", (), {"config": type("C", (), {"d_model": 192})()})())


def test_missing_package_does_not_construct_smollm(monkeypatch):
    monkeypatch.setattr("tidal.backend.try_import_growbrain", lambda: None)
    with pytest.raises(ImportError, match="SmolLM2Backend"):
        load_growbrain_stage0(device="cpu")
    with pytest.raises(ImportError, match="GrowbrainBackend"):
        LLMMemory()


def test_imported_package_without_a_loader_refuses_a_576_shell(monkeypatch):
    fake = types.ModuleType("growbrain")
    fake.tidal_adapter = object()
    monkeypatch.setitem(sys.modules, "growbrain", fake)
    for name in ("growbrain.model", "growbrain.tokenizer"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(RuntimeError, match="576"):
        load_growbrain_stage0(device="cpu")


def test_default_store_is_d_model_not_the_injector_slot(monkeypatch, tmp_path):
    from tidal.llm_memory import _growbrain_inject_path

    cfg = {
        "d_model": 192,
        "mem_dim": 64,
        "mem_key_dim": 32,
        "inject_layer": 2,
        "n_layers": 4,
        "head_dim": 48,
    }
    monkeypatch.setattr("tidal.llm_memory.load_growbrain_stage0", lambda **_kwargs: (_Char(), _Tok(), cfg))
    monkeypatch.setattr(LLMMemory, "_key_centers", _zeros)
    mem = LLMMemory()
    try:
        assert type(mem.backend) is GrowbrainBackend
        assert mem.model_name == GROWBRAIN_CHECKPOINT
        assert mem.hidden == mem.cue_dim == mem.store.dk == mem.store.dv == 192
        assert mem.backend.mem_dim == 64
        assert mem.store.dk != mem.backend.mem_dim
        assert mem.layer == 2 and mem.n_layers == 4
        assert mem.backend.mem_key_dim == 32 and mem.backend.head_dim == 48
        last = mem.inject.net[-1].weight.detach().cpu()
        assert tuple(last.shape) == (192, 192)
        shipped_path = _growbrain_inject_path()
        if shipped_path.is_file():
            shipped = torch.load(shipped_path, map_location="cpu", weights_only=False)
            expect = shipped["inject"]["net.2.weight"].detach().cpu()
            assert torch.equal(last, expect)
            assert int(torch.count_nonzero(last)) > 0
        else:
            assert int(torch.count_nonzero(last)) == 0
        hidden, enc, logits = mem.backend.encode_hidden("ab")
        assert tuple(hidden.shape) == (2, 192)
        assert hidden.dtype == torch.float32
        assert int(enc["input_ids"].shape[-1]) == 2
        assert logits.shape[-1] == 128
        cue = mem.cue_hidden("ab")
        assert cue.shape == (2, 192)
        assert cue.dtype == np.float64
        nll = mem.backend.answer_nll(logits, enc["input_ids"][0].tolist(), [False, True])
        assert nll > 0.0
        fields = mem.cue_fields("ab")
        assert int(fields["interface_version"]) == 6
        assert tuple(fields["vec"].shape) == (1, 2, 192)
    finally:
        mem.close()

    monkeypatch.setattr(
        "tidal.llm_memory._growbrain_inject_path",
        lambda: tmp_path / "absent.pt",
    )
    bare = LLMMemory()
    try:
        assert bare.hidden == 192
        assert bare.backend.mem_dim == 64
        assert bare.store.dk != bare.backend.mem_dim
        assert torch.count_nonzero(bare.inject.net[-1].weight) == 0
    finally:
        bare.close()


def test_cfg_d_model_replaces_the_pin_and_a_576_residual_is_refused(monkeypatch):
    wide = _Char(hidden=576, n_layers=4)
    wide.config.d_model = 320
    cfg = {"d_model": 320, "mem_dim": 64, "inject_layer": 2, "n_layers": 4}
    monkeypatch.setattr("tidal.llm_memory.load_growbrain_stage0", lambda **_kwargs: (wide, _Tok(), cfg))
    monkeypatch.setattr(LLMMemory, "_key_centers", _zeros)
    mem = LLMMemory()
    try:
        assert mem.hidden == 320
        assert mem.store.dk == 320
        assert mem.backend.mem_dim == 64
        with pytest.raises(RuntimeError, match="576"):
            mem.backend.encode_hidden("ab")
    finally:
        mem.close()


def test_model_idx_is_not_rewritten_when_the_module_rejects_it():
    class _KeywordOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.config = type("Cfg", (), {"d_model": 192, "n_layers": 4, "inject_layer": 2, "mem_dim": 64})()

        def forward(self, *, input_ids):
            del input_ids
            raise AssertionError("keyword signature ran")

    backend = GrowbrainBackend(
        _KeywordOnly(),
        _Tok(),
        layer=2,
        n_layers=4,
        layers=_KeywordOnly().layers,
        device=torch.device("cpu"),
        cfg={"d_model": 192, "n_layers": 4, "inject_layer": 2, "mem_dim": 64},
    )
    # The layers on the backend must be the module's own blocks.
    model = _KeywordOnly()
    backend.model = model
    backend.layers = model.layers
    with pytest.raises(RuntimeError, match="model\\(idx\\)"):
        backend.encode_hidden("ab")


def test_backend_import_of_growbrain_is_inside_the_try():
    source = open(os.path.join(ROOT, "tidal", "backend.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top.append(node.module.split(".")[0])
    assert "growbrain" not in top
    assert "environ" not in source
    memory = open(os.path.join(ROOT, "tidal", "llm_memory.py"), encoding="utf-8").read()
    assert "import growbrain" not in memory
    assert "TidalAdapter.encode_hidden" not in inspect_encode()


def test_stage0_banks_are_disjoint_and_explicit_is_separate():
    from tidal.llm_memory import _CALIB_NAMES
    from tidal.llm_memory_data import names

    assert PROTOCOL == "growbrain-stage0-v1"
    assert len(FIT_IDS) == 8 and len(HELD_IDS) == 50
    assert set(FIT_IDS).isdisjoint(HELD_IDS)
    assert FIT_IDS[0] == "growbrain-fit-000" and FIT_IDS[-1] == "growbrain-fit-007"
    assert HELD_IDS[0] == "growbrain-held-000" and HELD_IDS[-1] == "growbrain-held-049"
    banned = set(_CALIB_NAMES) | set(names("train")) | set(names("test"))
    used = {row["name"] for row in FIT + HELD}
    assert used.isdisjoint(banned)
    assert len(bank_sha256()) == 64
    families = {}
    for row in HELD:
        families[row["family"]] = families.get(row["family"], 0) + 1
    assert families == {"tone": 14, "choice": 14, "preference": 14, "phone": 4, "birthday": 4}
    assert GROWBRAIN_VOCAB_SIZE == 5135
    assert len(GROWBRAIN_STAGE0_SHA256) == 64


def test_stage0_checkpoint_loads_when_present():
    if try_import_growbrain() is None or stage0_checkpoint_files() is None:
        pytest.skip("stage0 package or checkpoint is not on this machine")
    model, tok, cfg = load_growbrain_stage0(device="cpu")
    assert cfg["d_model"] == 192 and cfg["mem_dim"] == 64
    assert cfg["inject_layer"] == 2 and cfg["n_layers"] == 4
    assert cfg["vocab_size"] == 5135 and len(tok) == 5135
    ids = torch.tensor([tok.encode("ab")], dtype=torch.long)
    logits = model(ids)
    assert tuple(logits.shape) == (1, 2, 5135)
    backend = GrowbrainBackend(
        model,
        tok,
        layer=2,
        n_layers=4,
        layers=model.blocks,
        device=torch.device("cpu"),
        cfg=cfg,
    )
    hidden, _enc, got = backend.encode_hidden("Hi")
    assert tuple(hidden.shape) == (2, 192)
    assert tuple(got.shape) == (1, 2, 5135)


def inspect_encode() -> str:
    import inspect

    return inspect.getsource(GrowbrainBackend.encode_hidden)
