"""Frozen-LM residual memory. CPU only. No SmolLM download."""
import ast
import inspect
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from tidal.llm_memory import LLMMemory, greedy_continuation
from tidal.llm_memory_train import fit_inject

ROOT = Path(__file__).resolve().parent.parent


class _Out:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class EyeLM(nn.Module):
    """Embedding, identity blocks, identity head. Hidden index == token id.

    Answer rows 'Y' and 'Z' are scaled so a residual add selects that token.
    """

    def __init__(self, n_layers=4):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, vocab)
        self.layers = nn.ModuleList(
            [nn.Linear(vocab, vocab, bias=False) for _ in range(n_layers)]
        )
        self.lm_head = nn.Linear(vocab, vocab, bias=False)
        eye = torch.eye(vocab)
        eye[ord("Y")] *= 5
        eye[ord("Z")] *= 5
        with torch.no_grad():
            self.embed.weight.copy_(eye)
            self.lm_head.weight.copy_(torch.eye(vocab))
            for layer in self.layers:
                layer.weight.copy_(torch.eye(vocab))
        self.config = type("Cfg", (), {"hidden_size": vocab, "vocab_size": vocab})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **_kwargs):
        del attention_mask, use_cache
        h = self.embed(input_ids)
        states = [h]
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        logits = self.lm_head(h)
        return _Out(logits, tuple(states) if output_hidden_states else None)


def _byte_memory(**kwargs):
    from tidal.llm_memory import ByteTokenizer

    model = EyeLM()
    mem = LLMMemory(model, ByteTokenizer(), overwrite_cos=1.0, **kwargs)
    return mem


def test_import_tidal_does_not_import_transformers():
    code = (
        "import sys\n"
        "import tidal\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'tidal.llm_memory' not in sys.modules\n"
        "import tidal.episodic, tidal.continuum, tidal.question_head, tidal.episodic_eval\n"
        "import tidal.llm_memory\n"
        "assert 'transformers' not in sys.modules\n"
        "import tidal.llm_memory_train\n"
        "assert 'transformers' not in sys.modules\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call([sys.executable, "-c", code], cwd=str(ROOT), env=env)


def test_llm_modules_do_not_import_transformers_at_load():
    import tidal.llm_memory as mem
    import tidal.llm_memory_train as train

    for mod in (mem, train):
        tree = ast.parse(inspect.getsource(mod))
        bad = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                bad.extend(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                bad.append(node.module.split(".")[0])
        assert "transformers" not in bad


def test_packaging_extra_and_missing_message(monkeypatch):
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extra = project["project"]["optional-dependencies"]["llm"]
    assert any(item.startswith("transformers") for item in extra)
    assert any(item.startswith("torch") for item in extra)
    assert (ROOT / "requirements-llm.txt").is_file()

    import tidal.optional as opt

    def boom(name):
        raise ImportError("no module named " + name)

    monkeypatch.setattr(opt.importlib, "import_module", boom)
    with pytest.raises(ImportError, match=r"tidal\[llm\]") as exc:
        opt.require("transformers", "llm")
    assert "requirements-llm.txt" in str(exc.value)


def test_empty_store_matches_base_model():
    mem = _byte_memory()
    model = mem.model
    tok = mem.tokenizer
    prompt = "Ask:"
    try:
        hooked = mem.continuation_ids(prompt, max_new_tokens=4)
        assert mem.generate(prompt, max_new_tokens=4) == "::::"
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        mem._delta = None
        logits_hooked = model(input_ids=ids, use_cache=False).logits
        mem.close()
        bare = greedy_continuation(model, tok, prompt, 4, tok.eos_token_id, torch.device("cpu"))
        logits_bare = model(input_ids=ids, use_cache=False).logits
    finally:
        mem.close()
    assert hooked == bare
    assert torch.equal(logits_hooked, logits_bare)


def test_same_identity_overwrite_returns_new_value():
    mem = _byte_memory()
    try:
        prompt = "Ask:"
        assert "Y" not in prompt and "Z" not in prompt
        assert mem.remember("code is Y", name="ada", rel="phone")
        assert len(mem.store) == 1
        assert mem.generate(prompt, max_new_tokens=4) == "YYYY"
        y = mem.encode_value("code is Y", "Y")
        assert y.shape == (260,)
        assert abs(float(np.linalg.norm(y)) - 5.0) < 1e-5
        assert np.allclose(mem.injection(prompt), y)
        assert mem.remember("code is Z", name="ada", rel="phone")
        assert len(mem.store) == 1
        z = mem.encode_value("code is Z", "Z")
        assert not np.allclose(y, z)
        slot = mem.store.slots[0]
        assert slot.name == "ada" and slot.rel == "phone"
        assert np.allclose(slot.v, z)
        assert np.allclose(mem.injection(prompt), z)
        assert mem.generate(prompt, max_new_tokens=4) == "ZZZZ"
        assert torch.allclose(mem.last_added, torch.tensor(z))
    finally:
        mem.close()


def test_other_identity_is_kept_and_soft_topk_mixes():
    mem = _byte_memory()
    mix = None
    try:
        assert mem.remember("code is Y", name="ada", rel="phone")
        assert mem.remember("town is Q", name="bea", rel="city")
        assert len(mem.store) == 2
        bea = [s for s in mem.store.slots if s.name == "bea"][0].v.copy()
        assert mem.remember("code is Z", name="ada", rel="phone")
        assert len(mem.store) == 2
        bea_after = [s for s in mem.store.slots if s.name == "bea"][0].v
        ada = [s for s in mem.store.slots if s.name == "ada"][0].v
        assert np.allclose(bea_after, bea)
        assert np.allclose(ada, mem.encode_value("code is Z", "Z"))

        mix = _byte_memory()
        v1 = mix.encode_value("red is Y", "Y")
        v2 = mix.encode_value("blue is Z", "Z")
        assert mix.remember("red is Y", name="a", rel="p", key="query")
        assert mix.remember("blue is Z", name="b", rel="c", key="query")
        assert len(mix.store) == 2
        got = mix.injection("query")
        assert np.allclose(got, 0.5 * (v1 + v2), atol=1e-5)
        assert "self.store.write" in inspect.getsource(LLMMemory.remember)
        assert "self.store.readout" in inspect.getsource(LLMMemory._arm)
    finally:
        mem.close()
        if mix is not None:
            mix.close()


def test_chat_write_is_dropped():
    mem = _byte_memory()
    try:
        assert mem.remember("code is Y", name="ada", rel="phone", kind="chat") is False
        assert len(mem.store) == 0
        assert np.all(mem.injection("Ask:") == 0)
    finally:
        mem.close()


def test_fit_inject_moves_zero_last_layer():
    mem = _byte_memory()
    try:
        w0 = mem.inject.net[-1].weight.detach().clone()
        assert torch.count_nonzero(w0) == 0
        facts = [
            {"text": "Ada's phone is 111", "a": "111", "q": "Ada's phone is "},
            {"text": "Bea's city is Oslo", "a": "Oslo", "q": "Bea's city is "},
            {"text": "Cal's color is green", "a": "green", "q": "Cal's color is "},
        ]
        # Answer pool is three '1' tokens, not the whole fact, and is not unit length.
        pooled = mem.encode_value(facts[0]["text"], facts[0]["a"])
        whole = mem.encode_key(facts[0]["text"])
        assert not np.allclose(pooled, whole)
        # '1' is a one-hot row. The span mean stays that row; it is not renormalized.
        assert np.allclose(pooled, mem.model.embed.weight[ord("1")].detach().numpy())
        log = fit_inject(mem, facts, steps=1, bs=2, lr=1e-2, neg_p=1.0, seed=0)
        assert np.isfinite(log[0]["loss"])
        assert not torch.equal(mem.inject.net[-1].weight, w0)
    finally:
        mem.close()


def test_untrained_inject_is_identity_copy():
    mem = _byte_memory()
    try:
        value = torch.arange(mem.hidden, dtype=torch.float32)
        got = mem.inject(value)
        assert torch.equal(got, value)
    finally:
        mem.close()


def test_tiny_llama_empty_store_matches_and_script_runs(tmp_path):
    pytest.importorskip("transformers")
    from tidal.llm_memory import build_tiny_lm

    model, tok = build_tiny_lm(seed=0, hidden=32, layers=4)
    mem = LLMMemory(model, tok)
    try:
        prompt = "Ask:"
        hooked = mem.continuation_ids(prompt, max_new_tokens=3)
        ids = tok(prompt, return_tensors="pt")["input_ids"]
        mem._delta = None
        logits_hooked = model(input_ids=ids, use_cache=False).logits
        # One slot: the armed residual is the answer-span pool, still with the fact out of the prompt.
        assert mem.remember("code is Y", name="ada", rel="phone")
        pooled = mem.encode_value("code is Y", "Y")
        assert "Y" not in prompt
        assert np.allclose(mem.injection(prompt), pooled)
        # Clearing the store leaves generation identical to the base forward.
        mem.store.clear()
        assert len(mem.store) == 0
        cleared = mem.continuation_ids(prompt, max_new_tokens=3)
        mem.close()
        bare = greedy_continuation(model, tok, prompt, 3, tok.eos_token_id, torch.device("cpu"))
        logits_bare = model(input_ids=ids, use_cache=False).logits
        assert hooked == bare == cleared
        assert torch.equal(logits_hooked, logits_bare)
    finally:
        mem.close()

    out = tmp_path / "inj.pt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "tidal.llm_memory_train",
            "--tiny",
            "--steps",
            "1",
            "--bs",
            "1",
            "--n-cache",
            "4",
            "--neg-p",
            "0",
            "--out",
            str(out),
        ],
        cwd=str(ROOT),
        env=env,
    )
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["hidden"] == 32
    assert ckpt["log"][0]["loss"] == ckpt["log"][0]["loss"]
    weight = ckpt["inject"]["net.2.weight"]
    assert torch.count_nonzero(weight) > 0


def test_pool_key_tokens_drops_only_the_sink():
    from tidal.llm_memory import pool_key_tokens

    rows = np.ones((4, 6), dtype=np.float64)
    rows[0, 3] = 8000
    pooled = pool_key_tokens(rows)
    assert abs(float(pooled[3]) - 1.0) < 1e-8
    equal = pool_key_tokens(np.eye(4))
    assert np.allclose(equal, np.eye(4).mean(axis=0))


def test_sink_token_does_not_collapse_distinct_facts():
    """A planted first-token spike used to make every key cosine > 0.9."""
    from tidal.llm_memory import ByteTokenizer, pool_masked

    class SinkLM(EyeLM):
        def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **kwargs):
            out = super().forward(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                use_cache=use_cache,
                **kwargs,
            )
            if not output_hidden_states or out.hidden_states is None:
                return out
            states = []
            for hidden in out.hidden_states:
                hidden = hidden.clone()
                hidden[:, 0, 7] = hidden[:, 0, 7] + 8000
                states.append(hidden)
            return _Out(out.logits, tuple(states))

    mem = LLMMemory(SinkLM(), ByteTokenizer())  # default overwrite_cos 0.9
    try:
        facts = [
            ("code is Y", "Y", "ada", "phone"),
            ("town is Q", "Q", "bea", "city"),
            ("team is M", "M", "cal", "team"),
            ("hue is R", "R", "dee", "color"),
        ]
        legacy = []
        fixed = []
        for text, answer, name, rel in facts:
            hidden, enc = mem._run_hidden(text)
            content = mem._content_mask(enc, hidden.shape[0])
            answer_m = mem._answer_mask(text, enc, answer, hidden.shape[0])
            mask = [c and not a for c, a in zip(content, answer_m)]
            legacy.append(pool_masked(hidden, mask).detach().cpu().numpy())
            assert mem.remember(text, answer, name=name, rel=rel)
            fixed.append(mem.encode_key(text, answer))
        assert _offdiag_max(legacy) > 0.98
        assert _offdiag_mean(fixed) < 0.5
        assert _offdiag_max(fixed) < 0.9
        assert len(mem.store) == len(facts)
        # Same identity still replaces the slot. The other three facts stay.
        assert mem.remember("code is Z", "Z", name="ada", rel="phone")
        assert len(mem.store) == len(facts)
        ada = [s for s in mem.store.slots if s.name == "ada"]
        assert len(ada) == 1
        assert np.allclose(ada[0].v, mem.encode_value("code is Z", "Z"))
        bea = [s for s in mem.store.slots if s.name == "bea"][0]
        assert np.allclose(bea.v, mem.encode_value("town is Q", "Q"))
        # The spike dim is not what the stored key points along.
        assert abs(float(fixed[0][7])) < 100
    finally:
        mem.close()


def test_held_out_facts_do_not_reuse_calibration_names():
    from tidal.llm_memory import _CALIB_NAMES
    from tidal.llm_memory_data import fact_stream, names, norm, session

    calib = set(_CALIB_NAMES)
    assert calib.isdisjoint(names("train"))
    assert calib.isdisjoint(names("test"))
    rng = random.Random(0)
    turns = session(rng, "test", 8)
    facts = [t for t in turns if t["kind"] == "fact"]
    assert facts
    assert all(t["n"] not in calib for t in facts)
    stream = fact_stream(rng, "test", 12)
    assert len({(t["n"], t["rel"]) for t in stream}) == 12
    assert len({norm(t["a"]) for t in stream}) == 12
    phone = next(t for t in stream if t["rel"] == "phone")
    assert phone["a"].count("-") >= 2
    job = next(t for t in stream if t["rel"] == "job")
    assert " " in job["a"]


def test_shipped_smol_key_centers_are_finite_and_distinct():
    from tidal.llm_memory import KEY_GEOM_VERSION, KEY_SINK_RATIO, _center_path

    path = _center_path()
    assert path.is_file()
    with np.load(path, allow_pickle=False) as z:
        assert str(z["version"].item()) == KEY_GEOM_VERSION
        assert str(z["model"].item()) == "HuggingFaceTB/SmolLM2-135M"
        assert int(z["layer"]) == 15
        assert int(z["hidden"]) == 576
        assert float(z["sink_ratio"]) == KEY_SINK_RATIO
        key_mean = np.asarray(z["key_mean"], dtype=np.float64)
        query_mean = np.asarray(z["query_mean"], dtype=np.float64)
    assert key_mean.shape == (576,)
    assert query_mean.shape == (576,)
    assert np.isfinite(key_mean).all() and np.isfinite(query_mean).all()
    assert not np.allclose(key_mean, query_mean)
    assert abs(float(key_mean[507])) > 10


def _offdiag_max(rows) -> float:
    return _offdiag(rows, "max")


def _offdiag_mean(rows) -> float:
    return _offdiag(rows, "mean")


def _offdiag(rows, which: str) -> float:
    X = np.stack(rows).astype(np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(nrm, 1e-12)
    C = U @ U.T
    v = C[~np.eye(len(X), dtype=bool)]
    return float(v.max() if which == "max" else v.mean())
