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

from tidal.llm_memory import (
    LLMMemory,
    RIF_COMPETITOR_KEEP,
    RIF_OVERWRITE_KEEP,
    answer_span,
    decode_answer,
    greedy_continuation,
    span_complete,
)
from tidal.llm_memory_train import fit_decision_value, fit_inject, fit_polarity

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
    packaged = project["tool"]["setuptools"]["package-data"]["tidal"]
    assert "*.npz" in packaged
    assert "*.pt" in packaged
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


def test_other_identity_is_kept_and_readout_is_top1():
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
        # Tied keys: hard top-1 still names the first slot, not a soft blend.
        # The gap is 0, so the residual scale is 0 and nothing is added.
        value, info = mix.store.read(mix._query_vector("query"), mode="top1")
        assert info["top"] == ("a", "p")
        assert np.allclose(value, v1, atol=1e-5)
        assert not np.allclose(value, 0.5 * (v1 + v2), atol=1e-5)
        got = mix.injection("query")
        assert np.allclose(got, 0.0, atol=1e-5)
        assert "self.store.write" in inspect.getsource(LLMMemory.remember)
        assert 'mode="top1"' in inspect.getsource(LLMMemory._rank)
    finally:
        mem.close()
        if mix is not None:
            mix.close()


def test_top1_decays_the_other_soft_top4_and_overwrite_drops_the_old_value():
    mem = _byte_memory()
    try:
        hidden = mem.hidden
        query = np.zeros(hidden, dtype=np.float64)
        query[0] = 1.0
        mem._query_vector = lambda _prompt: query
        # Cosines against query: 1.0, 0.8, 0.6, 0.4, and an outsider at 0.
        specs = (
            ("win", 1.0, 0.0),
            ("a", 0.8, 0.6),
            ("b", 0.6, 0.8),
            ("c", 0.4, np.sqrt(1.0 - 0.16)),
            ("out", 0.0, 1.0),
        )
        for name, x, y in specs:
            key = np.zeros(hidden, dtype=np.float64)
            key[0] = x
            key[1] = y
            value = np.zeros(hidden, dtype=np.float64)
            value[2] = 1.0
            assert mem.store.write(key, value, name=name, rel="fact")
        rank = mem._rank("probe")
        assert rank["top"] == ("win", "fact")
        assert rank["strength"] == 1.0
        strengths = {slot.name: slot.strength for slot in mem.store.slots}
        assert strengths["win"] == 1.0
        assert strengths["out"] == 1.0
        for name in ("a", "b", "c"):
            assert strengths[name] == pytest.approx(RIF_COMPETITOR_KEEP)
        # A second win decays the same three again. The outsider stays out.
        mem._rank("probe")
        strengths = {slot.name: slot.strength for slot in mem.store.slots}
        assert strengths["a"] == pytest.approx(RIF_COMPETITOR_KEEP ** 2)
        assert strengths["out"] == 1.0
        # A tie adds nothing and does not decay.
        same = _byte_memory()
        try:
            q = np.zeros(same.hidden, dtype=np.float64)
            q[0] = 1.0
            same._query_vector = lambda _prompt: q
            key = q.copy()
            for name in ("a", "b"):
                assert same.store.write(key, np.ones(same.hidden), name=name, rel="fact")
            assert np.allclose(same.injection("query"), 0.0)
            assert [slot.strength for slot in same.store.slots] == [1.0, 1.0]
        finally:
            same.close()
        assert RIF_OVERWRITE_KEEP < RIF_COMPETITOR_KEEP
        assert mem.remember("code is Y", name="ada", rel="phone")
        first = mem.encode_value("code is Y", "Y")
        assert mem.remember("code is Z", name="ada", rel="phone")
        updated = [slot for slot in mem.store.slots if slot.name == "ada"]
        assert len(updated) == 1
        assert updated[0].strength == 1.0
        assert np.allclose(updated[0].v, mem.encode_value("code is Z", "Z"))
        assert not np.allclose(updated[0].v, first)
    finally:
        mem.close()


def _legacy_v22(tok, ids) -> str:
    """Old free-gen stop: cut before a newline token, then on sentence punctuation."""
    out = []
    for nx in ids:
        s = tok.decode(out + [int(nx)], skip_special_tokens=True)
        if "\n" in s:
            break
        out.append(int(nx))
        if s.rstrip().endswith((".", "?", "!")) and len(s.strip()) > 1:
            break
    return tok.decode(out, skip_special_tokens=True)


def test_answer_span_skips_leading_newline_then_stops():
    assert answer_span("\n+1-415-555-0134") == "+1-415-555-0134"
    assert answer_span("\n\n  Oslo\nnope") == "Oslo"
    assert answer_span("\nThe color is green.\nextra") == "The color is green."
    assert answer_span("\nSure! The number is 5") == "Sure!"
    assert answer_span("\n") == ""
    assert answer_span("   \n\t") == ""
    assert answer_span("green") == "green"
    assert answer_span(".") == "."
    assert answer_span("..") == ".."
    assert answer_span("x?") == "x?"
    assert span_complete("\n") is False
    assert span_complete("\nHi") is False
    assert span_complete("\nHi!") is True
    assert span_complete("\nHi\nmore") is True
    assert span_complete(".") is False
    assert span_complete("..") is True


def test_answer_span_matches_v22_without_a_leading_newline():
    from tidal.llm_memory import ByteTokenizer

    tok = ByteTokenizer()
    samples = [
        "green",
        "green.",
        "harbor pilot",
        "March 4, 1991",
        "+1-415-555-0134",
        "Oslo\nextra",
        "ab.",
        ".",
        "..",
        "?",
        "x?",
        "Sure! more",
        "",
    ]
    for text in samples:
        ids = tok.encode(text)
        assert decode_answer(tok, ids) == _legacy_v22(tok, ids) == answer_span(text)


def test_decode_answer_recovers_text_after_a_leading_newline():
    from tidal.llm_memory import ByteTokenizer

    tok = ByteTokenizer()
    text = "\nYY.ZZ"
    ids = tok.encode(text)
    assert _legacy_v22(tok, ids) == ""
    assert decode_answer(tok, ids) == "YY."
    assert decode_answer(tok, []) == ""


class _ScriptedLM(nn.Module):
    """Emits a fixed id sequence. Logits ignore the residual, so the stop is isolated."""

    def __init__(self, prompt_len, script):
        super().__init__()
        self.prompt_len = int(prompt_len)
        self.script = [int(i) for i in script]
        self.n_forward = 0
        self.embed = nn.Embedding(260, 8)
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.config = type("Cfg", (), {"hidden_size": 8, "vocab_size": 260})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **_kwargs):
        del attention_mask, use_cache
        self.n_forward += 1
        h = self.embed(input_ids)
        states = [h]
        for layer in self.layers:
            h = layer(h)
            states.append(h)
        logits = torch.full((input_ids.shape[0], input_ids.shape[1], 260), -1.0e4)
        step = int(input_ids.shape[1]) - self.prompt_len
        nxt = self.script[min(max(step, 0), len(self.script) - 1)]
        logits[:, -1, nxt] = 0.0
        return _Out(logits, tuple(states) if output_hidden_states else None)


def test_generate_skips_leading_newline_and_stops():
    from tidal.llm_memory import ByteTokenizer

    tok = ByteTokenizer()
    prompt = "Ask:"
    assert len(tok.encode(prompt)) == 4
    script = [ord("\n"), ord("Y"), ord("Y"), ord("."), ord("Z")]
    model = _ScriptedLM(4, script)
    mem = LLMMemory(model, tok)
    try:
        model.n_forward = 0
        assert mem.generate(prompt, max_new_tokens=8) == "YY."
        assert model.n_forward == 4
        model.n_forward = 0
        ids = mem.continuation_ids(prompt, max_new_tokens=5)
        assert ids == script
        assert model.n_forward == 5
        assert _legacy_v22(tok, ids) == ""
        assert decode_answer(tok, ids) == "YY."
    finally:
        mem.close()


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
        key = torch.randn(mem.hidden)
        assert torch.equal(mem.inject(value, key), value)
    finally:
        mem.close()


def test_polarity_flips_the_same_value():
    mem = _byte_memory()
    try:
        value = torch.arange(mem.hidden, dtype=torch.float32)
        direction = torch.zeros(mem.hidden)
        direction[0] = 1.0
        with torch.no_grad():
            mem.inject.polarity.copy_(direction * 20)
        boosted = mem.inject(value, direction)
        flipped = mem.inject(value, -direction)
        assert torch.allclose(boosted, value * 3)
        assert torch.allclose(flipped, -value)
        key_pos = direction.numpy().astype(np.float32)
        mem.store.write(key_pos, value.numpy(), name="a", rel="p")
        assert np.allclose(mem.injection("Ask:"), value.numpy() * 3)
        mem.store.clear()
        mem.store.write(-key_pos, value.numpy(), name="a", rel="p")
        assert np.allclose(mem.injection("Ask:"), -value.numpy())
    finally:
        mem.close()


def test_missing_polarity_loads_as_zero(tmp_path):
    mem = _byte_memory()
    try:
        with torch.no_grad():
            mem.inject.polarity.fill_(4.0)
            mem.inject.net[-1].bias.fill_(0.2)
        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in mem.inject.state_dict().items()
            if key != "polarity"
        }
        path = tmp_path / "old.pt"
        torch.save({"inject": state, "layer": mem.layer, "hidden": mem.hidden}, path)
        fresh = _byte_memory()
        try:
            with torch.no_grad():
                fresh.inject.polarity.fill_(5.0)
            fresh.load_inject(path)
            assert torch.count_nonzero(fresh.inject.polarity) == 0
            assert torch.equal(fresh.inject.net[-1].bias.detach().cpu(), mem.inject.net[-1].bias.detach().cpu())
        finally:
            fresh.close()
    finally:
        mem.close()


def test_fit_polarity_trains_only_the_direction():
    mem = _byte_memory()
    try:
        net0 = mem.inject.net[-1].weight.detach().clone()
        pol0 = mem.inject.polarity.detach().clone()
        facts = [
            {"text": "Ada favors Y", "a": "Y", "q": "Pick:", "good": "Y", "bad": "Z", "family": "prefer"},
            {"text": "Bea refuses Y", "a": "Y", "q": "Pick:", "good": "Z", "bad": "Y", "family": "avoid"},
        ]
        incidental = [np.ones(mem.hidden, dtype=np.float32)]
        log = fit_polarity(mem, facts, incidental, steps=1, bs=1, lr=0.5, margin=0.0, seed=0)
        assert np.isfinite(log[0]["loss"])
        assert not torch.equal(mem.inject.polarity.detach(), pol0)
        assert torch.equal(mem.inject.net[-1].weight.detach(), net0)
    finally:
        mem.close()


def test_fit_inject_leaves_polarity_alone():
    mem = _byte_memory()
    try:
        with torch.no_grad():
            mem.inject.polarity.fill_(0.3)
        before = mem.inject.polarity.detach().clone()
        facts = [
            {"text": "Ada's phone is 111", "a": "111", "q": "Ada's phone is "},
            {"text": "Bea's city is Oslo", "a": "Oslo", "q": "Bea's city is "},
        ]
        fit_inject(mem, facts, steps=1, bs=1, lr=1e-2, neg_p=0.0, seed=0, weight_decay=1e-2)
        assert torch.equal(mem.inject.polarity.detach(), before)
    finally:
        mem.close()


def test_no_key_fit_leaves_polarity_alone():
    mem = _byte_memory()
    try:
        with torch.no_grad():
            mem.inject.polarity.fill_(0.4)
        before = mem.inject.polarity.detach().clone()
        facts = [
            {"text": "Ada favors Y", "a": "Y", "q": "Pick:", "good": "Y", "bad": "Z", "family": "prefer"},
            {"text": "Bea refuses Y", "a": "Y", "q": "Pick:", "good": "Z", "bad": "Y", "family": "avoid"},
        ]
        log = fit_decision_value(mem, facts, steps=1, bs=1, lr=1e-2, margin=0.0, seed=0)
        assert np.isfinite(log[0]["loss"])
        assert torch.equal(mem.inject.polarity.detach(), before)
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


class _ShapeLM(nn.Module):
    """Layer count and hidden size only. Init of a matching default must not forward."""

    def __init__(self, n_layers, hidden):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.config = type("Cfg", (), {"hidden_size": int(hidden)})()

    def forward(self, *args, **kwargs):
        raise AssertionError("LLMMemory init ran the language model")


class _UnusedTok:
    eos_token_id = 0


def _install_shape_loader(monkeypatch, n_layers=30, hidden=576):
    def fake(model_name, device="cpu", dtype=None):
        del model_name, device, dtype
        return _ShapeLM(n_layers, hidden), _UnusedTok()

    monkeypatch.setattr("tidal.llm_memory._load_pretrained", fake)


def _zero_centers(self):
    z = np.zeros(self.hidden, dtype=np.float64)
    return z, z.copy()


def _last_weight(mem):
    return mem.inject.net[-1].weight.detach().cpu()


def test_default_llm_memory_loads_shipped_inject(monkeypatch):
    """Default SmolLM2-135M shape loads the packaged MLP. No 135M download."""
    from tidal.llm_memory import DEFAULT_MODEL, InjectMLP, _inject_path

    path = _inject_path()
    assert path.is_file()
    assert path.name == "llm_inject_smollm2_135m.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert set(ckpt) == {"inject", "layer", "hidden"}
    assert int(ckpt["layer"]) == 15
    assert int(ckpt["hidden"]) == 576
    shipped = ckpt["inject"]["net.2.weight"]
    assert tuple(shipped.shape) == (576, 576)
    assert torch.count_nonzero(shipped) > 0
    assert float(ckpt["inject"]["scale"]) == 1.0
    assert float(ckpt["inject"]["phone_on"]) == 1.0
    assert tuple(ckpt["inject"]["phone_w"].shape) == (4, 10, 576)
    assert tuple(ckpt["inject"]["phone_b"].shape) == (4, 10)
    assert float(ckpt["inject"]["date_on"]) == 1.0
    assert tuple(ckpt["inject"]["date_month_w"].shape) == (12, 576)
    assert tuple(ckpt["inject"]["date_digit_w"].shape) == (6, 10, 576)
    assert tuple(ckpt["inject"]["date_digit_b"].shape) == (6, 10)

    _install_shape_loader(monkeypatch, n_layers=30, hidden=576)
    mem = LLMMemory()
    try:
        assert mem.model_name == DEFAULT_MODEL
        assert mem.n_layers == 30
        assert mem.layer == 15
        assert mem.hidden == 576
        assert torch.equal(_last_weight(mem), shipped)
        assert torch.equal(mem.inject.state_dict()["net.2.bias"].cpu(), ckpt["inject"]["net.2.bias"])
        blank = InjectMLP(576)
        value = torch.arange(576, dtype=torch.float32)
        assert torch.equal(blank(value), value)
        assert not torch.equal(mem.inject(value).cpu(), value)
        if "polarity" in ckpt["inject"]:
            assert torch.equal(mem.inject.polarity.detach().cpu(), ckpt["inject"]["polarity"])
        else:
            assert torch.count_nonzero(mem.inject.polarity) == 0
        assert float(mem.inject.phone_on) == 1.0
        assert float(mem.inject.date_on) == 1.0
    finally:
        mem.close()


def test_other_model_or_layer_keeps_zero_inject(monkeypatch):
    from tidal.llm_memory import _inject_path

    monkeypatch.setattr(LLMMemory, "_key_centers", _zero_centers)
    _install_shape_loader(monkeypatch, n_layers=30, hidden=576)

    other_layer = LLMMemory(layer=14)
    other_name = LLMMemory(model_name="other/model")
    passed = LLMMemory(_ShapeLM(30, 576), _UnusedTok())
    _install_shape_loader(monkeypatch, n_layers=4, hidden=32)
    other_shape = LLMMemory()
    _install_shape_loader(monkeypatch, n_layers=30, hidden=576)
    narrow = LLMMemory(inject_hidden=32)
    try:
        for mem in (other_layer, other_name, passed, other_shape, narrow):
            assert torch.count_nonzero(_last_weight(mem)) == 0
            value = torch.arange(mem.hidden, dtype=torch.float32)
            assert torch.equal(mem.inject(value).cpu(), value)
        assert other_layer.layer == 14 and other_layer.hidden == 576
        assert other_name.model_name == "other/model"
        assert passed.model_name is None and passed.layer == 15 and passed.hidden == 576
        assert other_shape.layer == 2 and other_shape.hidden == 32
        assert narrow.layer == 15 and narrow.inject.net[0].out_features == 32

        other_layer.load_inject(_inject_path())
        shipped = torch.load(_inject_path(), map_location="cpu", weights_only=False)
        assert torch.equal(_last_weight(other_layer), shipped["inject"]["net.2.weight"])
        assert not torch.equal(other_layer.inject(torch.zeros(576)), torch.zeros(576))
    finally:
        for mem in (other_layer, other_name, passed, other_shape, narrow):
            mem.close()


def test_load_inject_custom_path(tmp_path):
    mem = _byte_memory()
    try:
        assert torch.count_nonzero(_last_weight(mem)) == 0
        with torch.no_grad():
            mem.inject.net[-1].bias.fill_(0.25)
            mem.inject.net[-1].weight.fill_(0.01)
        path = tmp_path / "custom.pt"
        mem.save_inject(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert set(saved) == {"inject", "layer", "hidden"}
        assert int(saved["hidden"]) == mem.hidden
        assert int(saved["layer"]) == mem.layer

        fresh = _byte_memory()
        try:
            assert torch.count_nonzero(_last_weight(fresh)) == 0
            fresh.load_inject(path)
            assert torch.equal(_last_weight(fresh), _last_weight(mem))
            assert torch.equal(fresh.inject.net[-1].bias.detach().cpu(), mem.inject.net[-1].bias.detach().cpu())
            value = torch.arange(fresh.hidden, dtype=torch.float32)
            assert not torch.equal(fresh.inject(value).cpu(), value)
        finally:
            fresh.close()
    finally:
        mem.close()


def test_phone_pool_is_the_last_four_digit_tokens():
    mem = _byte_memory()
    try:
        from tidal.llm_memory import phone_block_sizes

        answer = "+1-415-555-1111"
        pooled = mem.encode_value(f"Ada's phone number is {answer}", answer)
        # Four identical digit tokens. Each block is that token's leading slice.
        emb = mem.model.embed.weight[ord("1")].detach().numpy()
        sizes = phone_block_sizes(mem.hidden)
        assert np.allclose(pooled, np.concatenate([emb[: width] for width in sizes]))
        assert not np.allclose(pooled, emb)
        city = mem.encode_value("Ada lives in Oslo", "Oslo")
        assert not np.allclose(city, mem.model.embed.weight[ord("o")].detach().numpy())
    finally:
        mem.close()


def test_phone_block_sizes_are_four_equal_parts():
    from tidal.llm_memory import phone_block_sizes

    assert phone_block_sizes(576) == [144, 144, 144, 144]
    assert phone_block_sizes(32) == [8, 8, 8, 8]
    assert phone_block_sizes(260) == [65, 65, 65, 65]
    assert sum(phone_block_sizes(7)) == 7
    assert phone_block_sizes(7) == [1, 2, 2, 2]


def test_phone_pool_puts_each_digit_in_its_own_block():
    """A ones-digit edit must not move the tens block. The mean does."""
    from tidal.llm_memory import assemble_phone_value, phone_block_sizes, pool_answer

    hidden = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)
    mask = [False, False, False, False, True, True, True, True]
    answer = "1234567"
    pooled = pool_answer(hidden, mask, answer)
    sizes = phone_block_sizes(32)
    rows = hidden[-4:]
    expect = assemble_phone_value([rows[i] for i in range(4)])
    assert torch.equal(pooled, expect)
    edited = hidden.clone()
    edited[-1] = edited[-1] + 1
    pooled_ones = pool_answer(edited, mask, answer)
    tens_at = sizes[0] + sizes[1]
    assert torch.equal(pooled_ones[: tens_at + sizes[2]], pooled[: tens_at + sizes[2]])
    assert not torch.equal(pooled_ones[tens_at + sizes[2] :], pooled[tens_at + sizes[2] :])
    mean = rows.mean(dim=0)
    assert not torch.allclose(pooled, mean)

    mem = _byte_memory()
    try:
        sizes = phone_block_sizes(mem.hidden)
        emb = mem.model.embed.weight.detach().numpy()
        answer = "+1-415-555-2060"
        pooled = mem.encode_value(f"Ada's phone number is {answer}", answer)
        parts = [emb[ord(ch)][: sizes[i]] for i, ch in enumerate("2060")]
        assert np.allclose(pooled, np.concatenate(parts))
        ones = mem.encode_value("Ada's phone number is +1-415-555-2061", "+1-415-555-2061")
        cut = sum(sizes[:3])
        assert np.array_equal(pooled[:cut], ones[:cut])
        assert not np.array_equal(pooled[cut:], ones[cut:])
        tens = mem.encode_value("Ada's phone number is +1-415-555-2070", "+1-415-555-2070")
        assert np.array_equal(pooled[: sizes[0] + sizes[1]], tens[: sizes[0] + sizes[1]])
        assert not np.array_equal(
            pooled[sizes[0] + sizes[1] : cut],
            tens[sizes[0] + sizes[1] : cut],
        )
    finally:
        mem.close()


def test_parse_date_is_the_generator_spelling():
    from tidal.llm_memory_data import parse_date

    assert parse_date("March 4, 1991")[:3] == (2, 4, 1991)
    assert parse_date("September 27, 2019")[:3] == (8, 27, 2019)
    assert parse_date("May 1, 1970")[:3] == (4, 1, 1970)
    for text in ("March 28, 1991", "March 04, 1991", "1991-03-04", "March 4 1991", "3 March 1991", "Oslo"):
        assert parse_date(text) is None


def test_date_block_sizes_match_the_measured_split():
    from tidal.llm_memory import date_block_sizes

    assert date_block_sizes(576) == [88, 44, 88, 89, 89, 89, 89]
    assert sum(date_block_sizes(260)) == 260
    assert sum(date_block_sizes(32)) == 32
    assert min(date_block_sizes(32)) >= 1


def test_date_pool_slices_month_day_and_year_tokens():
    from tidal.llm_memory import assemble_date_value, date_block_sizes

    mem = _byte_memory()
    try:
        sizes = date_block_sizes(mem.hidden)
        single = "March 4, 1991"
        pooled = mem.encode_value(f"Ada's birthday is {single}", single)
        emb = mem.model.embed.weight.detach().numpy()
        month = emb[[ord(ch) for ch in "March"]].mean(axis=0)
        parts = [month[: sizes[0]], np.zeros(sizes[1], dtype=np.float32), emb[ord("4")][: sizes[2]]]
        for i, ch in enumerate("1991"):
            parts.append(emb[ord(ch)][: sizes[3 + i]])
        assert np.allclose(pooled, np.concatenate(parts))
        span = emb[[ord(ch) for ch in single]].mean(axis=0)
        assert not np.allclose(pooled, span)

        double = "March 14, 1991"
        pooled = mem.encode_value(f"Ada's birthday is {double}", double)
        parts = [
            month[: sizes[0]],
            emb[ord("1")][: sizes[1]],
            emb[ord("4")][: sizes[2]],
        ]
        for i, ch in enumerate("1991"):
            parts.append(emb[ord(ch)][: sizes[3 + i]])
        assert np.allclose(pooled, np.concatenate(parts))

        hidden = mem.hidden
        month_v = torch.arange(hidden, dtype=torch.float32)
        day_v = torch.arange(hidden, dtype=torch.float32) + 100
        year_vs = [torch.arange(hidden, dtype=torch.float32) + 200 + 10 * i for i in range(4)]
        packed = assemble_date_value(month_v, [day_v], year_vs)
        assert int(packed.numel()) == hidden
        assert torch.equal(packed[: sizes[0]], month_v[: sizes[0]])
        assert torch.equal(packed[sizes[0] : sizes[0] + sizes[1]], torch.zeros(sizes[1]))
        city = mem.encode_value("Ada lives in Oslo", "Oslo")
        assert not np.allclose(city, pooled)
    finally:
        mem.close()


def test_date_readout_spells_birthday_and_skips_other_relations():
    mem = _byte_memory()
    try:
        mem.inject.date_on.fill_(1.0)
        mem.inject.date_sd.fill_(1.0)
        mem.inject.phone_on.fill_(1.0)
        mem.inject.phone_sd.fill_(1.0)
        # March is index 2. Day 04, year 1991.
        mem.inject.date_month_w[2, 0] = 1.0
        for pos, digit in enumerate((0, 4, 1, 9, 9, 1)):
            mem.inject.date_digit_w[pos, digit, 0] = 1.0
        for pos, digit in enumerate((2, 4, 9, 9)):
            mem.inject.phone_w[pos, digit, 0] = 1.0
        value = np.zeros(mem.hidden, dtype=np.float32)
        value[0] = 1.0
        key = np.ones(mem.hidden, dtype=np.float32)
        assert mem.store.write(key, value, name="ada", rel="birthday")
        assert mem.generate("When is Ada's birthday?") == "March 4, 1991"
        assert len(mem.continuation_ids("When is Ada's birthday?", max_new_tokens=4)) == 4
        mem.store.clear()
        assert mem.store.write(key, value, name="ada", rel="phone")
        assert mem.generate("What is Ada's phone number?") == "+1-415-555-2499"
        mem.store.clear()
        assert mem.store.write(key, value, name="ada", rel="city")
        assert mem.scripted_continuation("Which city does Ada live in?") is None
        mem.inject.date_on.fill_(0.0)
        mem.store.clear()
        assert mem.store.write(key, value, name="ada", rel="birthday")
        assert mem.scripted_continuation("When is Ada's birthday?") is None
    finally:
        mem.close()


def test_load_inject_without_date_keys_leaves_the_readout_off(tmp_path):
    mem = _byte_memory()
    try:
        with torch.no_grad():
            mem.inject.net[-1].bias.fill_(0.5)
        state = {
            key: tensor.detach().cpu().clone()
            for key, tensor in mem.inject.state_dict().items()
            if not key.startswith("date_")
        }
        path = tmp_path / "mlp.pt"
        torch.save({"inject": state, "layer": mem.layer, "hidden": mem.hidden}, path)
        fresh = _byte_memory()
        try:
            assert float(fresh.inject.date_on) == 0.0
            fresh.load_inject(path)
            assert float(fresh.inject.date_on) == 0.0
            assert torch.equal(fresh.inject.net[-1].bias.detach().cpu(), mem.inject.net[-1].bias.detach().cpu())
            assert fresh.inject.date_fields(torch.zeros(fresh.hidden)) is None
        finally:
            fresh.close()
    finally:
        mem.close()


def test_phone_readout_spells_digits_and_skips_other_relations():
    mem = _byte_memory()
    try:
        assert mem.scripted_continuation("Ask:", 8) is None
        mem.inject.phone_on.fill_(1.0)
        mem.inject.phone_sd.fill_(1.0)
        for pos, digit in enumerate((2, 4, 9, 9)):
            mem.inject.phone_w[pos, digit, 0] = 1.0
        value = np.zeros(mem.hidden, dtype=np.float32)
        value[0] = 1.0
        key = np.ones(mem.hidden, dtype=np.float32)
        assert mem.store.write(key, value, name="ada", rel="phone")
        assert mem.tokenizer.decode(mem.continuation_ids("What is Ada's phone number?")) == "+1-415-555-2499"
        assert mem.generate("What is Ada's phone number?") == "+1-415-555-2499"
        assert len(mem.continuation_ids("What is Ada's phone number?", max_new_tokens=4)) == 4
        mem.store.clear()
        assert mem.store.write(key, value, name="ada", rel="city")
        assert mem.scripted_continuation("Which city does Ada live in?") is None
    finally:
        mem.close()


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
