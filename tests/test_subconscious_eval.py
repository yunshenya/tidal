"""Subconscious bank and scorer. No SmolLM download."""
import ast
import inspect
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tidal.llm_memory import LLMMemory
from tidal.llm_memory_data import _CHAT, _CITIES, _COLORS, _JOBS, _TEAMS, names
from tidal.subconscious_data import (
    AVOID,
    PHONE_FACT,
    PREFER,
    bank_sha256,
    choice_words,
    completion_prompt,
    decision_items,
    distractors,
    incidental_facts,
    word_hit,
)
from tidal.subconscious_eval import (
    _count_path,
    phone_takeover,
    render_report,
    run_eval,
    score_pair,
)

ROOT = Path(__file__).resolve().parent.parent


class _Out:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class EyeLM(nn.Module):
    """Identity LM. Hidden rows Y and Z are scaled so a residual selects them."""

    def __init__(self, n_layers=4):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, vocab)
        self.layers = nn.ModuleList([nn.Linear(vocab, vocab, bias=False) for _ in range(n_layers)])
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


def _memory():
    from tidal.llm_memory import ByteTokenizer

    return LLMMemory(EyeLM(), ByteTokenizer(), overwrite_cos=0.9)


def _names():
    from tidal.llm_memory import _CALIB_NAMES

    taken = set(names("train")) | set(names("test")) | set(_CALIB_NAMES)
    return taken


def test_modules_do_not_import_transformers():
    import tidal.subconscious_data as data
    import tidal.subconscious_eval as ev

    for mod in (data, ev):
        tree = ast.parse(inspect.getsource(mod))
        bad = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                bad.extend(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                bad.append(node.module.split(".")[0])
        assert "transformers" not in bad


def test_bank_is_disjoint_and_stable():
    taken = _names()
    ours = {item["name"] for item in decision_items()}
    ours.add(PHONE_FACT["name"])
    incidental = incidental_facts()
    ours |= {row["name"] for row in incidental}
    assert ours.isdisjoint(taken)
    assert len(PREFER) == 8 and len(AVOID) == 8
    assert len(incidental) == 256
    assert len({(row["name"], row["rel"]) for row in incidental}) == 256
    words = choice_words()
    banned_answers = set(_CITIES) | set(_COLORS) | set(_JOBS) | set(_TEAMS) | set(_CHAT)
    for row in incidental:
        assert row["answer"] not in words
        assert row["answer"] not in banned_answers
        assert len(row["answer"]) >= 6
    for item in decision_items():
        assert not word_hit(item["prompt"], item["good"].strip())
        assert not word_hit(item["prompt"], item["bad"].strip())
        assert item["answer"] in item["text"]
        assert not word_hit(completion_prompt(item["text"], item["answer"]), item["answer"])
        if "update_answer" in item:
            assert item["update_answer"] in item["update_text"]
            assert not word_hit(item["prompt"], item["update_answer"])
    nested = distractors(0, 7, incidental)
    assert nested == distractors(0, 63, incidental)[:7]
    assert bank_sha256() == bank_sha256()
    prereg = (ROOT / "reports" / "subconscious_eval_prereg.md").read_text(encoding="utf-8")
    assert bank_sha256() in prereg
    assert "0.05" in prereg
    assert "cce20b6b" in prereg


def test_choice_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        return full_ids[len(prompt_ids) :]

    for item in decision_items():
        assert len(suffix(item["prompt"], item["good"])) == 1
        assert len(suffix(item["prompt"], item["bad"])) == 1
        prefix = completion_prompt(item["text"], item["answer"])
        assert len(suffix(prefix, " " + item["answer"])) == 1
        assert len(suffix(prefix, " " + item["other"])) == 1
        if "update_good" in item:
            assert len(suffix(item["prompt"], item["update_good"])) == 1
            assert len(suffix(item["prompt"], item["update_bad"])) == 1


def test_residual_raises_the_stored_byte():
    mem = _memory()
    prompt = "Go:"
    base = score_pair(mem, prompt, "Y", "Q")
    assert mem.remember("See Y.", answer="Y", name="Mara", rel="mark")
    armed = score_pair(mem, prompt, "Y", "Q", name="Mara", rel="mark")
    assert armed["retrieval"]["hit"] == 1
    assert armed["good_mean"] > base["good_mean"]
    assert armed["margin"] > base["margin"]
    mem.close()


def test_readout_queries_twice_unless_the_slot_is_spelled():
    mem = _memory()
    mem.inject.phone_on.fill_(1.0)
    assert mem.remember("Mara's favorite snack is apples.", answer="apples", name="Mara", rel="snack")
    snack = _count_path(mem, "Pick one snack for Mara.\nAnswer:", 4)
    assert snack["queries"] == 2
    assert snack["forwards"] == snack["queries"] + snack["new_tokens"]
    mem.store.clear()
    assert mem.remember(PHONE_FACT["text"], answer=PHONE_FACT["answer"], name="Quill", rel="phone")
    spelled = mem.continuation_ids("What should we cook tonight?\nAnswer:", max_new_tokens=32)
    assert phone_takeover(mem.tokenizer.decode(spelled))
    phone = _count_path(mem, "What should we cook tonight?\nAnswer:", 4)
    assert phone["queries"] == 1
    assert phone["forwards"] == 1
    mem.close()


def test_eye_eval_runs_and_renders():
    mem = _memory()

    def first(texts, _query):
        return texts[0]

    result = run_eval(mem, first, cost_repeats=1, cost_warmup=0, progress=None)
    assert result["bank_sha256"] == bank_sha256()
    assert result["phone_on"] is False
    assert result["axis1"]["prefer"]["summary"]["inject"]["n"] == 8
    assert result["chat"]["pass"] is True
    assert result["repeat"]["pass"] is True
    assert result["cost"]["path_counts"]["empty"]["queries"] == 0
    assert result["cost"]["path_counts"]["inject"]["queries"] == 1
    text = render_report(result)
    assert result["bank_sha256"] in text
    assert "## Axis 1" in text
    assert "## Recommendation" in text
    mem.close()
