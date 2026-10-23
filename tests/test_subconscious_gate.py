"""Gap scale and spelling gate. The held-out bank is locked before it is scored."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tidal.llm_memory import DEFAULT_GAP_TAU, completing_relation, retrieval_confidence
from tidal.llm_memory_data import names
from tidal.question_heldout_data import _tokens
from tidal.subconscious_data import PHONE_FACT, bank_payload, decision_items, incidental_facts
from tidal.subconscious_gate_data import held_bank

ROOT = Path(__file__).resolve().parent.parent
_FIELDS = {
    "text",
    "prompt",
    "q",
    "good",
    "bad",
    "continuation",
    "answer",
    "other",
    "update_text",
    "update_good",
    "update_bad",
    "update_answer",
    "a",
}


def _walk(obj, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _FIELDS and isinstance(value, str):
                out.append(value)
            else:
                _walk(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, str):
                out.append(value)
            else:
                _walk(value, out)


def _surface(bank):
    out = []
    _walk(list(bank.PREFER) + list(bank.AVOID), out)
    _walk(bank.incidental_facts(), out)
    _walk(list(bank.CHATS), out)
    _walk(list(bank.NEUTRAL_NLL), out)
    _walk(list(bank.NEUTRAL_CHOICE), out)
    _walk(list(bank.NEUTRAL_GEN), out)
    _walk(bank.PHONE_FACT, out)
    _walk(bank.BIRTHDAY_FACT, out)
    out.append(bank.PHONE_QUESTION)
    out.append(bank.BIRTHDAY_QUESTION)
    return out


def _frozen_surface():
    payload = bank_payload()
    out = []
    for key in (
        "prefer",
        "avoid",
        "incidental",
        "chats",
        "neutral_nll",
        "neutral_choice",
        "neutral_gen",
        "phone",
        "birthday",
    ):
        _walk(payload[key], out)
    out.append(payload["phone_question"])
    return out


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_prereg_locks_the_bank_before_scores():
    text = (ROOT / "reports" / "subconscious_gate_prereg.md").read_text(encoding="utf-8")
    assert held_bank().sha256() in text
    assert "e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66" in text
    assert "20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce" in text
    assert "0.80" in text
    assert "not been scored" in text


def test_gate_data_does_not_import_torch():
    import tidal.subconscious_gate_data as data

    tree = ast.parse(inspect.getsource(data))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "transformers" not in bad
    assert "torch" not in bad


def test_inject_file_is_the_baseline_checkpoint():
    digest = hashlib.sha256((ROOT / "tidal" / "llm_inject_smollm2_135m.pt").read_bytes()).hexdigest()
    assert digest == "cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539"
    source = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "key_proj" not in source
    assert DEFAULT_GAP_TAU == 0.80


def test_held_bank_is_disjoint_from_prior_banks():
    own = _surface(held_bank())
    prior = json.loads((ROOT / "tests" / "fixtures" / "subconscious_prior_surfaces.json").read_text(encoding="utf-8"))
    others = {
        "frozen": _frozen_surface(),
        "pr39": prior["strings"],
    }
    own_t = [(text, _tokens(text)) for text in own]
    for name, other in others.items():
        other_set = set(other)
        literal = [text for text in own if text in other_set]
        assert literal == [], (name, literal[:3])
        best = 0.0
        pair = ("", "")
        for src, src_t in own_t:
            for dst in other:
                score = _jaccard(src_t, _tokens(dst))
                if score > best:
                    best = score
                    pair = (src, dst)
        assert best < 0.50, (name, best, pair)
    bank = held_bank()
    assert bank.sha256() == held_bank().sha256()
    assert len(bank.incidental_facts()) == 256
    assert completing_relation(bank.PHONE_QUESTION) == "phone"
    assert completing_relation(bank.BIRTHDAY_QUESTION) == "birthday"
    for prompt in bank.NEUTRAL_GEN:
        assert completing_relation(prompt) is None


def test_names_stay_apart_from_the_memory_splits():
    from tidal.llm_memory import _CALIB_NAMES

    taken = set(names("train")) | set(names("test")) | set(_CALIB_NAMES)
    frozen = {item["name"] for item in decision_items()}
    frozen.add(PHONE_FACT["name"])
    frozen |= {row["name"] for row in incidental_facts()}
    bank = held_bank()
    ours = {item["name"] for item in bank.decision_items()}
    ours.add(bank.PHONE_FACT["name"])
    ours |= {row["name"] for row in bank.incidental_facts()}
    assert ours.isdisjoint(taken)
    assert ours.isdisjoint(frozen)


def test_completing_relation_and_confidence():
    assert completing_relation("What is Ada's phone number?") == "phone"
    assert completing_relation("When is Ada's birthday?") == "birthday"
    assert completing_relation("The telephone rang.") is None
    assert completing_relation("What should we cook tonight?") is None
    assert retrieval_confidence(1, 0.0, 0.80) == 1.0
    assert retrieval_confidence(2, 0.0, 0.80) == 0.0
    assert retrieval_confidence(2, 0.40, 0.80) == pytest.approx(0.5)
    assert retrieval_confidence(3, 2.0, 0.80) == 1.0
    with pytest.raises(ValueError):
        retrieval_confidence(2, 0.1, 0.0)


def test_decision_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    bank = held_bank()

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        return full_ids[len(prompt_ids) :]

    for item in bank.decision_items():
        assert len(suffix(item["prompt"], item["good"])) == 1
        assert len(suffix(item["prompt"], item["bad"])) == 1
        if "update_good" in item:
            assert len(suffix(item["prompt"], item["update_good"])) == 1
            assert len(suffix(item["prompt"], item["update_bad"])) == 1
    for item in bank.NEUTRAL_NLL:
        assert len(suffix(item["prompt"], item["continuation"])) == 1
    for item in bank.NEUTRAL_CHOICE:
        assert len(suffix(item["prompt"], item["good"])) == 1
        assert len(suffix(item["prompt"], item["bad"])) == 1
