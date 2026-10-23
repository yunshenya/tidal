"""Direction fix: disjoint banks, confidence, and a key-conditioned decision loss."""
import ast
import inspect

import pytest

from tidal.llm_memory import completing_relation, retrieval_confidence
from tidal.llm_memory_data import names
from tidal.question_heldout_data import _tokens
from tidal.subconscious_data import (
    PHONE_FACT,
    bank_payload,
    decision_items,
    incidental_facts,
)
from tidal.subconscious_fix_data import (
    dev_rows,
    fit_bank,
    held_bank,
    train_rows,
)

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
    _walk(bank.INCIDENTAL, out)
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


def _overlap(own, other):
    own_t = [_tokens(text) for text in own]
    other_t = [_tokens(text) for text in other]
    other_set = set(other)
    literal = sum(1 for text in own if text in other_set)
    best = 0.0
    pair = ("", "")
    for src, src_t in zip(own, own_t):
        for dst, dst_t in zip(other, other_t):
            score = _jaccard(src_t, dst_t)
            if score > best:
                best = score
                pair = (src, dst)
    return literal, best, pair


def _decision_words(rows):
    words = set()
    for row in rows:
        for key in ("answer", "other", "update_answer", "a"):
            if row.get(key):
                words.add(str(row[key]).lower())
        for key in ("good", "bad", "update_good", "update_bad"):
            if isinstance(row.get(key), str):
                words.add(row[key].strip().lower())
    return words


def test_prereg_locks_the_banks_before_scores():
    from pathlib import Path

    text = (Path(__file__).resolve().parent.parent / "reports" / "subconscious_fix_prereg.md").read_text(encoding="utf-8")
    assert fit_bank().sha256() in text
    assert held_bank().sha256() in text
    assert "e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66" in text
    assert "0.80" in text
    assert "not been scored" in text


def test_fix_data_does_not_import_transformers():
    import tidal.subconscious_fix_data as data

    tree = ast.parse(inspect.getsource(data))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "transformers" not in bad
    assert "torch" not in bad


def test_fit_and_held_banks_are_disjoint():
    frozen = _frozen_surface()
    fit = fit_bank()
    held = held_bank()
    fit_s = _surface(fit)
    held_s = _surface(held)
    for name, own, other in (
        ("fit", fit_s, frozen),
        ("held", held_s, frozen),
        ("held-fit", held_s, fit_s),
    ):
        literal, best, pair = _overlap(own, other)
        assert literal == 0, (name, pair)
        assert best < 0.50, (name, best, pair)
    assert fit.sha256() == fit_bank().sha256()
    assert held.sha256() == held_bank().sha256()
    assert len(fit.INCIDENTAL) == 256 and len(held.INCIDENTAL) == 256
    assert len(fit.CHATS) == 32 and len({row[0] for row in fit.CHATS}) == 32
    assert len(held.CHATS) == 32 and len({row[0] for row in held.CHATS}) == 32
    assert completing_relation(fit.PHONE_QUESTION) == "phone"
    assert completing_relation(fit.BIRTHDAY_QUESTION) == "birthday"
    assert completing_relation(held.PHONE_QUESTION) == "phone"
    assert completing_relation(held.BIRTHDAY_QUESTION) == "birthday"


def test_decision_words_and_names_stay_apart():
    from tidal.llm_memory import _CALIB_NAMES

    frozen_words = _decision_words(decision_items())
    groups = {
        "train": _decision_words(train_rows()),
        "dev": _decision_words(dev_rows()),
        "fit": _decision_words(list(fit_bank().PREFER) + list(fit_bank().AVOID)),
        "held": _decision_words(list(held_bank().PREFER) + list(held_bank().AVOID)),
        "frozen": frozen_words,
    }
    keys = list(groups)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            assert groups[left].isdisjoint(groups[right]), (left, right, groups[left] & groups[right])

    taken = set(names("train")) | set(names("test")) | set(_CALIB_NAMES)
    frozen_names = {item["name"] for item in decision_items()}
    frozen_names.add(PHONE_FACT["name"])
    frozen_names |= {row["name"] for row in incidental_facts()}

    def people(bank):
        found = {item["name"] for item in bank.decision_items()}
        found.add(bank.PHONE_FACT["name"])
        found |= {row["name"] for row in bank.INCIDENTAL}
        return found

    fit_names = people(fit_bank())
    held_names = people(held_bank())
    train_names = {row["name"] for row in train_rows()}
    dev_names = {row["name"] for row in dev_rows()}
    assert fit_names.isdisjoint(held_names)
    assert fit_names.isdisjoint(frozen_names | taken | train_names | dev_names)
    assert held_names.isdisjoint(frozen_names | taken | train_names | dev_names)
    assert train_names.isdisjoint(dev_names | taken | frozen_names)
    by_answer = {}
    for row in train_rows():
        by_answer.setdefault(row["a"], set()).add(row["family"])
    assert all(families == {"prefer", "avoid"} for families in by_answer.values())


def test_completing_relation_is_a_word():
    assert completing_relation("What is Ada's phone number?") == "phone"
    assert completing_relation("When is Ada's birthday?") == "birthday"
    assert completing_relation("The telephone rang.") is None
    assert completing_relation("What should we cook tonight?") is None
    assert completing_relation("") is None


def test_retrieval_confidence_scales_the_gap():
    assert retrieval_confidence(1, 0.0, 0.05) == 1.0
    assert retrieval_confidence(0, None, 0.05) == 1.0
    assert retrieval_confidence(2, 0.0, 0.05) == 0.0
    assert retrieval_confidence(4, 0.025, 0.05) == pytest.approx(0.5)
    assert retrieval_confidence(4, 0.2, 0.05) == 1.0
    with pytest.raises(ValueError):
        retrieval_confidence(3, 0.1, 0.0)


def test_decision_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    from tidal.subconscious_data import completion_prompt

    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def one(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        assert len(full_ids) - len(prompt_ids) == 1

    for row in train_rows() + dev_rows():
        one(row["q"], row["good"])
        one(row["q"], row["bad"])
    for bank in (fit_bank(), held_bank()):
        for item in bank.decision_items():
            one(item["prompt"], item["good"])
            one(item["prompt"], item["bad"])
            prefix = completion_prompt(item["text"], item["answer"])
            one(prefix, " " + item["answer"])
            one(prefix, " " + item["other"])
            if "update_good" in item:
                one(item["prompt"], item["update_good"])
                one(item["prompt"], item["update_bad"])
        for row in bank.NEUTRAL_NLL:
            one(row["prompt"], row["continuation"])
        for row in bank.NEUTRAL_CHOICE:
            one(row["prompt"], row["good"])
            one(row["prompt"], row["bad"])


def _eye_memory():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("test_subconscious_eval.py")
    spec = importlib.util.spec_from_file_location("eye_subconscious_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._memory()


def test_same_value_opposite_key_moves_apart():
    """A constraint and a preference that share v must not share a direction."""
    import numpy as np

    from tidal.llm_memory_train import fit_decision
    from tidal.subconscious_eval import score_pair

    mem = _eye_memory()
    try:
        like = "likes Y."
        avoid = "avoids Y."
        v_like = mem.encode_value(like, "Y")
        v_avoid = mem.encode_value(avoid, "Y")
        assert np.allclose(v_like, v_avoid)
        assert not np.allclose(mem.encode_key(like, "Y"), mem.encode_key(avoid, "Y"))
        facts = [
            {"text": like, "a": "Y", "q": "Go:", "good": "Y", "bad": "Z", "name": "a", "rel": "like"},
            {"text": avoid, "a": "Y", "q": "Go:", "good": "Z", "bad": "Y", "name": "b", "rel": "avoid"},
        ]
        fit_decision(mem, facts, steps=40, bs=2, lr=1e-2, margin=1.0, seed=0, weight_decay=0.0)

        def margin(text, name, rel):
            mem.store.clear()
            assert mem.remember(text, answer="Y", name=name, rel=rel)
            return score_pair(mem, "Go:", "Y", "Z", name=name, rel=rel)["margin"]

        like_m = margin(like, "a", "like")
        avoid_m = margin(avoid, "b", "avoid")
        assert like_m > avoid_m + 0.2
    finally:
        mem.close()
