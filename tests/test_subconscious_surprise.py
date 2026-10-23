"""Surprise-gated write banks. Locked before the held-out bank is scored."""
import ast
import inspect
from pathlib import Path

import pytest

from tidal.llm_memory import DEFAULT_GAP_TAU, DEFAULT_SURPRISE_TAU, POLARITY_SPAN, completing_relation
from tidal.llm_memory_data import names, parse_date
from tidal.question_heldout_data import _tokens
from tidal.subconscious_data import PHONE_FACT, bank_payload, decision_items, incidental_facts
from tidal.subconscious_direction_data import (
    bank_sha256 as direction_sha256,
    fit_bank as direction_fit_bank,
    held_bank as direction_held_bank,
    train_incidental,
    train_rows,
)
from tidal.subconscious_surprise_data import (
    DIRECTION_FIT_SHA,
    DIRECTION_HELD_SHA,
    DIRECTION_TRAIN_SHA,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    SURPRISE_GATES,
    bank_sha256,
    fit_bank,
    held_bank,
)

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


def _direction_surface(bank):
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


def _surprise_surface(bank):
    out = []
    _walk(list(bank.LOW) + list(bank.HIGH) + list(bank.PREFER) + list(bank.AVOID), out)
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


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _max_jaccard(left, right):
    best = 0.0
    pair = ("", "")
    right_t = [(text, _tokens(text)) for text in right]
    for src in left:
        src_t = _tokens(src)
        for dst, dst_t in right_t:
            score = _jaccard(src_t, dst_t)
            if score > best:
                best = score
                pair = (src, dst)
    return best, pair


def test_surprise_data_does_not_import_torch():
    import tidal.subconscious_surprise_data as data

    tree = ast.parse(inspect.getsource(data))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "torch" not in bad
    assert "transformers" not in bad


def test_surprise_banks_are_disjoint():
    import json

    prior = json.loads((ROOT / "tests" / "fixtures" / "subconscious_prior_surfaces.json").read_text(encoding="utf-8"))
    from tidal.subconscious_data import bank_sha256 as frozen_sha
    from tidal.subconscious_gate_data import bank_sha256 as gate_sha
    from tidal.subconscious_gate_data import held_bank as gate_held

    assert frozen_sha() == FROZEN_SHA
    assert gate_sha() == GATE_SHA
    assert direction_sha256("train") == DIRECTION_TRAIN_SHA
    assert direction_sha256("fit") == DIRECTION_FIT_SHA
    assert direction_sha256("held") == DIRECTION_HELD_SHA
    assert FIT39_SHA in prior["sources"][2]
    assert HELD39_SHA in prior["sources"][3]
    groups = {"fit": _surprise_surface(fit_bank()), "held": _surprise_surface(held_bank())}
    others = {
        "frozen": _frozen_surface(),
        "pr39": prior["strings"],
        "gate": _direction_surface(gate_held()),
        "dir-train": _walk_rows(train_rows()) + _walk_rows(train_incidental()),
        "dir-fit": _direction_surface(direction_fit_bank()),
        "dir-held": _direction_surface(direction_held_bank()),
    }
    for name, rows in groups.items():
        for other_name, other in others.items():
            literal = [text for text in rows if text in set(other)]
            assert literal == [], (name, other_name, literal[:3])
            best, pair = _max_jaccard(rows, other)
            assert best < 0.50, (name, other_name, best, pair)
    literal = [text for text in groups["fit"] if text in set(groups["held"])]
    assert literal == []
    best, pair = _max_jaccard(groups["fit"], groups["held"])
    assert best < 0.50, (best, pair)
    assert len(fit_bank().incidental_facts()) == 32
    assert len(held_bank().incidental_facts()) == 32
    assert len(fit_bank().LOW) == 8 and len(fit_bank().HIGH) == 8
    assert len(held_bank().LOW) == 8 and len(held_bank().HIGH) == 8


def _walk_rows(rows):
    out = []
    _walk(rows, out)
    return out


def test_names_stay_apart_from_earlier_banks():
    from tidal.llm_memory import _CALIB_NAMES

    taken = {n.lower() for n in names("train")} | {n.lower() for n in names("test")} | {n.lower() for n in _CALIB_NAMES}
    for item in decision_items():
        taken.add(item["name"].lower())
    taken.add(PHONE_FACT["name"].lower())
    for row in incidental_facts():
        taken.add(row["name"].lower())
    for bank in (direction_fit_bank(), direction_held_bank()):
        for item in bank.decision_items():
            taken.add(item["name"].lower())
        taken.add(bank.PHONE_FACT["name"].lower())
        for row in bank.incidental_facts():
            taken.add(row["name"].lower())
    for row in train_rows():
        taken.add(row["name"].lower())
    for row in train_incidental():
        taken.add(row["name"].lower())
    ours = set()
    for bank in (fit_bank(), held_bank()):
        for item in list(bank.PREFER) + list(bank.AVOID):
            ours.add(item["name"].lower())
        ours.add(bank.PHONE_FACT["name"].lower())
        for row in bank.incidental_facts():
            ours.add(row["name"].lower())
    assert ours.isdisjoint(taken)


def test_phone_and_birthday_questions_match_the_relation_gate():
    for bank in (fit_bank(), held_bank()):
        assert completing_relation(bank.PHONE_QUESTION) == "phone"
        assert completing_relation(bank.BIRTHDAY_QUESTION) == "birthday"
        assert parse_date(bank.BIRTHDAY_FACT["answer"]) is not None
        assert bank.BIRTHDAY_FACT["answer"] in bank.BIRTHDAY_FACT["text"]
        assert "phone" not in " ".join(bank.NEUTRAL_GEN).lower()
        assert "birthday" not in " ".join(bank.NEUTRAL_GEN).lower()
        for prompt in bank.NEUTRAL_GEN:
            assert completing_relation(prompt) is None
    assert DEFAULT_GAP_TAU == 0.80
    assert POLARITY_SPAN == 2.0
    assert DEFAULT_SURPRISE_TAU >= 0.0
    assert SURPRISE_GATES["high_written_min"] == 6
    assert SURPRISE_GATES["low_written_max"] == 2
    assert SURPRISE_GATES["prefer_mean_min"] == 8.0
    assert SURPRISE_GATES["avoid_mean_min"] == 0.6
    assert SURPRISE_GATES["nll_abs_mean_max"] == 0.05
    assert SURPRISE_GATES["nll_bad_count_max"] == 1
    assert SURPRISE_GATES["choice_flips_max"] == 0
    assert SURPRISE_GATES["leak_max"] == 0


def test_scored_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids, (prompt, continuation)
        return full_ids[len(prompt_ids) :]

    for bank in (fit_bank(), held_bank()):
        for item in list(bank.LOW) + list(bank.HIGH) + list(bank.PREFER) + list(bank.AVOID):
            assert len(suffix(item["prompt"], item["good"])) == 1
            assert len(suffix(item["prompt"], item["bad"])) == 1
            assert item["answer"] in item["text"]
        for item in bank.NEUTRAL_NLL:
            assert len(suffix(item["prompt"], item["continuation"])) == 1
        for item in bank.NEUTRAL_CHOICE:
            assert len(suffix(item["prompt"], item["good"])) == 1
            assert len(suffix(item["prompt"], item["bad"])) == 1


def test_fit_script_does_not_score_held_or_frozen():
    source = (ROOT / "scripts" / "fit_subconscious_surprise.py").read_text(encoding="utf-8")
    assert "held_bank" not in source
    assert "run_eval" not in source
    assert "eval_subconscious_surprise" not in source
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert "tidal.subconscious_data" not in modules
    assert "tidal.subconscious_gate_data" not in modules
    assert "tidal.subconscious_direction_data" not in modules


def test_prereg_locks_the_banks_before_scores():
    text = (ROOT / "reports" / "subconscious_surprise_prereg.md").read_text(encoding="utf-8")
    assert "not been scored" in text
    assert bank_sha256("fit") in text
    assert bank_sha256("held") in text
    assert FROZEN_SHA in text
    assert FIT39_SHA in text
    assert HELD39_SHA in text
    assert GATE_SHA in text
    assert DIRECTION_TRAIN_SHA in text
    assert DIRECTION_FIT_SHA in text
    assert DIRECTION_HELD_SHA in text
    assert f"{DEFAULT_SURPRISE_TAU:.2f}" in text
    assert "0.80" in text


def test_shipped_polarity_shape_is_unchanged():
    source = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "key_proj" not in source
    ckpt = __import__("torch").load(
        ROOT / "tidal" / "llm_inject_smollm2_135m.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert tuple(ckpt["inject"]["polarity"].shape) == (576,)
