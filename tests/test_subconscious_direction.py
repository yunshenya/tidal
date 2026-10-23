"""Prefer/avoid polarity banks. Locked before the held-out bank is scored."""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from tidal.llm_memory import DEFAULT_GAP_TAU, POLARITY_SPAN, completing_relation
from tidal.llm_memory_data import names
from tidal.question_heldout_data import _tokens
from tidal.subconscious_data import PHONE_FACT, bank_payload, decision_items, incidental_facts
from tidal.subconscious_direction_data import (
    DIRECTION_GATES,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    bank_sha256,
    fit_bank,
    held_bank,
    train_incidental,
    train_rows,
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
# Baseline MLP last-layer weight. Polarity may change the file hash; this tensor may not.
BASELINE_NET_SHA = "f8d4681c8ca9be922523e0e8064131060209a4d0c25e15956fb5ce97e4834143"


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


def _bank_surface(bank):
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


def _train_surface():
    out = []
    _walk(train_rows(), out)
    _walk(train_incidental(), out)
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


def test_direction_data_does_not_import_torch():
    import tidal.subconscious_direction_data as data

    tree = ast.parse(inspect.getsource(data))
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "torch" not in bad
    assert "transformers" not in bad


def test_direction_banks_are_disjoint():
    prior = json.loads((ROOT / "tests" / "fixtures" / "subconscious_prior_surfaces.json").read_text(encoding="utf-8"))
    assert FIT39_SHA in prior["sources"][2]
    assert HELD39_SHA in prior["sources"][3]
    from tidal.subconscious_data import bank_sha256 as frozen_sha
    from tidal.subconscious_gate_data import bank_sha256 as gate_sha

    assert frozen_sha() == FROZEN_SHA
    assert gate_sha() == GATE_SHA
    groups = {
        "train": _train_surface(),
        "fit": _bank_surface(fit_bank()),
        "held": _bank_surface(held_bank()),
    }
    others = {
        "frozen": _frozen_surface(),
        "pr39": prior["strings"],
        "gate": _bank_surface(__import__("tidal.subconscious_gate_data", fromlist=["held_bank"]).held_bank()),
    }
    for name, rows in groups.items():
        for other_name, other in others.items():
            literal = [text for text in rows if text in set(other)]
            assert literal == [], (name, other_name, literal[:3])
            best, pair = _max_jaccard(rows, other)
            assert best < 0.50, (name, other_name, best, pair)
    for left, right in (("train", "fit"), ("train", "held"), ("fit", "held")):
        literal = [text for text in groups[left] if text in set(groups[right])]
        assert literal == [], (left, right, literal[:3])
        best, pair = _max_jaccard(groups[left], groups[right])
        assert best < 0.50, (left, right, best, pair)
    assert len(fit_bank().incidental_facts()) == 256
    assert len(held_bank().incidental_facts()) == 256
    assert len(train_rows()) == 64


def test_answer_words_and_swaps_stay_apart():
    def words(rows):
        out = set()
        for row in rows:
            out.add(row["answer"])
            out.add(row.get("other"))
            if "update_answer" in row:
                out.add(row["update_answer"])
        return out

    train = train_rows()
    prefer = {row["answer"] for row in train if row["family"] == "prefer"}
    avoid = {row["answer"] for row in train if row["family"] == "avoid"}
    assert prefer == avoid
    assert words(fit_bank().decision_items()).isdisjoint(prefer)
    assert words(held_bank().decision_items()).isdisjoint(prefer)
    assert words(fit_bank().decision_items()).isdisjoint(words(held_bank().decision_items()))


def test_names_stay_apart_from_the_memory_splits():
    from tidal.llm_memory import _CALIB_NAMES

    taken = set(names("train")) | set(names("test")) | set(_CALIB_NAMES)
    frozen = {item["name"] for item in decision_items()}
    frozen.add(PHONE_FACT["name"])
    frozen |= {row["name"] for row in incidental_facts()}
    ours = set()
    for bank in (fit_bank(), held_bank()):
        ours |= {item["name"] for item in bank.decision_items()}
        ours.add(bank.PHONE_FACT["name"])
        ours |= {row["name"] for row in bank.incidental_facts()}
    ours |= {row["name"] for row in train_rows()}
    ours |= {row["name"] for row in train_incidental()}
    assert ours.isdisjoint(taken)
    assert ours.isdisjoint(frozen)


def test_phone_and_birthday_questions_match_the_relation_gate():
    for bank in (fit_bank(), held_bank()):
        assert completing_relation(bank.PHONE_QUESTION) == "phone"
        assert completing_relation(bank.BIRTHDAY_QUESTION) == "birthday"
        for prompt in bank.NEUTRAL_GEN:
            assert completing_relation(prompt) is None
    assert DEFAULT_GAP_TAU == 0.80
    assert POLARITY_SPAN == 2.0
    assert DIRECTION_GATES["prefer_delta_pos_min"] == 6
    assert DIRECTION_GATES["avoid_delta_pos_min"] == 6
    assert DIRECTION_GATES["forbidden_down_min"] == 6
    assert DIRECTION_GATES["choice_flips_max"] == 0
    assert DIRECTION_GATES["leak_max"] == 0


def test_decision_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        return full_ids[len(prompt_ids) :]

    for row in train_rows():
        assert len(suffix(row["q"], row["good"])) == 1
        assert len(suffix(row["q"], row["bad"])) == 1
    for bank in (fit_bank(), held_bank()):
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


def test_train_script_does_not_score_held_or_frozen():
    source = (ROOT / "scripts" / "train_subconscious_direction.py").read_text(encoding="utf-8")
    assert "run_eval" not in source
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert "tidal.subconscious_data" not in modules
    assert "tidal.subconscious_gate_data" not in modules


def test_prereg_locks_the_banks_before_scores():
    text = (ROOT / "reports" / "subconscious_direction_prereg.md").read_text(encoding="utf-8")
    assert "not been scored" in text
    assert bank_sha256("train") in text
    assert bank_sha256("fit") in text
    assert bank_sha256("held") in text
    assert FROZEN_SHA in text
    assert FIT39_SHA in text
    assert HELD39_SHA in text
    assert GATE_SHA in text
    assert "0.80" in text


def test_shipped_mlp_last_layer_stays_the_baseline():
    ckpt = __import__("torch").load(
        ROOT / "tidal" / "llm_inject_smollm2_135m.pt",
        map_location="cpu",
        weights_only=False,
    )
    weight = ckpt["inject"]["net.2.weight"].contiguous().numpy().tobytes()
    assert hashlib.sha256(weight).hexdigest() == BASELINE_NET_SHA
    assert tuple(ckpt["inject"]["polarity"].shape) == (576,)
    source = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "key_proj" not in source
