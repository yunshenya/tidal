"""Relative-surprise banks. Disjoint from every earlier subconscious bank.

The preregistration test reads the file the fit script writes before the
held-out bank is scored. It does not score that bank.
"""
import ast
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch

from tidal.episodic import CLOCK_SPAN
from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    LLMMemory,
    POLARITY_SPAN,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
    SURPRISE_MIN_COUNT,
    SURPRISE_QUANTILE,
    SURPRISE_WINDOW,
    VALENCE_SPAN,
    VALENCE_STEP,
    accept_relative_write,
    build_tiny_lm,
    completing_relation,
    surprise_quantile,
)
from tidal.llm_memory_data import _MONTHS, names, parse_date
from tidal.question_heldout_data import _tokens
from tidal.subconscious_circadian_data import (
    fit_bank as clock_fit_bank,
    held_bank as clock_held_bank,
)
from tidal.subconscious_data import PHONE_FACT, bank_payload, decision_items, incidental_facts
from tidal.subconscious_direction_data import (
    fit_bank as direction_fit_bank,
    held_bank as direction_held_bank,
    train_incidental,
    train_rows,
)
from tidal.subconscious_gate_data import held_bank as gate_held_bank
from tidal.subconscious_replay_data import (
    bank_sha256 as replay_sha,
    fit_bank as replay_fit_bank,
    held_bank as replay_held_bank,
)
from tidal.subconscious_spell_data import (
    bank_sha256 as spell_sha,
    fit_bank as spell_fit_bank,
    held_bank as spell_held_bank,
)
from tidal.subconscious_surprise_rel_data import (
    CIRCADIAN_FIT_SHA,
    CIRCADIAN_HELD_SHA,
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    DIRECTION_TRAIN_SHA,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    PRIOR_VALENCE_FIT_SHA,
    PRIOR_VALENCE_HELD_SHA,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FIT_SHA,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_HELD_SHA,
    REPLAY_PREFER_MEAN_MIN,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_GATES,
    SURPRISE_HELD_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    _PAYLOAD,
    bank_sha256,
    fit_bank,
    held_bank,
    train_bank,
)
from tidal.subconscious_valence_data import (
    bank_sha256 as valence_sha,
    fit_bank as valence_fit_bank,
    held_bank as valence_held_bank,
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
BASELINE_NET_SHA = "f8d4681c8ca9be922523e0e8064131060209a4d0c25e15956fb5ce97e4834143"
PHONE_RE = re.compile(r"\+1-(\d{3})-(\d{3})-(\d{4})")
DATE_RE = re.compile("(" + "|".join(_MONTHS) + r") (\d{1,2}), (\d{4})")


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
    _walk(list(getattr(bank, "CHATS", ())), out)
    _walk(list(bank.NEUTRAL_NLL), out)
    _walk(list(bank.NEUTRAL_CHOICE), out)
    _walk(list(bank.NEUTRAL_GEN), out)
    _walk(bank.PHONE_FACT, out)
    _walk(bank.BIRTHDAY_FACT, out)
    out.append(bank.PHONE_QUESTION)
    out.append(bank.BIRTHDAY_QUESTION)
    return out


def _replay_surface(bank):
    out = _direction_surface(bank)
    _walk(bank.replay_facts(), out)
    return out


def _plain_surface(bank):
    out = []
    _walk(bank.incidental_facts(), out)
    _walk(list(bank.NEUTRAL_NLL), out)
    _walk(list(bank.NEUTRAL_CHOICE), out)
    _walk(list(bank.NEUTRAL_GEN), out)
    _walk(bank.PHONE_FACT, out)
    _walk(bank.BIRTHDAY_FACT, out)
    out.append(bank.PHONE_QUESTION)
    out.append(bank.BIRTHDAY_QUESTION)
    return out


def _valence_surface(bank):
    out = []
    _walk(list(bank.TOWARD) + list(bank.AWAY), out)
    out.extend(_plain_surface(bank))
    return out


def _clock_surface(bank):
    out = []
    _walk(list(bank.PHASE), out)
    out.extend(_plain_surface(bank))
    return out


def _surprise_surface(bank):
    out = []
    _walk(list(bank.LOW) + list(bank.HIGH) + list(bank.KEEP), out)
    _walk(bank.incidental_facts(), out)
    _walk(list(bank.NEUTRAL_NLL), out)
    _walk(list(bank.NEUTRAL_CHOICE), out)
    _walk(list(bank.NEUTRAL_GEN), out)
    if bank.PHONE_FACT:
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


def _load_fixture(name):
    return json.loads((ROOT / "tests" / "fixtures" / name).read_text(encoding="utf-8"))


def _names_of(bank):
    found = {row["name"].lower() for row in bank.incidental_facts()}
    found |= {item["name"].lower() for item in list(bank.LOW) + list(bank.HIGH) + list(bank.KEEP)}
    if bank.PHONE_FACT:
        found.add(bank.PHONE_FACT["name"].lower())
        found.add(bank.BIRTHDAY_FACT["name"].lower())
    return found


def _prior_blobs():
    surprise = _load_fixture("subconscious_surprise_surfaces.json")
    rif = _load_fixture("subconscious_rif_surfaces.json")
    prior = _load_fixture("subconscious_prior_surfaces.json")
    valence = _load_fixture("subconscious_valence_surfaces.json")
    blobs = []
    blobs.extend(_frozen_surface())
    blobs.extend(prior["strings"])
    blobs.extend(_direction_surface(gate_held_bank()))
    blobs.extend(_train_surface())
    blobs.extend(_direction_surface(direction_fit_bank()))
    blobs.extend(_direction_surface(direction_held_bank()))
    blobs.extend(surprise["fit"])
    blobs.extend(surprise["held"])
    blobs.extend(rif["fit"])
    blobs.extend(rif["held"])
    blobs.extend(_replay_surface(replay_fit_bank()))
    blobs.extend(_replay_surface(replay_held_bank()))
    blobs.extend(_plain_surface(spell_fit_bank()))
    blobs.extend(_plain_surface(spell_held_bank()))
    blobs.extend(valence["fit"])
    blobs.extend(valence["held"])
    blobs.extend(_valence_surface(valence_fit_bank()))
    blobs.extend(_valence_surface(valence_held_bank()))
    blobs.extend(_clock_surface(clock_fit_bank()))
    blobs.extend(_clock_surface(clock_held_bank()))
    return blobs


def test_quantile_is_relative_and_warmup_accepts():
    assert surprise_quantile([1.0, 3.0, 5.0, 7.0], 0.0) == 1.0
    assert surprise_quantile([1.0, 3.0, 5.0, 7.0], 1.0) == 7.0
    assert abs(surprise_quantile([1.0, 3.0, 5.0, 7.0], 0.5) - 4.0) < 1e-9
    assert accept_relative_write("chat", 9.0, [1.0] * 32, quantile=0.0, min_count=32) is False
    assert accept_relative_write("fact", 0.1, [1.0] * 10, quantile=0.0, min_count=32) is True
    assert accept_relative_write("fact", 0.1, [1.0] * 32, quantile=0.0, min_count=32) is False
    assert accept_relative_write("fact", 1.0, [1.0] * 32, quantile=0.0, min_count=32) is True
    assert SURPRISE_WINDOW == 32
    assert SURPRISE_MIN_COUNT == 32
    assert SURPRISE_QUANTILE in (0.0, 0.25, 0.5)


def test_forced_window_rejects_without_a_caller_bypass():
    model, tok = build_tiny_lm(seed=0)
    mem = LLMMemory(model, tok)
    calls = {"n": 0}

    def _hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = mem.model.register_forward_hook(_hook)
    try:
        assert mem.remember("code is Y", name="ada", rel="phone", kind="chat", surprise=1e9) is False
        assert calls["n"] == 0
    finally:
        handle.remove()
    assert mem.remember("code is Y", name="ada", rel="phone", surprise=0.0) is True
    assert len(mem.surprise_window()) == 1
    mem._surprise_window = [1e9] * 32
    assert mem.remember("town is Q", name="bea", rel="city", surprise=1e18) is False
    assert len(mem.store) == 1
    assert mem.surprise_window() == tuple([1e9] * 32)
    assert "town is Q" not in json.dumps(list(mem.surprise_window()))
    mem._surprise_window = [-1.0] * 32
    assert mem.remember("team is M", name="cal", rel="team") is True
    assert len(mem.store) == 2
    assert len(mem.surprise_window()) == 32
    mem.store.clear()
    assert mem.remember("hue is R", name="dee", rel="color") is True
    assert len(mem.surprise_window()) == 1
    mem.close()


def test_data_does_not_import_torch():
    source = (ROOT / "tidal" / "subconscious_surprise_rel_data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "torch" not in bad
    assert "transformers" not in bad
    assert set(_PAYLOAD) == {"train", "fit", "held"}


def test_score_reads_the_live_quantile():
    source = (ROOT / "tidal" / "subconscious_surprise_rel_score.py").read_text(encoding="utf-8")
    assert "SURPRISE_QUANTILE" in source
    assert "held_bank" not in source
    assert "phone_exact" in source
    assert "birthday_exact" in source
    assert "surprise_window" in source


def test_banks_are_disjoint_from_every_prior_bank():
    prior = _load_fixture("subconscious_prior_surfaces.json")
    surprise = _load_fixture("subconscious_surprise_surfaces.json")
    rif = _load_fixture("subconscious_rif_surfaces.json")
    valence = _load_fixture("subconscious_valence_surfaces.json")
    assert surprise["fit_sha"] == SURPRISE_FIT_SHA
    assert surprise["held_sha"] == SURPRISE_HELD_SHA
    assert rif["fit_sha"] == RIF_FIT_SHA
    assert rif["held_sha"] == RIF_HELD_SHA
    assert valence["fit_sha"] == PRIOR_VALENCE_FIT_SHA
    assert valence["held_sha"] == PRIOR_VALENCE_HELD_SHA
    assert replay_sha("fit") == REPLAY_FIT_SHA
    assert replay_sha("held") == REPLAY_HELD_SHA
    assert spell_sha("fit") == SPELL_FIT_SHA
    assert spell_sha("held") == SPELL_HELD_SHA
    assert valence_sha("fit") == VALENCE_FIT_SHA
    assert valence_sha("held") == VALENCE_HELD_SHA
    from tidal.subconscious_circadian_data import bank_sha256 as clock_sha
    from tidal.subconscious_data import bank_sha256 as frozen_sha
    from tidal.subconscious_gate_data import bank_sha256 as gate_sha

    assert frozen_sha() == FROZEN_SHA
    assert gate_sha() == GATE_SHA
    assert clock_sha("fit") == CIRCADIAN_FIT_SHA
    assert clock_sha("held") == CIRCADIAN_HELD_SHA
    assert FIT39_SHA == "20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce"
    assert HELD39_SHA == "ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a"
    groups = {
        "train": _surprise_surface(train_bank()),
        "fit": _surprise_surface(fit_bank()),
        "held": _surprise_surface(held_bank()),
    }
    others = {
        "frozen": _frozen_surface(),
        "pr39": prior["strings"],
        "gate": _direction_surface(gate_held_bank()),
        "direction_train": _train_surface(),
        "direction_fit": _direction_surface(direction_fit_bank()),
        "direction_held": _direction_surface(direction_held_bank()),
        "surprise_fit": surprise["fit"],
        "surprise_held": surprise["held"],
        "rif_fit": rif["fit"],
        "rif_held": rif["held"],
        "replay_fit": _replay_surface(replay_fit_bank()),
        "replay_held": _replay_surface(replay_held_bank()),
        "spell_fit": _plain_surface(spell_fit_bank()),
        "spell_held": _plain_surface(spell_held_bank()),
        "prior_valence_fit": valence["fit"],
        "prior_valence_held": valence["held"],
        "valence_fit": _valence_surface(valence_fit_bank()),
        "valence_held": _valence_surface(valence_held_bank()),
        "circadian_fit": _clock_surface(clock_fit_bank()),
        "circadian_held": _clock_surface(clock_held_bank()),
    }
    for name, rows in groups.items():
        for other_name, other in others.items():
            literal = [text for text in rows if text in set(other)]
            assert literal == [], (name, other_name, literal[:3])
            best, pair = _max_jaccard(rows, other)
            assert best < 0.50, (name, other_name, best, pair)
    names = list(groups)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            literal = [text for text in groups[left] if text in set(groups[right])]
            assert literal == [], (left, right, literal[:3])
            best, pair = _max_jaccard(groups[left], groups[right])
            assert best < 0.50, (left, right, best, pair)


def test_names_and_items_stay_apart():
    from tidal.llm_memory import _CALIB_NAMES

    surprise = _load_fixture("subconscious_surprise_surfaces.json")
    rif = _load_fixture("subconscious_rif_surfaces.json")
    valence = _load_fixture("subconscious_valence_surfaces.json")
    taken = {item.lower() for item in names("train")} | {item.lower() for item in names("test")}
    taken |= {item.lower() for item in _CALIB_NAMES}
    taken |= {item["name"].lower() for item in decision_items()}
    taken.add(PHONE_FACT["name"].lower())
    taken |= {row["name"].lower() for row in incidental_facts()}
    for bank in (
        direction_fit_bank(),
        direction_held_bank(),
        gate_held_bank(),
        replay_fit_bank(),
        replay_held_bank(),
        spell_fit_bank(),
        spell_held_bank(),
        valence_fit_bank(),
        valence_held_bank(),
        clock_fit_bank(),
        clock_held_bank(),
    ):
        if hasattr(bank, "decision_items"):
            taken |= {item["name"].lower() for item in bank.decision_items()}
        if hasattr(bank, "PHASE"):
            taken |= {item["name"].lower() for item in bank.PHASE}
        taken.add(bank.PHONE_FACT["name"].lower())
        taken.add(bank.BIRTHDAY_FACT["name"].lower())
        taken |= {row["name"].lower() for row in bank.incidental_facts()}
        if hasattr(bank, "replay_facts"):
            taken |= {row["name"].lower() for row in bank.replay_facts()}
    taken |= {row["name"].lower() for row in train_rows()}
    taken |= {row["name"].lower() for row in train_incidental()}
    taken |= {item.lower() for item in surprise["fit_names"] + surprise["held_names"]}
    taken |= {item.lower() for item in rif["fit_names"] + rif["held_names"]}
    taken |= {item.lower() for item in valence["fit_names"] + valence["held_names"]}
    ours = set()
    assert len(train_bank().incidental_facts()) == 32
    assert len(train_bank().KEEP) == 8
    assert train_bank().LOW == ()
    for bank in (fit_bank(), held_bank()):
        assert len(bank.incidental_facts()) == 32
        assert len(bank.LOW) == 8
        assert len(bank.HIGH) == 8
        assert len(bank.NEUTRAL_NLL) == 8
        assert len(bank.NEUTRAL_CHOICE) == 8
        assert len(bank.NEUTRAL_GEN) == 8
        for fact in list(bank.incidental_facts()) + list(bank.LOW) + list(bank.HIGH):
            assert fact["answer"] in fact["text"]
    for bank in (train_bank(), fit_bank(), held_bank()):
        ours |= _names_of(bank)
        for fact in list(bank.incidental_facts()) + list(bank.KEEP):
            assert fact["answer"] in fact["text"]
    assert _names_of(train_bank()).isdisjoint(_names_of(fit_bank()))
    assert _names_of(train_bank()).isdisjoint(_names_of(held_bank()))
    assert _names_of(fit_bank()).isdisjoint(_names_of(held_bank()))
    assert ours.isdisjoint(taken)
    assert len(_names_of(train_bank())) == 40
    assert len(_names_of(fit_bank())) == 49
    assert len(_names_of(held_bank())) == 49


def test_phone_birthday_and_relation_gate():
    known = []
    prior_phones = []
    for text in _prior_blobs():
        for match in DATE_RE.finditer(text):
            known.append((_MONTHS.index(match.group(1)), int(match.group(2)), int(match.group(3))))
        for match in PHONE_RE.finditer(text):
            prior_phones.append((match.group(1), match.group(3)))
    for bank in (fit_bank(), held_bank()):
        assert completing_relation(bank.PHONE_QUESTION) == "phone"
        assert completing_relation(bank.BIRTHDAY_QUESTION) == "birthday"
        for prompt in list(bank.NEUTRAL_GEN) + [row["prompt"] for row in bank.NEUTRAL_NLL + bank.NEUTRAL_CHOICE]:
            assert completing_relation(prompt) is None
        parsed = parse_date(bank.BIRTHDAY_FACT["answer"])
        assert parsed is not None
        month, day, year, _spans = parsed
        for other in known:
            shared = int(other[0] == month) + int(other[1] == day) + int(other[2] == year)
            assert shared < 2
        known.append((month, day, year))
        match = PHONE_RE.fullmatch(bank.PHONE_FACT["answer"])
        assert match is not None
        assert match.group(2) == "555"
        assert not bank.PHONE_FACT["answer"].startswith("+1-415-555-")
        for old_area, old_last4 in prior_phones:
            assert match.group(1) != old_area
            assert match.group(3) != old_last4
        prior_phones.append((match.group(1), match.group(3)))
        assert bank.PHONE_FACT["answer"] in bank.PHONE_FACT["text"]
        assert bank.BIRTHDAY_FACT["answer"] in bank.BIRTHDAY_FACT["text"]
        assert bank.PHONE_FACT["name"] == bank.BIRTHDAY_FACT["name"]
    assert DEFAULT_GAP_TAU == 0.80
    assert POLARITY_SPAN == 2.0
    assert REPLAY_DECAY == 0.5
    assert REPLAY_EVERY == 32
    assert REPLAY_M == 1
    assert VALENCE_SPAN == 1.0
    assert VALENCE_STEP == 0.25
    assert CLOCK_SPAN == 0.25
    assert SURPRISE_GATES["low_written_max"] == 2
    assert SURPRISE_GATES["high_written_min"] == 6
    assert SURPRISE_GATES["window"] == 32
    assert SURPRISE_GATES["keep_written_min"] == 8
    assert SURPRISE_GATES["nll_abs_mean_max"] == 0.05
    assert SURPRISE_GATES["choice_flips_max"] == 0
    assert SURPRISE_GATES["leak_max"] == 0
    assert SURPRISE_GATES["phone_exact"] == 1
    assert SURPRISE_GATES["birthday_exact"] == 1
    assert SURPRISE_GATES["incidental_slots"] == 32
    assert DIRECTION_HELD_PREFER_MEAN == 10.905
    assert DIRECTION_HELD_AVOID_MEAN == 1.277
    assert DIRECTION_HELD_FORBIDDEN_LIFT == -1.300
    assert REPLAY_PREFER_MEAN_MIN == 8.0
    assert REPLAY_AVOID_MEAN_MIN == 0.6
    assert REPLAY_FORBIDDEN_DOWN_MIN == 6
    circadian = json.loads((ROOT / "reports" / "subconscious_circadian.json").read_text(encoding="utf-8"))
    prefer = circadian["replay_direction"]["prefer"]["summary"]["mean_delta"]
    avoid = circadian["replay_direction"]["avoid"]["summary"]["mean_delta"]
    forbid = circadian["replay_direction"]["avoid"]["summary"]["mean_stored_lift"]
    assert abs(prefer - 20.74961856333539) < 1e-9
    assert abs(avoid - 1.452641248703003) < 1e-9
    assert abs(forbid - (-1.5781219005584717)) < 1e-9
    assert REPLAY_HELD_PREFER_MEAN == 20.750
    assert REPLAY_HELD_AVOID_MEAN == 1.453
    assert REPLAY_HELD_FORBIDDEN_LIFT == -1.578


def test_neutral_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        return full_ids[len(prompt_ids) :]

    for bank in (fit_bank(), held_bank()):
        for item in bank.NEUTRAL_NLL:
            assert len(suffix(item["prompt"], item["continuation"])) == 1
        for item in bank.NEUTRAL_CHOICE:
            assert len(suffix(item["prompt"], item["good"])) == 1
            assert len(suffix(item["prompt"], item["bad"])) == 1


def test_fit_script_does_not_score_the_held_bank():
    source = (ROOT / "scripts" / "fit_subconscious_surprise_rel.py").read_text(encoding="utf-8")
    assert "held_bank" not in source
    assert "run_eval" not in source
    assert "eval_subconscious_surprise_rel" not in source
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert "tidal.subconscious_surprise_rel_score" in modules
    assert "tidal.subconscious_replay_data" not in modules


def test_prereg_locks_the_banks_before_scores():
    text = (ROOT / "reports" / "subconscious_surprise_rel_prereg.md").read_text(encoding="utf-8")
    assert "not been scored" in text
    assert "quantile of the last 32" in text
    assert "equals the stored" in text
    assert "not out of scope" in text
    assert bank_sha256("train") in text
    assert bank_sha256("fit") in text
    assert bank_sha256("held") in text
    assert f"surprise quantile: {float(SURPRISE_QUANTILE):g}" in text
    assert FROZEN_SHA in text
    assert SURPRISE_FIT_SHA in text
    assert SURPRISE_HELD_SHA in text
    assert RIF_FIT_SHA in text
    assert RIF_HELD_SHA in text
    assert REPLAY_FIT_SHA in text
    assert REPLAY_HELD_SHA in text
    assert PRIOR_VALENCE_FIT_SHA in text
    assert PRIOR_VALENCE_HELD_SHA in text
    assert SPELL_FIT_SHA in text
    assert SPELL_HELD_SHA in text
    assert VALENCE_FIT_SHA in text
    assert VALENCE_HELD_SHA in text
    assert CIRCADIAN_FIT_SHA in text
    assert CIRCADIAN_HELD_SHA in text
    assert DIRECTION_TRAIN_SHA in text


def test_shipped_inject_is_unchanged_and_the_gate_adds_no_parameter():
    ckpt = torch.load(
        ROOT / "tidal" / "llm_inject_smollm2_135m.pt",
        map_location="cpu",
        weights_only=False,
    )
    weight = ckpt["inject"]["net.2.weight"].contiguous().numpy().tobytes()
    assert hashlib.sha256(weight).hexdigest() == BASELINE_NET_SHA
    assert tuple(ckpt["inject"]["polarity"].shape) == (576,)
    source = (ROOT / "tidal" / "llm_memory.py").read_text(encoding="utf-8")
    assert "key_proj" not in source
    assert "surprise_thr" not in source
    assert "SURPRISE_QUANTILE" in source
    assert "SURPRISE_WINDOW" in source
