"""Retrieval-forgetting banks. Disjoint from every earlier subconscious bank.

The preregistration test reads the file the fit script writes before the
held-out bank is scored. It does not score that bank.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tidal.episodic import CLOCK_SPAN, Slot
from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    LLMMemory,
    POLARITY_SPAN,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
    RIF_MAX_SET,
    RIF_SALIENCE_EXEMPT,
    RIF_SPAN,
    SURPRISE_MIN_COUNT,
    SURPRISE_QUANTILE,
    SURPRISE_WINDOW,
    VALENCE_SPAN,
    VALENCE_STEP,
    build_tiny_lm,
    forget_rivals,
    rif_competitor_keep,
    rif_weight,
    rival_indexes,
)
from tidal.llm_memory_data import names
from tidal.question_heldout_data import _tokens
from tidal.subconscious_circadian_data import (
    bank_sha256 as clock_sha,
    fit_bank as clock_fit_bank,
    held_bank as clock_held_bank,
)
from tidal.subconscious_data import PHONE_FACT, bank_payload, bank_sha256 as frozen_sha, decision_items, incidental_facts
from tidal.subconscious_direction_data import (
    fit_bank as direction_fit_bank,
    held_bank as direction_held_bank,
    train_incidental,
    train_rows,
)
from tidal.subconscious_gate_data import bank_sha256 as gate_sha, held_bank as gate_held_bank
from tidal.subconscious_replay_data import (
    bank_sha256 as replay_sha,
    fit_bank as replay_fit_bank,
    held_bank as replay_held_bank,
)
from tidal.subconscious_rif_retry_data import (
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
    RIF_GATES,
    RIF_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    SURPRISE_REL_FIT_SHA,
    SURPRISE_REL_HELD_SHA,
    SURPRISE_REL_TRAIN_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    _PAYLOAD,
    bank_sha256,
    fit_bank,
    held_bank,
)
from tidal.subconscious_spell_data import (
    bank_sha256 as spell_sha,
    fit_bank as spell_fit_bank,
    held_bank as spell_held_bank,
)
from tidal.subconscious_surprise_rel_data import (
    SURPRISE_GATES,
    bank_sha256 as rel_sha,
    fit_bank as rel_fit_bank,
    held_bank as rel_held_bank,
    train_bank as rel_train_bank,
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


def _rif_surface(bank):
    out = []
    for group in bank.QUARTETS:
        _walk(list(group), out)
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


def _slot(rel, salience, key, strength=1.0):
    vec = np.asarray(key, dtype=np.float64)
    slot = Slot(
        k=vec,
        v=np.zeros(4, dtype=np.float32),
        name="n",
        rel=rel,
        t=0,
    )
    slot.salience = float(salience)
    slot.strength = float(strength)
    return slot


def test_weight_is_relative_inside_the_competitor_set():
    assert rif_weight(0.8, 0.8) == 1.0
    assert abs(rif_weight(0.0, 0.8) - 0.5) < 1e-9
    assert abs(rif_weight(-0.2, 0.8) - 0.5) < 1e-9
    assert rif_weight(0.1, 0.0) == 0.5
    assert abs(rif_competitor_keep(0.8, 0.8, 0.05) - 0.95) < 1e-9
    assert abs(rif_competitor_keep(0.0, 0.8, 0.05) - 0.975) < 1e-9
    with pytest.raises(ValueError):
        rif_competitor_keep(0.2, 0.2, 1.0)


def test_background_relations_and_salience_are_not_cues():
    notes = [_slot("note", 0.0, np.eye(8)[i % 8]) for i in range(32)]
    assert rival_indexes(notes, 0, RIF_MAX_SET, RIF_SALIENCE_EXEMPT) == []
    assert forget_rivals(notes, 0) == 0
    assert all(slot.strength == 1.0 for slot in notes)
    cue = [
        _slot("cue", 0.0, [1.0, 0.0, 0.0]),
        _slot("cue", 0.5, [0.0, 1.0, 0.0]),
        _slot("cue", 0.1, [0.0, 0.0, 1.0]),
        _slot("other", 0.0, [1.0, 1.0, 0.0]),
    ]
    assert rival_indexes(cue, 0, 3, 0.5) == [2]
    lone = [_slot("only", 0.0, [1.0, 0.0]), _slot("else", 0.0, [0.0, 1.0])]
    assert rival_indexes(lone, 0, 3, 0.5) == []


def test_closer_key_is_scaled_more_and_span_zero_is_a_no_op(monkeypatch):
    slots = [
        _slot("cue", 0.0, [1.0, 0.0]),
        _slot("cue", 0.0, [1.0, 0.1]),
        _slot("cue", 0.0, [0.0, 1.0]),
        _slot("else", 0.0, [0.2, 0.2]),
    ]
    monkeypatch.setattr("tidal.llm_memory.RIF_SPAN", 0.0)
    assert forget_rivals(slots, 0) == 0
    assert [slot.strength for slot in slots] == [1.0, 1.0, 1.0, 1.0]
    monkeypatch.setattr("tidal.llm_memory.RIF_SPAN", 0.5)
    changed = forget_rivals(slots, 0)
    assert changed == 2
    assert slots[0].strength == 1.0
    assert slots[3].strength == 1.0
    assert slots[1].strength < slots[2].strength < 1.0


def test_rank_forgets_only_a_strict_same_relation_win():
    model, tok = build_tiny_lm(seed=1)
    mem = LLMMemory(model, tok)
    try:
        assert mem.remember("code is Y", name="ada", rel="phone")
        mem._rank("code is Y")
        assert mem.store.slots[0].strength == 1.0
        assert mem.last_rif == {"n": 0}
        assert mem.remember("town is Q", name="bea", rel="city")
        mem._rank("town is Q")
        assert all(slot.strength == 1.0 for slot in mem.store.slots)
        assert mem.last_rif == {"n": 0}
        for index in range(5):
            assert mem.remember(f"note {index} is Z", name=f"n{index}", rel="note")
            slot = mem.store._slots[-1]
            key = np.zeros(np.asarray(slot.k).shape, dtype=np.float64)
            key[index] = 1.0
            slot.k = key
            slot.salience = 0.0
            slot.strength = 1.0
        notes = [slot for slot in mem.store.slots if slot.rel == "note"]
        assert len(notes) == 5
        mem._rank("note 0 is Z")
        assert all(slot.strength == 1.0 for slot in mem.store.slots)
        mem.store.clear()
        assert mem.remember("red is Y", name="a", rel="cue")
        assert mem.remember("blue is Z", name="b", rel="cue")
        slots = mem.store._slots
        query = np.asarray(mem._query_vector("red is Y"), dtype=np.float64)
        slots[0].k = query.copy()
        slots[0].salience = 0.0
        orth = np.zeros_like(query)
        denom = float(np.dot(query, query)) + 1e-12
        for axis in np.eye(query.size):
            proj = axis - query * (float(np.dot(axis, query)) / denom)
            if float(np.linalg.norm(proj)) > 1e-6:
                orth = proj
                break
        slots[1].k = orth
        slots[1].salience = 0.0
        info = mem._rank("red is Y")
        assert info["gap"] is not None and float(info["gap"]) > 0.0
        assert slots[0].strength == 1.0
        assert slots[1].strength < 1.0
        assert mem.last_rif["n"] == 1
        slots[1].strength = 1.0
        slots[1].salience = 1.0
        mem._rank("red is Y")
        assert slots[0].strength == 1.0
        assert slots[1].strength == 1.0
        assert mem.last_rif["n"] == 0
        slots[1].k = query.copy()
        slots[1].salience = 0.0
        mem._rank("red is Y")
        assert slots[0].strength == 1.0
        assert slots[1].strength == 1.0
        assert mem.last_rif["n"] == 0
        assert "forget_rivals" in inspect.getsource(LLMMemory._rank)
        assert 'mode="top1"' in inspect.getsource(LLMMemory._rank)
    finally:
        mem.close()


def test_data_does_not_import_torch():
    source = (ROOT / "tidal" / "subconscious_rif_retry_data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "torch" not in bad
    assert "transformers" not in bad
    assert set(_PAYLOAD) == {"fit", "held"}


def test_score_reads_the_live_span():
    source = (ROOT / "tidal" / "subconscious_rif_retry_score.py").read_text(encoding="utf-8")
    assert "RIF_SPAN" in source
    assert "held_bank" not in source
    assert "noncompetitor" in source
    assert "outsider" in source


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
    assert frozen_sha() == FROZEN_SHA
    assert gate_sha() == GATE_SHA
    assert clock_sha("fit") == CIRCADIAN_FIT_SHA
    assert clock_sha("held") == CIRCADIAN_HELD_SHA
    assert rel_sha("train") == SURPRISE_REL_TRAIN_SHA
    assert rel_sha("fit") == SURPRISE_REL_FIT_SHA
    assert rel_sha("held") == SURPRISE_REL_HELD_SHA
    groups = {
        "fit": _rif_surface(fit_bank()),
        "held": _rif_surface(held_bank()),
    }
    others = {
        "frozen": _frozen_surface(),
        "pr39": prior["strings"],
        "gate": _direction_surface(gate_held_bank()),
        "direction_train": _walk_train(),
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
        "surprise_rel_train": _surprise_surface(rel_train_bank()),
        "surprise_rel_fit": _surprise_surface(rel_fit_bank()),
        "surprise_rel_held": _surprise_surface(rel_held_bank()),
    }
    for name, rows in groups.items():
        for other_name, other in others.items():
            literal = [text for text in rows if text in set(other)]
            assert literal == [], (name, other_name, literal[:3])
            best, pair = _max_jaccard(rows, other)
            assert best < 0.50, (name, other_name, best, pair)
    literal = [text for text in groups["fit"] if text in set(groups["held"])]
    assert literal == [], literal[:3]
    best, pair = _max_jaccard(groups["fit"], groups["held"])
    assert best < 0.50, (best, pair)


def _walk_train():
    out = []
    _walk(train_rows(), out)
    _walk(train_incidental(), out)
    return out


def _names_of(bank):
    return {row["name"].lower() for group in bank.QUARTETS for row in group}


def test_names_and_quartets_stay_apart():
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
        rel_train_bank(),
        rel_fit_bank(),
        rel_held_bank(),
    ):
        if hasattr(bank, "decision_items"):
            taken |= {item["name"].lower() for item in bank.decision_items()}
        if hasattr(bank, "PHASE"):
            taken |= {item["name"].lower() for item in bank.PHASE}
        if hasattr(bank, "LOW"):
            taken |= {item["name"].lower() for item in list(bank.LOW) + list(bank.HIGH) + list(bank.KEEP)}
        if bank.PHONE_FACT:
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
    seen_rel = set()
    for bank in (fit_bank(), held_bank()):
        assert len(bank.QUARTETS) == 2
        for group in bank.QUARTETS:
            assert len(group) == 5
            assert [row["role"] for row in group].count("target") == 1
            assert [row["role"] for row in group].count("competitor") == 3
            assert [row["role"] for row in group].count("outsider") == 1
            assert group[0]["role"] == "target"
            cue = {row["rel"] for row in group if row["role"] != "outsider"}
            assert len(cue) == 1
            rel = cue.pop()
            assert rel not in ("note", "phone", "birthday", "")
            assert group[-1]["rel"] != rel
            assert group[-1]["rel"] not in ("note", "phone", "birthday", "")
            assert rel not in seen_rel
            assert group[-1]["rel"] not in seen_rel
            seen_rel.add(rel)
            seen_rel.add(group[-1]["rel"])
            for row in group:
                assert row["answer"] in row["text"]
                assert row["answer"] not in row["prompt"]
                assert row["good"] == " " + row["answer"]
                assert row["bad"].startswith(" ")
                assert row["rel"] not in row["text"]
                assert row["rel"] not in row["prompt"]
                assert row["rel"] not in row["good"]
                assert row["rel"] not in row["bad"]
        ours |= _names_of(bank)
    assert _names_of(fit_bank()).isdisjoint(_names_of(held_bank()))
    assert ours.isdisjoint(taken)
    assert len(_names_of(fit_bank())) == 10
    assert len(_names_of(held_bank())) == 10


def test_continuations_are_one_smollm_token():
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

    def suffix(prompt, continuation):
        prompt_ids = tok.encode(prompt, add_special_tokens=True)
        full_ids = tok.encode(prompt + continuation, add_special_tokens=True)
        assert full_ids[: len(prompt_ids)] == prompt_ids
        return full_ids[len(prompt_ids) :]

    for bank in (fit_bank(), held_bank()):
        for group in bank.QUARTETS:
            for row in group:
                assert len(suffix(row["prompt"], row["good"])) == 1
                assert len(suffix(row["prompt"], row["bad"])) == 1


def test_fit_script_does_not_score_the_held_bank():
    source = (ROOT / "scripts" / "fit_subconscious_rif_retry.py").read_text(encoding="utf-8")
    assert "held_bank" not in source
    assert "run_eval" not in source
    assert "eval_subconscious_rif_retry" not in source
    tree = ast.parse(source)
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert "tidal.subconscious_rif_retry_score" in modules
    assert "tidal.subconscious_replay_data" not in modules


def test_prereg_locks_the_banks_before_scores():
    text = (ROOT / "reports" / "subconscious_rif_retry_prereg.md").read_text(encoding="utf-8")
    assert "not been scored" in text
    assert "share the winner's relation" in text
    assert "not out of scope" in text
    assert "Anderson, Bjork, and Bjork (1994)" in text
    assert "Anderson, M. C. (2003)" in text
    assert "Norman, K. A., Newman, E. L., & Detre, G. (2007)" in text
    assert "Rp-" in text
    assert "Nrp" in text
    assert "37.5%" in text
    assert "48.4%" in text
    assert "15.7" in text
    assert "6.3" in text
    assert "https://doi.org/10.1037/0278-7393.20.5.1063" in text
    assert "https://doi.org/10.1016/j.jml.2003.08.006" in text
    assert "https://doi.org/10.1037/0033-295X.114.4.887" in text
    assert "fuzzy recall" in text
    assert "minimum drop at least 0.10" in text
    assert "mean drop at least 0.20" in text
    assert "absolute winner drop at most 0.05" in text
    assert "absolute outsider drop at most 0.05" in text
    fit_source = (ROOT / "scripts" / "fit_subconscious_rif_retry.py").read_text(encoding="utf-8")
    eval_source = (ROOT / "scripts" / "eval_subconscious_rif_retry.py").read_text(encoding="utf-8")
    assert "HUMAN_EFFECT" in fit_source
    assert "HUMAN_EFFECT" in eval_source
    assert "held_bank" not in fit_source
    assert "noncompetitor" in text or "noncompetitors" in text
    assert bank_sha256("fit") in text
    assert bank_sha256("held") in text
    assert f"rif span: {float(RIF_SPAN):g}" in text
    for sha in (
        FROZEN_SHA,
        FIT39_SHA,
        HELD39_SHA,
        GATE_SHA,
        DIRECTION_TRAIN_SHA,
        SURPRISE_FIT_SHA,
        SURPRISE_HELD_SHA,
        RIF_FIT_SHA,
        RIF_HELD_SHA,
        REPLAY_FIT_SHA,
        REPLAY_HELD_SHA,
        PRIOR_VALENCE_FIT_SHA,
        PRIOR_VALENCE_HELD_SHA,
        SPELL_FIT_SHA,
        SPELL_HELD_SHA,
        VALENCE_FIT_SHA,
        VALENCE_HELD_SHA,
        CIRCADIAN_FIT_SHA,
        CIRCADIAN_HELD_SHA,
        SURPRISE_REL_TRAIN_SHA,
        SURPRISE_REL_FIT_SHA,
        SURPRISE_REL_HELD_SHA,
    ):
        assert sha in text


def test_shipped_inject_and_the_older_gates_stay_put():
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
    assert "RIF_SPAN" in source
    assert "forget_rivals" in source
    assert RIF_SPAN in (0.05, 0.10, 0.20)
    assert RIF_MAX_SET == 3
    assert RIF_SALIENCE_EXEMPT == 0.5
    assert DEFAULT_GAP_TAU == 0.80
    assert POLARITY_SPAN == 2.0
    assert REPLAY_DECAY == 0.5
    assert REPLAY_EVERY == 32
    assert REPLAY_M == 1
    assert VALENCE_SPAN == 1.0
    assert VALENCE_STEP == 0.25
    assert CLOCK_SPAN == 0.25
    assert SURPRISE_WINDOW == 32
    assert SURPRISE_MIN_COUNT == 32
    assert SURPRISE_QUANTILE == 0.0
    assert RIF_GATES["competitor_drop_min"] == 0.10
    assert RIF_GATES["competitor_mean_drop_min"] == 0.20
    assert RIF_GATES["winner_abs_drop_max"] == 0.05
    assert RIF_GATES["noncompetitor_abs_drop_max"] == 0.05
    assert RIF_GATES["prefer_delta_pos_min"] == 8
    assert RIF_GATES["avoid_delta_pos_min"] == 8
    assert RIF_GATES["forbidden_down_min"] == 6
    assert SURPRISE_GATES["choice_flips_max"] == 0
    assert SURPRISE_GATES["leak_max"] == 0
    assert SURPRISE_GATES["phone_exact"] == 1
    assert SURPRISE_GATES["birthday_exact"] == 1
    assert SURPRISE_GATES["nll_abs_mean_max"] == 0.05
    assert DIRECTION_HELD_PREFER_MEAN == 10.905
    assert DIRECTION_HELD_AVOID_MEAN == 1.277
    assert DIRECTION_HELD_FORBIDDEN_LIFT == -1.300
    assert REPLAY_PREFER_MEAN_MIN == 8.0
    assert REPLAY_AVOID_MEAN_MIN == 0.6
    assert REPLAY_FORBIDDEN_DOWN_MIN == 6
    assert REPLAY_HELD_PREFER_MEAN == 20.750
    assert REPLAY_HELD_AVOID_MEAN == 1.453
    assert REPLAY_HELD_FORBIDDEN_LIFT == -1.578
    circadian = json.loads((ROOT / "reports" / "subconscious_circadian.json").read_text(encoding="utf-8"))
    prefer = circadian["replay_direction"]["prefer"]["summary"]["mean_delta"]
    avoid = circadian["replay_direction"]["avoid"]["summary"]["mean_delta"]
    forbid = circadian["replay_direction"]["avoid"]["summary"]["mean_stored_lift"]
    assert abs(prefer - 20.74961856333539) < 1e-9
    assert abs(avoid - 1.452641248703003) < 1e-9
    assert abs(forbid - (-1.5781219005584717)) < 1e-9
