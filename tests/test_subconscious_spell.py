"""Exact phone and birthday spelling banks.

The preregistration test reads the file the fit script writes before the
held-out bank is scored. It does not score that bank.
"""
import ast
import hashlib
import inspect
import json
import re
from pathlib import Path

import pytest

from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    POLARITY_SPAN,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
    LLMMemory,
    completing_relation,
)
from tidal.llm_memory_data import names, parse_date
from tidal.subconscious_eval import _tokens
from tidal.subconscious_data import PHONE_FACT, bank_payload, decision_items, incidental_facts
from tidal.subconscious_direction_data import (
    fit_bank as direction_fit_bank,
    held_bank as direction_held_bank,
    train_incidental,
    train_rows,
)
from tidal.subconscious_gate_data import held_bank as gate_held_bank
from tidal.subconscious_replay_data import fit_bank as replay_fit_bank
from tidal.subconscious_replay_data import held_bank as replay_held_bank
from tidal.subconscious_spell_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    REPLAY_FIT_SHA,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_HELD_SHA,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SPELL_GATES,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
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
BASELINE_NET_SHA = "f8d4681c8ca9be922523e0e8064131060209a4d0c25e15956fb5ce97e4834143"
DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December) ([1-9]|1[0-9]|2[0-7]), "
    r"(19[7-9][0-9]|200[0-9]|201[0-9])\b"
)
PHONE_RE = re.compile(r"\+1-(\d{3})-(\d{3})-(\d{4})")


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


def _spell_surface(bank):
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


def _dates(texts):
    found = []
    for text in texts:
        for match in DATE_RE.finditer(text):
            found.append((match.group(1), int(match.group(2)), int(match.group(3))))
    return found


def _phones(texts):
    found = []
    for text in texts:
        for match in PHONE_RE.finditer(text):
            found.append((match.group(1), match.group(2), match.group(3)))
    return found


def test_spell_data_does_not_import_torch():
    source = (ROOT / "tidal" / "subconscious_spell_data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    bad = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bad.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            bad.append(node.module.split(".")[0])
    assert "torch" not in bad
    assert "transformers" not in bad


def test_spell_banks_are_disjoint_from_every_prior_bank():
    prior = _load_fixture("subconscious_prior_surfaces.json")
    surprise = _load_fixture("subconscious_surprise_surfaces.json")
    rif = _load_fixture("subconscious_rif_surfaces.json")
    valence = _load_fixture("subconscious_valence_surfaces.json")
    assert surprise["fit_sha"] == SURPRISE_FIT_SHA
    assert surprise["held_sha"] == SURPRISE_HELD_SHA
    assert rif["fit_sha"] == RIF_FIT_SHA
    assert rif["held_sha"] == RIF_HELD_SHA
    assert valence["fit_sha"] == VALENCE_FIT_SHA
    assert valence["held_sha"] == VALENCE_HELD_SHA
    from tidal.subconscious_data import bank_sha256 as frozen_sha
    from tidal.subconscious_gate_data import bank_sha256 as gate_sha
    from tidal.subconscious_replay_data import bank_sha256 as replay_sha

    assert frozen_sha() == FROZEN_SHA
    assert gate_sha() == GATE_SHA
    assert replay_sha("fit") == REPLAY_FIT_SHA
    assert replay_sha("held") == REPLAY_HELD_SHA
    assert FIT39_SHA == "20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce"
    assert HELD39_SHA == "ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a"
    groups = {"fit": _spell_surface(fit_bank()), "held": _spell_surface(held_bank())}
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
        "valence_fit": valence["fit"],
        "valence_held": valence["held"],
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


def test_spell_names_dates_and_phones_stay_apart():
    from tidal.llm_memory import _CALIB_NAMES

    surprise = _load_fixture("subconscious_surprise_surfaces.json")
    rif = _load_fixture("subconscious_rif_surfaces.json")
    valence = _load_fixture("subconscious_valence_surfaces.json")
    taken = {item.lower() for item in names("train")} | {item.lower() for item in names("test")}
    taken |= {item.lower() for item in _CALIB_NAMES}
    taken |= {item["name"].lower() for item in decision_items()}
    taken.add(PHONE_FACT["name"].lower())
    taken |= {row["name"].lower() for row in incidental_facts()}
    for bank in (direction_fit_bank(), direction_held_bank(), gate_held_bank()):
        taken |= {item["name"].lower() for item in bank.decision_items()}
        taken.add(bank.PHONE_FACT["name"].lower())
        taken |= {row["name"].lower() for row in bank.incidental_facts()}
    taken |= {row["name"].lower() for row in train_rows()}
    taken |= {row["name"].lower() for row in train_incidental()}
    taken |= {item.lower() for item in surprise["fit_names"] + surprise["held_names"]}
    taken |= {item.lower() for item in rif["fit_names"] + rif["held_names"]}
    taken |= {item.lower() for item in valence["fit_names"] + valence["held_names"]}
    for bank in (replay_fit_bank(), replay_held_bank()):
        taken |= {item["name"].lower() for item in bank.decision_items()}
        taken.add(bank.PHONE_FACT["name"].lower())
        taken.add(bank.BIRTHDAY_FACT["name"].lower())
        taken |= {row["name"].lower() for row in bank.incidental_facts()}
        taken |= {row["name"].lower() for row in bank.replay_facts()}
    ours = set()
    dates = []
    phones = []
    for bank in (fit_bank(), held_bank()):
        names_here = {row["name"] for row in bank.incidental_facts()}
        names_here.add(bank.PHONE_FACT["name"])
        assert bank.BIRTHDAY_FACT["name"] == bank.PHONE_FACT["name"]
        assert len(bank.incidental_facts()) == 32
        assert len(names_here) == 33
        ours |= {item.lower() for item in names_here}
        parsed = parse_date(bank.BIRTHDAY_FACT["answer"])
        assert parsed is not None
        dates.append((parsed[0], parsed[1], parsed[2]))
        match = PHONE_RE.fullmatch(bank.PHONE_FACT["answer"])
        assert match is not None
        assert match.group(2) == "555"
        assert not bank.PHONE_FACT["answer"].startswith("+1-415-555-")
        phones.append((match.group(1), match.group(3)))
        assert bank.PHONE_FACT["answer"] in bank.PHONE_FACT["text"]
        assert bank.BIRTHDAY_FACT["answer"] in bank.BIRTHDAY_FACT["text"]
    assert len(ours) == 66
    assert ours.isdisjoint(taken)
    fit_names = {row["name"].lower() for row in fit_bank().incidental_facts()}
    fit_names.add(fit_bank().PHONE_FACT["name"].lower())
    held_names = {row["name"].lower() for row in held_bank().incidental_facts()}
    held_names.add(held_bank().PHONE_FACT["name"].lower())
    assert fit_names.isdisjoint(held_names)
    month_a, day_a, year_a = dates[0]
    month_b, day_b, year_b = dates[1]
    assert int(month_a == month_b) + int(day_a == day_b) + int(year_a == year_b) < 2
    assert phones[0][0] != phones[1][0]
    assert phones[0][1] != phones[1][1]
    prior_texts = []
    for rows in (
        _frozen_surface(),
        _load_fixture("subconscious_prior_surfaces.json")["strings"],
        _direction_surface(gate_held_bank()),
        _train_surface(),
        _direction_surface(direction_fit_bank()),
        _direction_surface(direction_held_bank()),
        _load_fixture("subconscious_surprise_surfaces.json")["fit"],
        _load_fixture("subconscious_surprise_surfaces.json")["held"],
        _load_fixture("subconscious_rif_surfaces.json")["fit"],
        _load_fixture("subconscious_rif_surfaces.json")["held"],
        _replay_surface(replay_fit_bank()),
        _replay_surface(replay_held_bank()),
        _load_fixture("subconscious_valence_surfaces.json")["fit"],
        _load_fixture("subconscious_valence_surfaces.json")["held"],
    ):
        prior_texts.extend(rows)
    prior_dates = _dates(prior_texts)
    months = (
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    for month_i, day, year in dates:
        for old_month, old_day, old_year in prior_dates:
            same = int(months[month_i] == old_month) + int(day == old_day) + int(year == old_year)
            assert same < 2, ((months[month_i], day, year), (old_month, old_day, old_year))
    prior_phones = _phones(prior_texts)
    for area, last4 in phones:
        for old_area, _exchange, old_last4 in prior_phones:
            assert area != old_area
            assert last4 != old_last4


def test_phone_birthday_relation_gate_and_floors():
    for bank in (fit_bank(), held_bank()):
        assert completing_relation(bank.PHONE_QUESTION) == "phone"
        assert completing_relation(bank.BIRTHDAY_QUESTION) == "birthday"
        for prompt in list(bank.NEUTRAL_GEN) + [row["prompt"] for row in bank.NEUTRAL_NLL + bank.NEUTRAL_CHOICE]:
            assert completing_relation(prompt) is None
    assert DEFAULT_GAP_TAU == 0.80
    assert POLARITY_SPAN == 2.0
    assert REPLAY_DECAY == 0.5
    assert REPLAY_EVERY == 32
    assert REPLAY_M == 1
    assert SPELL_GATES["phone_off_takeover_max"] == 0.0
    assert SPELL_GATES["birthday_off_takeover_max"] == 0.0
    assert SPELL_GATES["phone_exact"] == 1
    assert SPELL_GATES["birthday_exact"] == 1
    assert SPELL_GATES["phone_off_equal_max"] == 0
    assert SPELL_GATES["birthday_off_equal_max"] == 0
    assert SPELL_GATES["nll_abs_mean_max"] == 0.05
    assert SPELL_GATES["nll_bad_count_max"] == 1
    assert SPELL_GATES["choice_flips_max"] == 0
    assert SPELL_GATES["leak_max"] == 0
    assert SPELL_GATES["incidental_slots"] == 32
    assert SPELL_GATES["prefer_mean_min"] == 8.0
    assert SPELL_GATES["avoid_mean_min"] == 0.6
    assert SPELL_GATES["forbidden_down_min"] == 6
    assert SPELL_GATES["parameters_new"] == 0
    assert SPELL_GATES["remember_forwards"] == 2
    assert SPELL_GATES["query_forwards"] == 1
    assert DIRECTION_HELD_PREFER_MEAN == 10.905
    assert DIRECTION_HELD_AVOID_MEAN == 1.277
    assert DIRECTION_HELD_FORBIDDEN_LIFT == -1.300
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
    source = (ROOT / "scripts" / "fit_subconscious_spell.py").read_text(encoding="utf-8")
    assert "held_bank" not in source
    assert "run_eval" not in source
    assert "eval_subconscious_spell" not in source


def test_spelling_reconstructs_without_the_readout_or_the_raw_answer():
    source = inspect.getsource(LLMMemory._stored_spelling)
    assert "phone_digits" not in source
    assert "date_spelling" not in source
    assert "PHONE_TAIL" not in source
    assert "slot.answer" not in source
    assert "recall_surface" in source
    spell = inspect.getsource(LLMMemory._spell_ids)
    assert "phone_on" not in spell
    assert "date_on" not in spell


def test_prereg_locks_the_banks_before_scores():
    text = (ROOT / "reports" / "subconscious_spell_prereg.md").read_text(encoding="utf-8")
    assert "not been scored" in text
    assert bank_sha256("fit") in text
    assert bank_sha256("held") in text
    assert FROZEN_SHA in text
    assert SURPRISE_FIT_SHA in text
    assert SURPRISE_HELD_SHA in text
    assert RIF_FIT_SHA in text
    assert RIF_HELD_SHA in text
    assert REPLAY_FIT_SHA in text
    assert REPLAY_HELD_SHA in text
    assert VALENCE_FIT_SHA in text
    assert VALENCE_HELD_SHA in text
    assert "full number" in text
    assert "+1-415-555-" in text


def test_shipped_inject_is_unchanged_and_spelling_has_no_new_parameter():
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
    assert "surprise_thr" not in source


def test_replay_held_numbers_match_the_published_report():
    data = json.loads((ROOT / "reports" / "subconscious_replay.json").read_text(encoding="utf-8"))
    prefer = float(data["direction"]["prefer"]["summary"]["mean_delta"])
    avoid = float(data["direction"]["avoid"]["summary"]["mean_delta"])
    forbidden = float(data["direction"]["avoid"]["summary"]["mean_stored_lift"])
    assert abs(prefer - REPLAY_HELD_PREFER_MEAN) < 0.001
    assert abs(avoid - REPLAY_HELD_AVOID_MEAN) < 0.001
    assert abs(forbidden - REPLAY_HELD_FORBIDDEN_LIFT) < 0.001
    assert data["bank_sha256"] == REPLAY_HELD_SHA
