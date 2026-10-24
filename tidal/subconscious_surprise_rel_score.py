"""Score one relative-surprise bank.

The quantile is whatever ``SURPRISE_QUANTILE`` is in ``tidal.llm_memory`` when
this module runs. This module does not choose it and does not import the
held-out bank. Lows are attempted before highs, both after the incidental
seed, with no clear between them. Phone and birthday gates are exact string
equality with the stored answer. The prefix detector is recorded and is not
that gate. The surprise window is numbers only.
"""
from __future__ import annotations

import json
import time

import tidal.episodic as episodic
import tidal.llm_memory as llm_memory
from tidal.subconscious_eval import date_takeover, inject_census, phone_takeover
from tidal.subconscious_replay_score import _forward_count, score_neutral


def _mean(xs):
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _fact(item):
    return {
        "text": item["text"],
        "answer": item["answer"],
        "name": item["name"],
        "rel": item["rel"],
    }


def _attempt(mem, items, family):
    rows = []
    for item in items:
        wrote = bool(mem.remember(_fact(item)))
        surprise = mem.last_surprise
        rows.append(
            {
                "id": item["id"],
                "family": family,
                "written": int(wrote),
                "surprise": None if surprise is None else float(surprise),
            }
        )
    return rows


def _window_has_text(mem, items) -> bool:
    blob = json.dumps(list(mem.surprise_window()))
    for item in items:
        for piece in (item.get("text"), item.get("answer"), item.get("name")):
            if piece and piece in blob:
                return True
    return False


def score_gate(mem, bank, *, progress=None) -> dict:
    """Seed 32 incidentals, then lows, then highs. No clear between them."""
    incidentals = list(bank.incidental_facts())
    mem.store.clear()
    seed_written = 0
    seed_surprises = []
    for index, item in enumerate(incidentals):
        if progress is not None:
            progress(f"seed {index} {item['id']}")
        seed_written += int(bool(mem.remember(_fact(item))))
        if mem.last_surprise is not None:
            seed_surprises.append(float(mem.last_surprise))
    seed_slots = len(mem.store)
    window = list(mem.surprise_window())
    level = float(llm_memory.SURPRISE_QUANTILE)
    bar = None
    if len(window) >= int(llm_memory.SURPRISE_MIN_COUNT):
        bar = float(llm_memory.surprise_quantile(window, level))
    if progress is not None:
        progress("lows")
    low_rows = _attempt(mem, bank.LOW, "low")
    if progress is not None:
        progress("highs")
    high_rows = _attempt(mem, bank.HIGH, "high")
    leaked = _window_has_text(mem, list(incidentals) + list(bank.LOW) + list(bank.HIGH))
    mem.store.clear()
    return {
        "seed_written": int(seed_written),
        "seed_n": len(incidentals),
        "seed_slots": int(seed_slots),
        "window": len(window),
        "quantile": level,
        "bar": bar,
        "seed_min_surprise": _mean([]) if not seed_surprises else float(min(seed_surprises)),
        "seed_max_surprise": None if not seed_surprises else float(max(seed_surprises)),
        "low_written": int(sum(row["written"] for row in low_rows)),
        "low_n": len(low_rows),
        "high_written": int(sum(row["written"] for row in high_rows)),
        "high_n": len(high_rows),
        "low_mean_surprise": _mean([row["surprise"] for row in low_rows if row["surprise"] is not None]),
        "high_mean_surprise": _mean([row["surprise"] for row in high_rows if row["surprise"] is not None]),
        "window_has_text": int(leaked),
        "low_rows": low_rows,
        "high_rows": high_rows,
    }


def score_keeps(mem, bank, *, progress=None) -> dict:
    """After the train incidental window is full, ordinary keeps must still be written."""
    incidentals = list(bank.incidental_facts())
    mem.store.clear()
    seed_written = 0
    for index, item in enumerate(incidentals):
        if progress is not None:
            progress(f"train seed {index}")
        seed_written += int(bool(mem.remember(_fact(item))))
    slots = len(mem.store)
    rows = _attempt(mem, bank.KEEP, "keep")
    mem.store.clear()
    return {
        "seed_written": int(seed_written),
        "seed_n": len(incidentals),
        "seed_slots": int(slots),
        "keep_written": int(sum(row["written"] for row in rows)),
        "keep_n": len(rows),
        "rows": rows,
    }


def _off(rows, question):
    return [row for row in rows if row["prompt"] != question]


def exact_spelling(bank, neutral) -> dict:
    """On-relation string equality. Off-relation includes the cross question."""
    phone_rows = neutral["scripted"]["phone"]
    date_rows = neutral["scripted"]["birthday"]
    phone = bank.PHONE_FACT["answer"]
    birthday = bank.BIRTHDAY_FACT["answer"]
    phone_q = next(row for row in phone_rows if row["prompt"] == bank.PHONE_QUESTION)
    date_q = next(row for row in date_rows if row["prompt"] == bank.BIRTHDAY_QUESTION)
    phone_off = _off(phone_rows, bank.PHONE_QUESTION)
    date_off = _off(date_rows, bank.BIRTHDAY_QUESTION)
    return {
        "phone_stored": phone,
        "birthday_stored": birthday,
        "phone_question_text": phone_q["text"],
        "birthday_question_text": date_q["text"],
        "phone_exact": int(phone_q["text"] == phone),
        "birthday_exact": int(date_q["text"] == birthday),
        "phone_prefix": int(phone_takeover(phone_q["text"])),
        "birthday_prefix": int(date_takeover(date_q["text"])),
        "phone_off_takeover": _mean([phone_takeover(row["text"]) for row in phone_off]),
        "birthday_off_takeover": _mean([date_takeover(row["text"]) for row in date_off]),
        "phone_off_equal": int(any(row["text"] == phone for row in phone_off)),
        "birthday_off_equal": int(any(row["text"] == birthday for row in date_off)),
        "phone_off_n": len(phone_off),
        "birthday_off_n": len(date_off),
    }


def measure_surprise_cost(mem, bank) -> dict:
    """Chat is free. A full window rejects in one forward and accepts in two."""
    census = inject_census(mem.inject)
    facts = list(bank.incidental_facts())
    mem.store.clear()

    def _chat():
        fact = dict(_fact(facts[0]))
        fact["kind"] = "chat"
        return mem.remember(fact, surprise=1e9)

    _chat_wrote, chat_forwards = _forward_count(mem, _chat)

    def _warmup():
        return mem.remember(_fact(facts[0]))

    warmup_wrote, warmup_forwards = _forward_count(mem, _warmup)
    mem._surprise_window = [1.0e9] * int(llm_memory.SURPRISE_WINDOW)

    def _below():
        return mem.remember(_fact(facts[1]), surprise=1.0e18)

    below_wrote, below_forwards = _forward_count(mem, _below)
    mem._surprise_window = [-1.0] * int(llm_memory.SURPRISE_WINDOW)

    def _above():
        return mem.remember(_fact(facts[2]))

    above_wrote, above_forwards = _forward_count(mem, _above)
    blob = json.dumps(list(mem.surprise_window()))
    has_text = int(any(piece and piece in blob for piece in (facts[0]["text"], facts[0]["answer"], facts[2]["text"])))
    mem.store.clear()
    named = dict(census["named_parameters"])
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(named.get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "chat_forwards": int(chat_forwards),
        "chat_written": int(bool(_chat_wrote)),
        "warmup_forwards": int(warmup_forwards),
        "warmup_written": int(bool(warmup_wrote)),
        "below_forwards": int(below_forwards),
        "below_written": int(bool(below_wrote)),
        "above_forwards": int(above_forwards),
        "above_written": int(bool(above_wrote)),
        "window_has_text": int(has_text),
        "surprise_window": int(llm_memory.SURPRISE_WINDOW),
        "surprise_min_count": int(llm_memory.SURPRISE_MIN_COUNT),
        "surprise_quantile": float(llm_memory.SURPRISE_QUANTILE),
        "valence_span": float(llm_memory.VALENCE_SPAN),
        "valence_step": float(llm_memory.VALENCE_STEP),
        "clock_span": float(episodic.CLOCK_SPAN),
        "replay_every": int(llm_memory.REPLAY_EVERY),
        "replay_m": int(llm_memory.REPLAY_M),
        "replay_decay": float(llm_memory.REPLAY_DECAY),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """Relative gate, then the non-interference floors. One pass."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    say("gate")
    gate = score_gate(mem, bank, progress=progress)
    gate_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    say("exact")
    exact = exact_spelling(bank, neutral)
    neutral_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    say("cost")
    cost = measure_surprise_cost(mem, bank)
    cost_s = time.perf_counter() - t2
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "gate": gate,
        "neutral": neutral,
        "exact": exact,
        "cost": cost,
        "phases_s": {"gate": gate_s, "neutral": neutral_s, "cost": cost_s},
        "wall_s": float(gate_s + neutral_s + cost_s),
    }


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Count gates apply on fit and held."""
    del held
    gate = result["gate"]
    neutral = result["neutral"]
    exact = result["exact"]
    cost = result["cost"]
    nll = neutral["nll"]
    return [
        (
            "low surprise not written",
            int(gate["low_written"]) <= int(gates["low_written_max"]),
            f"{gate['low_written']}/{gate['low_n']} written, mean surprise {gate['low_mean_surprise']}",
        ),
        (
            "high surprise written",
            int(gate["high_written"]) >= int(gates["high_written_min"]),
            f"{gate['high_written']}/{gate['high_n']} written, mean surprise {gate['high_mean_surprise']}",
        ),
        (
            "seed window",
            int(gate["seed_written"]) == int(gate["seed_n"])
            and int(gate["seed_slots"]) == int(gates["incidental_slots"])
            and int(gate["window"]) == int(gates["window"])
            and gate["bar"] is not None
            and int(gate["window_has_text"]) == 0,
            (
                f"written {gate['seed_written']}/{gate['seed_n']}, "
                f"slots {gate['seed_slots']}, window {gate['window']}, bar {gate['bar']}"
            ),
        ),
        (
            "neutral NLL",
            abs(float(nll["mean_delta"])) <= float(gates["nll_abs_mean_max"])
            and int(nll["n_bad"]) <= int(gates["nll_bad_count_max"]),
            f"mean {nll['mean_delta']:.3f}, above {gates['nll_bad']}: {nll['n_bad']}",
        ),
        (
            "choice flips",
            int(neutral["choice"]["flips"]) <= int(gates["choice_flips_max"]),
            f"{neutral['choice']['flips']}/{neutral['choice']['n']}",
        ),
        (
            "leak",
            int(neutral["leak"]["n_leak"]) <= int(gates["leak_max"]),
            f"{neutral['leak']['n_leak']}/{neutral['leak']['n']}",
        ),
        (
            "phone off-relation takeover",
            float(exact["phone_off_takeover"]) <= float(gates["phone_off_takeover_max"]),
            f"{float(exact['phone_off_takeover']):.3f}",
        ),
        (
            "birthday off-relation takeover",
            float(exact["birthday_off_takeover"]) <= float(gates["birthday_off_takeover_max"]),
            f"{float(exact['birthday_off_takeover']):.3f}",
        ),
        (
            "phone exact",
            int(exact["phone_exact"]) == int(gates["phone_exact"]),
            f"{exact['phone_question_text']!r} vs {exact['phone_stored']!r}",
        ),
        (
            "birthday exact",
            int(exact["birthday_exact"]) == int(gates["birthday_exact"]),
            f"{exact['birthday_question_text']!r} vs {exact['birthday_stored']!r}",
        ),
        (
            "phone off-relation not stored",
            int(exact["phone_off_equal"]) <= int(gates["phone_off_equal_max"]),
            str(exact["phone_off_equal"]),
        ),
        (
            "birthday off-relation not stored",
            int(exact["birthday_off_equal"]) <= int(gates["birthday_off_equal_max"]),
            str(exact["birthday_off_equal"]),
        ),
        (
            "incidental slots",
            int(neutral["incidental_slots"]) == int(gates["incidental_slots"])
            and int(neutral["incidental_written"]) == int(neutral["incidental_n"]),
            f"{neutral['incidental_slots']} slots, written {neutral['incidental_written']}/{neutral['incidental_n']}",
        ),
        (
            "cost",
            int(cost["parameters_new"]) == 0
            and int(cost["chat_forwards"]) == 0
            and int(cost["chat_written"]) == 0
            and int(cost["below_forwards"]) == 1
            and int(cost["below_written"]) == 0
            and int(cost["above_forwards"]) == 2
            and int(cost["above_written"]) == 1
            and int(cost["warmup_forwards"]) == 2
            and int(cost["window_has_text"]) == 0,
            (
                f"new {cost['parameters_new']}, chat {cost['chat_forwards']}, "
                f"below {cost['below_forwards']}/{cost['below_written']}, "
                f"above {cost['above_forwards']}/{cost['above_written']}"
            ),
        ),
    ]


def passes(result, gates, *, held: bool) -> bool:
    return all(ok for _name, ok, _detail in gate_rows(result, gates, held=held))
