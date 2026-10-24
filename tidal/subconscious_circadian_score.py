"""Score one circadian bank.

The write gain and the decay rate are whatever ``CLOCK_SPAN`` is in
``tidal.episodic`` when this module runs. This module does not choose the
span and does not import the held-out bank. Phone and birthday gates are
exact string equality with the stored answer. The prefix detector is
recorded and is not that gate. The clock log is phase, angle, and gain.
It does not carry fact text.
"""
from __future__ import annotations

import json
import math
import time

from tidal import llm_memory
import tidal.episodic as episodic
from tidal.episodic import clock_decay_rate, clock_write_gain
from tidal.subconscious_eval import date_takeover, inject_census, phone_takeover, score_pair
from tidal.subconscious_replay_score import score_neutral

HIGH_ANGLE = math.pi / 2
LOW_ANGLE = -math.pi / 2
_LOG_KEYS = {"kind", "angle", "phase", "gain"}


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


def _margin(mem, item):
    return score_pair(
        mem,
        item["prompt"],
        item["good"],
        item["bad"],
        name=item["name"],
        rel=item["rel"],
    )


def _write_at(mem, item, angle: float):
    mem.store.clear()
    mem.store.clock_set(angle)
    if not mem.remember(_fact(item)):
        raise RuntimeError(f"{item['id']} was not written")
    return mem.store.slots[0]


def _audit_ok(mem, item) -> bool:
    """The log is a phase record. It does not contain the fact."""
    log = mem.store.clock_log()
    if not log:
        return False
    if any(set(event) != _LOG_KEYS for event in log):
        return False
    blob = json.dumps(log)
    for piece in (item["text"], item["answer"], item["prompt"], item["name"]):
        if piece and piece in blob:
            return False
    return True


def _gain_matches(slot, phase: float) -> bool:
    expected = float(clock_write_gain(phase))
    return abs(float(slot.strength) - expected) < 1e-9 and abs(float(slot.clock_gain) - expected) < 1e-9


def score_phases(mem, items, *, progress=None) -> dict:
    """Same facts at angle 0, +pi/2, and -pi/2. Decay is measured before refresh."""
    rows = []
    for index, item in enumerate(items):
        if progress is not None:
            progress(f"phase {index} {item['id']}")
        zero = _write_at(mem, item, 0.0)
        zero_audit = _audit_ok(mem, item)
        zero_match = _gain_matches(zero, 0.0)
        control = _margin(mem, item)
        high = _write_at(mem, item, HIGH_ANGLE)
        high_audit = _audit_ok(mem, item)
        high_match = _gain_matches(high, mem.store.clock_phase())
        high_margin = _margin(mem, item)
        high_strength = float(high.strength)
        low = _write_at(mem, item, LOW_ANGLE)
        low_audit = _audit_ok(mem, item)
        low_match = _gain_matches(low, mem.store.clock_phase())
        low_margin = _margin(mem, item)
        low_strength = float(low.strength)
        _write_at(mem, item, 0.0)
        mem.store.recompute_decay(llm_memory.REPLAY_DECAY)
        decay_zero = float(mem.store.slots[0].strength)
        _write_at(mem, item, HIGH_ANGLE)
        mem.store.recompute_decay(llm_memory.REPLAY_DECAY)
        decay_high = float(mem.store.slots[0].strength)
        _write_at(mem, item, LOW_ANGLE)
        mem.store.recompute_decay(llm_memory.REPLAY_DECAY)
        decay_low = float(mem.store.slots[0].strength)
        high_delta = float(high_margin["margin"] - control["margin"])
        low_delta = float(low_margin["margin"] - control["margin"])
        rows.append(
            {
                "id": item["id"],
                "high_delta": high_delta,
                "low_delta": low_delta,
                "strength_high": high_strength,
                "strength_low": low_strength,
                "decay_high": decay_high,
                "decay_zero": decay_zero,
                "decay_low": decay_low,
                "strength_match": bool(zero_match and high_match and low_match),
                "decay_order": bool(decay_high > decay_zero > decay_low and decay_zero == 0.5),
                "audit": bool(zero_audit and high_audit and low_audit),
                "hit": int(high_margin["retrieval"]["hit"]),
            }
        )
    mem.store.clear()
    high_gains = [row["strength_high"] for row in rows]
    low_gains = [row["strength_low"] for row in rows]
    content_free = bool(rows) and len(set(high_gains)) == 1 and len(set(low_gains)) == 1
    return {
        "rows": rows,
        "n": len(rows),
        "mean_high": _mean([row["high_delta"] for row in rows]),
        "mean_low": _mean([row["low_delta"] for row in rows]),
        "n_high_pos": int(sum(row["high_delta"] > 0.0 for row in rows)),
        "n_low_neg": int(sum(row["low_delta"] < 0.0 for row in rows)),
        "n_decay_order": int(sum(row["decay_order"] for row in rows)),
        "n_strength": int(sum(row["strength_match"] for row in rows)),
        "n_audit": int(sum(row["audit"] for row in rows)),
        "phase0_exact": bool(rows) and all(row["decay_zero"] == 0.5 for row in rows),
        "mean_decay_gap": _mean([row["decay_high"] - row["decay_low"] for row in rows]),
        "content_free": content_free,
        "write_gain_high": high_gains[0] if high_gains else None,
        "decay_rate_high": float(clock_decay_rate(math.sin(HIGH_ANGLE))),
        "decay_rate_low": float(clock_decay_rate(math.sin(LOW_ANGLE))),
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


def _forward_count(mem, fn):
    calls = {"n": 0}

    def _hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = mem.model.register_forward_hook(_hook)
    try:
        result = fn()
    finally:
        handle.remove()
        mem._delta = None
    return result, int(calls["n"])


def measure_clock_cost(mem, bank) -> dict:
    """The clock is a scalar on the store. A tick does not run the language model."""
    census = inject_census(mem.inject)
    item = bank.PHASE[0]
    mem.store.clear()

    def _write():
        return mem.remember(_fact(item))

    wrote, forwards = _forward_count(mem, _write)
    if not wrote:
        raise RuntimeError("cost probe did not accept the write")

    def _query():
        return mem._rank(item["prompt"])

    _ranked, query_forwards = _forward_count(mem, _query)

    def _tick():
        return mem.clock_tick()

    _event, tick_forwards = _forward_count(mem, _tick)
    mem.store.clear()
    named = dict(census["named_parameters"])
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(named.get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "remember_forwards": forwards,
        "query_forwards": query_forwards,
        "clock_tick_forwards": tick_forwards,
        "clock_span": float(episodic.CLOCK_SPAN),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """Phase margins, decay order, and the non-interference floors. One pass."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    say("phases")
    phases = score_phases(mem, bank.PHASE, progress=progress)
    phase_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    say("exact")
    exact = exact_spelling(bank, neutral)
    neutral_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    say("cost")
    cost = measure_clock_cost(mem, bank)
    cost_s = time.perf_counter() - t2
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "phases": phases,
        "neutral": neutral,
        "exact": exact,
        "cost": cost,
        "phases_s": {"phases": phase_s, "neutral": neutral_s, "cost": cost_s},
        "wall_s": float(phase_s + neutral_s + cost_s),
    }


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Mean floors apply only on the held-out bank."""
    phases = result["phases"]
    neutral = result["neutral"]
    exact = result["exact"]
    nll = neutral["nll"]
    rows = [
        (
            "high phase",
            int(phases["n_high_pos"]) >= int(gates["high_pos_min"]),
            f"{phases['n_high_pos']}/{phases['n']} > 0, mean {phases['mean_high']:.3f}",
        ),
    ]
    if held:
        rows.append(
            (
                "high mean",
                float(phases["mean_high"]) >= float(gates["high_mean_min"]),
                f"{phases['mean_high']:.3f} >= {gates['high_mean_min']}",
            )
        )
    rows.append(
        (
            "low phase",
            int(phases["n_low_neg"]) >= int(gates["low_neg_min"]),
            f"{phases['n_low_neg']}/{phases['n']} < 0, mean {phases['mean_low']:.3f}",
        )
    )
    if held:
        rows.append(
            (
                "low mean",
                float(phases["mean_low"]) <= float(gates["low_mean_max"]),
                f"{phases['mean_low']:.3f} <= {gates['low_mean_max']}",
            )
        )
    rows.append(
        (
            "decay order",
            int(phases["n_decay_order"]) >= int(gates["decay_order_min"]) and bool(phases["phase0_exact"]),
            (
                f"{phases['n_decay_order']}/{phases['n']} high > phase0 > low, "
                f"phase0 exact {int(bool(phases['phase0_exact']))}, "
                f"gap {phases['mean_decay_gap']:.3f}"
            ),
        )
    )
    if held:
        rows.append(
            (
                "decay gap mean",
                float(phases["mean_decay_gap"]) >= float(gates["decay_gap_mean_min"]),
                f"{phases['mean_decay_gap']:.3f} >= {gates['decay_gap_mean_min']}",
            )
        )
    rows.extend(
        [
            (
                "write gain",
                int(phases["n_strength"]) == int(phases["n"]),
                f"{phases['n_strength']}/{phases['n']} match the clock gain",
            ),
            (
                "content-free",
                bool(phases["content_free"]) and int(phases["n_audit"]) == int(phases["n"]),
                f"equal gains {int(bool(phases['content_free']))}, audit {phases['n_audit']}/{phases['n']}",
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
        ]
    )
    return rows


def passes(result, gates, *, held: bool) -> bool:
    return all(ok for _name, ok, _detail in gate_rows(result, gates, held=held))
