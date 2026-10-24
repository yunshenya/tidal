"""Score one experience-valence bank.

The update rule is whatever ``VALENCE_SPAN`` and ``VALENCE_STEP`` are in
``tidal.llm_memory`` when this module runs. This module does not choose them
and does not import the held-out bank. Phone and birthday gates are exact
string equality with the stored answer. The prefix detector is recorded and
is not that gate.
"""
from __future__ import annotations

import time

from tidal import llm_memory
from tidal.subconscious_eval import date_takeover, inject_census, phone_takeover, score_pair
from tidal.subconscious_replay_score import score_neutral


def _mean(xs):
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _fact(item, experience):
    row = {
        "text": item["text"],
        "answer": item["answer"],
        "name": item["name"],
        "rel": item["rel"],
    }
    if experience is not None:
        row["experience"] = experience
    return row


def _margin(mem, item):
    return score_pair(
        mem,
        item["prompt"],
        item["good"],
        item["bad"],
        name=item["name"],
        rel=item["rel"],
    )


def _audit_ok(slot, item, sign: float, repeats: int) -> bool:
    """History is the experiences just written, and the scalar matches their sum."""
    events = [event for event in slot.history if event["kind"] == "experience"]
    if len(events) != int(repeats):
        return False
    if any(event["source"] != item["text"] for event in events):
        return False
    ids = [int(event["episode_id"]) for event in events]
    if ids != sorted(ids) or len(set(ids)) != len(ids):
        return False
    if any("t" not in event for event in events):
        return False
    expected = float(sign) * float(repeats) * float(llm_memory.VALENCE_STEP)
    return abs(float(slot.valence) - expected) < 1e-6


def score_signed(mem, items, sign: float, repeats: int) -> dict:
    """Margin after repeated experiences, minus the same fact at valence 0."""
    rows = []
    for item in items:
        mem.store.clear()
        if not mem.remember(_fact(item, 0)):
            raise RuntimeError(f"{item['id']} anchor was not written")
        control = _margin(mem, item)
        for _ in range(int(repeats)):
            if not mem.remember(_fact(item, sign)):
                raise RuntimeError(f"{item['id']} experience was not written")
        armed = _margin(mem, item)
        slot = mem.store.slots[0]
        delta = float(armed["margin"] - control["margin"])
        rows.append(
            {
                "id": item["id"],
                "delta": delta,
                "valence": float(slot.valence),
                "audit": bool(_audit_ok(slot, item, sign, repeats)),
                "hit": int(armed["retrieval"]["hit"]),
            }
        )
    deltas = [row["delta"] for row in rows]
    return {
        "rows": rows,
        "n": len(rows),
        "mean_delta": _mean(deltas),
        "n_pos": int(sum(value > 0.0 for value in deltas)),
        "n_neg": int(sum(value < 0.0 for value in deltas)),
        "n_audit": int(sum(bool(row["audit"]) for row in rows)),
    }


def score_revise(mem, items, repeats: int) -> dict:
    """Equal contrary experiences must pull valence and the margin back."""
    rows = []
    for item in items:
        mem.store.clear()
        if not mem.remember(_fact(item, 0)):
            raise RuntimeError(f"{item['id']} anchor was not written")
        control = _margin(mem, item)
        for _ in range(int(repeats)):
            mem.remember(_fact(item, 1))
        peak = _margin(mem, item)
        peak_valence = float(mem.store.slots[0].valence)
        for _ in range(int(repeats)):
            mem.remember(_fact(item, -1))
        revised = _margin(mem, item)
        slot = mem.store.slots[0]
        peak_delta = float(peak["margin"] - control["margin"])
        revised_delta = float(revised["margin"] - control["margin"])
        kinds = [event["kind"] for event in slot.history]
        rows.append(
            {
                "id": item["id"],
                "peak_delta": peak_delta,
                "revised_delta": revised_delta,
                "peak_valence": peak_valence,
                "valence": float(slot.valence),
                "moved_back": int(abs(revised_delta) < abs(peak_delta)),
                "valence_revised": int(abs(float(slot.valence)) < abs(peak_valence)),
                "kinds": kinds,
            }
        )
    mem.store.clear()
    return {
        "rows": rows,
        "n": len(rows),
        "n_moved_back": int(sum(row["moved_back"] for row in rows)),
        "n_valence_revised": int(sum(row["valence_revised"] for row in rows)),
        "mean_peak_delta": _mean([row["peak_delta"] for row in rows]),
        "mean_revised_delta": _mean([row["revised_delta"] for row in rows]),
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


def measure_valence_cost(mem, bank) -> dict:
    """An experience is a scalar on the slot. It does not add a forward."""
    census = inject_census(mem.inject)
    item = bank.TOWARD[0]
    mem.store.clear()

    def _write():
        return mem.remember(_fact(item, 1))

    wrote, forwards = _forward_count(mem, _write)
    if not wrote:
        raise RuntimeError("cost probe did not accept the experience")

    def _query():
        return mem._rank(item["prompt"])

    _ranked, query_forwards = _forward_count(mem, _query)
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
        "valence_step": float(llm_memory.VALENCE_STEP),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """Toward, away, revise, and the non-interference floors. One pass."""

    def say(msg):
        if progress is not None:
            progress(msg)

    repeats = int(bank.REPEATS)
    phases = {}
    t0 = time.perf_counter()
    say("toward")
    toward = score_signed(mem, bank.TOWARD, 1.0, repeats)
    phases["toward"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("away")
    away = score_signed(mem, bank.AWAY, -1.0, repeats)
    phases["away"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    say("revise")
    revise = score_revise(mem, bank.TOWARD, repeats)
    phases["revise"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    say("exact")
    exact = exact_spelling(bank, neutral)
    phases["neutral"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    say("cost")
    cost = measure_valence_cost(mem, bank)
    phases["cost"] = time.perf_counter() - t4
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "repeats": repeats,
        "toward": toward,
        "away": away,
        "revise": revise,
        "neutral": neutral,
        "exact": exact,
        "cost": cost,
        "phases_s": phases,
        "wall_s": float(sum(phases.values())),
    }


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Mean floors apply only on the held-out bank."""
    toward = result["toward"]
    away = result["away"]
    revise = result["revise"]
    neutral = result["neutral"]
    exact = result["exact"]
    nll = neutral["nll"]
    rows = [
        (
            "toward experiences",
            int(toward["n_pos"]) >= int(gates["toward_pos_min"]) and int(toward["n_audit"]) == int(toward["n"]),
            f"{toward['n_pos']}/{toward['n']} > 0, mean {toward['mean_delta']:.3f}, audit {toward['n_audit']}/{toward['n']}",
        ),
        (
            "away experiences",
            int(away["n_neg"]) >= int(gates["away_neg_min"]) and int(away["n_audit"]) == int(away["n"]),
            f"{away['n_neg']}/{away['n']} < 0, mean {away['mean_delta']:.3f}, audit {away['n_audit']}/{away['n']}",
        ),
        (
            "contrary revises",
            int(revise["n_moved_back"]) >= int(gates["revise_min"])
            and int(revise["n_valence_revised"]) >= int(gates["revise_min"]),
            (
                f"margin back {revise['n_moved_back']}/{revise['n']}, "
                f"valence back {revise['n_valence_revised']}/{revise['n']}, "
                f"peak {revise['mean_peak_delta']:.3f} revised {revise['mean_revised_delta']:.3f}"
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
    ]
    if held:
        rows.insert(
            1,
            (
                "toward mean",
                float(toward["mean_delta"]) >= float(gates["toward_mean_min"]),
                f"{toward['mean_delta']:.3f} >= {gates['toward_mean_min']}",
            ),
        )
        rows.insert(
            3,
            (
                "away mean",
                float(away["mean_delta"]) <= float(gates["away_mean_max"]),
                f"{away['mean_delta']:.3f} <= {gates['away_mean_max']}",
            ),
        )
    return rows


def passes(result, gates, *, held: bool) -> bool:
    return all(ok for _name, ok, _detail in gate_rows(result, gates, held=held))
