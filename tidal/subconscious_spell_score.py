"""Score exact phone and birthday spelling on one bank.

The on-relation gate is string equality with the stored answer. The prefix
detector is recorded and is not that gate. This module does not choose a
hyperparameter and does not import the held-out bank.
"""
from __future__ import annotations

import time

from tidal.subconscious_eval import date_takeover, inject_census, phone_takeover
from tidal.subconscious_replay_score import _forward_count, score_neutral


def _mean(xs):
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _off(rows, question):
    return [row for row in rows if row["prompt"] != question]


def _exact(bank, neutral) -> dict:
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


def measure_cost(mem, bank) -> dict:
    """One remember and one phone question. Spelling adds no parameter and no second query."""
    census = inject_census(mem.inject)
    fact = bank.incidental_facts()[0]
    mem.store.clear()

    def _remember():
        return mem.remember(
            {
                "text": fact["text"],
                "answer": fact["answer"],
                "name": fact["name"],
                "rel": fact["rel"],
            }
        )

    wrote, remember_forwards = _forward_count(mem, _remember)
    if not wrote:
        raise RuntimeError("cost probe did not accept the remember")
    mem.store.clear()
    if not mem.remember(bank.PHONE_FACT):
        raise RuntimeError("cost probe did not accept the phone")

    def _query():
        return mem.generate(bank.PHONE_QUESTION, max_new_tokens=32)

    text, query_forwards = _forward_count(mem, _query)
    mem.store.clear()
    named = dict(census["named_parameters"])
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(named.get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "remember_forwards": int(remember_forwards),
        "query_forwards": int(query_forwards),
        "query_text": text,
    }


def score_spell(mem, bank, *, progress=None) -> dict:
    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    say("exact")
    exact = _exact(bank, neutral)
    say("cost")
    cost = measure_cost(mem, bank)
    mem.store.clear()
    wall = time.perf_counter() - t0
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "neutral": neutral,
        "exact": exact,
        "cost": cost,
        "wall_s": float(wall),
    }


def gate_rows(result, gates, *, direction=None):
    """Rows of (name, passed, detail). Direction floors apply when ``direction`` is set."""
    exact = result["exact"]
    neutral = result["neutral"]
    nll = neutral["nll"]
    cost = result["cost"]
    rows = [
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
            "incidental slots",
            int(neutral["incidental_slots"]) == int(gates["incidental_slots"])
            and int(neutral["incidental_written"]) == int(neutral["incidental_n"]),
            f"{neutral['incidental_slots']} slots, max cosine {float(neutral['incidental_max_key_cosine']):.3f}",
        ),
        (
            "new parameters",
            int(cost["parameters_new"]) == int(gates["parameters_new"]),
            str(cost["parameters_new"]),
        ),
        (
            "remember forwards",
            int(cost["remember_forwards"]) == int(gates["remember_forwards"]),
            str(cost["remember_forwards"]),
        ),
        (
            "query forwards",
            int(cost["query_forwards"]) == int(gates["query_forwards"]),
            str(cost["query_forwards"]),
        ),
    ]
    if direction is not None:
        prefer = direction["prefer"]["summary"]
        avoid = direction["avoid"]["summary"]
        rows.extend(
            [
                (
                    "prefer mean",
                    float(prefer["mean_delta"]) >= float(gates["prefer_mean_min"]),
                    f"{prefer['mean_delta']:.3f} >= {gates['prefer_mean_min']}",
                ),
                (
                    "avoid mean",
                    float(avoid["mean_delta"]) >= float(gates["avoid_mean_min"]),
                    f"{avoid['mean_delta']:.3f} >= {gates['avoid_mean_min']}",
                ),
                (
                    "forbidden logprob down",
                    int(avoid["n_stored_down"]) >= int(gates["forbidden_down_min"]),
                    f"{avoid['n_stored_down']}/{avoid['n']} down, mean stored lift {avoid['mean_stored_lift']:.3f}",
                ),
            ]
        )
    return rows


def passes(result, gates, *, direction=None) -> bool:
    return all(ok for _name, ok, _detail in gate_rows(result, gates, direction=direction))
