"""Score one fixed-budget replay bank.

N and m are whatever ``REPLAY_EVERY`` and ``REPLAY_M`` are in
``tidal.llm_memory`` when this module runs. This module does not choose
them and does not import the held-out bank.
"""
from __future__ import annotations

import time

import numpy as np

from tidal.subconscious_data import word_hit
from tidal.subconscious_eval import (
    GREEDY_TOKENS,
    SCRIPTED_TOKENS,
    date_takeover,
    inject_census,
    phone_takeover,
    score_one,
    score_pair,
)


def _mean(xs):
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _cosine(a, b) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _max_key_cosine(mem) -> float:
    keys = [slot.k for slot in mem.store.slots]
    best = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            best = max(best, _cosine(keys[i], keys[j]))
    return best


def _fact(row) -> dict:
    return {
        "text": row["text"],
        "answer": row["answer"],
        "name": row["name"],
        "rel": row["rel"],
    }


def _pair_margin(mem, row) -> dict:
    return score_pair(
        mem,
        row["prompt"],
        row["good"],
        row["bad"],
        name=row["name"],
        rel=row["rel"],
    )


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


def score_replay(mem, bank) -> dict:
    """One replay cycle on the probe, against the decay-only prefix of that cycle.

    The control is recorded inside the real consolidate: decay has been
    recomputed, and the refresh and the merge have not run yet. Production
    always continues into both. There is no switch.
    """
    facts = list(bank.replay_facts())
    mem.store.clear()
    for fact in facts:
        if not mem.remember(_fact(fact)):
            raise RuntimeError(f"replay probe {fact.get('name')} was not written")
    info = dict(mem.last_replay or {})
    if not info:
        raise RuntimeError("replay probe did not consolidate")
    info["accepted"] = int(mem._accepted)
    info["slots"] = len(mem.store)
    info["slot_drop"] = int(info["control_n_slots"]) - int(info["replay_n_slots"])
    info["near_dup_drop"] = int(info["control_near_dup_slots"]) - int(info["replay_near_dup_slots"])
    mem.store.clear()
    return info


def _direction_one(mem, item) -> dict:
    mem.store.clear()
    nomem = _pair_margin(mem, item)
    mem.store.clear()
    if not mem.remember(_fact(item)):
        raise RuntimeError(f"{item['id']} was not written")
    if len(mem.store) != 1:
        raise RuntimeError(f"{item['id']} did not occupy one slot")
    scored = _pair_margin(mem, item)
    stored_side = "bad_mean" if item["family"] == "avoid" else "good_mean"
    return {
        "id": item["id"],
        "family": item["family"],
        "delta": float(scored["margin"] - nomem["margin"]),
        "stored_lift": float(scored[stored_side] - nomem[stored_side]),
        "margin": float(scored["margin"]),
        "nomem_margin": float(nomem["margin"]),
        "hit": int(scored["retrieval"]["hit"]),
        "strength": float(mem.store.slots[0].strength),
    }


def _family(rows) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    return {
        "n": len(rows),
        "mean_delta": _mean(deltas),
        "n_delta_pos": int(sum(value > 0.0 for value in deltas)),
        "mean_stored_lift": _mean([row["stored_lift"] for row in rows]),
        "n_stored_down": int(sum(float(row["stored_lift"]) < 0.0 for row in rows)),
    }


def score_direction(mem, bank) -> dict:
    prefer = [_direction_one(mem, item) for item in bank.PREFER]
    avoid = [_direction_one(mem, item) for item in bank.AVOID]
    mem.store.clear()
    return {
        "prefer": {"rows": prefer, "summary": _family(prefer)},
        "avoid": {"rows": avoid, "summary": _family(avoid)},
    }


def score_neutral(mem, bank) -> dict:
    incidental = list(bank.incidental_facts())
    mem.store.clear()
    written = 0
    for fact in incidental:
        written += int(bool(mem.remember(_fact(fact))))
    slots = len(mem.store)
    maxcos = _max_key_cosine(mem) if slots >= 2 else 0.0
    snap = list(mem.store._slots)
    mem.store.clear()
    nll_base = [score_one(mem, item["prompt"], item["continuation"]) for item in bank.NEUTRAL_NLL]
    choice_base = [score_pair(mem, item["prompt"], item["good"], item["bad"]) for item in bank.NEUTRAL_CHOICE]
    nll_rows = []
    for item, base in zip(bank.NEUTRAL_NLL, nll_base):
        mem.store._slots = list(snap)
        armed = score_one(mem, item["prompt"], item["continuation"])
        nll_rows.append(
            {
                "id": item["id"],
                "delta": float(armed["nll"] - base["nll"]),
                "nll_nomem": float(base["nll"]),
                "nll_inject": float(armed["nll"]),
                "confidence": armed["retrieval"]["confidence"],
            }
        )
    choice_rows = []
    for item, base in zip(bank.NEUTRAL_CHOICE, choice_base):
        mem.store._slots = list(snap)
        armed = score_pair(mem, item["prompt"], item["good"], item["bad"])
        choice_rows.append(
            {
                "id": item["id"],
                "nomem_choose": int(base["choose"]),
                "inject_choose": int(armed["choose"]),
                "delta": float(armed["margin"] - base["margin"]),
                "flip": int(base["choose"] == 1 and armed["choose"] == 0),
            }
        )
    leak_src = incidental[: int(bank.LEAK_STORE_N)]
    mem.store.clear()
    for fact in leak_src:
        mem.remember(_fact(fact))
    leak_answers = [row["answer"] for row in leak_src]
    leak_rows = []
    for prompt in bank.NEUTRAL_GEN:
        text = mem.generate(prompt, max_new_tokens=GREEDY_TOKENS)
        hits = [answer for answer in leak_answers if word_hit(text, answer)]
        leak_rows.append({"prompt": prompt, "text": text, "hits": hits})
    script_prompts = list(bank.NEUTRAL_GEN[:4]) + [bank.PHONE_QUESTION, bank.BIRTHDAY_QUESTION]
    mem.store.clear()
    phone_written = bool(mem.remember(_fact(bank.PHONE_FACT)))
    phone_rows = []
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        phone_rows.append({"prompt": prompt, "text": text, "takeover": int(phone_takeover(text))})
    mem.store.clear()
    birthday_written = bool(mem.remember(_fact(bank.BIRTHDAY_FACT)))
    date_rows = []
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        date_rows.append({"prompt": prompt, "text": text, "takeover": int(date_takeover(text))})
    mem.store.clear()
    nll_deltas = [row["delta"] for row in nll_rows]
    return {
        "incidental_written": written,
        "incidental_n": len(incidental),
        "incidental_slots": slots,
        "incidental_max_key_cosine": maxcos,
        "nll": {
            "rows": nll_rows,
            "mean_delta": _mean(nll_deltas),
            "n_bad": int(sum(value > 0.5 for value in nll_deltas)),
        },
        "choice": {
            "rows": choice_rows,
            "flips": int(sum(row["flip"] for row in choice_rows)),
            "n": len(choice_rows),
        },
        "leak": {
            "rows": leak_rows,
            "n_leak": int(sum(bool(row["hits"]) for row in leak_rows)),
            "n": len(leak_rows),
        },
        "scripted": {
            "phone_written": phone_written,
            "birthday_written": birthday_written,
            "phone_unrelated": _mean([row["takeover"] for row in phone_rows if row["prompt"] != bank.PHONE_QUESTION]),
            "phone_question": next(row["takeover"] for row in phone_rows if row["prompt"] == bank.PHONE_QUESTION),
            "date_unrelated": _mean(
                [
                    row["takeover"]
                    for row in date_rows
                    if row["prompt"] != bank.PHONE_QUESTION and row["prompt"] != bank.BIRTHDAY_QUESTION
                ]
            ),
            "birthday_question": next(row["takeover"] for row in date_rows if row["prompt"] == bank.BIRTHDAY_QUESTION),
            "phone": phone_rows,
            "birthday": date_rows,
        },
    }


def measure_replay_cost(mem, bank) -> dict:
    """Replay is a bounded scan. It does not add a language-model forward."""
    census = inject_census(mem.inject)
    facts = list(bank.replay_facts())
    mem.store.clear()
    for fact in facts[:-1]:
        mem.remember(_fact(fact))

    def _consolidate_once():
        return mem.remember(_fact(facts[-1]))

    wrote, forwards = _forward_count(mem, _consolidate_once)
    if not wrote:
        raise RuntimeError("cost probe did not accept the last write")
    # The remember above encodes the key and the value, then consolidates.
    # A second consolidate is the scan alone and must not forward.
    def _scan_only():
        mem._consolidate()

    _ignored, scan_forwards = _forward_count(mem, _scan_only)
    replay_m = int(mem.last_replay["m"])
    rank_prompt = bank.PREFER[0]["prompt"]

    def _query():
        return mem._rank(rank_prompt)

    _ranked, query_forwards = _forward_count(mem, _query)
    samples = []
    for _ in range(20):
        t0 = time.perf_counter()
        mem._consolidate()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    mem.store.clear()
    named = dict(census["named_parameters"])
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(named.get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "remember_forwards": forwards,
        "consolidate_forwards": scan_forwards,
        "query_forwards": query_forwards,
        "consolidate_median_s": float(samples[len(samples) // 2]),
        "replay_m": replay_m,
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """Replay probe, direction, and the non-interference floors. One pass."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    phases = {}
    say("replay")
    replay = score_replay(mem, bank)
    phases["replay"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("direction")
    direction = score_direction(mem, bank)
    phases["direction"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    phases["neutral"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    say("cost")
    cost = measure_replay_cost(mem, bank)
    phases["cost"] = time.perf_counter() - t3
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "replay": replay,
        "direction": direction,
        "neutral": neutral,
        "cost": cost,
        "phases_s": phases,
        "wall_s": float(sum(phases.values())),
    }


def _replay_ok(replay, gates) -> bool:
    gain = float(replay["salient_gain"] or 0.0)
    drop = int(replay["slot_drop"])
    return gain >= float(gates["salient_gain_min"]) or drop >= int(gates["near_dup_drop_min"])


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Mean floors apply only on the held-out bank."""
    replay = result["replay"]
    prefer = result["direction"]["prefer"]["summary"]
    avoid = result["direction"]["avoid"]["summary"]
    neutral = result["neutral"]
    scripted = neutral["scripted"]
    nll = neutral["nll"]
    weaken = _replay_ok(replay, gates)
    rows = [
        (
            "replay consolidates",
            weaken,
            (
                f"salient gain {float(replay['salient_gain']):.3f}, "
                f"slot drop {int(replay['slot_drop'])}, "
                f"near-dup drop {int(replay['near_dup_drop'])}"
            ),
        ),
        (
            "prefer margin",
            int(prefer["n_delta_pos"]) >= int(gates["prefer_delta_pos_min"]),
            f"{prefer['n_delta_pos']}/{prefer['n']} > 0, mean {prefer['mean_delta']:.3f}",
        ),
        (
            "avoid margin",
            int(avoid["n_delta_pos"]) >= int(gates["avoid_delta_pos_min"]),
            f"{avoid['n_delta_pos']}/{avoid['n']} > 0, mean {avoid['mean_delta']:.3f}",
        ),
        (
            "forbidden logprob down",
            int(avoid["n_stored_down"]) >= int(gates["forbidden_down_min"]),
            f"{avoid['n_stored_down']}/{avoid['n']} down, mean stored lift {avoid['mean_stored_lift']:.3f}",
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
        ("leak", int(neutral["leak"]["n_leak"]) <= int(gates["leak_max"]), f"{neutral['leak']['n_leak']}/{neutral['leak']['n']}"),
        (
            "phone unrelated",
            float(scripted["phone_unrelated"]) <= float(gates["phone_unrelated_max"]),
            f"{float(scripted['phone_unrelated']):.3f}",
        ),
        (
            "birthday unrelated",
            float(scripted["date_unrelated"]) <= float(gates["birthday_unrelated_max"]),
            f"{float(scripted['date_unrelated']):.3f}",
        ),
        (
            "phone question",
            int(scripted["phone_question"]) == int(gates["phone_question"]),
            str(scripted["phone_question"]),
        ),
        (
            "birthday question",
            int(scripted["birthday_question"]) == int(gates["birthday_question"]),
            str(scripted["birthday_question"]),
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
            2,
            (
                "prefer mean",
                float(prefer["mean_delta"]) >= float(gates["prefer_mean_min"]),
                f"{prefer['mean_delta']:.3f} >= {gates['prefer_mean_min']}",
            ),
        )
        rows.insert(
            4,
            (
                "avoid mean",
                float(avoid["mean_delta"]) >= float(gates["avoid_mean_min"]),
                f"{avoid['mean_delta']:.3f} >= {gates['avoid_mean_min']}",
            ),
        )
    return rows


def passes(result, gates, *, held: bool) -> bool:
    rows = gate_rows(result, gates, held=held)
    if not held:
        rows = [row for row in rows if row[0] != "forbidden logprob down"]
    return all(ok for _name, ok, _detail in rows)
