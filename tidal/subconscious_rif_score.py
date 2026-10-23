"""Score one retrieval-induced-forgetting bank.

Competitor decay is whatever ``RIF_COMPETITOR_KEEP`` is in ``tidal.llm_memory``
when this module runs. This module does not choose that keep and does not
import the held-out bank.
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


def _restore(mem, values):
    for slot, value in zip(mem.store.slots, values):
        slot.strength = float(value)


def _max_key_cosine(mem):
    keys = [slot.k for slot in mem.store.slots]
    best = 0.0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            best = max(best, _cosine(keys[i], keys[j]))
    return best


def _fact(row, *, update: bool = False) -> dict:
    if update:
        return {
            "text": row["update_text"],
            "answer": row["update_answer"],
            "name": row["name"],
            "rel": row["rel"],
        }
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


def score_quartets(mem, bank) -> dict:
    """Competitor drop versus a strength-1 control. One decay query per quartet.

    The control restores strength 1 before every score, so the measurement
    itself does not accumulate decay. The treated arm applies one real top-1
    of the target prompt, then restores that snapshot before each score.
    """
    groups = []
    for index, group in enumerate(bank.QUARTETS):
        mem.store.clear()
        for fact in group:
            if not mem.remember(_fact(fact)):
                raise RuntimeError(f"{fact['id']} was not written")
        slots_n = len(mem.store)
        maxcos = _max_key_cosine(mem) if slots_n >= 2 else 0.0
        held_slots = list(mem.store._slots)
        mem.store.clear()
        nomem = [_pair_margin(mem, fact)["margin"] for fact in group]
        mem.store._slots = held_slots
        _restore(mem, [1.0] * slots_n)
        control = []
        for fact in group:
            _restore(mem, [1.0] * slots_n)
            control.append(_pair_margin(mem, fact)["margin"] - nomem[len(control)])
        _restore(mem, [1.0] * slots_n)
        decay = mem._rank(group[0]["prompt"])
        snapshot = [float(slot.strength) for slot in mem.store.slots]
        items = []
        for fact, base, ctrl in zip(group, nomem, control):
            _restore(mem, snapshot)
            scored = _pair_margin(mem, fact)
            treated = float(scored["margin"] - base)
            items.append(
                {
                    "id": fact["id"],
                    "role": fact["role"],
                    "name": fact["name"],
                    "rel": fact["rel"],
                    "control_delta": float(ctrl),
                    "treated_delta": treated,
                    "drop": float(ctrl - treated),
                    "top": scored["retrieval"]["top"],
                    "hit": int(scored["retrieval"]["hit"]),
                    "gap": scored["retrieval"]["gap"],
                }
            )
        winner = next(row for row in items if row["role"] == "target")
        competitors = [row for row in items if row["role"] != "target"]
        geometry_ok = (
            slots_n == len(group)
            and len(group) == 4
            and decay.get("gap") is not None
            and float(decay["gap"]) > 0.0
            and decay.get("top") == (group[0]["name"], group[0]["rel"])
            and all(row["hit"] == 1 for row in items)
        )
        groups.append(
            {
                "index": index,
                "slots": slots_n,
                "max_key_cosine": maxcos,
                "decay_top": list(decay["top"]) if decay.get("top") else None,
                "decay_gap": decay.get("gap"),
                "decay_confidence": decay.get("confidence"),
                "strengths": snapshot,
                "items": items,
                "winner_drop": float(winner["drop"]),
                "competitor_drops": [float(row["drop"]) for row in competitors],
                "geometry_ok": bool(geometry_ok),
            }
        )
        mem.store.clear()
    competitor_drops = [drop for group in groups for drop in group["competitor_drops"]]
    winner_abs = [abs(float(group["winner_drop"])) for group in groups]
    return {
        "quartets": groups,
        "competitor_drop_min": min(competitor_drops) if competitor_drops else None,
        "competitor_drop_mean": _mean(competitor_drops),
        "winner_abs_drop_max": max(winner_abs) if winner_abs else None,
        "geometry_ok": bool(groups) and all(group["geometry_ok"] for group in groups),
    }


def score_overwrite(mem, bank) -> dict:
    """Same-key correction. The slot must store the new value, and the new word must win."""
    rows = []
    for item in bank.OVERWRITE:
        mem.store.clear()
        nomem = score_pair(mem, item["prompt"], item["good"], item["bad"])
        mem.store.clear()
        if not mem.remember(_fact(item)):
            raise RuntimeError(f"{item['id']} old value was not written")
        if not mem.remember(_fact(item, update=True)):
            raise RuntimeError(f"{item['id']} new value was not written")
        slots = list(mem.store.slots)
        new_v = mem.encode_value(item["update_text"], item["update_answer"])
        old_v = mem.encode_value(item["text"], item["answer"])
        if len(slots) == 1:
            cos_new = _cosine(slots[0].v, new_v)
            cos_old = _cosine(slots[0].v, old_v)
            strength = float(slots[0].strength)
        else:
            cos_new = float("nan")
            cos_old = float("nan")
            strength = float("nan")
        scored = score_pair(mem, item["prompt"], item["good"], item["bad"])
        rows.append(
            {
                "id": item["id"],
                "slots": len(slots),
                "strength": strength,
                "cos_new": cos_new,
                "cos_old": cos_old,
                "margin": float(scored["margin"]),
                "nomem_margin": float(nomem["margin"]),
                "delta": float(scored["margin"] - nomem["margin"]),
                "follows_new": bool(len(slots) == 1 and cos_new >= 0.999 and strength == 1.0),
                "positive": bool(scored["margin"] > 0.0 and len(slots) == 1 and cos_new >= 0.999),
            }
        )
    mem.store.clear()
    return {
        "rows": rows,
        "n": len(rows),
        "n_pos": int(sum(row["positive"] for row in rows)),
        "n_follows": int(sum(row["follows_new"] for row in rows)),
        "mean_margin": _mean([row["margin"] for row in rows]),
        "mean_delta": _mean([row["delta"] for row in rows]),
    }


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
    return {"prefer": {"rows": prefer, "summary": _family(prefer)}, "avoid": {"rows": avoid, "summary": _family(avoid)}}


def score_neutral(mem, bank) -> dict:
    incidental = list(bank.incidental_facts())
    mem.store.clear()
    written = 0
    for fact in incidental:
        written += int(bool(mem.remember(fact)))
    slots = len(mem.store)
    maxcos = _max_key_cosine(mem) if slots >= 2 else 0.0
    snap = list(mem.store._slots)
    mem.store.clear()
    nll_base = [score_one(mem, item["prompt"], item["continuation"]) for item in bank.NEUTRAL_NLL]
    choice_base = [score_pair(mem, item["prompt"], item["good"], item["bad"]) for item in bank.NEUTRAL_CHOICE]
    nll_rows = []
    for item, base in zip(bank.NEUTRAL_NLL, nll_base):
        mem.store._slots = snap
        _restore(mem, [1.0] * len(snap))
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
        mem.store._slots = snap
        _restore(mem, [1.0] * len(snap))
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
        mem.remember(fact)
    leak_answers = [row["answer"] for row in leak_src]
    leak_rows = []
    for prompt in bank.NEUTRAL_GEN:
        text = mem.generate(prompt, max_new_tokens=GREEDY_TOKENS)
        hits = [answer for answer in leak_answers if word_hit(text, answer)]
        leak_rows.append({"prompt": prompt, "text": text, "hits": hits})
    script_prompts = list(bank.NEUTRAL_GEN[:4]) + [bank.PHONE_QUESTION, bank.BIRTHDAY_QUESTION]
    mem.store.clear()
    phone_written = bool(mem.remember(bank.PHONE_FACT))
    phone_rows = []
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        phone_rows.append({"prompt": prompt, "text": text, "takeover": int(phone_takeover(text))})
    mem.store.clear()
    birthday_written = bool(mem.remember(bank.BIRTHDAY_FACT))
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


def measure_rif_cost(mem, bank) -> dict:
    """The decay is three strength multiplies inside the query that already runs."""
    census = inject_census(mem.inject)
    group = bank.QUARTETS[0]
    mem.store.clear()
    for fact in group:
        mem.remember(_fact(fact))
    _restore(mem, [1.0] * len(mem.store))

    def once():
        return mem._rank(group[0]["prompt"])

    _rank, forwards = _forward_count(mem, once)
    losers = [i for i, slot in enumerate(mem.store.slots) if slot.name != group[0]["name"]]
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        mem.store.scale_strength(losers, 1.0)
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
        "rank_forwards": forwards,
        "rank_gap": _rank.get("gap"),
        "scale_strength_median_s": float(samples[len(samples) // 2]),
        "losers": len(losers),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """One pass over quartets, overwrite, direction, and the non-interference floors."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    phases = {}
    say("quartets")
    quartets = score_quartets(mem, bank)
    phases["quartets"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("overwrite")
    overwrite = score_overwrite(mem, bank)
    phases["overwrite"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    say("direction")
    direction = score_direction(mem, bank)
    phases["direction"] = time.perf_counter() - t2
    t3 = time.perf_counter()
    say("neutral")
    neutral = score_neutral(mem, bank)
    phases["neutral"] = time.perf_counter() - t3
    t4 = time.perf_counter()
    say("cost")
    cost = measure_rif_cost(mem, bank)
    phases["cost"] = time.perf_counter() - t4
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "quartets": quartets,
        "overwrite": overwrite,
        "direction": direction,
        "neutral": neutral,
        "cost": cost,
        "phases_s": phases,
        "wall_s": float(sum(phases.values())),
    }


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Mean floors apply only on the held-out bank."""
    quartets = result["quartets"]
    overwrite = result["overwrite"]
    prefer = result["direction"]["prefer"]["summary"]
    avoid = result["direction"]["avoid"]["summary"]
    neutral = result["neutral"]
    scripted = neutral["scripted"]
    nll = neutral["nll"]
    weaken = (
        bool(quartets["geometry_ok"])
        and float(quartets["competitor_drop_min"]) >= float(gates["competitor_drop_min"])
        and float(quartets["competitor_drop_mean"]) >= float(gates["competitor_mean_drop_min"])
        and float(quartets["winner_abs_drop_max"]) <= float(gates["winner_abs_drop_max"])
    )
    rows = [
        (
            "competitors weaken",
            weaken,
            (
                f"min drop {quartets['competitor_drop_min']:.3f}, "
                f"mean {quartets['competitor_drop_mean']:.3f}, "
                f"winner abs {quartets['winner_abs_drop_max']:.3f}, "
                f"geometry {'yes' if quartets['geometry_ok'] else 'no'}"
            ),
        ),
        (
            "overwrite follows new",
            int(overwrite["n_pos"]) >= int(gates["overwrite_pos_min"])
            and int(overwrite["n_follows"]) == int(overwrite["n"]),
            f"{overwrite['n_pos']}/{overwrite['n']} margin > 0, slots follow new {overwrite['n_follows']}/{overwrite['n']}",
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
        (
            "leak",
            int(neutral["leak"]["n_leak"]) <= int(gates["leak_max"]),
            f"{neutral['leak']['n_leak']}/{neutral['leak']['n']}",
        ),
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
            3,
            (
                "prefer mean",
                float(prefer["mean_delta"]) >= float(gates["prefer_mean_min"]),
                f"{prefer['mean_delta']:.3f} >= {gates['prefer_mean_min']}",
            ),
        )
        rows.insert(
            5,
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
