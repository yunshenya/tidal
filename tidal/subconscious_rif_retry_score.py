"""Score one retrieval-induced-forgetting bank.

The span is whatever ``RIF_SPAN`` is in ``tidal.llm_memory`` when this module
runs. This module does not choose that span and does not import the held-out
bank. Competitors are the other slots that share the winner's relation, and
only when that set has one, two, or three members.
"""
from __future__ import annotations

import json
import time

import numpy as np

import tidal.episodic as episodic
import tidal.llm_memory as llm_memory
from tidal.subconscious_eval import inject_census, score_pair


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


def score_quartets(mem, bank) -> dict:
    """Competitor drop versus a strength-1 control. One retrieval per quartet.

    The control restores strength 1 before every score, so the measurement
    itself does not accumulate suppression. The treated arm applies one real
    top-1 of the target prompt, then restores that snapshot before each score.
    """
    groups = []
    for index, group in enumerate(bank.QUARTETS):
        group = list(group)
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
        forgotten = int(mem.last_rif["n"]) if isinstance(getattr(mem, "last_rif", None), dict) else 0
        items = []
        for fact, base, ctrl, strength in zip(group, nomem, control, snapshot):
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
                    "strength": float(strength),
                    "top": scored["retrieval"]["top"],
                    "hit": int(scored["retrieval"]["hit"]),
                    "gap": scored["retrieval"]["gap"],
                }
            )
        winner = next(row for row in items if row["role"] == "target")
        competitors = [row for row in items if row["role"] == "competitor"]
        outsiders = [row for row in items if row["role"] == "outsider"]
        geometry_ok = (
            slots_n == len(group)
            and len(group) == 5
            and len(competitors) == 3
            and len(outsiders) == 1
            and group[0]["role"] == "target"
            and decay.get("gap") is not None
            and float(decay["gap"]) > 0.0
            and decay.get("top") == (group[0]["name"], group[0]["rel"])
            and all(row["hit"] == 1 for row in items)
            and float(winner["strength"]) == 1.0
            and float(outsiders[0]["strength"]) == 1.0
            and all(float(row["strength"]) < 1.0 for row in competitors)
            and forgotten == len(competitors)
        )
        groups.append(
            {
                "index": index,
                "slots": slots_n,
                "max_key_cosine": maxcos,
                "decay_top": list(decay["top"]) if decay.get("top") else None,
                "decay_gap": decay.get("gap"),
                "decay_confidence": decay.get("confidence"),
                "forgotten": forgotten,
                "strengths": snapshot,
                "items": items,
                "winner_drop": float(winner["drop"]),
                "competitor_drops": [float(row["drop"]) for row in competitors],
                "noncompetitor_drops": [float(row["drop"]) for row in outsiders],
                "geometry_ok": bool(geometry_ok),
            }
        )
        mem.store.clear()
    competitor_drops = [drop for group in groups for drop in group["competitor_drops"]]
    outsider_abs = [abs(drop) for group in groups for drop in group["noncompetitor_drops"]]
    winner_abs = [abs(float(group["winner_drop"])) for group in groups]
    return {
        "quartets": groups,
        "competitor_drop_min": min(competitor_drops) if competitor_drops else None,
        "competitor_drop_mean": _mean(competitor_drops),
        "winner_abs_drop_max": max(winner_abs) if winner_abs else None,
        "noncompetitor_abs_drop_max": max(outsider_abs) if outsider_abs else None,
        "geometry_ok": bool(groups) and all(group["geometry_ok"] for group in groups),
    }


def measure_cost(mem, bank) -> dict:
    """One query forward. Suppression reads stored keys and adds no parameter."""
    census = inject_census(mem.inject)
    group = list(bank.QUARTETS[0])
    mem.store.clear()
    for fact in group:
        if not mem.remember(_fact(fact)):
            raise RuntimeError(f"{fact['id']} was not written for the cost probe")

    def _query():
        return mem._rank(group[0]["prompt"])

    info, forwards = _forward_count(mem, _query)
    blob = json.dumps(mem.last_rif)
    has_text = int(
        any(
            piece and piece in blob
            for fact in group
            for piece in (fact["text"], fact["answer"], fact["name"], fact["prompt"])
        )
    )
    named = dict(census["named_parameters"])
    mem.store.clear()
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(named.get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "rank_forwards": int(forwards),
        "rif_n": int(info and mem.last_rif["n"]),
        "rif_has_text": int(has_text),
        "rif_span": float(llm_memory.RIF_SPAN),
        "rif_max_set": int(llm_memory.RIF_MAX_SET),
        "rif_salience_exempt": float(llm_memory.RIF_SALIENCE_EXEMPT),
        "gap_tau": float(llm_memory.DEFAULT_GAP_TAU),
        "polarity_span": float(llm_memory.POLARITY_SPAN),
        "valence_span": float(llm_memory.VALENCE_SPAN),
        "valence_step": float(llm_memory.VALENCE_STEP),
        "clock_span": float(episodic.CLOCK_SPAN),
        "replay_every": int(llm_memory.REPLAY_EVERY),
        "replay_m": int(llm_memory.REPLAY_M),
        "replay_decay": float(llm_memory.REPLAY_DECAY),
        "surprise_quantile": float(llm_memory.SURPRISE_QUANTILE),
        "surprise_window": int(llm_memory.SURPRISE_WINDOW),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """Quartet drops, then the fixed-cost probe. One pass."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    say("quartets")
    quartets = score_quartets(mem, bank)
    quartet_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    say("cost")
    cost = measure_cost(mem, bank)
    cost_s = time.perf_counter() - t1
    mem.store.clear()
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "quartets": quartets,
        "cost": cost,
        "phases_s": {"quartets": quartet_s, "cost": cost_s},
        "wall_s": float(quartet_s + cost_s),
    }


def gate_rows(result, gates, *, held: bool):
    """Rows of (name, passed, detail). Quartet floors apply on fit and held."""
    del held
    quartets = result["quartets"]
    cost = result["cost"]
    return [
        (
            "competitors weaken",
            bool(quartets["geometry_ok"])
            and float(quartets["competitor_drop_min"]) >= float(gates["competitor_drop_min"])
            and float(quartets["competitor_drop_mean"]) >= float(gates["competitor_mean_drop_min"])
            and float(quartets["winner_abs_drop_max"]) <= float(gates["winner_abs_drop_max"]),
            (
                f"min drop {quartets['competitor_drop_min']}, "
                f"mean {quartets['competitor_drop_mean']}, "
                f"winner abs {quartets['winner_abs_drop_max']}, "
                f"geometry {'yes' if quartets['geometry_ok'] else 'no'}"
            ),
        ),
        (
            "noncompetitors unaffected",
            bool(quartets["geometry_ok"])
            and float(quartets["noncompetitor_abs_drop_max"]) <= float(gates["noncompetitor_abs_drop_max"]),
            f"outsider abs {quartets['noncompetitor_abs_drop_max']}",
        ),
        (
            "cost",
            int(cost["parameters_new"]) == 0
            and int(cost["rank_forwards"]) == 1
            and int(cost["rif_has_text"]) == 0
            and int(cost["rif_max_set"]) == 3,
            (
                f"new {cost['parameters_new']}, forwards {cost['rank_forwards']}, "
                f"text {cost['rif_has_text']}, forgotten {cost['rif_n']}"
            ),
        ),
    ]


def passes(result, gates, *, held: bool) -> bool:
    return all(ok for _name, ok, _detail in gate_rows(result, gates, held=held))
