"""Diagnosis only. Does not change theta, half-life, lambda, or the inject path.

Writes three reports:
  reports/subconscious_cue_rif_trace.json
  reports/subconscious_cue_fit50.json
  reports/subconscious_cue_fresh_route.json
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "eval_subconscious_cue", ROOT / "scripts" / "eval_subconscious_cue.py"
)
ev = importlib.util.module_from_spec(SPEC)
sys.modules["eval_subconscious_cue"] = ev
SPEC.loader.exec_module(ev)

RIF_PATH = ev.REPORT_DIR / "subconscious_cue_rif_trace.json"
FIT50_PATH = ev.REPORT_DIR / "subconscious_cue_fit50.json"
ROUTE_PATH = ev.REPORT_DIR / "subconscious_cue_fresh_route.json"


def _constants():
    import tidal.llm_memory as mod

    return {
        "theta": float(mod.CUE_THETA),
        "half_life": float(mod.CUE_HALF_LIFE),
        "lambda": float(mod.CUE_LAMBDA),
        "a_trig": float(mod.CUE_A_TRIG),
        "fan_s": float(mod.CUE_FAN_S),
        "rif_span": float(mod.RIF_SPAN),
        "salience_exempt": float(mod.RIF_SALIENCE_EXEMPT),
        "floor": float(mod.CUE_FLOOR),
        "tau": float(mod.CUE_TAU),
    }


def _role_index(slots, fact):
    for index, slot in enumerate(slots):
        if slot.name == fact["name"] and slot.rel == fact["rel"]:
            return index
    return None


def _cosine_matrix(mem, slots):
    import numpy as np

    from tidal.episodic import cosine_scores

    rows = np.asarray(mem._last_cue_rows, dtype=np.float64)
    center = np.asarray(mem.query_mean, dtype=np.float64)
    centered = rows - center
    out = []
    for slot in slots:
        key = np.asarray(slot.k, dtype=np.float64).reshape(-1)
        out.append([float(value) for value in cosine_scores(centered, key).tolist()])
    return out


def _dummy_slots(slots):
    import numpy as np

    from tidal.episodic import Slot

    copies = []
    for slot in slots:
        copy = Slot(
            k=np.asarray(slot.k, dtype=np.float64).copy(),
            v=np.zeros(4, dtype=np.float32),
            name=slot.name,
            rel=slot.rel,
            t=int(slot.t),
        )
        copy.salience = float(slot.salience)
        copy.strength = float(slot.strength)
        copies.append(copy)
    return copies


def _replay_competition(slots, events):
    """Apply the real compete_retrieved to copies, in _rank's order."""
    from tidal.llm_memory import compete_retrieved

    copies = _dummy_slots(slots)
    seen = set()
    applied = []
    for event in events:
        if not event["triggered"]:
            continue
        winner = int(event["winner"])
        if winner in seen:
            applied.append({"token": event["token"], "fired": False, "reason": "winner already seen"})
            continue
        seen.add(winner)
        gap = event["gap"]
        strict = gap is not None and gap == gap and float(gap) > 0.0 and len(copies) >= 2
        if not strict:
            applied.append({"token": event["token"], "fired": False, "reason": "gap is not a strict win"})
            continue
        before = [float(copy.strength) for copy in copies]
        changed = int(compete_retrieved(copies, winner, list(event["candidates"])))
        after = [float(copy.strength) for copy in copies]
        applied.append(
            {
                "token": event["token"],
                "fired": True,
                "changed": changed,
                "winner": winner,
                "candidates": list(event["candidates"]),
                "scaled": [index for index, (left, right) in enumerate(zip(before, after)) if right != left],
            }
        )
    return [float(copy.strength) for copy in copies], applied


def rif_trace(mem) -> dict:
    import numpy as np

    from tidal.subconscious_interfere_score import _remember, _roles
    from tidal.subconscious_measure_data import RifMeasure

    started = ev.time.perf_counter()
    rows = []
    competitor_hits = 0
    competitor_slots = 0
    outsider_hits = 0
    strict_items = 0
    mismatch = 0
    for index, group in enumerate(RifMeasure.QUARTETS):
        ev._print(f"rif {index + 1}/50")
        target, competitors, outsider = _roles(group)
        _remember(mem, list(group))
        slots = list(mem.store._slots)
        by_role = {
            "target": _role_index(slots, target),
            "outsider": _role_index(slots, outsider),
            "competitors": [_role_index(slots, fact) for fact in competitors],
        }
        before = [float(slot.strength) for slot in slots]
        salience = [float(slot.salience) for slot in slots]
        fields = mem.cue_fields(target["prompt"])
        cosines = _cosine_matrix(mem, slots)
        triggered = [bool(flag) for flag in fields["triggered"][0].tolist()]
        memory = [int(value) for value in fields["memory_id"][0].tolist()]
        gaps = fields["gap"][0].tolist()
        winner_cos = fields["winner_cosine"][0].tolist()
        candidates = [list(row) for row in mem._last_candidates]
        ids = mem.tokenizer.encode(target["prompt"], add_special_tokens=True)
        tokens = [mem.tokenizer.decode([token]) for token in ids]
        if len(tokens) != len(triggered):
            raise RuntimeError(f"{target['id']} token count {len(tokens)} != cue {len(triggered)}")
        events = []
        for token, flag in enumerate(triggered):
            gap_value = float(gaps[token])
            gap = None if gap_value != gap_value else gap_value
            event = {
                "token": token,
                "text": tokens[token],
                "triggered": bool(flag),
                "winner": memory[token],
                "winner_cosine": None if not flag else float(winner_cos[token]),
                "gap": gap,
                "candidates": candidates[token] if token < len(candidates) else [],
                "cosines": {str(slot): float(cosines[slot][token]) for slot in range(len(slots))},
            }
            events.append(event)
        predicted, applied = _replay_competition(slots, events)
        rank = mem._rank(target["prompt"])
        after = [float(slot.strength) for slot in mem.store._slots]
        if after != predicted and not np.allclose(after, predicted):
            mismatch += 1
        forgotten = int(mem.last_rif["n"]) if isinstance(getattr(mem, "last_rif", None), dict) else None

        def max_cos(slot_index):
            if slot_index is None:
                return None
            return float(max(cosines[slot_index]))

        def entered(slot_index):
            if slot_index is None:
                return False
            return any(slot_index in event["candidates"] for event in events if event["triggered"])

        comp_entered = [entered(slot) for slot in by_role["competitors"]]
        comp_cos = [max_cos(slot) for slot in by_role["competitors"]]
        competitor_slots += len(comp_cos)
        competitor_hits += sum(1 for value in comp_cos if value is not None and value >= float(_constants()["theta"]))
        outsider_index = by_role["outsider"]
        if entered(outsider_index):
            outsider_hits += 1
        strict = any(row["fired"] for row in applied)
        if strict:
            strict_items += 1
        comp_strength = [after[slot] for slot in by_role["competitors"]]
        target_strength = after[by_role["target"]]
        outsider_strength = after[outsider_index]
        failure = (
            any(value != 1.0 for value in comp_strength)
            or target_strength != 1.0
            or outsider_strength != 1.0
        )
        row = {
            "id": target["id"],
            "prompt": target["prompt"],
            "relations": {
                "target": target["rel"],
                "competitors": [fact["rel"] for fact in competitors],
                "outsider": outsider["rel"],
            },
            "indexes": by_role,
            "strength_before": before,
            "strength_after": after,
            "predicted_strength": predicted,
            "match": bool(np.allclose(after, predicted)),
            "salience": salience,
            "forgotten": forgotten,
            "rank_top": list(rank["top"]) if rank.get("top") else None,
            "rank_gap": None if rank.get("gap") is None else float(rank["gap"]),
            "strict_win": bool(strict),
            "applied": applied,
            "competitor_max_cosine": comp_cos,
            "competitor_entered": comp_entered,
            "competitor_strength": comp_strength,
            "competitor_below_1": bool(all(value < 1.0 for value in comp_strength)),
            "target_strength": target_strength,
            "target_entered": entered(by_role["target"]),
            "outsider_max_cosine": max_cos(outsider_index),
            "outsider_entered": entered(outsider_index),
            "outsider_strength": outsider_strength,
            "outsider_same": bool(outsider_strength == 1.0),
            "winner_same": bool(target_strength == 1.0),
        }
        if failure:
            row["events"] = events
        rows.append(row)
        mem.store.clear()
    competitor_down = sum(1 for row in rows if row["competitor_below_1"])
    outsider_same = sum(1 for row in rows if row["outsider_same"])
    winner_same = sum(1 for row in rows if row["winner_same"])
    payload = {
        "kind": "rif-trace",
        "git": ev._git_sha(),
        "elapsed_s": ev.time.perf_counter() - started,
        "constants": _constants(),
        "constants_changed": False,
        "path": "score_quartet reads strength after LLMMemory._rank. _rank calls cue_fields, which does not change strength, then compete_retrieved. forget_rivals is not on this path.",
        "n": len(rows),
        "strict_win_items": strict_items,
        "competitor_instances_past_theta": competitor_hits,
        "competitor_instances": competitor_slots,
        "outsider_entered_items": outsider_hits,
        "strength_match_failures": mismatch,
        "competitor_down": competitor_down,
        "outsider_same": outsider_same,
        "winner_same": winner_same,
        "outsider_changed": [row["id"] for row in rows if not row["outsider_same"]],
        "winner_changed": [row["id"] for row in rows if not row["winner_same"]],
        "competitor_any_changed": [
            row["id"] for row in rows if any(value != 1.0 for value in row["competitor_strength"])
        ],
        "rows": rows,
    }
    ev._dump(RIF_PATH, payload)
    ev._print(
        json.dumps(
            {key: payload[key] for key in (
                "strict_win_items",
                "competitor_instances_past_theta",
                "competitor_instances",
                "outsider_entered_items",
                "strength_match_failures",
                "competitor_down",
                "outsider_same",
                "winner_same",
                "outsider_changed",
                "winner_changed",
                "competitor_any_changed",
            )},
            indent=2,
        )
    )
    return payload


def _decay_rows(mem, prefer, prefer_rows):
    import torch

    rows = []
    passed = 0
    for item, row in zip(prefer, prefer_rows):
        text = item["text"] + ev.FILLER
        ev._write(mem, [row])
        fields = mem.cue_fields(text)
        activation = fields["activation"][0].tolist()
        triggered = fields["triggered"][0].tolist()
        record = {"id": item["id"], "pass": False, "reason": ""}
        hits = [index for index, flag in enumerate(triggered) if flag]
        if not hits:
            record["reason"] = "no trigger"
            rows.append(record)
            continue
        start = int(hits[-1])
        if start + ev.DECAY_OFFSETS[-1] >= len(activation):
            record["reason"] = "sequence shorter than distance 8"
            rows.append(record)
            continue
        if any(triggered[start + 1 : start + ev.DECAY_OFFSETS[-1] + 1]):
            record["reason"] = "retrigger"
            rows.append(record)
            continue
        taken = [float(activation[start + offset]) for offset in ev.DECAY_OFFSETS]
        monotone = all(taken[i] >= taken[i + 1] - 1e-6 for i in range(len(taken) - 1))
        if not (taken[0] > 0.0 and taken[-1] < taken[0] and monotone):
            record["reason"] = "activation"
            record["activation"] = taken
            rows.append(record)
            continue
        quiet = ev._quiet_logits(mem, text)
        armed = ev._armed_logits(mem, text)
        l1_0 = float(torch.sum(torch.abs(armed[start] - quiet[start])))
        l1_8 = float(torch.sum(torch.abs(armed[start + 8] - quiet[start + 8])))
        record["l1_0"] = l1_0
        record["l1_8"] = l1_8
        record["activation"] = taken
        if not (l1_8 < l1_0):
            record["reason"] = "logit"
            rows.append(record)
            continue
        record["pass"] = True
        record["reason"] = "ok"
        passed += 1
        rows.append(record)
    return passed, rows


def fit50(mem) -> dict:
    from tidal.subconscious_cue_fit50_data import fit50_bank

    bank = fit50_bank()
    original = _constants()
    ev._bind(0.32, 30.0, 0.3, 1.0)
    started = ev.time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        rows = ev._encode(mem, [ev._fact(item) for item in prefer])
        mem.store.clear()
        empties = [
            ev._margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
            for item in prefer
        ]
        cued = []
        for index, (item, row, empty) in enumerate(zip(prefer, rows, empties)):
            ev._print(f"fit50 cued {index + 1}/50")
            ev._write(mem, [row])
            armed = ev._margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
            cued.append(float(armed - empty))
        mean = float(sum(cued) / len(cued))
        positive = int(sum(value > 0.0 for value in cued))
        ev._print("fit50 decay")
        decay_k, decay_rows = _decay_rows(mem, prefer, rows)
        fan_rows = []
        for index, (item, row, effect, empty) in enumerate(zip(prefer, rows, cued, empties)):
            ev._print(f"fit50 fan {index + 1}/50")
            measured = ev._fan_effects(mem, item, row, float(empty), float(ev.FROZEN_ALPHA))
            measured["id"] = item["id"]
            measured["fan1"] = float(effect)
            if measured["fan2"] is None or measured["fan4"] is None:
                measured["strict"] = False
            else:
                measured["strict"] = bool(float(effect) > float(measured["fan2"]) > float(measured["fan4"]))
            fan_rows.append(measured)
        fan = ev._fan_summary(fan_rows)
        mem.store.clear()
    finally:
        ev._bind(original["theta"], original["half_life"], original["lambda"], original["a_trig"])
    payload = {
        "kind": "fit50-diagnosis",
        "git": ev._git_sha(),
        "elapsed_s": ev.time.perf_counter() - started,
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "constants": _constants(),
        "constants_changed": _constants() != original,
        "note": "Diagnosis sample. The cell was not chosen from this bank.",
        "cued_positive": positive,
        "cued_n": len(cued),
        "cued_mean": mean,
        "cued_wilson": ev._wilson(positive, len(cued)),
        "cued_deltas": cued,
        "decay_pass": int(decay_k),
        "decay_n": len(prefer),
        "decay_wilson": ev._wilson(int(decay_k), len(prefer)),
        "decay_reasons": [
            {"id": row["id"], "pass": row["pass"], "reason": row["reason"]} for row in decay_rows
        ],
        "fan_k": int(fan["k"]),
        "fan_n": int(fan["n"]),
        "fan_wilson": ev._wilson(int(fan["k"]), int(fan["n"])) if int(fan["n"]) else None,
        "fan_means": fan["means"],
        "alpha": float(ev.FROZEN_ALPHA),
    }
    ev._dump(FIT50_PATH, payload)
    ev._print(
        json.dumps(
            {
                "cued": [payload["cued_positive"], payload["cued_n"], payload["cued_mean"], payload["cued_wilson"]],
                "decay": [payload["decay_pass"], payload["decay_n"], payload["decay_wilson"]],
                "fan": [payload["fan_k"], payload["fan_n"], payload["fan_means"], payload["fan_wilson"]],
            },
            indent=2,
        )
    )
    return payload


def _ids(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(value) for value in ids]


def _live_fields(mem, prompt, continuation):
    from tidal.subconscious_eval import continuation_logprobs

    scored = continuation_logprobs(mem, prompt, continuation)
    fields = mem.last_cue_fields
    prompt_ids = _ids(mem.tokenizer, prompt)
    full_ids = _ids(mem.tokenizer, prompt + continuation)
    predictors = [len(prompt_ids) - 1 + index for index in range(len(full_ids) - len(prompt_ids))]
    triggered = [bool(flag) for flag in fields["triggered"][0].tolist()]
    memory = [int(value) for value in fields["memory_id"][0].tolist()]
    activation = [float(value) for value in fields["activation"][0].tolist()]
    cosine = [float(value) for value in fields["winner_cosine"][0].tolist()]
    tokens = [mem.tokenizer.decode([token]) for token in full_ids]
    if len(tokens) != len(triggered):
        raise RuntimeError("live cue length does not match tokens")
    hits = [index for index, flag in enumerate(triggered) if flag]
    last_hit = None if not hits else int(hits[-1])

    def govern(position):
        earlier = [index for index in hits if index <= position]
        if not earlier:
            return None
        return int(earlier[-1])

    scored_rows = []
    for position in predictors:
        trigger = govern(position)
        scored_rows.append(
            {
                "position": int(position),
                "token": tokens[position],
                "predicts": tokens[position + 1] if position + 1 < len(tokens) else None,
                "activation": activation[position],
                "triggered": triggered[position],
                "memory_id": memory[position],
                "winner_cosine": None if trigger is None else cosine[trigger],
                "trigger_position": trigger,
                "distance": None if trigger is None else int(position - trigger),
            }
        )
    return {
        "logp_mean": float(scored["mean"]),
        "n_tokens": int(scored["n_tokens"]),
        "prompt_len": len(prompt_ids),
        "triggers": [
            {
                "position": int(index),
                "token": tokens[index],
                "memory_id": memory[index],
                "winner_cosine": cosine[index],
                "activation": activation[index],
            }
            for index in hits
        ],
        "last_trigger": last_hit,
        "scored": scored_rows,
        "intended_at_first": bool(scored_rows and scored_rows[0]["memory_id"] == 0),
    }


def fresh_route(mem) -> dict:
    from tidal.subconscious_cue_fresh_data import fresh_bank

    bank = fresh_bank()
    original = _constants()
    ev._bind(0.32, 30.0, 0.3, 1.0)
    started = ev.time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        encoded = ev._encode(mem, [ev._fact(item) for item in prefer])
        rows = []
        for index, (item, row) in enumerate(zip(prefer, encoded)):
            ev._print(f"fresh route {index + 1}/50")
            mem.store.clear()
            from tidal.subconscious_eval import continuation_logprobs

            empty_good = continuation_logprobs(mem, item["prompt"], item["good"])
            empty_bad = continuation_logprobs(mem, item["prompt"], item["bad"])
            ev._write(mem, [row])
            slot = mem.store._slots[0]
            offline = mem.cue_fields(item["prompt"])
            off_triggered = [bool(flag) for flag in offline["triggered"][0].tolist()]
            off_hits = [pos for pos, flag in enumerate(off_triggered) if flag]
            good = _live_fields(mem, item["prompt"], item["good"])
            bad = _live_fields(mem, item["prompt"], item["bad"])
            fact_ids = set(_ids(mem.tokenizer, item["text"]))
            good_ids = _ids(mem.tokenizer, item["prompt"] + item["good"])
            prompt_ids = _ids(mem.tokenizer, item["prompt"])
            continuation_ids = good_ids[len(prompt_ids) :]
            first = good["scored"][0]
            rows.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "rel": item["rel"],
                    "answer": item["answer"],
                    "stored_name": slot.name,
                    "stored_rel": slot.rel,
                    "prompt": item["prompt"],
                    "delta": float(good["logp_mean"] - bad["logp_mean"])
                    - float(empty_good["mean"] - empty_bad["mean"]),
                    "offline_triggers": off_hits,
                    "good_triggers": good["triggers"],
                    "bad_triggers": bad["triggers"],
                    "first_scored": first,
                    "good_scored": good["scored"],
                    "bad_scored": bad["scored"],
                    "continuation_ids": continuation_ids,
                    "continuation_in_fact": [int(token) in fact_ids for token in continuation_ids],
                    "intended_winner": bool(first["memory_id"] == 0 and slot.name == item["name"]),
                }
            )
        mem.store.clear()
    finally:
        ev._bind(original["theta"], original["half_life"], original["lambda"], original["a_trig"])
    deltas = [float(row["delta"]) for row in rows]
    activations = [float(row["first_scored"]["activation"]) for row in rows]
    distances = [row["first_scored"]["distance"] for row in rows]
    known_distance = [int(value) for value in distances if value is not None]
    triggered = sum(1 for row in rows if row["good_triggers"])
    intended = sum(1 for row in rows if row["intended_winner"])
    active = [row["delta"] for row in rows if row["first_scored"]["activation"] >= 0.05]
    quiet = [row["delta"] for row in rows if row["first_scored"]["activation"] < 0.05]
    zero = [row["delta"] for row in rows if row["first_scored"]["activation"] == 0.0]
    continuation_covered = sum(1 for row in rows if row["continuation_in_fact"] and all(row["continuation_in_fact"]))
    payload = {
        "kind": "fresh-route",
        "git": ev._git_sha(),
        "elapsed_s": ev.time.perf_counter() - started,
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "constants": _constants(),
        "constants_changed": _constants() != original,
        "n": len(rows),
        "mean_delta": float(sum(deltas) / len(deltas)),
        "positive": int(sum(value > 0.0 for value in deltas)),
        "wilson": ev._wilson(int(sum(value > 0.0 for value in deltas)), len(deltas)),
        "items_with_good_trigger": triggered,
        "intended_winner_at_first_scored": intended,
        "mean_activation_at_first_scored": float(sum(activations) / len(activations)),
        "activation_at_least_floor": int(sum(value >= 0.05 for value in activations)),
        "activation_zero": int(sum(value == 0.0 for value in activations)),
        "mean_distance": None if not known_distance else float(sum(known_distance) / len(known_distance)),
        "distance_zero": int(sum(value == 0 for value in distances if value is not None)),
        "distance_missing": int(sum(value is None for value in distances)),
        "mean_delta_when_activation_at_least_floor": None if not active else float(sum(active) / len(active)),
        "mean_delta_when_activation_below_floor": None if not quiet else float(sum(quiet) / len(quiet)),
        "mean_delta_when_activation_zero": None if not zero else float(sum(zero) / len(zero)),
        "continuation_tokens_all_in_fact": continuation_covered,
        "rows": rows,
    }
    ev._dump(ROUTE_PATH, payload)
    ev._print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "mean_delta",
                    "positive",
                    "wilson",
                    "items_with_good_trigger",
                    "intended_winner_at_first_scored",
                    "mean_activation_at_first_scored",
                    "activation_at_least_floor",
                    "activation_zero",
                    "mean_distance",
                    "distance_zero",
                    "distance_missing",
                    "mean_delta_when_activation_at_least_floor",
                    "mean_delta_when_activation_below_floor",
                    "mean_delta_when_activation_zero",
                    "continuation_tokens_all_in_fact",
                )
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    before = _constants()
    mem = ev._load_memory()
    try:
        rif_trace(mem)
        fit50(mem)
        fresh_route(mem)
    finally:
        ev._bind(before["theta"], before["half_life"], before["lambda"], before["a_trig"])
        mem.close()
    after = _constants()
    if after != before:
        raise SystemExit(f"constants changed: {before} -> {after}")


if __name__ == "__main__":
    main()
