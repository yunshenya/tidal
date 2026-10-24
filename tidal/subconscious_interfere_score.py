"""Choose the repeat threshold on the fit bank, then score held once.

Fit does not read the held quartets and does not score a log-prob margin.
Held refuses a second complete report and refuses to run until
``REPEAT_JACCARD`` is the threshold fit selected.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import tidal.episodic as episodic
from tidal.episodic import EpisodicStore
from tidal.subconscious_eval import Encoder, score_pair, write_rows
from tidal.subconscious_interfere_data import CASES, GRID, PROTOCOL, bank_sha256
from tidal.subconscious_measure_data import (
    RifMeasure,
    avoid_items,
    away_items,
    birthday_items,
    choice_items,
    cohens_dz,
    phone_items,
    prefer_items,
    preference_items,
    rate_row,
    subconscious_score,
    tone_items,
    toward_items,
    wilson_interval,
)
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_softmix_score import (
    MEASURE_INTERVALS,
    _Readout,
    _family_direction,
    _family_signed,
    _group_rows,
    _score_item,
    held_rate_table,
    preflight_groups,
    score_explicit_family,
)
from tidal.subconscious_valence_data import REPEATS
from tidal.subconscious_valence_score import score_signed

MEASURE_BANK = "0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228"
# 95% percentile interval of the #53 mixture subconscious score.
# 10000 resamples, numpy Generator seed 20261024, linear percentile.
# Computed from reports/subconscious_softmix_held.json before this held pass.
IMPLICIT_DZ_INTERVAL = (1.0171138974369223, 1.3856896713238247)
GEOMETRY_MIN = 26
PATTERN_WILSON_FLOOR = 0.5


def dump_json(path, payload) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest)


def expected_slots(case, threshold: float) -> int:
    """Slot count a passing cell must produce. Independent of the threshold.

    ``merge`` is a near-identical same-relation repeat and must collapse.
    ``stay`` is a different fact, a different relation, or a different group
    and must remain two slots. ``one`` is a same-identity update.
    ``threshold`` is unused; the store's behavior is what changes with it.
    """
    del threshold
    rule = case["rule"]
    if rule in ("one", "merge"):
        return 1
    if rule in ("two", "stay"):
        return 2
    raise ValueError(f"unknown fit rule {rule}")


def _run_case(threshold: float, case) -> int:
    store = EpisodicStore(capacity=8, dk=4, dv=4, overwrite_cos=0.9, repeat_jaccard=float(threshold))
    for index, write in enumerate(case["writes"]):
        key = np.asarray(write.get("key", np.eye(4)[index % 4]), dtype=np.float64)
        value = np.asarray(write.get("value", np.eye(4)[(index + 1) % 4]), dtype=np.float64)
        ok = store.write(
            key,
            value,
            name=write["name"],
            rel=write.get("rel"),
            group=write.get("group"),
            text=write["text"],
            answer=write.get("answer"),
        )
        if not ok:
            raise RuntimeError(f"{case['id']} write {index} was dropped")
    return len(store)


def score_grid() -> list:
    """Every grid threshold against the fit cases. No language model."""
    rows = []
    for threshold in GRID:
        cases = []
        for case in CASES:
            slots = _run_case(threshold, case)
            expect = expected_slots(case, threshold)
            cases.append(
                {
                    "id": case["id"],
                    "rule": case["rule"],
                    "slots": int(slots),
                    "expect": int(expect),
                    "pass": slots == expect,
                }
            )
        rows.append(
            {
                "threshold": float(threshold),
                "pass": all(item["pass"] for item in cases),
                "cases": cases,
            }
        )
    return rows


def select_threshold(rows):
    """Highest passing threshold. No passing row returns None.

    A higher threshold merges fewer distinct facts. The fit bank's
    near-copy still has to collapse, so the highest passing cell is the
    strictest repeat rule that catches that near-copy.
    """
    passing = [row for row in rows if row.get("pass")]
    if not passing:
        return None
    passing.sort(key=lambda row: float(row["threshold"]))
    return dict(passing[-1])


def summarize_fit(rows) -> dict:
    selected = select_threshold(rows)
    return {
        "complete": True,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "measure_bank": MEASURE_BANK,
        "grid": [float(threshold) for threshold in GRID],
        "cells": rows,
        "selected": None
        if selected is None
        else {"threshold": float(selected["threshold"])},
        "n_cases": len(CASES),
    }


def score_fit() -> dict:
    """Fit slot counts only. This function does not import held items."""
    return summarize_fit(score_grid())


def held_allowed(fit_report, repeat_jaccard: float, held_complete: bool):
    """Held runs once, and only at the fit selection."""
    if held_complete:
        return False, "held report already exists"
    if not isinstance(fit_report, dict) or not fit_report.get("complete"):
        return False, "fit report is not complete"
    if fit_report.get("bank_sha256") != bank_sha256():
        return False, "fit report is for a different bank"
    selected = fit_report.get("selected")
    if not selected:
        return False, "fit selected no threshold"
    if not math.isclose(float(selected["threshold"]), float(repeat_jaccard), rel_tol=0.0, abs_tol=1e-9):
        return False, "REPEAT_JACCARD is not the fit selection"
    return True, ""


def _fact(row) -> dict:
    return {
        "text": row["text"],
        "answer": row["answer"],
        "name": row["name"],
        "rel": row["rel"],
    }


def _remember(mem, facts) -> int:
    mem.store.clear()
    for fact in facts:
        if not mem.remember(_fact(fact)):
            raise RuntimeError(f"{fact['id']} was not written")
    return len(mem.store)


def _find_slot(mem, fact):
    hits = [slot for slot in mem.store.slots if slot.name == fact["name"] and slot.rel == fact["rel"]]
    if len(hits) != 1:
        return None
    return hits[0]


def _cosine(a, b) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _margin(mem, fact) -> float:
    scored = score_pair(
        mem,
        fact["prompt"],
        fact["good"],
        fact["bad"],
        name=fact["name"],
        rel=fact["rel"],
    )
    return float(scored["margin"])


def _roles(group):
    target = next(fact for fact in group if fact["role"] == "target")
    competitors = [fact for fact in group if fact["role"] == "competitor"]
    outsider = next(fact for fact in group if fact["role"] == "outsider")
    return target, competitors, outsider


def _unique_extreme(values):
    """Index of the unique max and the unique min, or None when either ties."""
    if len(values) < 2:
        return None
    top = max(values)
    bottom = min(values)
    if values.count(top) != 1 or values.count(bottom) != 1:
        return None
    if values.index(top) == values.index(bottom):
        return None
    return values.index(top), values.index(bottom)


def score_quartet(mem, group) -> dict:
    """Slot count, strength pattern, similarity contrast, and count contrast.

    The similarity and count margins are taken at strength 1. ``score_pair``
    ranks before it scales rivals, so those margins are not the RIF pattern.
    """
    target, competitors, outsider = _roles(group)
    if len(competitors) != 3:
        raise RuntimeError(f"{target['id']} does not have three competitors")
    slots_n = _remember(mem, list(group))
    present = {
        "target": _find_slot(mem, target) is not None,
        "outsider": _find_slot(mem, outsider) is not None,
        "competitors": sum(_find_slot(mem, fact) is not None for fact in competitors),
    }
    keys = {}
    for fact in (target, *competitors):
        slot = _find_slot(mem, fact)
        if slot is not None:
            keys[fact["id"]] = np.asarray(slot.k, dtype=np.float64)
    cosines = []
    if target["id"] in keys:
        for fact in competitors:
            if fact["id"] in keys:
                cosines.append(_cosine(keys[target["id"]], keys[fact["id"]]))
            else:
                cosines.append(None)
    else:
        cosines = [None] * len(competitors)
    maxcos = 0.0
    stored = [slot.k for slot in mem.store.slots]
    for i in range(len(stored)):
        for j in range(i + 1, len(stored)):
            maxcos = max(maxcos, _cosine(stored[i], stored[j]))
    decay = mem._rank(target["prompt"])
    forgotten = 0
    if isinstance(getattr(mem, "last_rif", None), dict):
        forgotten = int(mem.last_rif["n"])
    strength = {}
    for fact in (target, *competitors, outsider):
        slot = _find_slot(mem, fact)
        strength[fact["id"]] = None if slot is None else float(slot.strength)
    comp_strengths = [strength[fact["id"]] for fact in competitors]
    pattern_ok = (
        present["target"]
        and present["outsider"]
        and present["competitors"] == 3
        and strength[target["id"]] is not None
        and strength[outsider["id"]] is not None
        and all(value is not None for value in comp_strengths)
    )
    competitor_down = bool(pattern_ok and all(value < 1.0 for value in comp_strengths))
    winner_same = bool(pattern_ok and strength[target["id"]] == 1.0)
    outsider_same = bool(pattern_ok and strength[outsider["id"]] == 1.0)
    numeric = [value for value in cosines if value is not None]
    extreme = _unique_extreme(numeric) if len(numeric) == 3 else None

    mem.store.clear()
    nomem = _margin(mem, target)

    def armed(facts) -> float:
        _remember(mem, facts)
        value = _margin(mem, target)
        mem.store.clear()
        return value

    similarity = {"included": False}
    if extreme is not None:
        high_i, low_i = extreme
        high = armed([target, competitors[high_i]])
        low = armed([target, competitors[low_i]])
        similarity = {
            "included": True,
            "high_cosine": float(numeric[high_i]),
            "low_cosine": float(numeric[low_i]),
            "high_margin": high,
            "low_margin": low,
            "high_delta": high - nomem,
            "low_delta": low - nomem,
            "high_lower": bool((high - nomem) < (low - nomem)),
        }
    else:
        similarity = {"included": False, "reason": "cosine extreme was not unique"}

    fan_low = armed([target, competitors[0]])
    fan_high = armed([target, *competitors])
    fan = {
        "low_margin": fan_low,
        "high_margin": fan_high,
        "low_delta": fan_low - nomem,
        "high_delta": fan_high - nomem,
        "high_lower": bool((fan_high - nomem) < (fan_low - nomem)),
    }
    mem.store.clear()
    return {
        "id": target["id"],
        "slots": int(slots_n),
        "ge4": bool(slots_n >= 4),
        "max_key_cosine": float(maxcos),
        "present": present,
        "strength": strength,
        "cosines": [None if value is None else float(value) for value in cosines],
        "decay_top": list(decay["top"]) if decay.get("top") else None,
        "decay_gap": None if decay.get("gap") is None else float(decay["gap"]),
        "forgotten": int(forgotten),
        "pattern_ok": bool(pattern_ok),
        "competitor_down": competitor_down,
        "winner_same": winner_same,
        "outsider_same": outsider_same,
        "nomem_margin": nomem,
        "similarity": similarity,
        "fan": fan,
    }


def _rate_success(flags) -> dict:
    return rate_row(sum(1 for flag in flags if flag), len(flags))


def _wilson_above(row, floor: float) -> bool:
    if row is None or int(row["n"]) <= 0:
        return False
    low, _high = wilson_interval(int(row["k"]), int(row["n"]))
    return float(low) > float(floor)


def summarize_quartets(rows) -> dict:
    ordered = list(rows)
    n = len(ordered)
    ge4 = _rate_success([row["ge4"] for row in ordered]) if n else None
    pattern_rows = [row for row in ordered if row["pattern_ok"]]
    competitor = _rate_success([row["competitor_down"] for row in pattern_rows]) if pattern_rows else None
    winner = _rate_success([row["winner_same"] for row in pattern_rows]) if pattern_rows else None
    outsider = _rate_success([row["outsider_same"] for row in pattern_rows]) if pattern_rows else None
    sim_rows = [row for row in ordered if row["similarity"].get("included")]
    sim = _rate_success([row["similarity"]["high_lower"] for row in sim_rows]) if sim_rows else None
    fan = _rate_success([row["fan"]["high_lower"] for row in ordered]) if n else None
    geometry_pass = bool(ge4 is not None and ge4["n"] == 50 and ge4["k"] >= GEOMETRY_MIN)
    pattern_pass = bool(
        _wilson_above(competitor, PATTERN_WILSON_FLOOR)
        and _wilson_above(winner, PATTERN_WILSON_FLOOR)
        and _wilson_above(outsider, PATTERN_WILSON_FLOOR)
    )
    interference_pass = bool(
        _wilson_above(sim, PATTERN_WILSON_FLOOR) and _wilson_above(fan, PATTERN_WILSON_FLOOR)
    )
    return {
        "n": n,
        "ge4": ge4,
        "slots_5": sum(1 for row in ordered if row["slots"] == 5),
        "pattern_n": len(pattern_rows),
        "competitor_down": competitor,
        "winner_same": winner,
        "outsider_same": outsider,
        "similarity_n": len(sim_rows),
        "similarity_excluded": n - len(sim_rows),
        "similarity": sim,
        "fan": fan,
        "geometry_pass": geometry_pass,
        "pattern_pass": pattern_pass,
        "interference_pass": interference_pass,
    }


def _family_dz(records, margin_of):
    out = {}
    for family in ("tone", "choice", "preference"):
        diffs = []
        for record in records:
            if record["family"] != family:
                continue
            diffs.append(float(margin_of(record)) - float(record["nomem_margin"]))
        out[family] = cohens_dz(diffs)
    return out


def _mean(vals):
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def summarize_implicit(records) -> dict:
    ordered = list(records)
    soft_dz = _family_dz(ordered, lambda record: record["mixture"]["margin"])
    top_dz = _family_dz(ordered, lambda record: record["top1"]["margin"])
    soft_score = subconscious_score(soft_dz)
    top_score = subconscious_score(top_dz)
    slots_ok = all(int(record["mixture"]["n"]) == 4 and int(record["top1"]["n"]) == 4 for record in ordered)
    low, high = IMPLICIT_DZ_INTERVAL
    inside = soft_score is not None and float(low) <= float(soft_score) <= float(high)
    return {
        "n": len(ordered),
        "slots_ok": bool(slots_ok),
        "soft_dz": soft_score,
        "top1_dz": top_score,
        "soft_family_dz": soft_dz,
        "top1_family_dz": top_dz,
        "interval": [low, high],
        "inside_interval": bool(inside),
        "pass": bool(slots_ok and inside),
        "participation_mean": _mean([record["mixture"]["participation"] for record in ordered]),
        "hit": rate_row(sum(int(record["mixture"]["hit"]) for record in ordered), len(ordered)) if ordered else None,
        "top1_confidence_mean": _mean([record["top1"]["confidence"] for record in ordered]),
        "top1_gap_mean": _mean([record["top1"]["gap"] for record in ordered]),
    }


def _load_seed(seed):
    if not seed:
        return {}
    if seed.get("bank_sha256") not in (None, bank_sha256()):
        raise RuntimeError("held checkpoint is for a different fit bank")
    if seed.get("measure_bank") not in (None, MEASURE_BANK):
        raise RuntimeError("held checkpoint is for a different measure bank")
    return seed


def _held_probes():
    probes = []
    for family, items in (
        ("tone", tone_items()),
        ("choice", choice_items()),
        ("preference", preference_items()),
    ):
        for item in items:
            row = dict(item)
            row["family"] = family
            probes.append(row)
    return probes


def score_held(mem, fit_report, *, progress=None, save=None, seed=None) -> dict:
    """One held pass. Refuses a second complete report and a mismatched threshold."""
    allowed, reason = held_allowed(fit_report, episodic.REPEAT_JACCARD, False)
    if not allowed:
        raise RuntimeError(reason)
    if len(RifMeasure.QUARTETS) != 50:
        raise RuntimeError("RIF held bank is not n=50")
    state = _load_seed(seed)
    state.setdefault("quartets", [])
    for index, group in enumerate(RifMeasure.QUARTETS):
        if index < len(state["quartets"]):
            continue
        if progress is not None:
            progress(f"quartet {index}/50")
        state["quartets"].append(score_quartet(mem, group))
        if save is not None:
            save(state)
    quartets = summarize_quartets(state["quartets"])

    state.setdefault("implicit", [])
    probes = _held_probes()
    if len(probes) != 150:
        raise RuntimeError("implicit held bank is not 150 probes")
    encoder = Encoder(mem)
    if len(state["implicit"]) < len(probes):
        preflight_groups(mem, probes[len(state["implicit"]) :], progress=progress)
    for index, probe in enumerate(probes):
        if index < len(state["implicit"]):
            continue
        if progress is not None:
            progress(f"held implicit {index}/{len(probes)} {probe['id']}")
        mem.store.clear()
        nomem = _score_item(mem, probe)
        encoded = encoder.many(_group_rows(probe))
        write_rows(mem, encoded)
        with _Readout(mem, "mixture"):
            mixture = _score_item(mem, probe)
        write_rows(mem, encoded)
        with _Readout(mem, "top1"):
            top1 = _score_item(mem, probe)
        mem.store.clear()
        state["implicit"].append(
            {
                "id": probe["id"],
                "family": probe["family"],
                "nomem_margin": nomem["margin"],
                "mixture": mixture,
                "top1": top1,
            }
        )
        if save is not None:
            save(state)
    implicit = summarize_implicit(state["implicit"])

    def _save():
        if save is not None:
            save(state)

    if "prefer" not in state:
        state["prefer"] = []
        _save()
    for index, item in enumerate(prefer_items()):
        if index < len(state["prefer"]):
            continue
        if progress is not None:
            progress(f"prefer {index}")
        part = score_direction(mem, _DirectionBank([item], []))
        state["prefer"].append(part["prefer"]["rows"][0])
        _save()
    if "avoid" not in state:
        state["avoid"] = []
        _save()
    for index, item in enumerate(avoid_items()):
        if index < len(state["avoid"]):
            continue
        if progress is not None:
            progress(f"avoid {index}")
        part = score_direction(mem, _DirectionBank([], [item]))
        state["avoid"].append(part["avoid"]["rows"][0])
        _save()
    direction = {
        "prefer": {"rows": state["prefer"], "summary": _family_direction(state["prefer"])},
        "avoid": {"rows": state["avoid"], "summary": _family_direction(state["avoid"])},
    }
    if "toward" not in state:
        state["toward"] = []
        _save()
    for index, item in enumerate(toward_items()):
        if index < len(state["toward"]):
            continue
        if progress is not None:
            progress(f"toward {index}")
        part = score_signed(mem, [item], 1.0, int(REPEATS))
        state["toward"].append(part["rows"][0])
        _save()
    if "away" not in state:
        state["away"] = []
        _save()
    for index, item in enumerate(away_items()):
        if index < len(state["away"]):
            continue
        if progress is not None:
            progress(f"away {index}")
        part = score_signed(mem, [item], -1.0, int(REPEATS))
        state["away"].append(part["rows"][0])
        _save()
    valence = {
        "repeats": int(REPEATS),
        "toward": _family_signed(state["toward"]),
        "away": _family_signed(state["away"]),
    }
    rates = held_rate_table(direction, valence)
    if "phone" not in state:
        state["phone"] = score_explicit_family(mem, phone_items(), progress=progress)
        _save()
    if "birthday" not in state:
        state["birthday"] = score_explicit_family(mem, birthday_items(), progress=progress)
        _save()
    mem.store.clear()
    passed = bool(
        quartets["geometry_pass"]
        and quartets["pattern_pass"]
        and quartets["interference_pass"]
        and implicit["pass"]
        and rates["pass"]
    )
    return {
        "complete": True,
        "bank_sha256": bank_sha256(),
        "measure_bank": MEASURE_BANK,
        "repeat_jaccard": float(episodic.REPEAT_JACCARD),
        "selected": fit_report["selected"],
        "quartets": quartets,
        "quartet_rows": state["quartets"],
        "implicit": implicit,
        "implicit_records": state["implicit"],
        "direction": direction,
        "valence": valence,
        "rates": rates,
        "explicit": {
            "excluded_from_subconscious_score": True,
            "phone": state["phone"],
            "birthday": state["birthday"],
        },
        "pass": passed,
    }


class _DirectionBank:
    def __init__(self, prefer, avoid):
        self.PREFER = tuple(prefer)
        self.AVOID = tuple(avoid)
