"""Score the strength-weighted mixture against top-1 in the same multi-slot store.

Fit chooses temperature and k. Held is scored once, and only after those
constants are the ones fit selected. Explicit phone and birthday recall is
reported by the held pass and is not an input to the subconscious score.
"""
from __future__ import annotations

import json
from pathlib import Path

from tidal.llm_memory import MIX_K, MIX_TEMP
from tidal.subconscious_eval import Encoder, score_pair, write_rows
from tidal.subconscious_measure_data import (
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
)
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_softmix_data import bank_sha256, distractors_for, fit_items
from tidal.subconscious_valence_data import REPEATS
from tidal.subconscious_valence_score import score_signed

PARTICIPATION_MIN = 1.05
GRID = tuple((k, temperature) for k in (2, 4, 8) for temperature in (0.05, 0.10, 0.20, 0.40))
MEASURE_BANK = "0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228"
# Inclusive 95% Wilson intervals from the unmerged measurement report.
# A held rate passes when its point estimate lies inside the interval.
MEASURE_INTERVALS = {
    "prefer_delta_pos": (0.9286524008666414, 1.0),
    "avoid_delta_pos": (0.6475845041874634, 0.8724608402978559),
    "avoid_stored_down": (0.7148578393696501, 0.916625793219666),
    "toward_signed": (0.7863976856252035, 0.9565242350681096),
    "away_signed": (0.8653990931249297, 0.9889611156723802),
}
MEASURE_COUNTS = {
    "prefer_delta_pos": (50, 50),
    "avoid_delta_pos": (39, 50),
    "avoid_stored_down": (42, 50),
    "toward_signed": (45, 50),
    "away_signed": (48, 50),
}


def cell_key(k: int, temperature: float) -> str:
    return f"{int(k)}:{float(temperature):.2f}"


def rate_inside(k: int, n: int, bounds) -> bool:
    """True when k/n is inside the inclusive Wilson interval."""
    low, high = bounds
    if int(n) <= 0:
        return False
    rate = int(k) / int(n)
    return float(low) <= rate <= float(high)


def select_cell(rows):
    """Largest mean participation, then smallest k, then smallest temperature.

    A row passes when both subconscious scores are numbers, the mixture
    score is at least the top-1 score, mean participation is at least 1.05,
    and every probe kept four slots. No passing row returns None.
    """
    passing = []
    for row in rows:
        soft = row.get("soft_dz")
        top = row.get("top1_dz")
        part = row.get("participation_mean")
        if soft is None or top is None or part is None:
            continue
        if not row.get("slots_ok"):
            continue
        if float(soft) < float(top):
            continue
        if float(part) < PARTICIPATION_MIN:
            continue
        passing.append(row)
    if not passing:
        return None
    passing.sort(key=lambda row: (-float(row["participation_mean"]), int(row["k"]), float(row["temperature"])))
    return dict(passing[0])


def held_allowed(fit_report, mix_k, mix_temp, held_complete: bool):
    """Held runs once, and only on the fit selection."""
    if held_complete:
        return False, "held report already exists"
    if not isinstance(fit_report, dict) or not fit_report.get("complete"):
        return False, "fit report is not complete"
    selected = fit_report.get("selected")
    if not selected:
        return False, "fit selected no cell"
    if int(selected["k"]) != int(mix_k) or float(selected["temperature"]) != float(mix_temp):
        return False, "MIX_K and MIX_TEMP are not the fit selection"
    return True, ""


def dump_json(path, payload) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest)


def _fact(item) -> dict:
    return {
        "text": item["text"],
        "answer": item["answer"],
        "name": item["name"],
        "rel": item["rel"],
    }


def _group_rows(probe) -> list:
    rows = [_fact(probe)]
    for item in distractors_for(probe["id"]):
        rows.append(_fact(item))
    return rows


def _pull(scored) -> dict:
    info = scored["retrieval"]
    gap = info.get("gap")
    maxcos = info.get("maxcos")
    participation = info.get("participation")
    return {
        "margin": float(scored["margin"]),
        "n": int(info["n"]),
        "hit": int(info["hit"]),
        "confidence": float(info["confidence"]),
        "gap": None if gap is None else float(gap),
        "maxcos": None if maxcos is None else float(maxcos),
        "participation": None if participation is None else float(participation),
    }


class _Readout:
    def __init__(self, mem, readout: str, k=None, temperature=None):
        self.mem = mem
        self.readout = readout
        self.k = k
        self.temperature = temperature
        self._old = None

    def __enter__(self):
        import tidal.llm_memory as lm

        self._old = (lm.MIX_K, lm.MIX_TEMP, self.mem.injection_readout)
        if self.k is not None:
            lm.MIX_K = int(self.k)
        if self.temperature is not None:
            lm.MIX_TEMP = float(self.temperature)
        self.mem.injection_readout = self.readout
        return self

    def __exit__(self, *_exc):
        import tidal.llm_memory as lm

        lm.MIX_K, lm.MIX_TEMP, self.mem.injection_readout = self._old
        self.mem._delta = None
        return False


def _score_item(mem, probe) -> dict:
    return _pull(
        score_pair(
            mem,
            probe["prompt"],
            probe["good"],
            probe["bad"],
            name=probe["name"],
            rel=probe["rel"],
        )
    )


def preflight_groups(mem, probes, *, progress=None) -> None:
    """Write each group and require the probe to win. No continuation is scored."""
    encoder = Encoder(mem)
    try:
        for index, probe in enumerate(probes):
            if progress is not None and index % 20 == 0:
                progress(f"preflight {index}/{len(probes)} {probe['id']}")
            encoded = encoder.many(_group_rows(probe))
            written = write_rows(mem, encoded)
            if int(written["slots"]) != 4:
                raise RuntimeError(f"{probe['id']} kept {written['slots']} slots")
            with _Readout(mem, "mixture"):
                rank = mem._rank(probe["prompt"])
            top = rank.get("top")
            if list(top) != [probe["name"], probe["rel"]] or int(rank["n"]) != 4:
                raise RuntimeError(f"{probe['id']} winner {top} n={rank['n']}")
    finally:
        mem.store.clear()
        mem._delta = None


def _fit_probes():
    probes = []
    for family in ("tone", "choice", "preference"):
        probes.extend(fit_items(family))
    return probes


def _mean(vals):
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


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


def summarize_fit(records) -> dict:
    ordered = list(records)
    top_dz = _family_dz(ordered, lambda record: record["top1"]["margin"])
    top_score = subconscious_score(top_dz)
    rows = []
    for k, temperature in GRID:
        key = cell_key(k, temperature)
        soft_dz = _family_dz(ordered, lambda record, key=key: record["cells"][key]["margin"])
        soft_score = subconscious_score(soft_dz)
        slots_ok = all(int(record["top1"]["n"]) == 4 for record in ordered) and all(
            int(record["cells"][key]["n"]) == 4 for record in ordered
        )
        parts = [record["cells"][key]["participation"] for record in ordered]
        row = {
            "k": int(k),
            "temperature": float(temperature),
            "slots_ok": bool(slots_ok),
            "participation_mean": _mean(parts),
            "participation_min": None if not parts else float(min(parts)),
            "soft_dz": soft_score,
            "top1_dz": top_score,
            "soft_family_dz": soft_dz,
            "top1_family_dz": top_dz,
            "hit": rate_row(sum(int(record["cells"][key]["hit"]) for record in ordered), len(ordered)),
            "top1_confidence_mean": _mean([record["top1"]["confidence"] for record in ordered]),
        }
        row["pass"] = select_cell([row]) is not None
        rows.append(row)
    selected = select_cell(rows)
    return {
        "complete": True,
        "bank_sha256": bank_sha256(),
        "measure_bank": MEASURE_BANK,
        "grid": [{"k": k, "temperature": temperature} for k, temperature in GRID],
        "participation_min": PARTICIPATION_MIN,
        "n_items": len(ordered),
        "top1_dz": top_score,
        "top1_family_dz": top_dz,
        "cells": rows,
        "selected": selected,
        "records": ordered,
    }


def score_fit(mem, *, progress=None, save=None, seed=None) -> dict:
    """Margins on the fit probes only. Does not read the held banks."""
    sha = bank_sha256()
    if seed and seed.get("bank_sha256") not in (None, sha):
        raise RuntimeError("fit checkpoint is for a different bank")
    done = dict((seed or {}).get("items") or {})
    probes = _fit_probes()
    pending = [probe for probe in probes if probe["id"] not in done]
    if pending:
        preflight_groups(mem, pending, progress=progress)
    encoder = Encoder(mem)
    for index, probe in enumerate(probes):
        if probe["id"] in done:
            continue
        if progress is not None:
            progress(f"fit {index}/{len(probes)} {probe['id']}")
        mem.store.clear()
        nomem = _score_item(mem, probe)
        encoded = encoder.many(_group_rows(probe))
        cells = {}
        for k, temperature in GRID:
            written = write_rows(mem, encoded)
            if int(written["slots"]) != 4:
                raise RuntimeError(f"{probe['id']} kept {written['slots']} slots at {k}:{temperature}")
            with _Readout(mem, "mixture", k, temperature):
                cells[cell_key(k, temperature)] = _score_item(mem, probe)
        written = write_rows(mem, encoded)
        if int(written["slots"]) != 4:
            raise RuntimeError(f"{probe['id']} kept {written['slots']} slots on top-1")
        with _Readout(mem, "top1"):
            top1 = _score_item(mem, probe)
        mem.store.clear()
        done[probe["id"]] = {
            "id": probe["id"],
            "family": probe["family"],
            "nomem_margin": nomem["margin"],
            "top1": top1,
            "cells": cells,
        }
        if save is not None:
            save({"bank_sha256": sha, "items": done})
    ordered = [done[probe["id"]] for probe in probes]
    mem.store.clear()
    return summarize_fit(ordered)


class _DirectionBank:
    def __init__(self, prefer, avoid):
        self.PREFER = tuple(prefer)
        self.AVOID = tuple(avoid)


def _held_implicit_probes():
    return tone_items() + choice_items() + preference_items()


def summarize_held_implicit(records) -> dict:
    ordered = list(records)
    soft_dz = _family_dz(ordered, lambda record: record["mixture"]["margin"])
    top_dz = _family_dz(ordered, lambda record: record["top1"]["margin"])
    soft_score = subconscious_score(soft_dz)
    top_score = subconscious_score(top_dz)
    slots_ok = all(int(record["mixture"]["n"]) == 4 and int(record["top1"]["n"]) == 4 for record in ordered)
    passed = (
        slots_ok
        and soft_score is not None
        and top_score is not None
        and float(soft_score) >= float(top_score)
    )
    return {
        "n": len(ordered),
        "slots_ok": bool(slots_ok),
        "soft_dz": soft_score,
        "top1_dz": top_score,
        "soft_family_dz": soft_dz,
        "top1_family_dz": top_dz,
        "pass": bool(passed),
        "participation_mean": _mean([record["mixture"]["participation"] for record in ordered]),
        "hit": rate_row(sum(int(record["mixture"]["hit"]) for record in ordered), len(ordered)) if ordered else None,
        "top1_confidence_mean": _mean([record["top1"]["confidence"] for record in ordered]),
        "top1_gap_mean": _mean([record["top1"]["gap"] for record in ordered]),
    }


def _direction_rate(summary, key):
    if key == "prefer_delta_pos":
        return rate_row(int(summary["n_delta_pos"]), int(summary["n"]))
    if key == "avoid_delta_pos":
        return rate_row(int(summary["n_delta_pos"]), int(summary["n"]))
    if key == "avoid_stored_down":
        return rate_row(int(summary["n_stored_down"]), int(summary["n"]))
    raise KeyError(key)


def _signed_rate(summary, *, positive: bool):
    k = int(summary["n_pos"] if positive else summary["n_neg"])
    return rate_row(k, int(summary["n"]))


def held_rate_table(direction, valence) -> dict:
    """Point estimates against the measurement Wilson intervals."""
    rows = {
        "prefer_delta_pos": _direction_rate(direction["prefer"]["summary"], "prefer_delta_pos"),
        "avoid_delta_pos": _direction_rate(direction["avoid"]["summary"], "avoid_delta_pos"),
        "avoid_stored_down": _direction_rate(direction["avoid"]["summary"], "avoid_stored_down"),
        "toward_signed": _signed_rate(valence["toward"], positive=True),
        "away_signed": _signed_rate(valence["away"], positive=False),
    }
    judged = {}
    for name, row in rows.items():
        low, high = MEASURE_INTERVALS[name]
        judged[name] = {
            "rate": row,
            "measure_k": MEASURE_COUNTS[name][0],
            "measure_n": MEASURE_COUNTS[name][1],
            "measure_wilson95": [low, high],
            "inside": rate_inside(row["k"], row["n"], (low, high)),
        }
    judged["pass"] = all(item["inside"] for item in judged.values() if isinstance(item, dict) and "inside" in item)
    return judged


def score_explicit_family(mem, items, *, progress=None) -> dict:
    rows = []
    for index, item in enumerate(items):
        if progress is not None:
            progress(f"explicit {item['family']} {index}")
        mem.store.clear()
        if not mem.remember(_fact(item)):
            raise RuntimeError(f"{item['id']} was not written")
        text = mem.generate(item["prompt"], max_new_tokens=32)
        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
                "exact": int(text == item["answer"]),
                "text": text,
                "answer": item["answer"],
            }
        )
    mem.store.clear()
    n = len(rows)
    return {
        "n": n,
        "exact": rate_row(sum(row["exact"] for row in rows), n) if n else None,
        "rows": rows,
        "excluded_from_subconscious_score": True,
        "note": "generate copies the stored phone or birthday when the prompt completes that relation",
    }


def _load_seed(seed):
    if not seed:
        return {}
    if seed.get("bank_sha256") not in (None, bank_sha256()):
        raise RuntimeError("held checkpoint is for a different bank")
    return seed


def score_held(mem, fit_report, *, progress=None, save=None, seed=None) -> dict:
    """One held pass. Refuses a second complete report and a mismatched selection."""
    allowed, reason = held_allowed(fit_report, MIX_K, MIX_TEMP, False)
    if not allowed:
        raise RuntimeError(reason)
    state = _load_seed(seed)
    state.setdefault("implicit", [])
    encoder = Encoder(mem)
    probes = _held_implicit_probes()
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
    implicit = summarize_held_implicit(state["implicit"])

    def _save_direction():
        if save is not None:
            save(state)

    if "prefer" not in state:
        state["prefer"] = []
        _save_direction()
    for index, item in enumerate(prefer_items()):
        if index < len(state["prefer"]):
            continue
        if progress is not None:
            progress(f"prefer {index}")
        part = score_direction(mem, _DirectionBank([item], []))
        state["prefer"].append(part["prefer"]["rows"][0])
        _save_direction()
    if "avoid" not in state:
        state["avoid"] = []
        _save_direction()
    for index, item in enumerate(avoid_items()):
        if index < len(state["avoid"]):
            continue
        if progress is not None:
            progress(f"avoid {index}")
        part = score_direction(mem, _DirectionBank([], [item]))
        state["avoid"].append(part["avoid"]["rows"][0])
        _save_direction()
    direction = {
        "prefer": {"rows": state["prefer"], "summary": _family_direction(state["prefer"])},
        "avoid": {"rows": state["avoid"], "summary": _family_direction(state["avoid"])},
    }

    if "toward" not in state:
        state["toward"] = []
        _save_direction()
    for index, item in enumerate(toward_items()):
        if index < len(state["toward"]):
            continue
        if progress is not None:
            progress(f"toward {index}")
        part = score_signed(mem, [item], 1.0, int(REPEATS))
        state["toward"].append(part["rows"][0])
        _save_direction()
    if "away" not in state:
        state["away"] = []
        _save_direction()
    for index, item in enumerate(away_items()):
        if index < len(state["away"]):
            continue
        if progress is not None:
            progress(f"away {index}")
        part = score_signed(mem, [item], -1.0, int(REPEATS))
        state["away"].append(part["rows"][0])
        _save_direction()
    valence = {
        "repeats": int(REPEATS),
        "toward": _family_signed(state["toward"]),
        "away": _family_signed(state["away"]),
    }
    rates = held_rate_table(direction, valence)
    if "phone" not in state:
        state["phone"] = score_explicit_family(mem, phone_items(), progress=progress)
        _save_direction()
    if "birthday" not in state:
        state["birthday"] = score_explicit_family(mem, birthday_items(), progress=progress)
        _save_direction()
    mem.store.clear()
    return {
        "complete": True,
        "bank_sha256": bank_sha256(),
        "measure_bank": MEASURE_BANK,
        "mix_k": int(MIX_K),
        "mix_temp": float(MIX_TEMP),
        "selected": fit_report["selected"],
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
        "pass": bool(implicit["pass"] and rates["pass"]),
    }


def _family_direction(rows) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    lifts = [float(row["stored_lift"]) for row in rows]
    return {
        "n": len(rows),
        "mean_delta": _mean(deltas),
        "n_delta_pos": int(sum(value > 0.0 for value in deltas)),
        "mean_stored_lift": _mean(lifts),
        "n_stored_down": int(sum(value < 0.0 for value in lifts)),
    }


def _family_signed(rows) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    return {
        "n": len(rows),
        "mean_delta": _mean(deltas),
        "n_pos": int(sum(value > 0.0 for value in deltas)),
        "n_neg": int(sum(value < 0.0 for value in deltas)),
        "n_audit": int(sum(bool(row["audit"]) for row in rows)),
        "rows": rows,
    }
