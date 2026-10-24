"""Measurement grid for fixed K=4 against cumulative top-p.

Production ``MIX_K`` and ``MIX_TEMP`` stay at 4 and 0.05. A top-p cell
patches ``activation_weights`` for the duration of one arm and then restores
it. Fit probes and held probes are separate lists. Geometry encodes keys and
does not score a continuation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tidal.llm_memory import MIX_K, MIX_TEMP, activation_weights, participation_ratio
from tidal.subconscious_eval import Encoder, continuation_logprobs
from tidal.subconscious_kablation_data import (
    COMPANION_NS,
    FAMILIES,
    FILLER_N,
    SELECTORS,
    SIMILAR_N,
    STORE_SIZES,
    TEMPERATURES,
    TOP_P,
    all_probes,
    bank_sha256,
    companion_facts,
    fit_probes,
    held_probes,
    store_facts,
)
from tidal.subconscious_measure_data import cohens_dz, rate_row, subconscious_score

CELLS = tuple(
    {"mode": row["mode"], "k": row.get("k"), "k_max": row.get("k_max", row.get("k")), "temperature": temperature}
    for row in SELECTORS
    for temperature in TEMPERATURES
)


def cell_key(cell: dict) -> str:
    if cell["mode"] == "fixed":
        return f"fixed:{int(cell['k'])}:{float(cell['temperature']):.2f}"
    return f"top_p:{int(cell['k_max'])}:{float(cell['temperature']):.2f}"


def cumulative_weights(cosines, strengths, *, temperature: float, k_max: int, p: float = TOP_P) -> np.ndarray:
    """Smallest cosine-order prefix whose softmax mass reaches ``p``, capped at ``k_max``.

    The softmax is over every positive-strength slot. The kept prefix is
    renormalized, so a prefix that matches a fixed-k set has the same weights
    as :func:`tidal.llm_memory.activation_weights` on that set. Ties break
    toward the earlier slot.
    """
    if not 0.0 < float(p) <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    take = int(k_max)
    if take < 1:
        raise ValueError("k_max must be >= 1")
    if float(temperature) <= 0.0:
        raise ValueError("mixture temperature must be positive")
    cosine = np.asarray(cosines, dtype=np.float64).reshape(-1)
    strength = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if cosine.shape != strength.shape:
        raise ValueError("cosines and strengths must have the same shape")
    weights = np.zeros(cosine.shape[0], dtype=np.float64)
    eligible = [index for index in range(int(cosine.size)) if float(strength[index]) > 0.0]
    if not eligible:
        return weights
    temp = float(temperature)
    logits = np.asarray(
        [(np.log(float(strength[index])) + float(cosine[index])) / temp for index in eligible],
        dtype=np.float64,
    )
    logits -= logits.max()
    raw = np.exp(logits)
    raw_sum = float(raw.sum())
    if raw_sum <= 0.0:
        return weights
    raw /= raw_sum
    order = sorted(range(len(eligible)), key=lambda pos: (-float(raw[pos]), eligible[pos]))
    chosen = []
    cum = 0.0
    limit = float(p)
    for pos in order:
        chosen.append(pos)
        cum += float(raw[pos])
        # 1e-12 absorbs a binary sum that lands just below p, such as 9 * 0.1.
        if len(chosen) >= take or cum + 1e-12 >= limit:
            break
    kept = raw[np.asarray(chosen, dtype=np.int64)]
    kept = kept / kept.sum()
    for pos, weight in zip(chosen, kept):
        weights[eligible[pos]] = float(weight)
    return weights


class Selector:
    """Temporary temperature, and a top-p patch that fixed-k does not install."""

    def __init__(self, cell: dict):
        self.cell = cell
        self._temp = None
        self._fn = None
        self._module = None

    def __enter__(self):
        import tidal.llm_memory as lm

        self._module = lm
        self._temp = lm.MIX_TEMP
        self._fn = lm.activation_weights
        lm.MIX_TEMP = float(self.cell["temperature"])
        if self.cell["mode"] == "top_p":
            k_max = int(self.cell["k_max"])

            def patched(cosines, strengths, *, temperature, k, _k_max=k_max):
                del k
                return cumulative_weights(cosines, strengths, temperature=temperature, k_max=_k_max, p=TOP_P)

            lm.activation_weights = patched
        elif self.cell["mode"] != "fixed":
            raise ValueError(f"unknown selector {self.cell['mode']}")
        return self

    def __exit__(self, *_exc):
        self._module.MIX_TEMP = self._temp
        self._module.activation_weights = self._fn
        return False


class _Calls:
    def __init__(self, mem):
        self.mem = mem
        self.mlp_runs = 0
        self.broadcast_tokens = 0
        self._forward = None
        self._add = None
        self._module = None

    def __enter__(self):
        import tidal.llm_memory as lm

        self._module = lm
        inject = self.mem.inject
        forward = inject.forward

        def wrapped_forward(*args, **kwargs):
            self.mlp_runs += 1
            return forward(*args, **kwargs)

        inject.forward = wrapped_forward
        self._forward = forward
        add = lm._add_delta

        def wrapped_add(output, delta):
            hidden = output[0] if isinstance(output, tuple) else output
            if hasattr(hidden, "shape") and len(hidden.shape) >= 2:
                self.broadcast_tokens += int(hidden.shape[-2])
            return add(output, delta)

        lm._add_delta = wrapped_add
        self._add = add
        return self

    def __exit__(self, *_exc):
        self.mem.inject.forward = self._forward
        self._module._add_delta = self._add
        return False


def dump_json(path, payload) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest)


def _cosine(left, right):
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if nx < 1e-8 or ny < 1e-8:
        return None
    return float(np.dot(x, y) / (nx * ny))


def pearson(xs, ys):
    """Pearson r. None when fewer than 3 finite pairs or either side is constant."""
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None, len(pairs)
    x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None, int(x.size)
    return float(np.corrcoef(x, y)[0, 1]), int(x.size)


def _mean(vals):
    xs = [float(x) for x in vals if x is not None]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _median(vals):
    xs = sorted(float(x) for x in vals)
    if not xs:
        return None
    mid = len(xs) // 2
    if len(xs) % 2:
        return float(xs[mid])
    return float(0.5 * (xs[mid - 1] + xs[mid]))


def _write(mem, rows: list, group: str) -> int:
    mem.store.clear()
    mem._delta = None
    for row in rows:
        ok = mem.store.write(
            row["key"],
            row["value"],
            name=row["name"],
            rel=row["rel"],
            kind="fact",
            group=group,
            answer=row["answer"],
            text=row["text"],
        )
        if not ok:
            raise RuntimeError(f"write rejected {row['name']}")
    if len(mem.store) != len(rows):
        raise RuntimeError(f"store kept {len(mem.store)} of {len(rows)}")
    return len(mem.store)


def _encode(encoder: Encoder, facts: list) -> list:
    return [encoder.one(fact) for fact in facts]


def score_margin(mem, prompt: str, good: str, bad: str, *, name: str = None, rel: str = None) -> dict:
    """One arm. Strengths are restored so retrieval-induced forgetting does not carry over."""
    saved = [float(slot.strength) for slot in mem.store.slots]
    calls = _Calls(mem)
    try:
        with calls:
            rank = mem._rank(prompt)
            mem._set_delta(rank)
            delta = None
            if mem._delta is not None:
                delta = mem._delta.detach().float().cpu().numpy().copy()
            good_row = continuation_logprobs(mem, prompt, good)
            bad_row = continuation_logprobs(mem, prompt, bad)
    finally:
        mem._delta = None
        for slot, strength in zip(mem.store.slots, saved):
            slot.strength = float(strength)
    weights = rank.get("weights") or []
    active = sum(float(weight) > 0.0 for weight in weights)
    top = rank.get("top")
    mlp = int(calls.mlp_runs)
    return {
        "margin": float(good_row["mean"] - bad_row["mean"]),
        "n": int(rank["n"]),
        "hit": int(list(top) == [name, rel]) if top is not None and name is not None else 0,
        "participation": participation_ratio(weights) if weights else 0.0,
        "active": int(active),
        "mlp_runs": mlp,
        "broadcast_tokens": int(calls.broadcast_tokens),
        "mlp_runs_per_neighbor": None if active == 0 else float(mlp / active),
        "mlp_runs_per_broadcast_token": None
        if calls.broadcast_tokens == 0
        else float(mlp / calls.broadcast_tokens),
        "delta": delta,
    }


def _public_margin(row: dict) -> dict:
    return {key: value for key, value in row.items() if key != "delta"}


def _family_dz(rows, family_of, margin_of, nomem_of):
    out = {}
    for family in FAMILIES:
        diffs = []
        for row in rows:
            if family_of(row) != family:
                continue
            diffs.append(float(margin_of(row)) - float(nomem_of(row)))
        out[family] = cohens_dz(diffs)
    return out


def summarize_split(records: list) -> dict:
    implicit_rows = []
    for size in STORE_SIZES:
        for cell in CELLS:
            key = cell_key(cell)
            chosen = []
            for record in records:
                for row in record["implicit"]:
                    if int(row["size"]) == int(size) and row["cell"] == key:
                        chosen.append((record, row))
            margins = _family_dz(
                chosen,
                lambda item: item[0]["family"],
                lambda item: item[1]["margin"],
                lambda item: item[0]["nomem_margin"],
            )
            help_xs = []
            align_ys = []
            for record, row in chosen:
                help_xs.append(float(row["margin"]) - float(record["target_only_margin"]))
                align_ys.append(row["alignment"])
            corr, corr_n = pearson(align_ys, help_xs)
            implicit_rows.append(
                {
                    "size": int(size),
                    "cell": key,
                    "mode": cell["mode"],
                    "k": None if cell["k"] is None else int(cell["k"]),
                    "k_max": int(cell["k_max"]),
                    "temperature": float(cell["temperature"]),
                    "n": len(chosen),
                    "dz": subconscious_score(margins),
                    "tone": margins["tone"],
                    "choice": margins["choice"],
                    "preference": margins["preference"],
                    "participation": _mean([row["participation"] for _record, row in chosen]),
                    "active": _mean([row["active"] for _record, row in chosen]),
                    "mlp_runs": _mean([row["mlp_runs"] for _record, row in chosen]),
                    "mlp_runs_per_neighbor": _mean([row["mlp_runs_per_neighbor"] for _record, row in chosen]),
                    "mlp_runs_per_broadcast_token": _mean(
                        [row["mlp_runs_per_broadcast_token"] for _record, row in chosen]
                    ),
                    "alignment": _mean([row["alignment"] for _record, row in chosen]),
                    "help_mean": _mean(help_xs),
                    "pearson_r": corr,
                    "pearson_n": int(corr_n),
                    "hit": _mean([row["hit"] for _record, row in chosen]),
                }
            )
    fixed = {
        (row["size"], row["temperature"]): row
        for row in implicit_rows
        if row["mode"] == "fixed"
    }
    comparisons = []
    for row in implicit_rows:
        if row["mode"] != "top_p":
            continue
        base = fixed[(row["size"], row["temperature"])]
        ready = None not in (row["alignment"], base["alignment"], row["dz"], base["dz"])
        alignment_lower = bool(ready and row["alignment"] < base["alignment"])
        score_higher = bool(ready and row["dz"] > base["dz"])
        comparisons.append(
            {
                "size": row["size"],
                "cell": row["cell"],
                "fixed_cell": base["cell"],
                "alignment": row["alignment"],
                "fixed_alignment": base["alignment"],
                "dz": row["dz"],
                "fixed_dz": base["dz"],
                "alignment_lower": alignment_lower,
                "score_higher": score_higher,
                "implication_holds": None if not ready else not (alignment_lower and score_higher),
            }
        )
    interference_rows = []
    contrasts = []
    for cell in CELLS:
        key = cell_key(cell)
        for band in ("high", "low"):
            for n_companions in COMPANION_NS:
                chosen = []
                for record in records:
                    for row in record["interference"]:
                        if row["cell"] == key and row["band"] == band and int(row["n_companions"]) == int(n_companions):
                            chosen.append((record, row))
                deltas = [float(row["margin"]) - float(record["nomem_margin"]) for record, row in chosen]
                interference_rows.append(
                    {
                        "cell": key,
                        "band": band,
                        "n_companions": int(n_companions),
                        "n": len(chosen),
                        "mean_delta": _mean(deltas),
                        "mean_key_cosine": _mean([row["mean_key_cosine"] for _record, row in chosen]),
                        "participation": _mean([row["participation"] for _record, row in chosen]),
                        "active": _mean([row["active"] for _record, row in chosen]),
                        "mlp_runs": _mean([row["mlp_runs"] for _record, row in chosen]),
                    }
                )
        for n_companions in COMPANION_NS:
            wins = 0
            total = 0
            for record in records:
                high = _interference_margin(record, key, "high", n_companions)
                low = _interference_margin(record, key, "low", n_companions)
                if high is None or low is None:
                    continue
                total += 1
                wins += int(high < low)
            contrasts.append(
                {
                    "cell": key,
                    "name": f"similarity_n{n_companions}",
                    "detail": "high-similarity margin strictly below low-similarity margin",
                    **(_empty_rate() if total == 0 else rate_row(wins, total)),
                }
            )
        for band in ("high", "low"):
            wins = 0
            total = 0
            for record in records:
                few = _interference_margin(record, key, band, 1)
                many = _interference_margin(record, key, band, 7)
                if few is None or many is None:
                    continue
                total += 1
                wins += int(many < few)
            contrasts.append(
                {
                    "cell": key,
                    "name": f"count_{band}_7_below_1",
                    "detail": "seven companions strictly below one companion",
                    **(_empty_rate() if total == 0 else rate_row(wins, total)),
                }
            )
    return {
        "implicit_rows": implicit_rows,
        "comparisons": comparisons,
        "interference_rows": interference_rows,
        "contrasts": contrasts,
        "target_only_mlp_runs": _mean([record["target_only_mlp_runs"] for record in records]),
        "nomem_mlp_runs": _mean([record["nomem_mlp_runs"] for record in records]),
        "n_probes": len(records),
    }


def _empty_rate() -> dict:
    return {"k": 0, "n": 0, "rate": None, "wilson95": None}


def _interference_margin(record, key, band, n_companions):
    for row in record["interference"]:
        if row["cell"] == key and row["band"] == band and int(row["n_companions"]) == int(n_companions):
            return float(row["margin"])
    return None


def _geometry_stats(values: list) -> dict:
    return {
        "n": len(values),
        "min": None if not values else float(min(values)),
        "median": _median(values),
        "max": None if not values else float(max(values)),
        "mean": _mean(values),
    }


def collect_geometry(mem, probes=None, progress=None) -> dict:
    """Key cosines, slot counts, and tokenizer prefixes. No continuation margin."""
    from tidal.episodic import EpisodicStore, token_jaccard

    chosen = all_probes() if probes is None else tuple(probes)
    rows = []
    prefix_fail = 0
    separated = 0
    argmax_hits = {str(size): 0 for size in STORE_SIZES}
    similar_values = []
    filler_values = []
    gaps = []
    jaccard_max = 0.0
    for probe_index, probe in enumerate(chosen):
        if progress is not None and probe_index % 6 == 0:
            progress(f"geometry {probe_index}/{len(chosen)} {probe['id']}")
        facts = store_facts(probe, 32)
        keys = [mem.encode_key(fact["text"], fact["answer"]) for fact in facts]
        query = mem._query_vector(probe["prompt"])
        target_key = keys[0]
        similar_n = SIMILAR_N[32]
        similar = [_cosine(target_key, keys[index]) for index in range(1, 1 + similar_n)]
        fillers = [_cosine(target_key, keys[index]) for index in range(1 + similar_n, len(keys))]
        similar_values.extend(similar)
        filler_values.extend(fillers)
        gap = float(min(similar) - max(fillers))
        gaps.append(gap)
        separated += int(gap > 0.0)
        store = EpisodicStore(capacity=64, dk=4, dv=4)
        ones = np.ones(4, dtype=np.float64)
        for fact in facts:
            store.write(ones, ones, name=fact["name"], rel=fact["rel"], group=probe["group"], answer=fact["answer"], text=fact["text"])
            left = fact["text"]
            for other in facts:
                if other["name"] == fact["name"]:
                    continue
                jaccard_max = max(jaccard_max, token_jaccard(left, other["text"]))
        query_cos = [_cosine(query, key) for key in keys]
        argmax = {}
        query_gap = {}
        slot_counts = {"32": len(store)}
        for size in STORE_SIZES:
            if size == 32:
                indexes = list(range(32))
            else:
                indexes = [0] + list(range(1, 1 + SIMILAR_N[size])) + list(range(16, 16 + FILLER_N[size]))
            scored = [(query_cos[index], index) for index in indexes]
            winner = max(scored, key=lambda item: (item[0], -item[1]))
            argmax[str(size)] = bool(winner[1] == 0 and scored[0][0] > max(item[0] for item in scored[1:]))
            query_gap[str(size)] = float(scored[0][0] - max(item[0] for item in scored[1:]))
            if argmax[str(size)]:
                argmax_hits[str(size)] += 1
            if size != 32:
                slot_counts[str(size)] = None
        prompt_ids = mem.tokenizer.encode(probe["prompt"], add_special_tokens=True)
        prefix_ok = True
        for continuation in (probe["good"], probe["bad"]):
            full_ids = mem.tokenizer.encode(probe["prompt"] + continuation, add_special_tokens=True)
            if full_ids[: len(prompt_ids)] != prompt_ids:
                prefix_ok = False
        prefix_fail += int(not prefix_ok)
        rows.append(
            {
                "id": probe["id"],
                "family": probe["family"],
                "split": probe["split"],
                "separated": bool(gap > 0.0),
                "similar_min": float(min(similar)),
                "similar_median": _median(similar),
                "filler_max": float(max(fillers)),
                "gap": gap,
                "argmax": argmax,
                "query_gap": query_gap,
                "slots_32": int(len(store)),
                "prefix_ok": bool(prefix_ok),
            }
        )
        del slot_counts
    return {
        "complete": True,
        "bank_sha256": bank_sha256(),
        "margins_scored": False,
        "n_probes": len(rows),
        "separated": int(separated),
        "separation_ok": bool(separated == len(rows) and len(rows) > 0 and all(row["slots_32"] == 32 for row in rows)),
        "prefix_fail": int(prefix_fail),
        "prefix_ok": bool(prefix_fail == 0 and len(rows) > 0),
        "argmax_hits": argmax_hits,
        "slots_32_min": None if not rows else min(row["slots_32"] for row in rows),
        "jaccard_max_within_probe": float(jaccard_max),
        "similar_cosine": _geometry_stats(similar_values),
        "filler_cosine": _geometry_stats(filler_values),
        "gap": _geometry_stats(gaps),
        "rows": rows,
    }


def _score_probe(mem, encoder: Encoder, probe: dict, *, progress=None) -> dict:
    if progress is not None:
        progress(f"probe {probe['id']}")
    encoded = {fact["name"]: encoder.one(fact) for fact in store_facts(probe, 32)}
    mem.store.clear()
    mem._delta = None
    nomem = score_margin(mem, probe["prompt"], probe["good"], probe["bad"], name=probe["target"]["name"], rel=probe["rel"])
    _write(mem, [encoded[probe["target"]["name"]]], probe["group"])
    alone = score_margin(mem, probe["prompt"], probe["good"], probe["bad"], name=probe["target"]["name"], rel=probe["rel"])
    alone_delta = alone.pop("delta")
    implicit = []
    for size in STORE_SIZES:
        facts = store_facts(probe, size)
        _write(mem, [encoded[fact["name"]] for fact in facts], probe["group"])
        for cell in CELLS:
            with Selector(cell):
                scored = score_margin(
                    mem,
                    probe["prompt"],
                    probe["good"],
                    probe["bad"],
                    name=probe["target"]["name"],
                    rel=probe["rel"],
                )
            delta = scored.pop("delta")
            implicit.append(
                {
                    "size": int(size),
                    "cell": cell_key(cell),
                    "alignment": _cosine(delta, alone_delta),
                    **_public_margin(scored),
                }
            )
    interference = []
    target_vec = encoded[probe["target"]["name"]]["key"]
    for band in ("high", "low"):
        for n_companions in COMPANION_NS:
            facts = companion_facts(probe, band, n_companions)
            companions = facts[1:]
            cosines = [_cosine(target_vec, encoded[fact["name"]]["key"]) for fact in companions]
            _write(mem, [encoded[fact["name"]] for fact in facts], probe["group"])
            for cell in CELLS:
                with Selector(cell):
                    scored = score_margin(
                        mem,
                        probe["prompt"],
                        probe["good"],
                        probe["bad"],
                        name=probe["target"]["name"],
                        rel=probe["rel"],
                    )
                scored.pop("delta", None)
                interference.append(
                    {
                        "band": band,
                        "n_companions": int(n_companions),
                        "cell": cell_key(cell),
                        "mean_key_cosine": _mean(cosines),
                        **_public_margin(scored),
                    }
                )
    mem.store.clear()
    mem._delta = None
    return {
        "id": probe["id"],
        "family": probe["family"],
        "split": probe["split"],
        "nomem_margin": float(nomem["margin"]),
        "nomem_mlp_runs": int(nomem["mlp_runs"]),
        "target_only_margin": float(alone["margin"]),
        "target_only_mlp_runs": int(alone["mlp_runs"]),
        "target_only_active": int(alone["active"]),
        "implicit": implicit,
        "interference": interference,
    }


def _run(mem, probes, *, progress=None, save=None, seed=None) -> dict:
    done = {}
    if isinstance(seed, dict):
        for record in seed.get("records") or []:
            done[record["id"]] = record
    encoder = Encoder(mem)
    records = []
    for index, probe in enumerate(probes):
        if probe["id"] in done:
            records.append(done[probe["id"]])
            continue
        record = _score_probe(mem, encoder, probe, progress=progress)
        records.append(record)
        if save is not None:
            save({"complete": False, "n_done": len(records), "n_probes": len(probes), "records": records})
        if progress is not None:
            progress(f"done {index + 1}/{len(probes)} {probe['id']}")
    summary = summarize_split(records)
    return {
        "complete": True,
        "bank_sha256": bank_sha256(),
        "top_p": TOP_P,
        "mix_k": int(MIX_K),
        "mix_temp": float(MIX_TEMP),
        "cells": [{"cell": cell_key(cell), **{key: cell[key] for key in ("mode", "k", "k_max", "temperature")}} for cell in CELLS],
        "store_sizes": list(STORE_SIZES),
        "summary": summary,
        "records": records,
    }


def score_fit(mem, *, progress=None, save=None, seed=None) -> dict:
    """Margins on the fit split only."""
    report = _run(mem, fit_probes(), progress=progress, save=save, seed=seed)
    report["split"] = "fit"
    return report


def score_held(mem, *, progress=None, save=None, seed=None) -> dict:
    """Margins on the held split only."""
    report = _run(mem, held_probes(), progress=progress, save=save, seed=seed)
    report["split"] = "held"
    return report


def _cell_ids(report) -> list:
    return [row["cell"] for row in report.get("cells") or []]


def held_allowed(fit_report, geometry, held_complete: bool):
    """Held runs once, after fit, on the same bank, without a new default."""
    if held_complete:
        return False, "held report already exists"
    if int(MIX_K) != 4 or float(MIX_TEMP) != 0.05:
        return False, "production mixture constants changed"
    if not isinstance(geometry, dict) or not geometry.get("complete"):
        return False, "geometry report is not complete"
    if not geometry.get("separation_ok") or not geometry.get("prefix_ok"):
        return False, "geometry did not keep similar memories apart from fillers"
    if geometry.get("margins_scored"):
        return False, "geometry scored a margin"
    if not isinstance(fit_report, dict) or not fit_report.get("complete"):
        return False, "fit report is not complete"
    if fit_report.get("split") != "fit":
        return False, "fit report is not the fit split"
    if fit_report.get("bank_sha256") != bank_sha256() or geometry.get("bank_sha256") != bank_sha256():
        return False, "bank sha changed after the report"
    if float(fit_report.get("top_p")) != float(TOP_P):
        return False, "top-p changed after fit"
    if int(fit_report.get("mix_k")) != 4 or float(fit_report.get("mix_temp")) != 0.05:
        return False, "fit was not scored at the production defaults"
    expected = [cell_key(cell) for cell in CELLS]
    if _cell_ids(fit_report) != expected:
        return False, "fit grid does not match the preregistered cells"
    return True, ""


def production_weights_unchanged(cosines, strengths) -> bool:
    """The unpatched fixed-k rule is still the shipped function at the shipped constants."""
    import tidal.llm_memory as lm

    if lm.activation_weights is not activation_weights:
        return False
    if int(lm.MIX_K) != 4 or float(lm.MIX_TEMP) != 0.05:
        return False
    left = activation_weights(cosines, strengths, temperature=0.05, k=4)
    right = lm.activation_weights(cosines, strengths, temperature=lm.MIX_TEMP, k=lm.MIX_K)
    return bool(np.allclose(left, right))
