"""Score the additional measurement banks. No parameter is written.

The locked n=8 banks stay where they are. Rates, key cosines, implicit
margins, and explicit spelling all go through the functions that already
ship: ``score_direction``, ``score_signed``, ``score_gate``, ``score_quartets``,
``score_pair``, and ``generate``.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from tidal.llm_memory import (
    DEFAULT_OVERWRITE_COS,
    _logits_of,
    completing_relation,
)
from tidal.subconscious_eval import score_pair, token_ids
from tidal.subconscious_measure_data import (
    SurpriseMeasure,
    avoid_items,
    away_items,
    birthday_items,
    choice_items,
    cohens_dz,
    phone_items,
    prefer_items,
    preference_items,
    rate_row,
    rif_quartets,
    subconscious_score,
    surprise_highs,
    surprise_lows,
    surprise_seeds,
    tone_items,
    toward_items,
)
from tidal.subconscious_replay_data import held_bank as replay_held_bank
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_rif_retry_score import score_quartets
from tidal.subconscious_surprise_rel_score import score_gate
from tidal.subconscious_valence_data import REPEATS
from tidal.subconscious_valence_score import score_signed

HIST_BINS = [-1.0, 0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]
OUTSIDER_EPS = 1e-6


def _mean(vals):
    xs = [float(x) for x in vals]
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _sd(vals):
    xs = [float(x) for x in vals]
    n = len(xs)
    if n < 2:
        return None
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return float(math.sqrt(var))


def _fact(item):
    return {
        "text": item["text"],
        "answer": item["answer"],
        "name": item["name"],
        "rel": item["rel"],
    }


def _pairwise(vectors) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float64)
    mat = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors], 0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.maximum(norms, 1e-12)
    sims = unit @ unit.T
    idx = np.triu_indices(unit.shape[0], k=1)
    return sims[idx]


def cosine_distribution(values) -> dict:
    """Pairwise cosine summary. ``all_gt_0.99`` is true only when every pair is."""
    arr = np.asarray(list(values), dtype=np.float64).reshape(-1)
    n_pairs = int(arr.size)
    if n_pairs == 0:
        hist = [0] * (len(HIST_BINS) - 1)
        return {
            "n_pairs": 0,
            "all_gt_0.99": False,
            "hist_bins": list(HIST_BINS),
            "hist_counts": hist,
            "overflow": 0,
        }
    inside = arr[(arr >= -1.0) & (arr <= 1.0)]
    counts, _edges = np.histogram(inside, bins=HIST_BINS)
    qs = {
        "min": 0.0,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
        "max": 1.0,
    }
    out = {
        "n_pairs": n_pairs,
        "mean": float(arr.mean()),
        "all_gt_0.99": bool(np.all(arr > 0.99)),
        "n_gt_0.99": int(np.sum(arr > 0.99)),
        "frac_gt_0.99": float(np.mean(arr > 0.99)),
        "n_gt_0.9": int(np.sum(arr > 0.9)),
        "frac_gt_0.9": float(np.mean(arr > 0.9)),
        "hist_bins": list(HIST_BINS),
        "hist_counts": [int(x) for x in counts.tolist()],
        "overflow": int(n_pairs - inside.size),
    }
    for name, q in qs.items():
        out[name] = float(np.quantile(arr, q))
    return out


def _key_rows():
    replay = replay_held_bank()
    valence = valence_held_bank()
    rows = []
    for source, items in (
        ("measure-prefer", prefer_items()),
        ("measure-avoid", avoid_items()),
        ("replay-held-prefer", list(replay.PREFER)),
        ("replay-held-avoid", list(replay.AVOID)),
        ("valence-held-toward", list(valence.TOWARD)),
        ("valence-held-away", list(valence.AWAY)),
    ):
        for item in items:
            rows.append((source, item))
    return rows


def score_key_cosines(mem, *, progress=None) -> dict:
    """Encoder keys, then the slots ``remember`` actually keeps.

    The encoder pass subtracts the fixed key center and does not write.
    ``remember`` then applies the surprise gate and cosine overwrite.
    A third pass writes those same encoder keys with ``store.write``, so the
    overwrite rule can be seen without the surprise gate.
    """
    rows = _key_rows()
    encoded = []
    by_source = {}
    for index, (source, item) in enumerate(rows):
        if progress is not None and index % 20 == 0:
            progress(f"encode_key {index}/{len(rows)} {source}")
        key = np.asarray(mem.encode_key(item["text"], item["answer"]), dtype=np.float64).copy()
        encoded.append(key)
        by_source.setdefault(source, []).append(key)
    new_items = [item for source, item in rows if source.startswith("measure-")]
    new_keys = [key for (source, _item), key in zip(rows, encoded) if source.startswith("measure-")]
    mem.store.clear()
    written = 0
    for index, item in enumerate(new_items):
        if progress is not None and index % 10 == 0:
            progress(f"remember {index}/{len(new_items)}")
        written += int(bool(mem.remember(_fact(item))))
    remembered = [np.asarray(slot.k, dtype=np.float64).copy() for slot in mem.store.slots]
    remembered_n = len(remembered)
    mem.store.clear()
    direct = 0
    for item, key in zip(new_items, new_keys):
        direct += int(
            bool(
                mem.store.write(
                    np.asarray(key, dtype=np.float32),
                    np.zeros(mem.hidden, dtype=np.float32),
                    name=item["name"],
                    rel=item["rel"],
                    kind="fact",
                    answer=item["answer"],
                )
            )
        )
    direct_keys = [np.asarray(slot.k, dtype=np.float64).copy() for slot in mem.store.slots]
    direct_n = len(direct_keys)
    mem.store.clear()
    per_source = {source: cosine_distribution(_pairwise(keys)) for source, keys in by_source.items()}
    for source, dist in per_source.items():
        dist["n_vectors"] = len(by_source[source])
    pooled = cosine_distribution(_pairwise(encoded))
    pooled["n_vectors"] = len(encoded)
    remembered_dist = cosine_distribution(_pairwise(remembered))
    remembered_dist["n_vectors"] = remembered_n
    direct_dist = cosine_distribution(_pairwise(direct_keys))
    direct_dist["n_vectors"] = direct_n
    return {
        "overwrite_cos": float(DEFAULT_OVERWRITE_COS),
        "encoder": pooled,
        "encoder_by_source": per_source,
        "remember": {
            "attempted": len(new_items),
            "written": int(written),
            "n_slots": remembered_n,
            "distribution": remembered_dist,
        },
        "overwrite_only": {
            "attempted": len(new_items),
            "accepted": int(direct),
            "n_slots": direct_n,
            "distribution": direct_dist,
        },
    }


class _DirectionBank:
    def __init__(self, prefer, avoid):
        self.PREFER = tuple(prefer)
        self.AVOID = tuple(avoid)


def _direction_summary(rows) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    lifts = [float(row["stored_lift"]) for row in rows]
    pos = int(sum(value > 0.0 for value in deltas))
    down = int(sum(value < 0.0 for value in lifts))
    n = len(rows)
    return {
        "n": n,
        "mean_delta": _mean(deltas),
        "sd_delta": _sd(deltas),
        "delta_pos": rate_row(pos, n),
        "mean_stored_lift": _mean(lifts),
        "stored_down": rate_row(down, n),
        "rows": rows,
    }


def score_prefer_avoid(mem, *, progress=None, save=None, seed=None) -> dict:
    prefer_rows = list((seed or {}).get("prefer") or [])
    avoid_rows = list((seed or {}).get("avoid") or [])
    state = {"prefer": prefer_rows, "avoid": avoid_rows}
    if save is not None:
        save(state)
    for index, item in enumerate(prefer_items()):
        if index < len(prefer_rows):
            continue
        if progress is not None:
            progress(f"prefer {index}")
        part = score_direction(mem, _DirectionBank([item], []))
        prefer_rows.append(part["prefer"]["rows"][0])
        if save is not None:
            save(state)
    for index, item in enumerate(avoid_items()):
        if index < len(avoid_rows):
            continue
        if progress is not None:
            progress(f"avoid {index}")
        part = score_direction(mem, _DirectionBank([], [item]))
        avoid_rows.append(part["avoid"]["rows"][0])
        if save is not None:
            save(state)
    mem.store.clear()
    return {"prefer": _direction_summary(prefer_rows), "avoid": _direction_summary(avoid_rows)}


def _valence_summary(rows, *, positive: bool) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    n = len(rows)
    hits = int(sum(value > 0.0 for value in deltas)) if positive else int(sum(value < 0.0 for value in deltas))
    return {
        "n": n,
        "mean_delta": _mean(deltas),
        "sd_delta": _sd(deltas),
        "signed_rate": rate_row(hits, n),
        "n_audit": int(sum(bool(row["audit"]) for row in rows)),
        "rows": rows,
    }


def score_valence(mem, *, progress=None, save=None, seed=None) -> dict:
    repeats = int(REPEATS)
    toward_rows = list((seed or {}).get("toward") or [])
    away_rows = list((seed or {}).get("away") or [])
    state = {"repeats": repeats, "toward": toward_rows, "away": away_rows}
    if save is not None:
        save(state)
    for index, item in enumerate(toward_items()):
        if index < len(toward_rows):
            continue
        if progress is not None:
            progress(f"toward {index}")
        part = score_signed(mem, [item], 1.0, repeats)
        toward_rows.append(part["rows"][0])
        if save is not None:
            save(state)
    for index, item in enumerate(away_items()):
        if index < len(away_rows):
            continue
        if progress is not None:
            progress(f"away {index}")
        part = score_signed(mem, [item], -1.0, repeats)
        away_rows.append(part["rows"][0])
        if save is not None:
            save(state)
    mem.store.clear()
    return {
        "repeats": repeats,
        "toward": _valence_summary(toward_rows, positive=True),
        "away": _valence_summary(away_rows, positive=False),
    }


def score_surprise(mem, *, progress=None) -> dict:
    raw = score_gate(mem, SurpriseMeasure(), progress=progress)
    low_n = int(raw["low_n"])
    high_n = int(raw["high_n"])
    low_rejected = low_n - int(raw["low_written"])
    return {
        "seed_n": int(raw["seed_n"]),
        "seed_written": int(raw["seed_written"]),
        "seed_slots": int(raw["seed_slots"]),
        "window": int(raw["window"]),
        "quantile": float(raw["quantile"]),
        "bar": None if raw["bar"] is None else float(raw["bar"]),
        "seed_min_surprise": raw["seed_min_surprise"],
        "seed_max_surprise": raw["seed_max_surprise"],
        "low_mean_surprise": raw["low_mean_surprise"],
        "high_mean_surprise": raw["high_mean_surprise"],
        "window_has_text": int(raw["window_has_text"]),
        "low_rejected": rate_row(low_rejected, low_n),
        "high_written": rate_row(int(raw["high_written"]), high_n),
        "low_rows": raw["low_rows"],
        "high_rows": raw["high_rows"],
        "n_low_items": len(surprise_lows()),
        "n_high_items": len(surprise_highs()),
        "n_seeds": len(surprise_seeds()),
    }


class _OneQuartet:
    def __init__(self, group):
        self.QUARTETS = (tuple(group),)


def _rif_summary(groups) -> dict:
    competitors = []
    winners = []
    outsiders = []
    for group in groups:
        for item in group["items"]:
            drop = float(item["drop"])
            if item["role"] == "competitor":
                competitors.append(drop)
            elif item["role"] == "target":
                winners.append(drop)
            elif item["role"] == "outsider":
                outsiders.append(drop)
    geometry = int(sum(bool(group["geometry_ok"]) for group in groups))
    return {
        "n_quartets": len(groups),
        "geometry_ok": rate_row(geometry, len(groups)) if groups else None,
        "competitor_drop_pos": rate_row(int(sum(v > 0.0 for v in competitors)), len(competitors)) if competitors else None,
        "winner_not_weakened": rate_row(int(sum(v <= 0.0 for v in winners)), len(winners)) if winners else None,
        "outsider_unaffected": rate_row(int(sum(abs(v) <= OUTSIDER_EPS for v in outsiders)), len(outsiders)) if outsiders else None,
        "competitor_drop_mean": _mean(competitors),
        "competitor_drop_min": None if not competitors else float(min(competitors)),
        "competitor_drop_sd": _sd(competitors),
        "winner_drop_mean": _mean(winners),
        "winner_abs_max": None if not winners else float(max(abs(v) for v in winners)),
        "outsider_abs_max": None if not outsiders else float(max(abs(v) for v in outsiders)),
        "groups": groups,
    }


def score_rif(mem, *, progress=None, save=None, seed=None) -> dict:
    groups = list(seed or [])
    if save is not None:
        save(groups)
    for index, group in enumerate(rif_quartets()):
        if index < len(groups):
            continue
        if progress is not None:
            progress(f"rif {index}")
        part = score_quartets(mem, _OneQuartet(group))
        groups.append(part["quartets"][0])
        if save is not None:
            save(groups)
    mem.store.clear()
    return _rif_summary(groups)


def _arm_margin(mem, prompt, item) -> dict:
    scored = score_pair(mem, prompt, item["good"], item["bad"], name=item["name"], rel=item["rel"])
    info = scored["retrieval"]
    return {
        "margin": float(scored["margin"]),
        "choose": int(scored["choose"]),
        "maxcos": None if info.get("maxcos") is None else float(info["maxcos"]),
        "confidence": float(info.get("confidence") or 0.0),
        "gap": None if info.get("gap") is None else float(info["gap"]),
        "hit": int(info.get("hit") or 0),
        "n": int(info.get("n") or 0),
    }


def score_implicit_item(mem, item, retrieved: str) -> dict:
    """Four margins. The prompt does not mention the stored fact.

    Inject is one slot, so ``retrieval_confidence`` is 1 even when the query
    cosine is low. In-context and RAG prepend a line and leave the store empty.
    """
    prompt = item["prompt"]
    mem.store.clear()
    nomem = _arm_margin(mem, prompt, item)
    mem.store.clear()
    if not mem.remember(_fact(item)):
        raise RuntimeError(f"{item['id']} was not written")
    if len(mem.store) != 1:
        raise RuntimeError(f"{item['id']} did not occupy one slot")
    injected = _arm_margin(mem, prompt, item)
    mem.store.clear()
    context = _arm_margin(mem, item["text"] + "\n" + prompt, item)
    mem.store.clear()
    rag = _arm_margin(mem, retrieved + "\n" + prompt, item)
    mem.store.clear()
    return {
        "id": item["id"],
        "family": item["family"],
        "sign": item.get("sign"),
        "nomem_margin": nomem["margin"],
        "inject_margin": injected["margin"],
        "context_margin": context["margin"],
        "rag_margin": rag["margin"],
        "nomem_choose": nomem["choose"],
        "inject_choose": injected["choose"],
        "context_choose": context["choose"],
        "rag_choose": rag["choose"],
        "inject_maxcos": injected["maxcos"],
        "inject_confidence": injected["confidence"],
        "inject_gap": injected["gap"],
        "inject_hit": injected["hit"],
        "rag_hit": int(retrieved == item["text"]),
        "rag_retrieved": retrieved,
    }


def _effect(diffs) -> dict:
    xs = [float(x) for x in diffs]
    pos = int(sum(v > 0.0 for v in xs))
    return {
        "n": len(xs),
        "mean": _mean(xs),
        "sd": _sd(xs),
        "dz": cohens_dz(xs),
        "delta_pos": rate_row(pos, len(xs)) if xs else None,
    }


def summarize_implicit(rows) -> dict:
    families = []
    dz = {}
    per = {}
    for family in ("tone", "choice", "preference"):
        group = [row for row in rows if row["family"] == family]
        families.append(family)
        contrasts = {
            "inject_vs_nomem": _effect([row["inject_margin"] - row["nomem_margin"] for row in group]),
            "context_vs_nomem": _effect([row["context_margin"] - row["nomem_margin"] for row in group]),
            "rag_vs_nomem": _effect([row["rag_margin"] - row["nomem_margin"] for row in group]),
            "inject_vs_context": _effect([row["inject_margin"] - row["context_margin"] for row in group]),
            "inject_vs_rag": _effect([row["inject_margin"] - row["rag_margin"] for row in group]),
        }
        dz[family] = contrasts["inject_vs_nomem"]["dz"]
        choose = {}
        for cond, key in (
            ("nomem", "nomem_margin"),
            ("inject", "inject_margin"),
            ("context", "context_margin"),
            ("rag", "rag_margin"),
        ):
            choose[cond] = rate_row(int(sum(float(row[key]) > 0.0 for row in group)), len(group))
        per[family] = {
            "n": len(group),
            "contrasts": contrasts,
            "margin_pos": choose,
            "mean_margin": {
                "nomem": _mean([row["nomem_margin"] for row in group]),
                "inject": _mean([row["inject_margin"] for row in group]),
                "context": _mean([row["context_margin"] for row in group]),
                "rag": _mean([row["rag_margin"] for row in group]),
            },
            "inject_maxcos_mean": _mean([row["inject_maxcos"] for row in group if row["inject_maxcos"] is not None]),
            "inject_maxcos_min": None
            if not group
            else float(min(row["inject_maxcos"] for row in group if row["inject_maxcos"] is not None)),
            "inject_confidence_mean": _mean([row["inject_confidence"] for row in group]),
            "rag_hit": rate_row(int(sum(row["rag_hit"] for row in group)), len(group)) if group else None,
            "rag_unique": len({row["rag_retrieved"] for row in group}),
        }
    return {
        "families": per,
        "subconscious_score": subconscious_score(dz),
        "score_dz": dz,
        "rows": rows,
    }


def score_implicit(mem, bge, *, progress=None, save=None, seed=None) -> dict:
    rows = list(seed or [])
    if save is not None:
        save(rows)
    bundles = (
        ("tone", tone_items()),
        ("choice", choice_items()),
        ("preference", preference_items()),
    )
    encoded = {}
    for family, items in bundles:
        if progress is not None:
            progress(f"bge encode {family}")
        texts = [item["text"] for item in items]
        encoded[family] = (texts, bge.encode(texts))
    flat = 0
    for family, items in bundles:
        texts, vecs = encoded[family]
        for index, item in enumerate(items):
            if flat < len(rows):
                flat += 1
                continue
            if progress is not None:
                progress(f"{family} {index}")
            retrieved = bge.top1(vecs, texts, item["prompt"])
            rows.append(score_implicit_item(mem, item, retrieved))
            flat += 1
            if save is not None:
                save(rows)
    return summarize_implicit(rows)


def score_explicit_family(mem, items, *, progress=None, save=None) -> dict:
    rows = []
    if save is not None:
        save(rows)
    for index, item in enumerate(items):
        if progress is not None:
            progress(f"{item['family']} {index}")
        mem.store.clear()
        if not mem.remember(_fact(item)):
            raise RuntimeError(f"{item['id']} was not written")
        text = mem.generate(item["prompt"], max_new_tokens=32)
        answer_ids = token_ids(mem.tokenizer, item["answer"])
        # token_ids adds the special token. Count the surface the shortcut copies.
        surface_ids = mem.tokenizer.encode(item["answer"], add_special_tokens=False)
        if hasattr(surface_ids, "tolist"):
            surface_ids = surface_ids.tolist()
        rows.append(
            {
                "id": item["id"],
                "family": item["family"],
                "exact": int(text == item["answer"]),
                "text": text,
                "answer": item["answer"],
                "completing": completing_relation(item["prompt"]),
                "answer_id_len": len(list(surface_ids)),
                "fits_32": int(len(list(surface_ids)) <= 32),
                "prompt_ids": len(answer_ids),
            }
        )
        if save is not None:
            save(rows)
    mem.store.clear()
    n = len(rows)
    return {
        "n": n,
        "exact": rate_row(int(sum(row["exact"] for row in rows)), n) if n else None,
        "rows": rows,
        "excluded_from_subconscious_score": True,
    }


def score_explicit(mem, *, progress=None, save=None) -> dict:
    out = {"excluded_from_subconscious_score": True, "phone": None, "birthday": None}

    def _keep(family):
        def _save(rows):
            out[family] = rows
            if save is not None:
                save(out)

        return _save

    out["phone"] = score_explicit_family(mem, phone_items(), progress=progress, save=None if save is None else _keep("phone"))
    out["birthday"] = score_explicit_family(
        mem,
        birthday_items(),
        progress=progress,
        save=None if save is None else _keep("birthday"),
    )
    if save is not None:
        save(out)
    return out


def _bench(fn, repeats: int, warmup: int) -> dict:
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return {
        "median_s": float(samples[len(samples) // 2]),
        "min_s": float(samples[0]),
        "max_s": float(samples[-1]),
        "n": len(samples),
    }


def _n_tokens(mem, text: str) -> int:
    return len(token_ids(mem.tokenizer, text))


def _forward(mem, text: str, delta) -> None:
    ids = torch.tensor([token_ids(mem.tokenizer, text)], dtype=torch.long, device=mem._device)
    mem._delta = delta
    try:
        with torch.no_grad():
            logits = _logits_of(mem.model(input_ids=ids, use_cache=False))
            float(logits[0, -1, 0])
    finally:
        mem._delta = None


def _length_ids(mem, n: int) -> torch.Tensor:
    piece = "The kettle whistled softly. "
    text = piece
    while _n_tokens(mem, text) < n:
        text += piece
    ids = token_ids(mem.tokenizer, text)[:n]
    return torch.tensor([ids], dtype=torch.long, device=mem._device)


def measure_compute(mem, bge, *, repeats: int = 7, warmup: int = 2, progress=None) -> dict:
    """Wall clock of the residual add, one retrieval, in-context, and RAG.

    The length sweep uses the same ids with and without the residual, so the
    difference is the broadcast. Retrieval is timed separately and divided by
    the continuation-token count of one implicit pair. Nothing here changes
    the model.
    """
    if progress is not None:
        progress("compute compose")
    item = tone_items()[0]
    choice = choice_items()[0]
    mem.store.clear()
    if not mem.remember(_fact(item)):
        raise RuntimeError("compute fact was not written")
    rank = mem._rank(item["prompt"])
    mem._set_delta(rank)
    delta = None if mem._delta is None else mem._delta.detach().clone()
    mem._delta = None
    if delta is None:
        raise RuntimeError("single-slot residual was empty")
    query_prompt = item["prompt"]
    rank_bench = _bench(lambda: mem._rank(query_prompt), repeats, warmup)
    mem._delta = None
    value = rank["value"]
    key = rank["key"]
    compose_bench = _bench(lambda: mem._compose(value, key=key, grad=False), repeats, warmup)
    read_query = mem._query_vector(query_prompt)
    read_bench = _bench(lambda: mem.store.read(read_query, mode="top1"), repeats, warmup)
    mem.store.clear()
    mem._delta = None

    forwards = {}
    for length in (16, 32, 64):
        if progress is not None:
            progress(f"compute length {length}")
        ids = _length_ids(mem, length)

        def step(armed, _ids=ids, _delta=delta):
            mem._delta = _delta if armed else None
            try:
                with torch.no_grad():
                    logits = _logits_of(mem.model(input_ids=_ids, use_cache=False))
                    float(logits[0, -1, 0])
            finally:
                mem._delta = None

        empty = _bench(lambda: step(False), repeats, warmup)
        armed = _bench(lambda: step(True), repeats, warmup)
        extra = float(armed["median_s"] - empty["median_s"])
        forwards[str(length)] = {
            "empty_median_s": empty["median_s"],
            "armed_median_s": armed["median_s"],
            "extra_s": extra,
            "extra_per_token_s": extra / float(length),
        }
    per_token = [row["extra_per_token_s"] for row in forwards.values()]

    def _pair_cost(sample, label):
        if progress is not None:
            progress(f"compute prompt {label}")
        short = sample["prompt"] + sample["good"]
        long = sample["text"] + "\n" + short
        short_n = _n_tokens(mem, short)
        long_n = _n_tokens(mem, long)
        prompt_n = _n_tokens(mem, sample["prompt"])
        context_prompt_n = _n_tokens(mem, sample["text"] + "\n" + sample["prompt"])
        good_n = _n_tokens(mem, sample["prompt"] + sample["good"]) - prompt_n
        bad_n = _n_tokens(mem, sample["prompt"] + sample["bad"]) - prompt_n
        cont_n = int(good_n + bad_n)
        short_bench = _bench(lambda: _forward(mem, short, None), repeats, warmup)
        long_bench = _bench(lambda: _forward(mem, long, None), repeats, warmup)
        extra = float(long_bench["median_s"] - short_bench["median_s"])
        added_prompt = int(context_prompt_n - prompt_n)
        return {
            "family": sample["family"],
            "short_tokens": int(short_n),
            "long_tokens": int(long_n),
            "continuation_tokens": cont_n,
            "added_prompt_tokens": added_prompt,
            "short_median_s": short_bench["median_s"],
            "long_median_s": long_bench["median_s"],
            "extra_s": extra,
            "extra_per_continuation_token_s": None if cont_n <= 0 else extra / float(cont_n),
            "extra_per_added_prompt_token_s": None if added_prompt <= 0 else extra / float(added_prompt),
        }

    prompt_costs = [_pair_cost(item, "tone"), _pair_cost(choice, "choice"), _pair_cost(preference_items()[0], "preference")]
    cont_for_rank = int(prompt_costs[0]["continuation_tokens"])
    rank_per = None if cont_for_rank <= 0 else float(rank_bench["median_s"]) / float(cont_for_rank)
    compose_per = None if cont_for_rank <= 0 else float(compose_bench["median_s"]) / float(cont_for_rank)
    read_per = None if cont_for_rank <= 0 else float(read_bench["median_s"]) / float(cont_for_rank)

    if progress is not None:
        progress("compute bge")
    corpus = [row["text"] for row in tone_items()]
    query = tone_items()[0]["prompt"]

    def _bge_full():
        vecs = bge.encode(corpus)
        bge.top1(vecs, corpus, query)

    def _bge_query():
        bge.top1(corpus_vecs, corpus, query)

    bge_full = _bench(_bge_full, repeats, warmup)
    corpus_vecs = bge.encode(corpus)
    bge_query = _bench(_bge_query, repeats, warmup)
    bge_full_per = None if cont_for_rank <= 0 else float(bge_full["median_s"]) / float(cont_for_rank)
    bge_query_per = None if cont_for_rank <= 0 else float(bge_query["median_s"]) / float(cont_for_rank)
    context_per = [row["extra_per_continuation_token_s"] for row in prompt_costs]
    mem.store.clear()
    mem._delta = None
    return {
        "repeats": int(repeats),
        "warmup": int(warmup),
        "hidden": int(mem.hidden),
        "residual_adds_per_token": int(mem.hidden),
        "inject_parameters": int(sum(p.numel() for p in mem.inject.parameters())),
        "forward_by_length": forwards,
        "residual_extra_per_token_s": _mean(per_token),
        "retrieval": {
            "rank_median_s": rank_bench["median_s"],
            "compose_median_s": compose_bench["median_s"],
            "read_top1_median_s": read_bench["median_s"],
            "continuation_tokens": cont_for_rank,
            "rank_per_continuation_token_s": rank_per,
            "compose_per_continuation_token_s": compose_per,
            "read_per_continuation_token_s": read_per,
            "inject_added_per_continuation_token_s": None
            if rank_per is None
            else float(rank_per + compose_per + _mean(per_token)),
        },
        "prompt_length": prompt_costs,
        "context_extra_per_continuation_token_s": _mean([v for v in context_per if v is not None]),
        "bge": {
            "corpus_n": len(corpus),
            "full_median_s": bge_full["median_s"],
            "query_median_s": bge_query["median_s"],
            "full_per_continuation_token_s": bge_full_per,
            "query_per_continuation_token_s": bge_query_per,
        },
        "rag_added_per_continuation_token_s": None
        if bge_full_per is None
        else float(bge_full_per + _mean([v for v in context_per if v is not None])),
    }
