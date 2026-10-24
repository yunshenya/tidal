"""Fit and held scoring for cue-triggered injection.

``histogram`` and ``fit`` read the direction fit bank only. Held banks are
imported inside ``held`` and nowhere else. Theta, tau, and a_trig are applied
by writing the module globals that the cue trace reads at call time. The floor
stays 0.05.

The add is the same-token add documented on ``LLMMemory``: the cue is copied
from the inject-layer residual, then the residual is written at that index.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT_DIR = ROOT / "reports"
HISTOGRAM_PATH = REPORT_DIR / "subconscious_cue_histogram.json"
FIT_PATH = REPORT_DIR / "subconscious_cue_fit.json"
HELD_PATH = REPORT_DIR / "subconscious_cue_held.json"

# Bracketed by the fit-bank cosine histogram before any cell was scored.
# Unrelated-prompt max cosine is 0.303 and filler max is 0.294, so 0.32 is
# the low end. The lowest cued-prompt max is 0.436, so 0.43 is the high end.
# Tau and a_trig are not chosen from a margin.
GRID = tuple(
    {"theta": theta, "tau": tau, "a_trig": a_trig}
    for theta in (0.32, 0.35, 0.40, 0.43)
    for tau in (2.0, 4.0, 8.0)
    for a_trig in (0.5, 1.0)
)

FILLER = "\nLater the clerk filed the spare form and left the room quietly today."
DECAY_OFFSETS = (0, 1, 2, 4, 8)
FALSE_TRIGGER_CEILING = 0.10
UNRELATED_ABS_MAX = 0.05
CUED_POS_MIN = 6
DECAY_PASS_MIN = 6
STORE_DROP_MAX = 0.05
QUIET_NLL_MAX = 0.02
STORE_SIZES = (8, 16, 32)
COST_WARMUP = 2
COST_REPEATS = 7
WILSON_Z = 1.959963984540054

# Point counts from the merged interfere held pass (#54). Not worse means
# the cue pass is at least this many successes. n is 50.
HELD_FLOORS = {
    "prefer_up": 50,
    "avoid_up": 39,
    "avoid_stored_down": 42,
    "toward_up": 45,
    "away_down": 48,
    "rif_ge4": 50,
    "competitor_down": 50,
    "winner_same": 50,
    "outsider_same": 50,
    "similarity": 42,
    "count": 48,
}


def _print(message: str) -> None:
    print(message, flush=True)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _wilson(k: int, n: int) -> list:
    k = int(k)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    phat = k / n
    z2 = WILSON_Z * WILSON_Z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = WILSON_Z * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    if low < 1e-12:
        low = 0.0
    if high > 1.0 - 1e-12:
        high = 1.0
    return [low, high]


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fit_bank():
    from tidal.subconscious_direction_data import fit_bank

    return fit_bank()


def _load_memory():
    import torch

    from tidal.llm_memory import LLMMemory

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    mem = LLMMemory(dtype=torch.float32)
    _install_cue_cache(mem)
    return mem


def _install_cue_cache(mem) -> None:
    """Cue rows do not depend on theta. Reuse them across grid cells."""
    original = mem.cue_hidden
    cache = {}

    def cached(text, _original=original, _cache=cache):
        hit = _cache.get(text)
        if hit is None:
            hit = _original(text)
            _cache[text] = hit
        return hit

    mem.cue_hidden = cached


def _bind(theta: float, tau: float, a_trig: float) -> None:
    import tidal.llm_memory as mod

    mod.CUE_THETA = float(theta)
    mod.CUE_TAU = float(tau)
    mod.CUE_A_TRIG = float(a_trig)


def _clear_bind() -> None:
    import tidal.llm_memory as mod

    _bind(mod.CUE_THETA, mod.CUE_TAU, mod.CUE_A_TRIG)


def _encode(mem, facts) -> list:
    from tidal.subconscious_eval import Encoder

    return Encoder(mem).many(facts)


def _write(mem, rows) -> int:
    from tidal.subconscious_eval import write_rows

    return int(write_rows(mem, rows)["slots"])


def _snapshot(mem) -> list:
    return [float(slot.strength) for slot in mem.store._slots]


def _restore(mem, saved) -> None:
    for slot, value in zip(mem.store._slots, saved):
        slot.strength = float(value)


def _margin(mem, prompt: str, good: str, bad: str, name=None, rel=None) -> float:
    from tidal.subconscious_eval import score_pair

    saved = _snapshot(mem)
    try:
        scored = score_pair(mem, prompt, good, bad, name=name, rel=rel)
    finally:
        _restore(mem, saved)
    return float(scored["margin"])


def _percentiles(values) -> dict:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"n": 0}

    def at(q: float) -> float:
        pos = q * (len(ordered) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)

    return {
        "n": len(ordered),
        "min": ordered[0],
        "p50": at(0.50),
        "p90": at(0.90),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": float(sum(ordered) / len(ordered)),
    }


def _cosines(mem, text: str, key) -> list:
    import numpy as np

    from tidal.episodic import cosine_scores

    rows = np.asarray(mem.cue_hidden(text), dtype=np.float64)
    center = np.asarray(mem.query_mean, dtype=np.float64)
    stored = np.asarray(key, dtype=np.float64).reshape(-1)
    return [float(value) for value in cosine_scores(rows - center, stored).tolist()]


def histogram() -> dict:
    """Fit-bank cosine bracket. Does not score a grid cell."""
    import numpy as np

    bank = _fit_bank()
    mem = _load_memory()
    started = time.perf_counter()
    fact_max = []
    prompt_max = []
    prompt_all = []
    unrelated_max = []
    unrelated_all = []
    try:
        prefer = list(bank.PREFER)
        neutrals = list(bank.NEUTRAL_CHOICE)
        encoded = _encode(mem, [_fact(item) for item in prefer])
        for item, row in zip(prefer, encoded):
            _write(mem, [row])
            key = mem.store._slots[0].k
            fact_cos = _cosines(mem, item["text"], key)
            prompt_cos = _cosines(mem, item["prompt"], key)
            fact_max.append(max(fact_cos) if fact_cos else float("nan"))
            prompt_max.append(max(prompt_cos) if prompt_cos else float("nan"))
            prompt_all.extend(prompt_cos)
            for prompt in neutrals:
                other = _cosines(mem, prompt["prompt"], key)
                unrelated_max.append(max(other) if other else float("nan"))
                unrelated_all.extend(other)
            _print(
                f"histogram {item['id']} fact_max={fact_max[-1]:.4f} prompt_max={prompt_max[-1]:.4f}"
            )
        mem.store.clear()
    finally:
        mem.close()
    payload = {
        "kind": "histogram",
        "selected": None,
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-fit",
        "n_prefer": len(fact_max),
        "fact_max": _percentiles(fact_max),
        "prompt_max": _percentiles(prompt_max),
        "prompt_tokens": _percentiles(prompt_all),
        "unrelated_max": _percentiles(unrelated_max),
        "unrelated_tokens": _percentiles(unrelated_all),
        "prompt_max_values": prompt_max,
        "fact_max_values": fact_max,
        "note": "Grid bracket only. No cell was scored.",
    }
    del np
    _dump(HISTOGRAM_PATH, payload)
    _print(json.dumps({k: payload[k] for k in ("fact_max", "prompt_max", "unrelated_max")}, indent=2))
    return payload


def _fact(item) -> dict:
    return {
        "text": item["text"],
        "answer": item["answer"],
        "name": item["name"],
        "rel": item["rel"],
    }


def _armed_logits(mem, text: str):
    import torch

    from tidal.llm_memory import _batch_ids, _batch_mask, _logits_of

    enc = mem._tokenize(text)
    ids = _batch_ids(enc).to(mem._device)
    attn = _batch_mask(enc, mem._device)
    kwargs = {"input_ids": ids, "use_cache": False}
    if attn is not None:
        kwargs["attention_mask"] = attn
    mem._cue_live = True
    try:
        with torch.no_grad():
            logits = _logits_of(mem.model(**kwargs)).float()
    finally:
        mem._cue_live = False
    return logits[0].detach().cpu()


def _quiet_logits(mem, text: str):
    import torch

    from tidal.llm_memory import _batch_ids, _batch_mask, _logits_of

    enc = mem._tokenize(text)
    ids = _batch_ids(enc).to(mem._device)
    attn = _batch_mask(enc, mem._device)
    kwargs = {"input_ids": ids, "use_cache": False}
    if attn is not None:
        kwargs["attention_mask"] = attn
    mem._cue_live = False
    with torch.no_grad():
        logits = _logits_of(mem.model(**kwargs)).float()
    return logits[0].detach().cpu()


def _continuation_logps(mem, prompt: str, continuation: str, *, live: bool) -> tuple:
    import torch

    from tidal.llm_memory import _batch_ids, _batch_mask, _logits_of
    from tidal.subconscious_eval import token_ids

    prompt_ids = token_ids(mem.tokenizer, prompt)
    full_ids = token_ids(mem.tokenizer, prompt + continuation)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("continuation retokenizes the prompt")
    cont = full_ids[len(prompt_ids) :]
    ids = torch.tensor([full_ids], dtype=torch.long, device=mem._device)
    mem._cue_live = bool(live)
    try:
        with torch.no_grad():
            logits = _logits_of(mem.model(input_ids=ids, use_cache=False)).float()
            pos = [len(prompt_ids) - 1 + i for i in range(len(cont))]
            pred = logits[0, pos]
            logp = torch.log_softmax(pred, dim=-1)
            target = torch.tensor(cont, dtype=torch.long, device=logp.device)
            taken = logp.gather(1, target.unsqueeze(-1)).squeeze(-1)
            values = [float(x) for x in taken.detach().cpu().tolist()]
    finally:
        mem._cue_live = False
    return values, len(prompt_ids)


def _score_cell(mem, bank, prefer_rows, incidental_rows, base) -> dict:
    import torch

    prefer = list(bank.PREFER)
    neutrals = list(bank.NEUTRAL_CHOICE)
    cued = []
    for item, row, empty in zip(prefer, prefer_rows, base["cued_empty"]):
        _write(mem, [row])
        armed = _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
        cued.append(float(armed - empty))
    mean_cued = float(sum(cued) / len(cued))
    n_pos = int(sum(value > 0.0 for value in cued))
    gate1_cued = bool(n_pos >= CUED_POS_MIN and mean_cued > 0.0)

    _write(mem, prefer_rows)
    unrelated = []
    triggers = 0
    for item, empty in zip(neutrals, base["unrelated_empty"]):
        armed = _margin(mem, item["prompt"], item["good"], item["bad"])
        unrelated.append(abs(float(armed - empty)))
        fields = mem.cue_fields(item["prompt"])
        if bool(torch.any(fields["triggered"]).item()):
            triggers += 1
    unrelated_abs = float(sum(unrelated) / len(unrelated))
    gate1 = bool(gate1_cued and unrelated_abs <= UNRELATED_ABS_MAX)
    gate2_rate = triggers / len(neutrals)
    gate2 = bool(gate2_rate <= FALSE_TRIGGER_CEILING)

    decay_rows = []
    decay_pass = 0
    for item, row in zip(prefer, prefer_rows):
        text = item["text"] + FILLER
        _write(mem, [row])
        fields = mem.cue_fields(text)
        activation = fields["activation"][0].tolist()
        triggered = fields["triggered"][0].tolist()
        record = {"id": item["id"], "pass": False, "reason": ""}
        hits = [index for index, flag in enumerate(triggered) if flag]
        if not hits:
            record["reason"] = "no trigger"
            decay_rows.append(record)
            continue
        # Distance is measured from the last triggering token. The fact's own
        # tokens are a run of matches, so the first match is still inside the cue.
        start = int(hits[-1])
        if start + DECAY_OFFSETS[-1] >= len(activation):
            record["reason"] = "sequence shorter than distance 8"
            decay_rows.append(record)
            continue
        if any(triggered[start + 1 : start + DECAY_OFFSETS[-1] + 1]):
            record["reason"] = "retrigger"
            decay_rows.append(record)
            continue
        taken = [float(activation[start + offset]) for offset in DECAY_OFFSETS]
        monotone = all(taken[i] >= taken[i + 1] - 1e-6 for i in range(len(taken) - 1))
        if not (taken[0] > 0.0 and taken[-1] < taken[0] and monotone):
            record["reason"] = "activation"
            record["activation"] = taken
            decay_rows.append(record)
            continue
        quiet = base["decay_quiet"][item["id"]]
        armed_logits = _armed_logits(mem, text)
        l1_0 = float(torch.sum(torch.abs(armed_logits[start] - quiet[start])))
        l1_8 = float(torch.sum(torch.abs(armed_logits[start + 8] - quiet[start + 8])))
        record["l1_0"] = l1_0
        record["l1_8"] = l1_8
        record["activation"] = taken
        if not (l1_8 < l1_0):
            record["reason"] = "logit"
            decay_rows.append(record)
            continue
        record["pass"] = True
        record["reason"] = "ok"
        decay_pass += 1
        decay_rows.append(record)
    gate3 = bool(decay_pass >= DECAY_PASS_MIN)

    size_means = {}
    pool = incidental_rows[:31]
    for size in STORE_SIZES:
        deltas = []
        width = int(size) - 1
        for item, row, empty in zip(prefer, prefer_rows, base["cued_empty"]):
            slots = _write(mem, [row] + pool[:width])
            if slots != int(size):
                raise RuntimeError(f"store size {size} collapsed to {slots}")
            armed = _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
            deltas.append(float(armed - empty))
        size_means[str(size)] = float(sum(deltas) / len(deltas))
    mean8 = size_means["8"]
    mean16 = size_means["16"]
    mean32 = size_means["32"]
    gate4 = bool(
        mean16 > 0.0
        and mean32 > 0.0
        and mean16 >= mean8 - STORE_DROP_MAX
        and mean32 >= mean8 - STORE_DROP_MAX
    )

    _write(mem, prefer_rows)
    quiet_deltas = []
    for item in bank.NEUTRAL_NLL:
        fields = mem.cue_fields(item["prompt"] + item["continuation"])
        activation = fields["activation"][0]
        live, prompt_len = _continuation_logps(mem, item["prompt"], item["continuation"], live=True)
        base_lp = base["nll_empty"][item["id"]]
        if len(live) != len(base_lp):
            raise RuntimeError(f"{item['id']} token count changed")
        if int(activation.shape[0]) != prompt_len + len(live):
            raise RuntimeError(f"{item['id']} cue length does not match the scored string")
        for index, (empty_logp, memory_logp) in enumerate(zip(base_lp, live)):
            predictor = prompt_len - 1 + index
            if float(activation[predictor]) == 0.0:
                # nll is the negative log-prob. Positive means the memory is worse.
                quiet_deltas.append(float(empty_logp - memory_logp))
    if any(not math.isfinite(value) for value in quiet_deltas):
        raise RuntimeError("quiet nll was not finite")
    n_quiet = len(quiet_deltas)
    mean_quiet = float(sum(quiet_deltas) / n_quiet) if n_quiet else None
    gate5 = bool(n_quiet > 0 and mean_quiet is not None and mean_quiet <= QUIET_NLL_MAX)
    passed = bool(gate1 and gate2 and gate3 and gate4 and gate5)
    return {
        "pass": passed,
        "mean_cued_delta": mean_cued,
        "cued_deltas": cued,
        "n_cued_pos": n_pos,
        "unrelated_abs_mean": unrelated_abs,
        "gate1_pass": gate1,
        "false_triggers": int(triggers),
        "false_trigger_n": len(neutrals),
        "false_trigger_rate": float(gate2_rate),
        "false_trigger_wilson": _wilson(triggers, len(neutrals)),
        "gate2_pass": gate2,
        "decay_pass": int(decay_pass),
        "decay_n": len(prefer),
        "decay_rows": decay_rows,
        "gate3_pass": gate3,
        "store_means": size_means,
        "gate4_pass": gate4,
        "quiet_nll_mean": mean_quiet,
        "quiet_nll_n": int(n_quiet),
        "gate5_pass": gate5,
    }


def _baselines(mem, bank, prefer_rows) -> dict:
    prefer = list(bank.PREFER)
    cued_empty = []
    mem.store.clear()
    for item in prefer:
        cued_empty.append(_margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"]))
    unrelated_empty = []
    for item in bank.NEUTRAL_CHOICE:
        unrelated_empty.append(_margin(mem, item["prompt"], item["good"], item["bad"]))
    decay_quiet = {}
    for item in prefer:
        text = item["text"] + FILLER
        decay_quiet[item["id"]] = _quiet_logits(mem, text)
    nll_empty = {}
    for item in bank.NEUTRAL_NLL:
        values, _prompt_len = _continuation_logps(mem, item["prompt"], item["continuation"], live=False)
        nll_empty[item["id"]] = values
    del prefer_rows
    return {
        "cued_empty": cued_empty,
        "unrelated_empty": unrelated_empty,
        "decay_quiet": decay_quiet,
        "nll_empty": nll_empty,
    }


def _select(cells: list) -> dict:
    passing = [cell for cell in cells if cell.get("pass")]
    if not passing:
        return {"selected": None, "reason": "no cell passed gates 1-5"}

    def key(cell):
        return (
            float(cell["mean_cued_delta"]),
            float(cell["theta"]),
            float(cell["tau"]),
            -float(cell["a_trig"]),
        )

    best = max(passing, key=key)
    return {
        "selected": {
            "theta": best["theta"],
            "tau": best["tau"],
            "a_trig": best["a_trig"],
            "mean_cued_delta": best["mean_cued_delta"],
        },
        "reason": "largest mean cued margin; ties prefer higher theta, then larger tau, then smaller a_trig",
    }


def _cost(mem, rows, prompt: str) -> dict:
    import torch

    from tidal.llm_memory import _batch_ids, _batch_mask, _logits_of

    enc = mem._tokenize(prompt)
    ids = _batch_ids(enc).to(mem._device)
    attn = _batch_mask(enc, mem._device)
    steps = int(ids.shape[-1])
    kwargs = {"input_ids": ids, "use_cache": False}
    if attn is not None:
        kwargs["attention_mask"] = attn

    def step(live: bool) -> None:
        mem._cue_live = bool(live)
        try:
            with torch.no_grad():
                logits = _logits_of(mem.model(**kwargs))
                float(logits[0, -1, 0])
        finally:
            mem._cue_live = False

    out = {"tokens": steps, "sizes": {}}
    for size in STORE_SIZES:
        _write(mem, rows[: int(size)])
        for _ in range(COST_WARMUP):
            step(False)
            step(True)
        quiet = []
        armed = []
        for _ in range(COST_REPEATS):
            t0 = time.perf_counter()
            step(False)
            quiet.append(time.perf_counter() - t0)
            t1 = time.perf_counter()
            step(True)
            armed.append(time.perf_counter() - t1)
        quiet.sort()
        armed.sort()
        q = quiet[len(quiet) // 2]
        a = armed[len(armed) // 2]
        out["sizes"][str(size)] = {
            "quiet_median_s": float(q),
            "armed_median_s": float(a),
            "extra_s": float(a - q),
            "extra_s_per_token": float(a - q) / float(steps),
            "armed_s_per_token": float(a) / float(steps),
        }
        _print(
            f"cost store {size}: extra {out['sizes'][str(size)]['extra_s_per_token']:.6f} s/token"
        )
    mem.store.clear()
    return out


def fit() -> dict:
    """Gates 1-5 on the direction fit bank. Does not score a held bank."""
    import tidal.llm_memory as mod

    original = (float(mod.CUE_THETA), float(mod.CUE_TAU), float(mod.CUE_A_TRIG))
    bank = _fit_bank()
    mem = _load_memory()
    started = time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _print("baselines")
        base = _baselines(mem, bank, prefer_rows)
        cells = []
        for index, cell in enumerate(GRID):
            _bind(cell["theta"], cell["tau"], cell["a_trig"])
            _print(f"cell {index + 1}/{len(GRID)} theta={cell['theta']} tau={cell['tau']} a_trig={cell['a_trig']}")
            scored = _score_cell(mem, bank, prefer_rows, incidental_rows, base)
            scored.update(cell)
            cells.append(scored)
            _print(
                "  "
                + " ".join(
                    [
                        f"pass={scored['pass']}",
                        f"cued={scored['mean_cued_delta']:.4f}",
                        f"false={scored['false_triggers']}/{scored['false_trigger_n']}",
                        f"decay={scored['decay_pass']}/{scored['decay_n']}",
                        f"store={scored['store_means']}",
                        f"quiet={scored['quiet_nll_mean']}",
                    ]
                )
            )
        choice = _select(cells)
        chosen = choice["selected"]
        if chosen is None:
            _bind(*original)
        else:
            _bind(chosen["theta"], chosen["tau"], chosen["a_trig"])
        cost_rows = prefer_rows + incidental_rows
        cost = _cost(mem, cost_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        mem.close()
    payload = {
        "kind": "fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-fit",
        "grid": list(GRID),
        "cells": cells,
        "choice": choice,
        "cost": cost,
        "held_scored": False,
    }
    _dump(FIT_PATH, payload)
    return payload


def held() -> dict:
    """Gate 6 on the n=50 banks. Refuses to run unless fit selected a cell."""
    from tidal.subconscious_interfere_score import score_quartet
    from tidal.subconscious_measure_data import (
        RifMeasure,
        away_items,
        avoid_items,
        prefer_items,
        toward_items,
        wilson_interval,
    )
    from tidal.subconscious_replay_score import score_direction
    from tidal.subconscious_valence_data import REPEATS
    from tidal.subconscious_valence_score import score_signed

    if not FIT_PATH.is_file():
        raise RuntimeError("fit report is missing")
    fit_report = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    chosen = (fit_report.get("choice") or {}).get("selected")
    if not chosen:
        raise RuntimeError("fit did not select a cell; held is not scored")
    _bind(chosen["theta"], chosen["tau"], chosen["a_trig"])

    class _Bank:
        def __init__(self, prefer, avoid):
            self.PREFER = tuple(prefer)
            self.AVOID = tuple(avoid)

    mem = _load_memory()
    started = time.perf_counter()
    try:
        quartet_rows = []
        for index, group in enumerate(RifMeasure.QUARTETS):
            _print(f"quartet {index + 1}/50")
            quartet_rows.append(score_quartet(mem, group))
        prefer_rows = []
        for index, item in enumerate(prefer_items()):
            _print(f"prefer {index + 1}/50")
            part = score_direction(mem, _Bank([item], []))
            prefer_rows.append(part["prefer"]["rows"][0])
        avoid_rows = []
        for index, item in enumerate(avoid_items()):
            _print(f"avoid {index + 1}/50")
            part = score_direction(mem, _Bank([], [item]))
            avoid_rows.append(part["avoid"]["rows"][0])
        toward_rows = []
        for index, item in enumerate(toward_items()):
            _print(f"toward {index + 1}/50")
            part = score_signed(mem, [item], 1.0, int(REPEATS))
            toward_rows.append(part["rows"][0])
        away_rows = []
        for index, item in enumerate(away_items()):
            _print(f"away {index + 1}/50")
            part = score_signed(mem, [item], -1.0, int(REPEATS))
            away_rows.append(part["rows"][0])
    finally:
        mem.close()

    def _count(rows, pred) -> dict:
        flags = [bool(pred(row)) for row in rows]
        k = int(sum(flags))
        n = len(flags)
        if n <= 0:
            return {"k": 0, "n": 0, "rate": None, "wilson": None}
        low, high = wilson_interval(k, n)
        return {"k": k, "n": n, "rate": k / n, "wilson": [low, high]}

    prefer = _count(prefer_rows, lambda row: float(row["delta"]) > 0.0)
    avoid_up = _count(avoid_rows, lambda row: float(row["delta"]) > 0.0)
    avoid_down = _count(avoid_rows, lambda row: float(row["stored_lift"]) < 0.0)
    toward = _count(toward_rows, lambda row: float(row["delta"]) > 0.0)
    away = _count(away_rows, lambda row: float(row["delta"]) < 0.0)
    ge4 = _count(quartet_rows, lambda row: bool(row["ge4"]))
    pattern = [row for row in quartet_rows if row["pattern_ok"]]
    competitor = _count(pattern, lambda row: bool(row["competitor_down"]))
    winner = _count(pattern, lambda row: bool(row["winner_same"]))
    outsider = _count(pattern, lambda row: bool(row["outsider_same"]))
    sim_rows = [row for row in quartet_rows if row["similarity"].get("included")]
    similarity = _count(sim_rows, lambda row: bool(row["similarity"]["high_lower"]))
    count = _count(quartet_rows, lambda row: bool(row["fan"]["high_lower"]))
    rates = {
        "prefer_up": prefer,
        "avoid_up": avoid_up,
        "avoid_stored_down": avoid_down,
        "toward_up": toward,
        "away_down": away,
        "rif_ge4": ge4,
        "competitor_down": competitor,
        "winner_same": winner,
        "outsider_same": outsider,
        "similarity": similarity,
        "count": count,
    }
    failures = []
    for name, floor in HELD_FLOORS.items():
        row = rates[name]
        if int(row["n"]) < 50 or int(row["k"]) < int(floor):
            failures.append({"gate": name, "k": row["k"], "n": row["n"], "floor": int(floor)})
    payload = {
        "kind": "held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "selected": chosen,
        "rates": rates,
        "floors": HELD_FLOORS,
        "failures": failures,
        "pass": not failures,
        "means": {
            "prefer_delta": float(sum(float(row["delta"]) for row in prefer_rows) / len(prefer_rows)),
            "avoid_delta": float(sum(float(row["delta"]) for row in avoid_rows) / len(avoid_rows)),
            "avoid_stored_lift": float(sum(float(row["stored_lift"]) for row in avoid_rows) / len(avoid_rows)),
            "toward_delta": float(sum(float(row["delta"]) for row in toward_rows) / len(toward_rows)),
            "away_delta": float(sum(float(row["delta"]) for row in away_rows) / len(away_rows)),
        },
    }
    _dump(HELD_PATH, payload)
    _print(json.dumps({"pass": payload["pass"], "failures": failures, "rates": {k: [v["k"], v["n"]] for k, v in rates.items()}}, indent=2))
    return payload


def main(argv: list) -> None:
    mode = argv[1] if len(argv) > 1 else "fit"
    if mode == "histogram":
        histogram()
    elif mode == "fit":
        fit()
    elif mode == "held":
        held()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
