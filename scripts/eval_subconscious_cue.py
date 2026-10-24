"""Fit and held scoring for cue-triggered injection.

``histogram`` and ``fit`` read the direction fit bank only. Held banks are
imported inside ``held`` and nowhere else. The amendment fit freezes theta,
then sweeps half-life and lambda. It does not co-sweep theta. The floor
stays 0.05. Lambda scales only the residual this wrapper adds to the model.

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
SCHEDULE_FIT_PATH = REPORT_DIR / "subconscious_cue_schedule_fit.json"
SCHEDULE_HELD_PATH = REPORT_DIR / "subconscious_cue_schedule_held.json"
HALFLIFE_FIT_PATH = REPORT_DIR / "subconscious_cue_halflife_fit.json"
ANISOTROPY_PATH = REPORT_DIR / "subconscious_cue_anisotropy.json"
KWTA_FIT_PATH = REPORT_DIR / "subconscious_cue_kwta_fit.json"
DIRECTION_HELD_PATH = REPORT_DIR / "subconscious_cue_direction_held.json"
REVISION_PATH = REPORT_DIR / "subconscious_cue_revision_held.json"
WORD_HELD_PATH = REPORT_DIR / "subconscious_cue_word_held.json"
EVENT_FIT_PATH = REPORT_DIR / "subconscious_cue_event_fit.json"
EVENT_HELD_PATH = REPORT_DIR / "subconscious_cue_event_held.json"
COMPETITION_FIT_PATH = REPORT_DIR / "subconscious_cue_competition_fit.json"
KWTA_KS = (1, 2, 3, 4)
HELD_N_MIN = 50
FROZEN_ALPHA = 0.25

# First schedule, already scored. fit() does not iterate this.
HISTORICAL_GRID = tuple(
    {"theta": theta, "tau": tau, "a_trig": a_trig}
    for theta in (0.32, 0.35, 0.40, 0.43)
    for tau in (2.0, 4.0, 8.0)
    for a_trig in (0.5, 1.0)
)

# Theta is frozen at the low end of the cosine bracket. Half-life is the
# hypothesis sweep in tokens. Lambda is the injection scale. a_trig stays 1.
FROZEN_THETA = 0.32
# Retired sweep. fit() does not iterate this.
HISTORICAL_SCHEDULE = tuple(
    {"theta": FROZEN_THETA, "half_life": half_life, "lambda": lam, "a_trig": 1.0}
    for half_life in (10.0, 20.0, 30.0)
    for lam in (0.10, 0.15, 0.20, 0.25, 0.30)
)
SCHEDULE = tuple(
    {"theta": FROZEN_THETA, "half_life": half_life, "lambda": lam, "a_trig": 1.0}
    for half_life in (2.0, 4.0, 8.0, 16.0, 30.0)
    for lam in (0.10, 0.15, 0.20, 0.25, 0.30)
)

FILLER = "\nLater the clerk filed the spare form and left the room quietly today."
DECAY_OFFSETS = (0, 1, 2, 4, 8)
FALSE_TRIGGER_CEILING = 0.10
UNRELATED_ABS_MAX = 0.05
CUED_POS_MIN = 6
DECAY_PASS_MIN = 6
STORE_DROP_MAX = 0.05
QUIET_NLL_MAX = 0.02
# Fresh direction gates. 38/50 is the fit 6/8 proportion on n=50.
# Fixed before the fresh bank is scored. Not taken from the seen n=50 floors.
FRESH_N_MIN = 50
FRESH_CUED_MIN = 38
FRESH_DECAY_MIN = 38
FRESH_FAN_MIN = 38
FRESH_FALSE_TRIGGER_CEILING = FALSE_TRIGGER_CEILING
FRESH_UNRELATED_ABS_MAX = UNRELATED_ABS_MAX
FRESH_STORE_DROP_MAX = STORE_DROP_MAX
FRESH_QUIET_NLL_MAX = QUIET_NLL_MAX
STORE_SIZES = (8, 16, 32)
COST_WARMUP = 2
COST_REPEATS = 7
WILSON_Z = 1.959963984540054
FAN_ALPHAS = (0.25, 0.125, 0.0625, 0.03125, 0.015625)
FAN_STRENGTH = 0.5
FAN_RELATION = "fan-norm"
FAN_FIT_MIN = 6
HELD_FAN_MIN = 38

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


def _bind(theta: float, half_life: float, lam: float, a_trig: float = 1.0) -> None:
    import tidal.llm_memory as mod

    mod.CUE_THETA = float(theta)
    mod.CUE_HALF_LIFE = float(half_life)
    mod.CUE_TAU = float(half_life) / math.log(2.0)
    mod.CUE_LAMBDA = float(lam)
    mod.CUE_A_TRIG = float(a_trig)


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


def _write_target(mem, row) -> None:
    mem.store.clear()
    ok = mem.store.write(
        row["key"],
        row["value"],
        name=row["name"],
        rel=row["rel"],
        kind="fact",
        answer=row.get("answer"),
    )
    if not ok or len(mem.store) != 1:
        raise RuntimeError("fan target was not written")


def _add_companions(mem, row, match, n_extra: int, alpha: float) -> None:
    import numpy as np

    key = np.asarray(row["key"], dtype=np.float64).reshape(-1)
    match = np.asarray(match, dtype=np.float64).reshape(-1)
    unit = match / max(float(np.linalg.norm(match)), 1e-12)
    scale = float(np.linalg.norm(key))
    value = np.zeros_like(np.asarray(row["value"], dtype=np.float32))
    for index in range(int(n_extra)):
        mixed = (1.0 - float(alpha)) * key + float(alpha) * unit * scale
        mixed = np.array(mixed, dtype=np.float64, copy=True)
        mixed[(index + 1) % mixed.shape[0]] += 1e-4 * scale
        mixed = mixed / max(float(np.linalg.norm(mixed)), 1e-12) * scale
        wrote = mem.store.write(
            mixed,
            value,
            name=f"fan-{index}",
            rel=FAN_RELATION,
            kind="fact",
            answer=f"fan-{index}",
        )
        if not wrote:
            raise RuntimeError("fan companion was not written")
        mem.store._slots[-1].strength = float(FAN_STRENGTH)


def _trigger_match(mem, prompt: str):
    import numpy as np

    fields = mem.cue_fields(prompt)
    triggered = fields["triggered"][0].tolist()
    hits = [index for index, flag in enumerate(triggered) if flag]
    if not hits:
        return None, None
    hit = int(hits[-1])
    rows = np.asarray(mem._last_cue_rows, dtype=np.float64)
    match = rows[hit] - np.asarray(mem.query_mean, dtype=np.float64)
    return hit, match


def _fan2_ok(mem, prompt: str, hit: int, expected: float) -> bool:
    fields = mem.cue_fields(prompt)
    activation = fields["activation"][0].tolist()
    triggered = [bool(flag) for flag in fields["triggered"][0].tolist()]
    memory = fields["memory_id"][0].tolist()
    if hit >= len(activation) or not triggered[hit]:
        return False
    if int(memory[hit]) != 0:
        return False
    if any(triggered[hit + 1 :]):
        return False
    if abs(float(activation[hit]) - float(expected)) > 1e-5:
        return False
    if abs(float(fields["strength_confidence"][0, hit]) - 1.0) > 1e-5:
        return False
    return True


def _fan4_ok(mem, prompt: str, hit: int) -> bool:
    fields = mem.cue_fields(prompt)
    activation = fields["activation"][0].tolist()
    triggered = [bool(flag) for flag in fields["triggered"][0].tolist()]
    if any(triggered[hit:]):
        return False
    return all(abs(float(value)) <= 1e-8 for value in activation[hit:])


def _alpha_works(mem, items, rows, alpha: float) -> bool:
    from tidal.cue_inject import fan_factor
    import tidal.llm_memory as mod

    expected = float(mod.CUE_A_TRIG) * fan_factor(2, float(mod.CUE_FAN_S))
    for item, row in zip(items, rows):
        _write_target(mem, row)
        hit, match = _trigger_match(mem, item["prompt"])
        if hit is None or match is None:
            return False
        _add_companions(mem, row, match, 1, alpha)
        if not _fan2_ok(mem, item["prompt"], hit, expected):
            return False
        _write_target(mem, row)
        _add_companions(mem, row, match, 3, alpha)
        if not _fan4_ok(mem, item["prompt"], hit):
            return False
    mem.store.clear()
    return True


def _freeze_alpha(mem, items, rows) -> dict:
    """Largest alpha whose clean trace keeps fan 2 on the fan-1 trigger.

    Fan 4 must clear from that trigger onward, because ``S - ln 4`` puts the
    product under the floor. This check does not read a continuation margin.
    """
    tried = []
    selected = None
    for alpha in FAN_ALPHAS:
        ok = _alpha_works(mem, items, rows, float(alpha))
        tried.append({"alpha": float(alpha), "pass": bool(ok)})
        _print(f"fan alpha {alpha}: {ok}")
        if ok and selected is None:
            selected = float(alpha)
    mem.store.clear()
    return {"selected": selected, "tried": tried}


def _fan_effects(mem, item, row, empty: float, alpha: float) -> dict:
    _write_target(mem, row)
    first = float(_margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"]) - empty)
    _hit, match = _trigger_match(mem, item["prompt"])
    if match is None:
        return {"fan1": first, "fan2": None, "fan4": None, "strict": False}
    _add_companions(mem, row, match, 1, alpha)
    second = float(_margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"]) - empty)
    _write_target(mem, row)
    _add_companions(mem, row, match, 3, alpha)
    fourth = float(_margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"]) - empty)
    return {
        "fan1": first,
        "fan2": second,
        "fan4": fourth,
        "strict": bool(first > second > fourth),
    }


def _fan_summary(rows: list) -> dict:
    usable = [row for row in rows if row["fan2"] is not None and row["fan4"] is not None]
    if not usable:
        return {
            "n": 0,
            "k": 0,
            "means": {"1": None, "2": None, "4": None},
            "pass_means": False,
            "rows": rows,
        }
    means = {}
    for key, label in (("fan1", "1"), ("fan2", "2"), ("fan4", "4")):
        means[label] = float(sum(float(row[key]) for row in usable) / len(usable))
    return {
        "n": len(rows),
        "k": int(sum(bool(row["strict"]) for row in rows)),
        "means": means,
        "pass_means": bool(means["1"] > means["2"] > means["4"]),
        "rows": rows,
    }


def _decay_records(mem, prefer, prefer_rows) -> list:
    """Activation decay from the last trigger. Logit L1 is not a gate.

    Offsets 0, 1, 2, 4, and 8 are measured from the last triggering token.
    Offsets 1 through 8 must not trigger. Activation at 0 is positive, the
    series is monotone non-increasing within 1e-6, and activation at 8 is
    strictly below activation at 0.
    """
    records = []
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
            records.append(record)
            continue
        start = int(hits[-1])
        if start + DECAY_OFFSETS[-1] >= len(activation):
            record["reason"] = "sequence shorter than distance 8"
            records.append(record)
            continue
        if any(triggered[start + 1 : start + DECAY_OFFSETS[-1] + 1]):
            record["reason"] = "retrigger"
            records.append(record)
            continue
        taken = [float(activation[start + offset]) for offset in DECAY_OFFSETS]
        monotone = all(taken[i] >= taken[i + 1] - 1e-6 for i in range(len(taken) - 1))
        record["activation"] = taken
        if not (taken[0] > 0.0 and taken[-1] < taken[0] and monotone):
            record["reason"] = "activation"
            records.append(record)
            continue
        record["pass"] = True
        record["reason"] = "ok"
        records.append(record)
    return records


def _score_cell(mem, bank, prefer_rows, incidental_rows, base, alpha) -> dict:
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

    decay_rows = _decay_records(mem, prefer, prefer_rows)
    decay_pass = int(sum(bool(row["pass"]) for row in decay_rows))
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
    if alpha is None:
        fan = {"n": 0, "k": 0, "means": {"1": None, "2": None, "4": None}, "pass_means": False, "rows": []}
    else:
        fan_rows = []
        for item, row, effect, empty in zip(prefer, prefer_rows, cued, base["cued_empty"]):
            measured = _fan_effects(mem, item, row, float(empty), float(alpha))
            measured["id"] = item["id"]
            measured["fan1"] = float(effect)
            if measured["fan2"] is None or measured["fan4"] is None:
                measured["strict"] = False
            else:
                measured["strict"] = bool(float(effect) > float(measured["fan2"]) > float(measured["fan4"]))
            fan_rows.append(measured)
        fan = _fan_summary(fan_rows)
    fan_pass = bool(
        alpha is not None
        and fan["pass_means"]
        and int(fan["k"]) >= FAN_FIT_MIN
        and int(fan["n"]) >= FAN_FIT_MIN
    )
    passed = bool(gate1 and gate2 and gate3 and gate4 and gate5 and fan_pass)
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
        "fan_k": int(fan["k"]),
        "fan_n": int(fan["n"]),
        "fan_means": fan["means"],
        "fan_pass": fan_pass,
        "fan_rows": fan["rows"],
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
    # Decay is scored on activation. Quiet logits are not part of that gate.
    decay_quiet = {}
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
        return {"selected": None, "reason": "no cell passed gates 1-5 and the fit fan check"}

    def key(cell):
        return (
            float(cell["mean_cued_delta"]),
            -abs(float(cell["half_life"]) - 4.0),
            -abs(float(cell["lambda"]) - 0.25),
        )

    best = max(passing, key=key)
    return {
        "selected": {
            "theta": best["theta"],
            "half_life": best["half_life"],
            "tau": float(best["half_life"]) / math.log(2.0),
            "lambda": best["lambda"],
            "a_trig": best["a_trig"],
            "mean_cued_delta": best["mean_cued_delta"],
            "fan_means": best["fan_means"],
        },
        "reason": "largest mean cued margin; ties prefer half-life closer to 4, then lambda closer to 0.25",
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
    """Gates 1-5 and the fit fan check. Does not score a held bank.

    Theta stays at the frozen cosine-bracket value. The loop is half-life
    by lambda, from SCHEDULE.
    """
    import tidal.llm_memory as mod

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = _fit_bank()
    mem = _load_memory()
    started = time.perf_counter()
    cells = []
    alpha_report = {"selected": None, "tried": []}
    try:
        prefer = list(bank.PREFER)
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(mod.CUE_HALF_LIFE), float(mod.CUE_LAMBDA), 1.0)
        alpha_report = _freeze_alpha(mem, prefer, prefer_rows)
        _print("baselines")
        base = _baselines(mem, bank, prefer_rows)
        if alpha_report["selected"] is None:
            _print("no fan alpha kept the clean trace; cells are not scored")
        else:
            for index, cell in enumerate(SCHEDULE):
                _bind(cell["theta"], cell["half_life"], cell["lambda"], cell["a_trig"])
                _print(
                    f"cell {index + 1}/{len(SCHEDULE)} "
                    f"half_life={cell['half_life']} lambda={cell['lambda']}"
                )
                scored = _score_cell(
                    mem, bank, prefer_rows, incidental_rows, base, alpha_report["selected"]
                )
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
                            f"fan={scored['fan_k']}/{scored['fan_n']}",
                        ]
                    )
                )
        choice = _select(cells)
        chosen = choice["selected"]
        if chosen is None:
            _bind(*original)
        else:
            _bind(chosen["theta"], chosen["half_life"], chosen["lambda"], chosen["a_trig"])
        cost_rows = prefer_rows + incidental_rows
        cost = _cost(mem, cost_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        mem.close()
    payload = {
        "kind": "halflife-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-fit",
        "theta": FROZEN_THETA,
        "grid": list(SCHEDULE),
        "alpha": alpha_report,
        "cells": cells,
        "choice": choice,
        "cost": cost,
        "held_scored": False,
    }
    _dump(HALFLIFE_FIT_PATH, payload)
    return payload


def held(dest: Path = None) -> dict:
    """Gates 6 and 7 on the n=50 banks. Refuses to run unless the schedule fit selected a cell."""
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

    if not SCHEDULE_FIT_PATH.is_file():
        raise RuntimeError("schedule fit report is missing")
    fit_report = json.loads(SCHEDULE_FIT_PATH.read_text(encoding="utf-8"))
    chosen = (fit_report.get("choice") or {}).get("selected")
    alpha = (fit_report.get("alpha") or {}).get("selected")
    if not chosen or alpha is None:
        raise RuntimeError("fit did not select a cell; held is not scored")
    _bind(chosen["theta"], chosen["half_life"], chosen["lambda"], chosen.get("a_trig", 1.0))

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
        fan_rows = []
        prefer_list = list(prefer_items())
        encoded = _encode(mem, [_fact(item) for item in prefer_list])
        mem.store.clear()
        empties = [
            _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
            for item in prefer_list
        ]
        for index, (item, row, empty) in enumerate(zip(prefer_list, encoded, empties)):
            _print(f"fan {index + 1}/{len(prefer_list)}")
            measured = _fan_effects(mem, item, row, float(empty), float(alpha))
            measured["id"] = item["id"]
            fan_rows.append(measured)
        mem.store.clear()
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
    fan = _fan_summary(fan_rows)
    fan_rate = _count(fan_rows, lambda row: bool(row["strict"]))
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
        "fan_monotonic": fan_rate,
    }
    failures = []
    for name, floor in HELD_FLOORS.items():
        row = rates[name]
        if int(row["n"]) < 50 or int(row["k"]) < int(floor):
            failures.append({"gate": name, "k": row["k"], "n": row["n"], "floor": int(floor)})
    if int(fan_rate["n"]) < 50 or int(fan_rate["k"]) < HELD_FAN_MIN or not fan["pass_means"]:
        failures.append(
            {
                "gate": "fan_monotonic",
                "k": fan_rate["k"],
                "n": fan_rate["n"],
                "floor": HELD_FAN_MIN,
                "means": fan["means"],
            }
        )
    payload = {
        "kind": "schedule-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "selected": chosen,
        "alpha": alpha,
        "rates": rates,
        "floors": {**HELD_FLOORS, "fan_monotonic": HELD_FAN_MIN},
        "fan_means": fan["means"],
        "fan_rows": fan_rows,
        "failures": failures,
        "pass": not failures,
        "means": {
            "prefer_delta": float(sum(float(row["delta"]) for row in prefer_rows) / len(prefer_rows)),
            "avoid_delta": float(sum(float(row["delta"]) for row in avoid_rows) / len(avoid_rows)),
            "avoid_stored_lift": float(sum(float(row["stored_lift"]) for row in avoid_rows) / len(avoid_rows)),
            "toward_delta": float(sum(float(row["delta"]) for row in toward_rows) / len(toward_rows)),
            "away_delta": float(sum(float(row["delta"]) for row in away_rows) / len(away_rows)),
            "fan1": fan["means"].get("1"),
            "fan2": fan["means"].get("2"),
            "fan4": fan["means"].get("4"),
        },
    }
    _dump(SCHEDULE_HELD_PATH if dest is None else dest, payload)
    _print(json.dumps({"pass": payload["pass"], "failures": failures, "rates": {k: [v["k"], v["n"]] for k, v in rates.items()}}, indent=2))
    return payload


def _mix_kwta_residual(mem, cues, fields, k: int):
    """Replace the one-winner residual while the shipped trace is active.

    The on/off gate, the activation, and the winner's confidence stay the
    shipped trace. The vector becomes the k-winner-take-all mix. ``k`` of 1
    is not routed here; the caller keeps the shipped residual.
    """
    import numpy as np

    import tidal.llm_memory as mod
    from tidal.cue_inject import kwta_weights
    from tidal.episodic import cosine_scores

    keys, strengths, valences, values, _modalities = mem._slot_arrays()
    rows = np.asarray(cues, dtype=np.float64)
    center = np.asarray(mem.query_mean, dtype=np.float64).reshape(-1)
    activation = np.asarray(fields["activation"], dtype=np.float64).reshape(-1)
    confidence_scale = np.asarray(fields["strength_confidence"], dtype=np.float64).reshape(-1)
    memory = np.asarray(fields["memory_id"]).reshape(-1)
    steps, width = int(activation.shape[0]), int(mem.hidden)
    out = np.zeros((steps, width), dtype=np.float32)
    cache = {}
    active_tokens = 0
    positive_slots = 0
    for index in range(steps):
        level = float(activation[index])
        winner = int(memory[index])
        if winner < 0 or level == 0.0 or keys.shape[0] == 0:
            continue
        winner_strength = float(strengths[winner]) if 0 <= winner < strengths.shape[0] else 0.0
        if winner_strength <= 0.0:
            continue
        confidence = float(confidence_scale[index]) / winner_strength
        scores = cosine_scores(keys, rows[index] - center)
        weights = kwta_weights(scores, strengths, int(k))
        active_tokens += 1
        positive_slots += int(np.count_nonzero(weights > 0.0))
        for slot, weight in enumerate(weights.tolist()):
            if float(weight) == 0.0:
                continue
            offset = float(mod.VALENCE_SPAN) * math.tanh(float(valences[slot]))
            scale = (
                float(mod.CUE_LAMBDA)
                * level
                * confidence
                * float(weight)
                * float(strengths[slot])
                * (1.0 + offset)
            )
            if scale == 0.0:
                continue
            if slot not in cache:
                cache[slot] = (
                    mem._compose(values[slot], key=keys[slot], grad=False)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            out[index] = out[index] + cache[slot] * np.float32(scale)
    return out, active_tokens, positive_slots


def _install_kwta(mem, k: int):
    original = mem._trace_numpy
    stats = {"tokens": 0, "positive": 0}

    def wrapped(cues, state=None, _original=original, _k=int(k), _stats=stats):
        fields, residual, final = _original(cues, state)
        if _k == 1:
            return fields, residual, final
        mixed, tokens, positive = _mix_kwta_residual(mem, cues, fields, _k)
        _stats["tokens"] += int(tokens)
        _stats["positive"] += int(positive)
        return fields, mixed, final

    mem._trace_numpy = wrapped
    return original, stats


def anisotropy() -> dict:
    """Related versus unrelated cue cosines after the shipped key centers.

    Fit bank only. Does not write theta.
    """
    import numpy as np

    import tidal.llm_memory as mod
    from tidal.cue_inject import rank_auc
    from tidal.llm_memory import _center_path, _load_key_centers

    bank = _fit_bank()
    mem = _load_memory()
    started = time.perf_counter()
    theta_before = float(mod.CUE_THETA)
    related = []
    unrelated = []
    related_max = []
    unrelated_max = []
    try:
        loaded = _load_key_centers(
            _center_path(),
            model_name=mem.model_name,
            layer=mem.layer,
            hidden=mem.hidden,
        )
        if loaded is None:
            raise RuntimeError("llm_key_center.npz does not match this model")
        key_mean, query_mean = loaded
        if not np.allclose(mem.key_mean, key_mean) or not np.allclose(mem.query_mean, query_mean):
            raise RuntimeError("the memory centers are not llm_key_center.npz")
        prefer = list(bank.PREFER)
        neutrals = list(bank.NEUTRAL_CHOICE)
        encoded = _encode(mem, [_fact(item) for item in prefer])
        for item, row in zip(prefer, encoded):
            _write(mem, [row])
            key = mem.store._slots[0].k
            prompt_cos = _cosines(mem, item["prompt"], key)
            related.extend(prompt_cos)
            related_max.append(max(prompt_cos))
            for prompt in neutrals:
                other = _cosines(mem, prompt["prompt"], key)
                unrelated.extend(other)
                unrelated_max.append(max(other))
            _print(f"anisotropy {item['id']} related_max={related_max[-1]:.4f}")
        mem.store.clear()
        theta_after = float(mod.CUE_THETA)
        if theta_after != theta_before:
            raise RuntimeError("anisotropy changed theta")
    finally:
        mem.close()
    payload = {
        "kind": "anisotropy",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-fit",
        "centering": "tidal/llm_key_center.npz",
        "key_center": "key_mean",
        "cue_center": "query_mean",
        "theta_before": theta_before,
        "theta_after": theta_after,
        "theta_changed": False,
        "related_tokens": _percentiles(related),
        "unrelated_tokens": _percentiles(unrelated),
        "related_max": _percentiles(related_max),
        "unrelated_max": _percentiles(unrelated_max),
        "auc_tokens": rank_auc(related, unrelated),
        "auc_max": rank_auc(related_max, unrelated_max),
        "note": "Distribution only. Theta was not changed.",
    }
    _dump(ANISOTROPY_PATH, payload)
    _print(json.dumps({k: payload[k] for k in ("auc_tokens", "auc_max", "related_tokens", "unrelated_tokens", "theta_changed")}, indent=2))
    return payload


def compete() -> dict:
    """Winners=1 versus k-winner-take-all on the direction fit bank only.

    Does not import a held bank and does not select a k.
    """
    import tidal.llm_memory as mod

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = _fit_bank()
    mem = _load_memory()
    started = time.perf_counter()
    columns = []
    alpha_report = {"selected": None, "tried": []}
    try:
        prefer = list(bank.PREFER)
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(mod.CUE_HALF_LIFE), float(mod.CUE_LAMBDA), 1.0)
        alpha_report = _freeze_alpha(mem, prefer, prefer_rows)
        _print("baselines")
        base = _baselines(mem, bank, prefer_rows)
        if alpha_report["selected"] is None:
            _print("no fan alpha kept the clean trace; columns are not scored")
        else:
            for k in KWTA_KS:
                _print(f"kwta k={k}")
                restore, stats = _install_kwta(mem, k)
                try:
                    scored = _score_cell(
                        mem, bank, prefer_rows, incidental_rows, base, alpha_report["selected"]
                    )
                finally:
                    mem._trace_numpy = restore
                tokens = int(stats["tokens"])
                scored["k"] = int(k)
                scored["mean_positive_slots"] = (
                    None if tokens == 0 else float(stats["positive"]) / float(tokens)
                )
                columns.append(scored)
                _print(
                    "  "
                    + " ".join(
                        [
                            f"pass={scored['pass']}",
                            f"cued={scored['mean_cued_delta']:.4f}",
                            f"store={scored['store_means']}",
                            f"fan={scored['fan_means']}",
                            f"active={scored['mean_positive_slots']}",
                        ]
                    )
                )
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    payload = {
        "kind": "kwta-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-fit",
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "ks": list(KWTA_KS),
        "alpha": alpha_report,
        "columns": [
            {
                "k": row["k"],
                "pass": row["pass"],
                "mean_cued_delta": row["mean_cued_delta"],
                "n_cued_pos": row["n_cued_pos"],
                "unrelated_abs_mean": row["unrelated_abs_mean"],
                "false_triggers": row["false_triggers"],
                "false_trigger_n": row["false_trigger_n"],
                "decay_pass": row["decay_pass"],
                "decay_n": row["decay_n"],
                "store_means": row["store_means"],
                "quiet_nll_mean": row["quiet_nll_mean"],
                "quiet_nll_n": row["quiet_nll_n"],
                "fan_k": row["fan_k"],
                "fan_n": row["fan_n"],
                "fan_means": row["fan_means"],
                "gate1_pass": row["gate1_pass"],
                "gate2_pass": row["gate2_pass"],
                "gate3_pass": row["gate3_pass"],
                "gate4_pass": row["gate4_pass"],
                "gate5_pass": row["gate5_pass"],
                "fan_pass": row["fan_pass"],
                "mean_positive_slots": row["mean_positive_slots"],
            }
            for row in columns
        ],
        "selected_k": None,
        "held_scored": False,
        "theta_changed": False,
        "note": "Diagnostic only. Floors were not moved and no k was selected.",
    }
    _dump(KWTA_FIT_PATH, payload)
    _print(json.dumps({"columns": payload["columns"], "selected_k": None}, indent=2))
    return payload


def direction_held() -> dict:
    """Gates 1–5 on the direction held bank. Does not choose a cell.

    The 6-of-8 point rules stay the fit rules because this bank has 8 prefer
    items and 8 neutral-choice items. A rate whose n is below 50 fails even
    when that point rule passes. Theta, half-life, and lambda are not written
    back, and the n=50 floor table is not rescored.
    """
    import tidal.llm_memory as mod
    from tidal.subconscious_direction_data import held_bank

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = held_bank()
    mem = _load_memory()
    started = time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        neutrals = list(bank.NEUTRAL_CHOICE)
        nll_items = list(bank.NEUTRAL_NLL)
        if len(prefer) != 8 or len(neutrals) != 8 or len(nll_items) != 12:
            raise RuntimeError("direction held bank size changed")
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        base = _baselines(mem, bank, prefer_rows)
        scored = _score_cell(mem, bank, prefer_rows, incidental_rows, base, FROZEN_ALPHA)
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()

    def _rate(name: str, k: int, n: int, point_pass: bool) -> dict:
        return {
            "gate": name,
            "k": int(k),
            "n": int(n),
            "wilson": _wilson(int(k), int(n)) if int(n) > 0 else None,
            "point_pass": bool(point_pass),
            "n50_pass": bool(int(n) >= HELD_N_MIN and point_pass),
        }

    cued_point = bool(
        int(scored["n_cued_pos"]) >= CUED_POS_MIN and float(scored["mean_cued_delta"]) > 0.0
    )
    unrelated_point = bool(float(scored["unrelated_abs_mean"]) <= UNRELATED_ABS_MAX)
    rates = [
        _rate("cued_positive", scored["n_cued_pos"], scored["decay_n"], cued_point),
        _rate(
            "false_trigger",
            scored["false_triggers"],
            scored["false_trigger_n"],
            bool(scored["gate2_pass"]),
        ),
        _rate("decay", scored["decay_pass"], scored["decay_n"], bool(scored["gate3_pass"])),
        _rate("fan_strict", scored["fan_k"], scored["fan_n"], bool(scored["fan_pass"])),
    ]
    counts = {
        "unrelated_abs_mean": float(scored["unrelated_abs_mean"]),
        "unrelated_n": int(scored["false_trigger_n"]),
        "unrelated_point_pass": unrelated_point,
        "unrelated_n50_pass": bool(
            int(scored["false_trigger_n"]) >= HELD_N_MIN and unrelated_point
        ),
        "store_means": scored["store_means"],
        "store_n": int(scored["decay_n"]),
        "store_point_pass": bool(scored["gate4_pass"]),
        "store_n50_pass": bool(int(scored["decay_n"]) >= HELD_N_MIN and scored["gate4_pass"]),
        "quiet_nll_mean": scored["quiet_nll_mean"],
        "quiet_nll_n": int(scored["quiet_nll_n"]),
        "quiet_point_pass": bool(scored["gate5_pass"]),
        "quiet_n50_pass": bool(
            int(scored["quiet_nll_n"]) >= HELD_N_MIN and scored["gate5_pass"]
        ),
        "mean_cued_delta": float(scored["mean_cued_delta"]),
        "fan_means": scored["fan_means"],
    }
    n50_fail = [row["gate"] for row in rates if not row["n50_pass"]]
    for name in ("unrelated", "store", "quiet"):
        if not counts[f"{name}_n50_pass"]:
            n50_fail.append(name)
    payload = {
        "kind": "direction-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "bank": "direction-held",
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "rates": rates,
        "counts": counts,
        "decay_reasons": [
            {"id": row.get("id"), "pass": row.get("pass"), "reason": row.get("reason")}
            for row in scored["decay_rows"]
        ],
        "point_pass": bool(scored["pass"] and unrelated_point),
        "n50_pass": not n50_fail,
        "n50_fail": n50_fail,
        "theta_changed": False,
        "floors_rescored": False,
        "note": "Point rules are the fit rules. n below 50 fails. No cell was selected.",
    }
    _dump(DIRECTION_HELD_PATH, payload)
    _print(
        json.dumps(
            {
                "point_pass": payload["point_pass"],
                "n50_pass": payload["n50_pass"],
                "mean_cued_delta": counts["mean_cued_delta"],
                "unrelated_abs_mean": counts["unrelated_abs_mean"],
                "store_means": counts["store_means"],
                "quiet_nll_mean": counts["quiet_nll_mean"],
                "quiet_nll_n": counts["quiet_nll_n"],
                "rates": rates,
            },
            indent=2,
        )
    )
    return payload


def _fresh_summary(scored) -> dict:
    """Apply the pre-registered n=50 floors. ``_score_cell`` still uses the n=8 cut."""

    def _rate(name: str, k: int, n: int, floor: int) -> dict:
        return {
            "gate": name,
            "k": int(k),
            "n": int(n),
            "floor": int(floor),
            "wilson": _wilson(int(k), int(n)) if int(n) > 0 else None,
            "pass": bool(int(n) >= FRESH_N_MIN and int(k) >= int(floor)),
        }

    cued = _rate("cued_positive", scored["n_cued_pos"], scored["decay_n"], FRESH_CUED_MIN)
    cued["mean"] = float(scored["mean_cued_delta"])
    cued["pass"] = bool(cued["pass"] and float(scored["mean_cued_delta"]) > 0.0)
    false_trigger = _rate(
        "false_trigger",
        scored["false_triggers"],
        scored["false_trigger_n"],
        0,
    )
    false_rate = float(scored["false_trigger_rate"])
    false_trigger["rate"] = false_rate
    false_trigger["pass"] = bool(
        int(scored["false_trigger_n"]) >= FRESH_N_MIN and false_rate <= FRESH_FALSE_TRIGGER_CEILING
    )
    decay = _rate("decay", scored["decay_pass"], scored["decay_n"], FRESH_DECAY_MIN)
    fan = _rate("fan_strict", scored["fan_k"], scored["fan_n"], FRESH_FAN_MIN)
    fan["means"] = scored["fan_means"]
    means = scored["fan_means"]
    fan["pass"] = bool(
        fan["pass"]
        and means["1"] is not None
        and float(means["1"]) > float(means["2"]) > float(means["4"])
    )
    store_n = int(scored["decay_n"])
    store_means = scored["store_means"]
    store_pass = bool(
        store_n >= FRESH_N_MIN
        and float(store_means["16"]) > 0.0
        and float(store_means["32"]) > 0.0
        and float(store_means["16"]) >= float(store_means["8"]) - FRESH_STORE_DROP_MAX
        and float(store_means["32"]) >= float(store_means["8"]) - FRESH_STORE_DROP_MAX
    )
    quiet_n = int(scored["quiet_nll_n"])
    quiet_mean = scored["quiet_nll_mean"]
    quiet_pass = bool(
        quiet_n >= FRESH_N_MIN
        and quiet_mean is not None
        and float(quiet_mean) <= FRESH_QUIET_NLL_MAX
    )
    unrelated_n = int(scored["false_trigger_n"])
    unrelated_mean = float(scored["unrelated_abs_mean"])
    unrelated_pass = bool(
        unrelated_n >= FRESH_N_MIN and unrelated_mean <= FRESH_UNRELATED_ABS_MAX
    )
    gates = {
        "cued_positive": cued,
        "unrelated": {
            "n": unrelated_n,
            "mean_abs": unrelated_mean,
            "max": FRESH_UNRELATED_ABS_MAX,
            "pass": unrelated_pass,
        },
        "false_trigger": false_trigger,
        "decay": decay,
        "store": {"n": store_n, "means": store_means, "pass": store_pass},
        "quiet_nll": {"n": quiet_n, "mean": quiet_mean, "pass": quiet_pass},
        "fan_strict": fan,
    }
    return {
        "gates": gates,
        "pass": all(row["pass"] for row in gates.values()),
        "decay_reasons": [
            {"id": row.get("id"), "pass": row.get("pass"), "reason": row.get("reason")}
            for row in scored["decay_rows"]
        ],
    }


def revision() -> dict:
    """One held pass: fresh direction gates, then the previous n=50 floors.

    Does not read the fresh bank while choosing a cell. Theta, half-life,
    lambda, and the floors stay the values already selected on fit.
    """
    from tidal.subconscious_cue_fresh_data import PROTOCOL, fresh_bank

    prereg = (REPORT_DIR / "subconscious_cue_prereg.md").read_text(encoding="utf-8")
    if PROTOCOL not in prereg or "38/50" not in prereg:
        raise RuntimeError("fresh gates are not in the preregistration")
    import tidal.llm_memory as mod

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = fresh_bank()
    mem = _load_memory()
    started = time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        if len(prefer) < FRESH_N_MIN or len(list(bank.NEUTRAL_CHOICE)) < FRESH_N_MIN:
            raise RuntimeError("fresh bank is below 50")
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        base = _baselines(mem, bank, prefer_rows)
        scored = _score_cell(mem, bank, prefer_rows, incidental_rows, base, FROZEN_ALPHA)
        fresh = _fresh_summary(scored)
        cost = _cost(mem, prefer_rows + incidental_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    floors = held(dest=REPORT_DIR / "subconscious_cue_revision_floors.json")
    payload = {
        "kind": "revision-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank.sha256(),
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "fresh": fresh,
        "cost": cost,
        "floors_pass": bool(floors["pass"]),
        "floor_failures": floors["failures"],
        "floor_rates": floors["rates"],
        "floor_means": floors["means"],
        "pass": bool(fresh["pass"] and floors["pass"]),
        "theta_changed": False,
        "note": "One pass. The fresh gates and the previous floors were not used to choose the cell.",
    }
    _dump(REVISION_PATH, payload)
    _print(
        json.dumps(
            {
                "pass": payload["pass"],
                "fresh_pass": fresh["pass"],
                "floors_pass": payload["floors_pass"],
                "fresh": {
                    name: {"pass": row["pass"], **{k: row[k] for k in row if k in ("k", "n", "mean", "mean_abs", "rate")}}
                    for name, row in fresh["gates"].items()
                },
                "floor_failures": floors["failures"],
            },
            indent=2,
        )
    )
    return payload


def _prefer_slice(mem, bank) -> dict:
    """Cued margin, activation decay, and fan on one prefer list.

    Alpha stays the shipped 0.25. This does not call ``_freeze_alpha``.
    """
    prefer = list(bank.PREFER)
    rows = _encode(mem, [_fact(item) for item in prefer])
    mem.store.clear()
    empties = [
        _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
        for item in prefer
    ]
    cued = []
    for index, (item, row, empty) in enumerate(zip(prefer, rows, empties)):
        _print(f"scope cued {index + 1}/{len(prefer)}")
        _write(mem, [row])
        armed = _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
        cued.append(float(armed - empty))
    mean = float(sum(cued) / len(cued))
    positive = int(sum(value > 0.0 for value in cued))
    _print("scope decay")
    decay_rows = _decay_records(mem, prefer, rows)
    decay_k = int(sum(bool(row["pass"]) for row in decay_rows))
    fan_rows = []
    for index, (item, row, effect, empty) in enumerate(zip(prefer, rows, cued, empties)):
        _print(f"scope fan {index + 1}/{len(prefer)}")
        measured = _fan_effects(mem, item, row, float(empty), float(FROZEN_ALPHA))
        measured["id"] = item["id"]
        measured["fan1"] = float(effect)
        if measured["fan2"] is None or measured["fan4"] is None:
            measured["strict"] = False
        else:
            measured["strict"] = bool(float(effect) > float(measured["fan2"]) > float(measured["fan4"]))
        fan_rows.append(measured)
    fan = _fan_summary(fan_rows)
    reasons = {}
    for row in decay_rows:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {
        "n": len(prefer),
        "cued_positive": positive,
        "cued_mean": mean,
        "cued_wilson": _wilson(positive, len(prefer)),
        "decay_pass": decay_k,
        "decay_wilson": _wilson(decay_k, len(prefer)),
        "decay_reasons": reasons,
        "fan_k": int(fan["k"]),
        "fan_n": int(fan["n"]),
        "fan_means": fan["means"],
        "fan_wilson": _wilson(int(fan["k"]), int(fan["n"])) if int(fan["n"]) else None,
        "alpha": float(FROZEN_ALPHA),
    }


def scope_fit() -> dict:
    """Shipped cell on the n=8 direction fit bank and on fit50.

    Does not re-select alpha, theta, half-life, lambda, or fan S, and does
    not score a held bank.
    """
    import tidal.llm_memory as mod
    from tidal.subconscious_cue_fit50_data import fit50_bank

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    mem = _load_memory()
    started = time.perf_counter()
    try:
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        _print("scope n=8")
        small = _prefer_slice(mem, _fit_bank())
        _print("scope fit50")
        wide = _prefer_slice(mem, fit50_bank())
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    payload = {
        "kind": "scope-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "alpha_reselected": False,
        "decay_definition": (
            "From the last triggering token, offsets 0, 1, 2, 4, and 8 exist; "
            "no retrigger on offsets 1 through 8; activation at 0 is positive; "
            "activation is monotone non-increasing within 1e-6; activation at 8 "
            "is strictly below activation at 0. Logit L1 is not part of the gate."
        ),
        "n8": small,
        "fit50": wide,
        "held_scored": False,
        "note": "Design check only. Floors were not moved.",
    }
    _dump(REPORT_DIR / "subconscious_cue_scope_fit.json", payload)
    _print(json.dumps({"n8": small, "fit50": wide}, indent=2))
    return payload


def event_fit() -> dict:
    """Shipped cell on the n=8 direction fit bank and on fit50.

    Checks the retrieval-event rule without re-selecting alpha or scoring held.
    """
    import tidal.llm_memory as mod
    from tidal.subconscious_cue_fit50_data import fit50_bank

    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    mem = _load_memory()
    started = time.perf_counter()
    try:
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        _print("event n=8")
        small = _prefer_slice(mem, _fit_bank())
        _print("event fit50")
        wide = _prefer_slice(mem, fit50_bank())
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    payload = {
        "kind": "event-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "alpha_reselected": False,
        "floors_moved": False,
        "event": (
            "RIF runs once per cue retrieval event, not per token. "
            "A retrieval event is a trigger onset or a winner change. "
            "Retriggers of the same winner within its active decay window do not rerun competition."
        ),
        "n8": small,
        "fit50": wide,
        "held_scored": False,
        "note": "Design check only. Floors were not moved. The event bank was not scored.",
    }
    _dump(EVENT_FIT_PATH, payload)
    _print(json.dumps({"n8": small, "fit50": wide}, indent=2))
    return payload


def word_held() -> dict:
    """One held pass of the real-word direction bank, then the previous floors.

    Refuses unless the decay definition and the gates are already in the
    preregistration. Does not move theta, half-life, lambda, alpha, or a floor.
    """
    import tidal.llm_memory as mod
    from tidal.subconscious_cue_word_data import PROTOCOL, word_bank

    prereg = (REPORT_DIR / "subconscious_cue_prereg.md").read_text(encoding="utf-8")
    for needle in (
        PROTOCOL,
        "38/50",
        "Logit L1 is not part of the gate.",
        "distribution mismatch: injection cannot steer toward unknown nonce tokens",
        "The word bank was not scored.",
    ):
        if needle not in prereg:
            raise RuntimeError("word-bank gates are not in the preregistration")
    if WORD_HELD_PATH.is_file():
        raise RuntimeError("the word bank was already scored")
    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = word_bank()
    mem = _load_memory()
    started = time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        if len(prefer) < FRESH_N_MIN or len(list(bank.NEUTRAL_CHOICE)) < FRESH_N_MIN:
            raise RuntimeError("word bank is below 50")
        if len(list(bank.NEUTRAL_NLL)) < FRESH_N_MIN or len(list(bank.incidental_facts())) < 31:
            raise RuntimeError("word bank controls are short")
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        base = _baselines(mem, bank, prefer_rows)
        scored = _score_cell(mem, bank, prefer_rows, incidental_rows, base, FROZEN_ALPHA)
        summary = _fresh_summary(scored)
        cost = _cost(mem, prefer_rows + incidental_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    floors = held(dest=REPORT_DIR / "subconscious_cue_word_floors.json")
    payload = {
        "kind": "word-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank.sha256(),
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "constants_changed": False,
        "direction": summary,
        "cost": cost,
        "floors_pass": bool(floors["pass"]),
        "floor_failures": floors["failures"],
        "floor_rates": floors["rates"],
        "floor_means": floors["means"],
        "pass": bool(summary["pass"] and floors["pass"]),
        "theta_changed": False,
        "note": "One pass. Fit counts were not used to move a gate or a floor.",
    }
    _dump(WORD_HELD_PATH, payload)
    _print(
        json.dumps(
            {
                "pass": payload["pass"],
                "direction_pass": summary["pass"],
                "floors_pass": payload["floors_pass"],
                "direction": {
                    name: {
                        "pass": row["pass"],
                        **{key: row[key] for key in row if key in ("k", "n", "mean", "mean_abs", "rate")},
                    }
                    for name, row in summary["gates"].items()
                },
                "floor_failures": floors["failures"],
                "cost": cost,
            },
            indent=2,
        )
    )
    return payload


def event_held() -> dict:
    """One rescore of the word direction bank and the previous floors.

    Refuses unless the retrieval-event definition is already in the
    preregistration. Does not move theta, half-life, lambda, alpha, or a floor.
    Does not rewrite the earlier word-bank reports.
    """
    import tidal.llm_memory as mod
    from tidal.subconscious_cue_word_data import PROTOCOL, word_bank

    prereg = (REPORT_DIR / "subconscious_cue_prereg.md").read_text(encoding="utf-8")
    for needle in (
        "RIF runs once per cue retrieval event, not per token.",
        "Retriggers of the same winner within its active decay window do not rerun competition.",
        "Anderson, Bjork, and Bjork (1994) treat retrieval-induced forgetting as a consequence of a retrieval attempt.",
        "The event bank was not scored.",
        "38/50",
    ):
        if needle not in prereg:
            raise RuntimeError("event gates are not in the preregistration")
    if EVENT_HELD_PATH.is_file():
        raise RuntimeError("the event bank was already scored")
    original = (
        float(mod.CUE_THETA),
        float(mod.CUE_HALF_LIFE),
        float(mod.CUE_LAMBDA),
        float(mod.CUE_A_TRIG),
    )
    bank = word_bank()
    mem = _load_memory()
    started = time.perf_counter()
    try:
        prefer = list(bank.PREFER)
        if len(prefer) < FRESH_N_MIN or len(list(bank.NEUTRAL_CHOICE)) < FRESH_N_MIN:
            raise RuntimeError("word bank is below 50")
        if len(list(bank.NEUTRAL_NLL)) < FRESH_N_MIN or len(list(bank.incidental_facts())) < 31:
            raise RuntimeError("word bank controls are short")
        prefer_rows = _encode(mem, [_fact(item) for item in prefer])
        incidental_rows = _encode(mem, list(bank.incidental_facts())[:31])
        _bind(FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        base = _baselines(mem, bank, prefer_rows)
        scored = _score_cell(mem, bank, prefer_rows, incidental_rows, base, FROZEN_ALPHA)
        summary = _fresh_summary(scored)
        cost = _cost(mem, prefer_rows + incidental_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        _bind(*original)
        mem.close()
    floors = held(dest=REPORT_DIR / "subconscious_cue_event_floors.json")
    payload = {
        "kind": "event-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank.sha256(),
        "theta": FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": FROZEN_ALPHA,
        "constants_changed": False,
        "floors_moved": False,
        "direction": summary,
        "cost": cost,
        "floors_pass": bool(floors["pass"]),
        "floor_failures": floors["failures"],
        "floor_rates": floors["rates"],
        "floor_means": floors["means"],
        "pass": bool(summary["pass"] and floors["pass"]),
        "theta_changed": False,
        "note": "One pass. The retrieval-event definition was not changed after this score.",
    }
    _dump(EVENT_HELD_PATH, payload)
    _print(
        json.dumps(
            {
                "pass": payload["pass"],
                "direction_pass": summary["pass"],
                "floors_pass": payload["floors_pass"],
                "direction": {
                    name: {
                        "pass": row["pass"],
                        **{key: row[key] for key in row if key in ("k", "n", "mean", "mean_abs", "rate")},
                    }
                    for name, row in summary["gates"].items()
                },
                "floor_failures": floors["failures"],
                "cost": cost,
            },
            indent=2,
        )
    )
    return payload


def main(argv: list) -> None:
    mode = argv[1] if len(argv) > 1 else "fit"
    if mode == "histogram":
        histogram()
    elif mode == "fit":
        fit()
    elif mode == "held":
        held()
    elif mode == "anisotropy":
        anisotropy()
    elif mode == "compete":
        compete()
    elif mode == "direction-held":
        direction_held()
    elif mode == "revision":
        revision()
    elif mode == "scope-fit":
        scope_fit()
    elif mode == "word-held":
        word_held()
    elif mode == "event-fit":
        event_fit()
    elif mode == "event-held":
        event_held()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
