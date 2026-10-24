"""Fit and held gates for wall-clock decay and the drifting clock.

Fit ids and held ids are disjoint. The schedule is numeric. Held refuses
unless the preregistration still contains the unscored sentence, and it
refuses if its report already exists. The sleep guard, the arousal guard,
and the cue guard do the same with their own sentences and paths.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidal.clock_drift import (  # noqa: E402
    DRIFT_RATE,
    ENTRAIN_GAIN,
    WALL_TAU,
    clock_activation,
    elapsed_since,
)
from tidal.cue_inject import CueState, trace_sequence  # noqa: E402
from tidal.llm_memory import (  # noqa: E402
    ByteTokenizer,
    CUE_FLOOR,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_TAU,
    CUE_THETA,
    LLMMemory,
)
from tidal.subconscious_clock_data import (  # noqa: E402
    BOTH_TOKEN_STRONG,
    BOTH_WALL_STRONG,
    ENTRAIN_SECONDS,
    FIT_IDS,
    HELD_IDS,
    PROTOCOL,
    TOKEN_STEPS,
    WALL_CLEAR,
    WALL_SHORT,
    bank_sha256,
)
from tidal.subconscious_measure_data import wilson_interval  # noqa: E402

REPORT_DIR = ROOT / "reports"
FIT_PATH = REPORT_DIR / "subconscious_clock_fit.json"
HELD_PATH = REPORT_DIR / "subconscious_clock_held.json"
SLEEP_GUARD_PATH = REPORT_DIR / "subconscious_clock_sleep_held.json"
SLEEP_RECORDED_PATH = REPORT_DIR / "subconscious_sleep_held.json"
AROUSAL_GUARD_PATH = REPORT_DIR / "subconscious_clock_arousal_held.json"
AROUSAL_RECORDED_PATH = REPORT_DIR / "subconscious_arousal_held.json"
CUE_HELD_PATH = REPORT_DIR / "subconscious_clock_cue_held.json"
CUE_FLOORS_PATH = REPORT_DIR / "subconscious_clock_cue_floors.json"
PREREG_PATH = REPORT_DIR / "subconscious_clock_prereg.md"
GATE_NAMES = ("wall", "token", "both", "entrain", "fresh")
FORBIDDEN_FIELDS = ("clock_error", "wall_seconds", "internal_seconds", "internal")


class _Eye(nn.Module):
    """Byte id ``i`` is hidden axis ``i``. No download."""

    def __init__(self):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, vocab)
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.lm_head = nn.Linear(vocab, vocab, bias=False)
        eye = torch.eye(vocab)
        with torch.no_grad():
            self.embed.weight.copy_(eye)
            self.lm_head.weight.copy_(eye)
        self.config = type("Cfg", (), {"hidden_size": vocab, "vocab_size": vocab})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **_kwargs):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        return type("Out", (), {"logits": logits, "hidden_states": tuple(states) if output_hidden_states else None})()


def _patch_centers():
    original = LLMMemory._key_centers

    def zeros(self):
        row = np.zeros(self.hidden, dtype=np.float64)
        return row, row.copy()

    LLMMemory._key_centers = zeros
    return original


def _memory():
    return LLMMemory(_Eye(), ByteTokenizer())


def _git_sha() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roundtrip(payload):
    return json.loads(json.dumps(payload))


def _close(mem) -> None:
    mem.close()


def _armed(mem) -> CueState:
    return CueState(
        a=1.0,
        dt=0,
        memory_id=0,
        confidence=1.0,
        strength=1.0,
        valence=0.0,
        gap=1.0,
        winner_cosine=1.0,
        fan_factor=1.0,
        vec=np.ones(mem.cue_dim, dtype=np.float32),
        trigger_wall=0.0,
    )


def _wall_gate() -> dict:
    at_zero = clock_activation(1.0, 0, 0.0, CUE_TAU, WALL_TAU)
    at_short = clock_activation(1.0, 0, float(WALL_SHORT[0]), CUE_TAU, WALL_TAU)
    at_mid = clock_activation(1.0, 0, float(WALL_SHORT[1]), CUE_TAU, WALL_TAU)
    at_clear = clock_activation(1.0, 0, float(WALL_CLEAR), CUE_TAU, WALL_TAU)
    drifted = clock_activation(1.0, 0, float(WALL_SHORT[1]) * DRIFT_RATE, CUE_TAU, WALL_TAU)
    mem = _memory()
    try:
        mem.cue_state = _armed(mem)
        mem.advance_wall(float(WALL_SHORT[1]))
        live = float(mem.cue_state.a)
        live_dt = int(mem.cue_state.dt)
        live_wall = float(mem.wall_seconds)
        live_internal = float(mem.internal_seconds)
    finally:
        _close(mem)
    cleared = _memory()
    try:
        cleared.cue_state = _armed(cleared)
        cleared.advance_wall(float(WALL_CLEAR))
        cleared_id = int(cleared.cue_state.memory_id)
    finally:
        _close(cleared)
    passed = (
        at_zero == 1.0
        and at_zero > at_short > at_mid > float(CUE_FLOOR)
        and at_clear < float(CUE_FLOOR)
        and live_dt == 0
        and live_wall == float(WALL_SHORT[1])
        and live_internal == float(WALL_SHORT[1]) * DRIFT_RATE
        and live == at_mid
        and live != drifted
        and cleared_id == -1
    )
    return {
        "pass": bool(passed),
        "wall_0": at_zero,
        "wall_1": at_short,
        "wall_4": at_mid,
        "wall_clear": at_clear,
        "live_4": live,
        "live_dt": live_dt,
        "live_wall": live_wall,
        "live_internal": live_internal,
        "cleared_id": cleared_id,
    }


def _token_gate() -> dict:
    levels = []
    exact = True
    for dt in TOKEN_STEPS:
        got = clock_activation(1.0, int(dt), 0.0, CUE_TAU, WALL_TAU)
        want = math.exp(-int(dt) / CUE_TAU)
        levels.append(got)
        exact = exact and got == want
    values = np.ones((1, 3), dtype=np.float32)
    state = CueState(
        a=1.0,
        dt=0,
        memory_id=0,
        confidence=1.0,
        strength=1.0,
        fan_factor=1.0,
        vec=values[0].copy(),
        trigger_wall=0.0,
    )
    from tidal.cue_inject import step_cue

    stepped, _row = step_cue(
        state,
        [0.0],
        [1.0],
        [0.0],
        values,
        theta=0.5,
        tau=CUE_TAU,
        a_trig=1.0,
        floor=CUE_FLOOR,
        gap_tau=0.8,
        valence_span=1.0,
        now=0.0,
    )
    passed = (
        exact
        and float(CUE_HALF_LIFE) == 30.0
        and CUE_TAU == 30.0 / math.log(2.0)
        and float(CUE_THETA) == 0.32
        and float(CUE_LAMBDA) == 0.3
        and stepped.a == math.exp(-1.0 / CUE_TAU)
        and stepped.dt == 1
    )
    return {
        "pass": bool(passed),
        "token_0": levels[0],
        "token_1": levels[1],
        "token_4": levels[2],
        "stepped": float(stepped.a),
    }


def _both_gate() -> dict:
    pairs = {}
    passed = True
    for name, pair in (("wall_strong", BOTH_WALL_STRONG), ("token_strong", BOTH_TOKEN_STRONG)):
        dt, elapsed = int(pair[0]), float(pair[1])
        token_only = clock_activation(1.0, dt, 0.0, CUE_TAU, WALL_TAU)
        wall_only = clock_activation(1.0, 0, elapsed, CUE_TAU, WALL_TAU)
        both = clock_activation(1.0, dt, elapsed, CUE_TAU, WALL_TAU)
        pairs[name] = {"token_only": token_only, "wall_only": wall_only, "both": both}
        passed = passed and both == min(token_only, wall_only) and both <= token_only and both <= wall_only
        passed = passed and both > token_only * wall_only
    passed = passed and pairs["wall_strong"]["wall_only"] < pairs["wall_strong"]["token_only"]
    passed = passed and pairs["token_strong"]["token_only"] < pairs["token_strong"]["wall_only"]
    passed = passed and elapsed_since(5.0, 4.0) == 0.0
    passed = passed and clock_activation(1.0, 1, elapsed_since(5.0, 4.0), CUE_TAU, WALL_TAU) == math.exp(
        -1.0 / CUE_TAU
    )
    cues = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    keys = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    values = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    fields, final = trace_sequence(
        cues,
        keys,
        [1.0],
        [0.0],
        values,
        np.zeros(3),
        theta=0.5,
        tau=CUE_TAU,
        a_trig=1.0,
        floor=CUE_FLOOR,
        gap_tau=0.8,
        valence_span=1.0,
        now=100.0,
    )
    fresh = float(fields["activation"][1])
    passed = passed and fresh == math.exp(-1.0 / CUE_TAU) and final.trigger_wall == 100.0
    return {
        "pass": bool(passed),
        "wall_strong": pairs["wall_strong"]["both"],
        "token_strong": pairs["token_strong"]["both"],
        "fresh_token": fresh,
    }


def _entrain_gate() -> dict:
    mem = _memory()
    try:
        mem.wake_for(4.0)
        mem.observe_valence(1.5)
        pressure = float(mem.sleep_pressure)
        arousal = float(mem.arousal)
        mem.cue_state = _armed(mem)
        mem.advance_wall(float(ENTRAIN_SECONDS))
        open_error = float(mem.clock_error)
        carried = float(mem.cue_state.a)
        mem.calibrate(mem.wall_seconds)
        once = float(mem.clock_error)
        mem.calibrate(mem.wall_seconds)
        twice = float(mem.clock_error)
        after = float(mem.cue_state.a)
        passed = (
            float(mem.wall_seconds) == float(ENTRAIN_SECONDS)
            and float(mem.internal_seconds) != float(mem.wall_seconds)
            and open_error == 2.0
            and once == open_error * (1.0 - ENTRAIN_GAIN)
            and twice == once * (1.0 - ENTRAIN_GAIN)
            and after == carried
            and mem.sleep_pressure == pressure
            and mem.arousal == arousal
        )
    finally:
        _close(mem)
    return {
        "pass": bool(passed),
        "error_open": open_error,
        "error_once": once,
        "error_twice": twice,
        "activation": after,
    }


def _write_match(mem, text: str) -> None:
    hidden = mem.cue_hidden(text)
    key = hidden[0] - mem.query_mean
    value = np.zeros(mem.cue_dim, dtype=np.float32)
    value[3] = 4.0
    assert mem.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")


def _fresh_gate() -> dict:
    early = _memory()
    late = _memory()
    try:
        text = "Ab"
        _write_match(early, text)
        _write_match(late, text)
        before = early.cue_fields(text)
        late.advance_wall(float(ENTRAIN_SECONDS))
        after = late.cue_fields(text)
        forbidden = not any(name in before or name in after for name in FORBIDDEN_FIELDS)
        activation_0 = float(before["activation"][0, 0])
        passed = (
            int(before["interface_version"]) == int(after["interface_version"]) == int(CUE_INTERFACE_VERSION) == 5
            and torch.equal(before["activation"], after["activation"])
            and forbidden
            and float(late.wall_seconds) == float(ENTRAIN_SECONDS)
            and float(late.internal_seconds) == float(ENTRAIN_SECONDS) * DRIFT_RATE
        )
    finally:
        _close(early)
        _close(late)
    return {"pass": bool(passed), "activation_0": activation_0, "interface_version": int(CUE_INTERFACE_VERSION)}


def score_item(item_id: str) -> dict:
    wall = _wall_gate()
    token = _token_gate()
    both = _both_gate()
    entrain = _entrain_gate()
    fresh = _fresh_gate()
    gates = {
        "wall": bool(wall["pass"]),
        "token": bool(token["pass"]),
        "both": bool(both["pass"]),
        "entrain": bool(entrain["pass"]),
        "fresh": bool(fresh["pass"]),
    }
    canonical = {
        "wall_0": wall["wall_0"],
        "wall_1": wall["wall_1"],
        "wall_4": wall["wall_4"],
        "wall_clear": wall["wall_clear"],
        "live_4": wall["live_4"],
        "live_internal": wall["live_internal"],
        "token_0": token["token_0"],
        "token_1": token["token_1"],
        "token_4": token["token_4"],
        "wall_strong": both["wall_strong"],
        "token_strong": both["token_strong"],
        "fresh_token": both["fresh_token"],
        "error_open": entrain["error_open"],
        "error_once": entrain["error_once"],
        "error_twice": entrain["error_twice"],
        "calibrated_activation": entrain["activation"],
        "fresh_activation_0": fresh["activation_0"],
        "interface_version": fresh["interface_version"],
    }
    return {
        "id": item_id,
        "pass": all(gates.values()),
        "gates": gates,
        "canonical": _roundtrip(canonical),
    }


def _summarize(rows: list) -> dict:
    summary = {}
    for name in GATE_NAMES:
        flags = [bool(row["gates"][name]) for row in rows]
        k = int(sum(flags))
        n = len(flags)
        low, high = wilson_interval(k, n)
        summary[name] = {"k": k, "n": n, "rate": k / n, "wilson": [low, high], "pass": k == n}
    item_flags = [bool(row["pass"]) for row in rows]
    k = int(sum(item_flags))
    n = len(item_flags)
    low, high = wilson_interval(k, n)
    summary["item"] = {"k": k, "n": n, "rate": k / n, "wilson": [low, high], "pass": k == n}
    return summary


def _score_ids(ids: tuple) -> list:
    rows = []
    for index, item_id in enumerate(ids):
        print(f"item {index + 1}/{len(ids)} {item_id}", flush=True)
        rows.append(score_item(item_id))
    return rows


def _payload_gates(summary: dict) -> dict:
    return {name: [summary[name]["k"], summary[name]["n"]] for name in summary}


def fit() -> dict:
    if FIT_PATH.is_file():
        raise RuntimeError("clock fit report already exists")
    original = _patch_centers()
    started = time.perf_counter()
    try:
        rows = _score_ids(FIT_IDS)
    finally:
        LLMMemory._key_centers = original
    summary = _summarize(rows)
    canonical = rows[0]["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    payload = {
        "kind": "clock-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "ids": list(FIT_IDS),
        "gates": summary,
        "canonical": canonical,
        "canonical_same": bool(same),
        "pass": bool(summary["item"]["pass"] and same),
        "note": "Fit only. The held bank was not scored.",
    }
    _dump(FIT_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "gates": _payload_gates(summary), "canonical": canonical}, indent=2))
    return payload


def held() -> dict:
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The held bank was not scored." not in prereg:
        raise RuntimeError("the held bank is not locked in the preregistration")
    if HELD_PATH.is_file():
        raise RuntimeError("the held bank was already scored")
    if not FIT_PATH.is_file():
        raise RuntimeError("clock fit report is missing")
    fit_report = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    if not fit_report.get("pass"):
        raise RuntimeError("fit did not pass; held is not scored")
    original = _patch_centers()
    started = time.perf_counter()
    try:
        rows = _score_ids(HELD_IDS)
    finally:
        LLMMemory._key_centers = original
    summary = _summarize(rows)
    canonical = fit_report["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    failures = []
    for name, row in summary.items():
        if int(row["n"]) < 50 or int(row["k"]) != int(row["n"]):
            failures.append({"gate": name, "k": row["k"], "n": row["n"]})
    if not same:
        failures.append({"gate": "canonical", "k": 0, "n": len(rows)})
    payload = {
        "kind": "clock-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "ids": list(HELD_IDS),
        "gates": summary,
        "canonical": rows[0]["canonical"] if rows else None,
        "fit_canonical": canonical,
        "canonical_same": bool(same),
        "failures": failures,
        "pass": not failures,
        "note": "One pass. Constants were not changed after the fit lock.",
    }
    _dump(HELD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": failures, "gates": _payload_gates(summary)}, indent=2))
    return payload


def _load_module(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_rows(mod, label: str) -> tuple:
    original = mod._patch_centers()
    started = time.perf_counter()
    try:
        rows = []
        for index, item_id in enumerate(mod.HELD_IDS):
            print(f"{label} {index + 1}/{len(mod.HELD_IDS)} {item_id}", flush=True)
            rows.append(mod.score_item(item_id))
    finally:
        mod.LLMMemory._key_centers = original
    return rows, time.perf_counter() - started


def _guard_payload(kind: str, mod, rows: list, elapsed: float, recorded: dict, note: str) -> dict:
    summary = mod._summarize(rows)
    canonical = recorded["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    failures = []
    for name, row in summary.items():
        if int(row["n"]) < 50 or int(row["k"]) != int(row["n"]):
            failures.append({"gate": name, "k": row["k"], "n": row["n"]})
    if not same:
        failures.append({"gate": "canonical", "k": 0, "n": len(rows)})
    return {
        "kind": kind,
        "git": _git_sha(),
        "elapsed_s": elapsed,
        "protocol": mod.PROTOCOL,
        "bank_sha256": mod.bank_sha256(),
        "ids": list(mod.HELD_IDS),
        "gates": summary,
        "canonical": rows[0]["canonical"] if rows else None,
        "recorded_canonical": canonical,
        "canonical_same": bool(same),
        "failures": failures,
        "pass": not failures,
        "note": note,
    }


def sleep_guard() -> dict:
    """One rescore of the sleep held schedule into a new file.

    Does not call the sleep held() entry, and does not overwrite
    reports/subconscious_sleep_held.json.
    """
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The sleep guard was not scored." not in prereg:
        raise RuntimeError("the sleep guard is not locked in the preregistration")
    if SLEEP_GUARD_PATH.is_file():
        raise RuntimeError("the sleep guard was already scored")
    if not SLEEP_RECORDED_PATH.is_file():
        raise RuntimeError("the recorded sleep held report is missing")
    recorded = json.loads(SLEEP_RECORDED_PATH.read_text(encoding="utf-8"))
    mod = _load_module("eval_subconscious_sleep_clock_guard", "eval_subconscious_sleep.py")
    rows, elapsed = _guard_rows(mod, "sleep")
    payload = _guard_payload(
        "clock-sleep-guard",
        mod,
        rows,
        elapsed,
        recorded,
        "One pass. Process S formulas were not changed.",
    )
    _dump(SLEEP_GUARD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": payload["failures"], "gates": _payload_gates(payload["gates"])}, indent=2))
    return payload


def arousal_guard() -> dict:
    """One rescore of the arousal held schedule into a new file.

    Does not call the arousal held() entry, and does not overwrite
    reports/subconscious_arousal_held.json.
    """
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The arousal guard was not scored." not in prereg:
        raise RuntimeError("the arousal guard is not locked in the preregistration")
    if AROUSAL_GUARD_PATH.is_file():
        raise RuntimeError("the arousal guard was already scored")
    if not AROUSAL_RECORDED_PATH.is_file():
        raise RuntimeError("the recorded arousal held report is missing")
    recorded = json.loads(AROUSAL_RECORDED_PATH.read_text(encoding="utf-8"))
    mod = _load_module("eval_subconscious_arousal_clock_guard", "eval_subconscious_arousal.py")
    rows, elapsed = _guard_rows(mod, "arousal")
    payload = _guard_payload(
        "clock-arousal-guard",
        mod,
        rows,
        elapsed,
        recorded,
        "One pass. Arousal weights were not changed.",
    )
    _dump(AROUSAL_GUARD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": payload["failures"], "gates": _payload_gates(payload["gates"])}, indent=2))
    return payload


def cue_guard() -> dict:
    """One rescore of the word direction bank and the cue-triggered floors.

    Writes new files. Does not call backend_held and does not overwrite the
    merged cue reports or the earlier cue-guard reports.
    """
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The cue guard was not scored." not in prereg:
        raise RuntimeError("the cue guard is not locked in the preregistration")
    if CUE_HELD_PATH.is_file() or CUE_FLOORS_PATH.is_file():
        raise RuntimeError("the cue guard was already scored")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mod = _load_module("eval_subconscious_cue_clock_guard", "eval_subconscious_cue.py")
    from tidal.subconscious_cue_word_data import word_bank

    import tidal.llm_memory as memory

    original = (
        float(memory.CUE_THETA),
        float(memory.CUE_HALF_LIFE),
        float(memory.CUE_LAMBDA),
        float(memory.CUE_A_TRIG),
    )
    bank = word_bank()
    started = time.perf_counter()
    mem = mod._load_memory()
    try:
        prefer = list(bank.PREFER)
        if len(prefer) < mod.FRESH_N_MIN or len(list(bank.NEUTRAL_CHOICE)) < mod.FRESH_N_MIN:
            raise RuntimeError("word bank is below 50")
        if len(list(bank.NEUTRAL_NLL)) < mod.FRESH_N_MIN or len(list(bank.incidental_facts())) < 31:
            raise RuntimeError("word bank controls are short")
        prefer_rows = mod._encode(mem, [mod._fact(item) for item in prefer])
        incidental_rows = mod._encode(mem, list(bank.incidental_facts())[:31])
        mod._bind(mod.FROZEN_THETA, float(original[1]), float(original[2]), 1.0)
        base = mod._baselines(mem, bank, prefer_rows)
        scored = mod._score_cell(mem, bank, prefer_rows, incidental_rows, base, mod.FROZEN_ALPHA)
        summary = mod._fresh_summary(scored)
        cost = mod._cost(mem, prefer_rows + incidental_rows, prefer[0]["prompt"])
        mem.store.clear()
    finally:
        mod._bind(*original)
        mem.close()
    floors = mod.held(dest=CUE_FLOORS_PATH)
    mod._bind(*original)
    diffs = mod._pr57_diffs(summary, floors)
    payload = {
        "kind": "clock-cue-guard",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "theta": mod.FROZEN_THETA,
        "half_life": original[1],
        "lambda": original[2],
        "alpha": mod.FROZEN_ALPHA,
        "constants_changed": False,
        "direction": summary,
        "cost": cost,
        "floors_pass": bool(floors["pass"]),
        "floor_failures": floors["failures"],
        "floor_rates": floors["rates"],
        "floor_means": floors["means"],
        "winner_events": floors["winner_events"],
        "matches_pr57": not diffs,
        "diffs": diffs,
        "pass": bool(summary["pass"] and floors["pass"] and not diffs),
        "note": "One pass. Wall decay is internal to a carried trace. A fresh retrieval stamps the current wall time.",
    }
    _dump(CUE_HELD_PATH, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "matches_pr57": payload["matches_pr57"],
                "diffs": diffs,
                "floors_pass": payload["floors_pass"],
                "floor_rates": {name: [row["k"], row["n"]] for name, row in floors["rates"].items()},
                "winner_events": floors["winner_events"],
                "floor_means": floors["means"],
                "cost": cost,
            },
            indent=2,
        )
    )
    return payload


def main(argv: list) -> None:
    mode = argv[1] if len(argv) > 1 else "fit"
    if mode == "fit":
        fit()
    elif mode == "held":
        held()
    elif mode == "sleep-guard":
        sleep_guard()
    elif mode == "arousal-guard":
        arousal_guard()
    elif mode == "cue-guard":
        cue_guard()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
