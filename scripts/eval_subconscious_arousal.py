"""Fit and held gates for LC-NE arousal. Sleep and cue floors are separate guards.

Fit ids and held ids are disjoint. The schedule is numeric. Held refuses
unless the preregistration still contains the unscored sentence, and it
refuses if its report already exists. The sleep guard and the cue guard
do the same with their own sentences and paths.
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

from tidal.arousal import (  # noqa: E402
    AROUSAL_BASELINE,
    apply_trace,
)
from tidal.llm_memory import ByteTokenizer, CUE_INTERFACE_VERSION, LLMMemory  # noqa: E402
from tidal.sleep import SLEEP_FLOOR  # noqa: E402
from tidal.subconscious_arousal_data import (  # noqa: E402
    FIT_IDS,
    HELD_IDS,
    PROTOCOL,
    QUIET_LONG,
    QUIET_SHORT,
    SURPRISE_HIGH,
    SURPRISE_SEEDS,
    TRIGGER_HIGH,
    TRIGGER_LOW,
    VALENCE_HIGH,
    VALENCE_LOW,
    bank_sha256,
)
from tidal.subconscious_measure_data import wilson_interval  # noqa: E402

REPORT_DIR = ROOT / "reports"
FIT_PATH = REPORT_DIR / "subconscious_arousal_fit.json"
HELD_PATH = REPORT_DIR / "subconscious_arousal_held.json"
SLEEP_GUARD_PATH = REPORT_DIR / "subconscious_arousal_sleep_held.json"
SLEEP_RECORDED_PATH = REPORT_DIR / "subconscious_sleep_held.json"
CUE_HELD_PATH = REPORT_DIR / "subconscious_arousal_cue_held.json"
CUE_FLOORS_PATH = REPORT_DIR / "subconscious_arousal_cue_floors.json"
PREREG_PATH = REPORT_DIR / "subconscious_arousal_prereg.md"
GATE_NAMES = ("surprise", "quiet", "valence_trigger", "export")
EXPORT_ABS_TOL = 1e-6


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


def _seeded(mem) -> float:
    for value in SURPRISE_SEEDS:
        mem.observe_surprise(float(value))
    return float(mem.arousal)


def _surprise_gate() -> dict:
    mem = _memory()
    try:
        start = float(mem.arousal)
        after_seeds = _seeded(mem)
        after_high = float(mem.observe_surprise(float(SURPRISE_HIGH)))
        passed = (
            start == float(AROUSAL_BASELINE)
            and after_seeds == float(AROUSAL_BASELINE)
            and after_high > float(AROUSAL_BASELINE)
            and after_high <= 1.0
            and mem.sleep_pressure == float(SLEEP_FLOOR)
            and mem.surprise_window() == ()
        )
    finally:
        _close(mem)
    return {"pass": bool(passed), "after_seeds": after_seeds, "after_high": after_high}


def _quiet_gate(after_high: float) -> dict:
    short = _memory()
    long = _memory()
    try:
        for mem in (short, long):
            _seeded(mem)
            mem.observe_surprise(float(SURPRISE_HIGH))
            assert mem.arousal == after_high
        quiet_short = float(short.quiet_ticks(int(QUIET_SHORT)))
        quiet_long = float(long.quiet_ticks(int(QUIET_LONG)))
        trace = apply_trace(after_high, [False] * int(QUIET_LONG))
        passed = (
            quiet_long < quiet_short
            and (after_high - quiet_short) > 0.0
            and (after_high - quiet_long) > 0.0
            and quiet_long > float(AROUSAL_BASELINE)
            and quiet_short > float(AROUSAL_BASELINE)
            and quiet_short <= 1.0
            and quiet_long <= 1.0
            and trace[-1] == quiet_long
            and short.sleep_pressure == float(SLEEP_FLOOR)
            and long.sleep_pressure == float(SLEEP_FLOOR)
        )
    finally:
        _close(short)
        _close(long)
    return {
        "pass": bool(passed),
        "quiet_short": quiet_short,
        "quiet_long": quiet_long,
        "drop_short": after_high - quiet_short,
        "drop_long": after_high - quiet_long,
    }


def _valence_trigger_gate() -> dict:
    zero = _memory()
    low = _memory()
    high = _memory()
    hold = _memory()
    none = _memory()
    one = _memory()
    many = _memory()
    try:
        valence_zero = float(zero.observe_valence(0.0))
        valence_low = float(low.observe_valence(float(VALENCE_LOW)))
        valence_high = float(high.observe_valence(float(VALENCE_HIGH)))
        elevated = float(hold.observe_valence(float(VALENCE_HIGH)))
        held = float(hold.observe_triggers(0))
        trigger_zero = float(none.observe_triggers(0))
        trigger_low = float(one.observe_triggers(int(TRIGGER_LOW)))
        trigger_high = float(many.observe_triggers(int(TRIGGER_HIGH)))
        quiet = apply_trace(float(AROUSAL_BASELINE), [False] * int(TRIGGER_HIGH))
        single = apply_trace(float(AROUSAL_BASELINE), [True])
        dense = apply_trace(float(AROUSAL_BASELINE), [True] * int(TRIGGER_HIGH))
        passed = (
            valence_zero == float(AROUSAL_BASELINE)
            and float(AROUSAL_BASELINE) < valence_low < valence_high <= 1.0
            and held == elevated
            and trigger_zero == float(AROUSAL_BASELINE)
            and float(AROUSAL_BASELINE) < trigger_low < trigger_high <= 1.0
            and quiet[-1] == float(AROUSAL_BASELINE)
            and dense[-1] > single[-1] > float(AROUSAL_BASELINE)
            and zero.sleep_pressure == float(SLEEP_FLOOR)
        )
    finally:
        for mem in (zero, low, high, hold, none, one, many):
            _close(mem)
    return {
        "pass": bool(passed),
        "valence_zero": valence_zero,
        "valence_low": valence_low,
        "valence_high": valence_high,
        "trigger_hold": held,
        "trigger_zero": trigger_zero,
        "trigger_low": trigger_low,
        "trigger_high": trigger_high,
        "trace_quiet": quiet[-1],
        "trace_one": single[-1],
        "trace_four": dense[-1],
    }


def _close_enough(got, expected) -> bool:
    return math.isclose(float(got), float(expected), rel_tol=0.0, abs_tol=EXPORT_ABS_TOL)


def _write_match(mem, text: str) -> None:
    hidden = mem.cue_hidden(text)
    key = hidden[0] - mem.query_mean
    value = np.zeros(mem.cue_dim, dtype=np.float32)
    value[3] = 4.0
    assert mem.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")


def _export_gate() -> dict:
    left = _memory()
    right = _memory()
    empty = _memory()
    asleep = _memory()
    try:
        text = "Ab"
        _write_match(left, text)
        _write_match(right, text)
        right.observe_valence(float(VALENCE_HIGH))
        assert right.arousal > left.arousal
        pressure = float(left.sleep_pressure)
        before = float(left.arousal)
        full = left.cue_fields(text)
        moved = right.cue_fields(text)
        flags = [bool(item) for item in full["triggered"][0].tolist()]
        expected = apply_trace(before, flags)
        exported = [float(item) for item in full["arousal"].detach().reshape(-1).tolist()]
        series_ok = bool(exported) and len(exported) == len(expected) and all(
            _close_enough(got, want) for got, want in zip(exported, expected)
        )
        in_range = all(0.0 <= item <= 1.0 for item in exported)
        quiet = empty.cue_fields(text)
        quiet_exported = [float(item) for item in quiet["arousal"].detach().reshape(-1).tolist()]
        quiet_ok = bool(quiet_exported) and all(_close_enough(item, float(AROUSAL_BASELINE)) for item in quiet_exported)
        level = float(asleep.observe_valence(float(VALENCE_HIGH)))
        woke = float(asleep.wake_for(4.0))
        wake_same = asleep.arousal == level and woke > float(SLEEP_FLOOR)
        slept = float(asleep.sleep_for(3.0)["pressure_after"])
        sleep_same = asleep.arousal == level and float(SLEEP_FLOOR) < slept < woke
        passed = (
            int(full["interface_version"]) == int(CUE_INTERFACE_VERSION) == 5
            and series_ok
            and in_range
            and "cue" in full["source"][0]
            and torch.equal(full["activation"], moved["activation"])
            and torch.equal(left.last_added, right.last_added)
            and left.sleep_pressure == pressure
            and right.sleep_pressure == pressure
            and quiet["source"] == [["none"] * int(quiet["memory_id"].shape[1])]
            and quiet_ok
            and empty.arousal == float(AROUSAL_BASELINE)
            and empty.sleep_pressure == float(SLEEP_FLOOR)
            and wake_same
            and sleep_same
        )
    finally:
        _close(left)
        _close(right)
        _close(empty)
        _close(asleep)
    return {"pass": bool(passed), "export_0": exported[0], "export_last": exported[-1]}


def score_item(item_id: str) -> dict:
    surprise = _surprise_gate()
    quiet = _quiet_gate(float(surprise["after_high"]))
    valence = _valence_trigger_gate()
    export = _export_gate()
    gates = {
        "surprise": bool(surprise["pass"]),
        "quiet": bool(quiet["pass"]),
        "valence_trigger": bool(valence["pass"]),
        "export": bool(export["pass"]),
    }
    canonical = {
        "after_seeds": surprise["after_seeds"],
        "after_high": surprise["after_high"],
        "quiet_short": quiet["quiet_short"],
        "quiet_long": quiet["quiet_long"],
        "drop_short": quiet["drop_short"],
        "drop_long": quiet["drop_long"],
        "valence_zero": valence["valence_zero"],
        "valence_low": valence["valence_low"],
        "valence_high": valence["valence_high"],
        "trigger_hold": valence["trigger_hold"],
        "trigger_zero": valence["trigger_zero"],
        "trigger_low": valence["trigger_low"],
        "trigger_high": valence["trigger_high"],
        "trace_quiet": valence["trace_quiet"],
        "trace_one": valence["trace_one"],
        "trace_four": valence["trace_four"],
        "export_0": export["export_0"],
        "export_last": export["export_last"],
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
        raise RuntimeError("arousal fit report already exists")
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
        "kind": "arousal-fit",
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
        raise RuntimeError("arousal fit report is missing")
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
        "kind": "arousal-held",
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
    mod = _load_module("eval_subconscious_sleep_arousal_guard", "eval_subconscious_sleep.py")
    original = mod._patch_centers()
    started = time.perf_counter()
    try:
        rows = []
        for index, item_id in enumerate(mod.HELD_IDS):
            print(f"sleep {index + 1}/{len(mod.HELD_IDS)} {item_id}", flush=True)
            rows.append(mod.score_item(item_id))
    finally:
        mod.LLMMemory._key_centers = original
    summary = mod._summarize(rows)
    canonical = recorded["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    failures = []
    for name, row in summary.items():
        if int(row["n"]) < 50 or int(row["k"]) != int(row["n"]):
            failures.append({"gate": name, "k": row["k"], "n": row["n"]})
    if not same:
        failures.append({"gate": "canonical", "k": 0, "n": len(rows)})
    payload = {
        "kind": "arousal-sleep-guard",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": mod.PROTOCOL,
        "bank_sha256": mod.bank_sha256(),
        "ids": list(mod.HELD_IDS),
        "gates": summary,
        "canonical": rows[0]["canonical"] if rows else None,
        "recorded_canonical": canonical,
        "canonical_same": bool(same),
        "failures": failures,
        "pass": not failures,
        "note": "One pass. Process S formulas were not changed.",
    }
    _dump(SLEEP_GUARD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": failures, "gates": {name: [summary[name]["k"], summary[name]["n"]] for name in summary}}, indent=2))
    return payload


def cue_guard() -> dict:
    """One rescore of the word direction bank and the cue-triggered floors.

    Writes new files. Does not call backend_held and does not overwrite the
    merged cue reports or the sleep cue-guard reports.
    """
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The cue guard was not scored." not in prereg:
        raise RuntimeError("the cue guard is not locked in the preregistration")
    if CUE_HELD_PATH.is_file() or CUE_FLOORS_PATH.is_file():
        raise RuntimeError("the cue guard was already scored")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mod = _load_module("eval_subconscious_cue_arousal_guard", "eval_subconscious_cue.py")
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
        "kind": "arousal-cue-guard",
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
        "note": "One pass. Arousal is exported and is not part of the residual.",
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
    elif mode == "cue-guard":
        cue_guard()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
