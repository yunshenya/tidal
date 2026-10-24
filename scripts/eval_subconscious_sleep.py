"""Fit and held gates for process S. Cue floors are a separate one-shot guard.

Fit ids and held ids are disjoint. The schedule is numeric. Held refuses
unless the preregistration still contains the unscored sentence, and it
refuses if its report already exists.
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

from tidal.llm_memory import ByteTokenizer, LLMMemory  # noqa: E402
from tidal.sleep import (  # noqa: E402
    REPLAY_BLEND,
    SLEEP_FLOOR,
    SPONTANEOUS_CEILING,
    SPONTANEOUS_PERIOD,
    STRENGTH_STEP,
    sleep_after,
    wake_after,
)
from tidal.subconscious_measure_data import wilson_interval  # noqa: E402
from tidal.subconscious_sleep_data import (  # noqa: E402
    FIT_IDS,
    HELD_IDS,
    OFFLINE_TICKS,
    PROTOCOL,
    SLEEP_DURATIONS,
    WAKE_DURATIONS,
    bank_sha256,
)

REPORT_DIR = ROOT / "reports"
FIT_PATH = REPORT_DIR / "subconscious_sleep_fit.json"
HELD_PATH = REPORT_DIR / "subconscious_sleep_held.json"
CUE_HELD_PATH = REPORT_DIR / "subconscious_sleep_cue_held.json"
CUE_FLOORS_PATH = REPORT_DIR / "subconscious_sleep_cue_floors.json"
PREREG_PATH = REPORT_DIR / "subconscious_sleep_prereg.md"
GATE_NAMES = ("wake", "sleep", "replay", "spontaneous", "cue_source")


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


def _wake_gate(mem) -> dict:
    start = float(mem.sleep_pressure)
    observed = []
    for duration in WAKE_DURATIONS:
        trial = _memory()
        try:
            assert trial.sleep_pressure == start
            observed.append(float(trial.wake_for(float(duration))))
        finally:
            trial.close()
    passed = start < observed[0] < observed[1] < observed[2] <= 1.0
    return {"pass": bool(passed), "start": start, "pressures": observed}


def _sleep_gate() -> dict:
    left = _memory()
    right = _memory()
    try:
        before = float(left.wake_for(float(WAKE_DURATIONS[-1])))
        assert float(right.wake_for(float(WAKE_DURATIONS[-1]))) == before
        assert len(left.store) == 0 and len(right.store) == 0
        short = left.sleep_for(float(SLEEP_DURATIONS[0]))
        long = right.sleep_for(float(SLEEP_DURATIONS[1]))
    finally:
        left.close()
        right.close()
    drop_short = float(short["drop"])
    drop_long = float(long["drop"])
    after_short = float(short["pressure_after"])
    after_long = float(long["pressure_after"])
    passed = (
        drop_long > drop_short > 0.0
        and after_short > float(SLEEP_FLOOR)
        and after_long > float(SLEEP_FLOOR)
        and after_short > 0.0
        and after_long > 0.0
        and after_long < after_short < before
    )
    return {
        "pass": bool(passed),
        "before": before,
        "after": [after_short, after_long],
        "drop": [drop_short, drop_long],
    }


def _basis(dim: int, index: int) -> np.ndarray:
    row = np.zeros(int(dim), dtype=np.float64)
    row[int(index)] = 1.0
    return row


def _replay_gate() -> dict:
    mem = _memory()
    try:
        dim = int(mem.hidden)
        low_k = _basis(dim, 0)
        high_k = _basis(dim, 2)
        high_v = _basis(dim, 3)
        assert mem.store.write(
            low_k,
            low_k.copy(),
            name="low",
            rel="note",
            group="low",
            answer="LOW-ANSWER",
            text="raw-low-sleep",
        )
        assert mem.store.write(
            high_k,
            high_v,
            name="high",
            rel="note",
            group="high",
            answer="HIGH-ANSWER",
            text="raw-high-sleep",
        )
        report = mem.sleep_for(1.0)
        low, high = mem.store.slots
        secrets = ("raw-low-sleep", "raw-high-sleep", "LOW-ANSWER", "HIGH-ANSWER")
        blob = json.dumps(mem.sleep_audit())
        keys_ok = all(
            set(row) == {"index", "error", "updated", "strength_before", "strength_after"}
            for row in mem.sleep_audit()
        )
        expected = (1.0 - float(REPLAY_BLEND)) * high_v + float(REPLAY_BLEND) * high_k
        passed = (
            low.text == "raw-low-sleep"
            and low.answer == "LOW-ANSWER"
            and high.text == "raw-high-sleep"
            and high.answer == "HIGH-ANSWER"
            and low.strength == 1.0
            and np.array_equal(np.asarray(low.v), low_k)
            and high.strength == math.exp(-float(STRENGTH_STEP))
            and np.allclose(np.asarray(high.v), expected)
            and report["replay"][0]["updated"] is False
            and report["replay"][1]["updated"] is True
            and keys_ok
            and all(secret not in blob for secret in secrets)
            and not hasattr(low, "replay_raw")
            and not hasattr(high, "replay_raw")
            and mem.sleep_pressure == float(SLEEP_FLOOR)
        )
        fields = mem.cue_fields("Zz")
        passed = bool(passed) and "sleep_audit" not in fields and all(secret not in json.dumps(fields["source"]) for secret in secrets)
    finally:
        mem.close()
    return {
        "pass": bool(passed),
        "high_strength": math.exp(-float(STRENGTH_STEP)),
        "low_updated": False,
        "high_updated": True,
    }


def _spontaneous_gate() -> dict:
    mem = _memory()
    try:
        pressure = float(mem.wake_for(float(WAKE_DURATIONS[-1])))
        key = _basis(mem.hidden, 0)
        assert mem.store.write(
            key, key.copy(), name="wander", rel="note", answer="STAY", text="raw-spontaneous-sleep"
        )
        ticks = mem.offline_ticks(int(OFFLINE_TICKS))
        sources = [tick["source"][0][0] for tick in ticks]
        fires = int(sources.count("spontaneous"))
        rate = fires / int(OFFLINE_TICKS)
        passed = (
            fires == 1
            and sources[0] == "spontaneous"
            and sources.count("none") == int(OFFLINE_TICKS) - 1
            and 0.0 < rate < float(SPONTANEOUS_CEILING)
            and int(OFFLINE_TICKS) == int(SPONTANEOUS_PERIOD)
            and mem.sleep_pressure == pressure
            and mem.store.slots[0].text == "raw-spontaneous-sleep"
            and mem.store.slots[0].strength == 1.0
            and "raw-spontaneous-sleep" not in json.dumps(mem.sleep_audit())
            and mem.sleep_audit()[0]["updated"] is False
        )
    finally:
        mem.close()
    return {"pass": bool(passed), "fires": fires, "ticks": int(OFFLINE_TICKS), "rate": rate, "pressure": pressure}


def _cue_gate() -> dict:
    mem = _memory()
    try:
        text = "Ab"
        hidden = mem.cue_hidden(text)
        key = hidden[0] - mem.query_mean
        value = np.zeros(mem.cue_dim, dtype=np.float32)
        value[3] = 4.0
        assert mem.store.write(key, value, name="ada", rel="note", answer="A", text="awake-cue-text")
        fields = mem.cue_fields(text)
        sources = list(fields["source"][0])
        pressure = float(mem.sleep_pressure)
        exported = [float(item) for item in fields["sleep_pressure"].detach().reshape(-1).tolist()]
        # The export is float32. 0.1 is not a float32 value, so exact == fails.
        pressure_ok = bool(exported) and all(math.isclose(item, pressure, rel_tol=0.0, abs_tol=1e-6) for item in exported)
        passed = (
            fields["interface_version"] == 4
            and set(sources) <= {"cue", "none"}
            and "cue" in sources
            and "spontaneous" not in sources
            and pressure_ok
            and "awake-cue-text" not in json.dumps(sources)
        )
    finally:
        mem.close()
    return {"pass": bool(passed), "interface_version": 4, "pressure": pressure}


def score_item(item_id: str) -> dict:
    mem = _memory()
    try:
        wake = _wake_gate(mem)
    finally:
        mem.close()
    sleep = _sleep_gate()
    replay = _replay_gate()
    spontaneous = _spontaneous_gate()
    cue = _cue_gate()
    gates = {
        "wake": bool(wake["pass"]),
        "sleep": bool(sleep["pass"]),
        "replay": bool(replay["pass"]),
        "spontaneous": bool(spontaneous["pass"]),
        "cue_source": bool(cue["pass"]),
    }
    canonical = {
        "wake_pressures": wake["pressures"],
        "sleep_before": sleep["before"],
        "sleep_after": sleep["after"],
        "sleep_drop": sleep["drop"],
        "spontaneous_fires": spontaneous["fires"],
        "spontaneous_rate": spontaneous["rate"],
        "replay_high_strength": replay["high_strength"],
        "formula_wake_4": wake_after(float(SLEEP_FLOOR), float(WAKE_DURATIONS[-1])),
        "formula_sleep_1": sleep_after(sleep["before"], float(SLEEP_DURATIONS[0])),
        "formula_sleep_3": sleep_after(sleep["before"], float(SLEEP_DURATIONS[1])),
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


def fit() -> dict:
    if FIT_PATH.is_file():
        raise RuntimeError("sleep fit report already exists")
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
        "kind": "sleep-fit",
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
    print(json.dumps({"pass": payload["pass"], "gates": {name: [summary[name]["k"], summary[name]["n"]] for name in summary}}, indent=2))
    return payload


def held() -> dict:
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The held bank was not scored." not in prereg:
        raise RuntimeError("the held bank is not locked in the preregistration")
    if HELD_PATH.is_file():
        raise RuntimeError("the held bank was already scored")
    if not FIT_PATH.is_file():
        raise RuntimeError("sleep fit report is missing")
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
        "kind": "sleep-held",
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
    print(json.dumps({"pass": payload["pass"], "failures": failures, "gates": {name: [summary[name]["k"], summary[name]["n"]] for name in summary}}, indent=2))
    return payload


def _load_cue_eval():
    path = ROOT / "scripts" / "eval_subconscious_cue.py"
    spec = importlib.util.spec_from_file_location("eval_subconscious_cue_sleep_guard", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cue_guard() -> dict:
    """One rescore of the word direction bank and the cue-triggered floors.

    Writes new files. Does not call backend_held and does not overwrite the
    merged cue reports.
    """
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The cue guard was not scored." not in prereg:
        raise RuntimeError("the cue guard is not locked in the preregistration")
    if CUE_HELD_PATH.is_file() or CUE_FLOORS_PATH.is_file():
        raise RuntimeError("the cue guard was already scored")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mod = _load_cue_eval()
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
    diffs = mod._pr57_diffs(summary, floors)
    payload = {
        "kind": "sleep-cue-guard",
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
        "note": "One pass. Sleep pressure is exported and is not part of the residual.",
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
    elif mode == "cue-guard":
        cue_guard()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
