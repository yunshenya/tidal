"""Fit and held gates for reconstructive phone and birthday recall.

Fit ids and held ids are disjoint. The schedule is numeric. Held refuses
unless the preregistration still contains the unscored sentence, and it
refuses if its report already exists. The sleep, arousal, clock, and cue
guards do the same with their own sentences and paths.
"""
from __future__ import annotations

import importlib.util
import json
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

from tidal.llm_memory import ByteTokenizer, CUE_INTERFACE_VERSION, LLMMemory  # noqa: E402
from tidal.reconstruct import birthday_parts, grade, phone_parts, recall_surface, residue_from_answer  # noqa: E402
from tidal.subconscious_measure_data import wilson_interval  # noqa: E402
from tidal.subconscious_reconstruct_data import (  # noqa: E402
    BIRTHDAY,
    BIRTHDAY_PROMPT,
    EMOTION,
    FIT_IDS,
    HELD_IDS,
    HIGH,
    LONG,
    LOW,
    LURE_BIRTHDAY,
    LURE_PHONE,
    OFF_PROMPT,
    PHONE,
    PHONE_PROMPT,
    PROTOCOL,
    SCHEDULES,
    SLIP,
    bank_sha256,
    canonical_surfaces,
)

REPORT_DIR = ROOT / "reports"
FIT_PATH = REPORT_DIR / "subconscious_reconstruct_fit.json"
HELD_PATH = REPORT_DIR / "subconscious_reconstruct_held.json"
SLEEP_GUARD_PATH = REPORT_DIR / "subconscious_reconstruct_sleep_held.json"
SLEEP_RECORDED_PATH = REPORT_DIR / "subconscious_sleep_held.json"
AROUSAL_GUARD_PATH = REPORT_DIR / "subconscious_reconstruct_arousal_held.json"
AROUSAL_RECORDED_PATH = REPORT_DIR / "subconscious_arousal_held.json"
CLOCK_GUARD_PATH = REPORT_DIR / "subconscious_reconstruct_clock_held.json"
CLOCK_RECORDED_PATH = REPORT_DIR / "subconscious_clock_held.json"
CUE_HELD_PATH = REPORT_DIR / "subconscious_reconstruct_cue_held.json"
CUE_FLOORS_PATH = REPORT_DIR / "subconscious_reconstruct_cue_floors.json"
PREREG_PATH = REPORT_DIR / "subconscious_reconstruct_prereg.md"
GATE_NAMES = ("high", "low", "long", "emotion", "slip", "intrusion", "live")
EXPORT_KEYS = frozenset(
    {
        "interface_version",
        "cue_dim",
        "memory_id",
        "vec",
        "activation",
        "strength_confidence",
        "valence",
        "modality",
        "circadian_phase",
        "triggered",
        "gap",
        "winner_cosine",
        "sleep_pressure",
        "arousal",
        "source",
        "fingerprint",
        "fusion_confidence",
    }
)


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


def _plant(mem, prompt: str, rel: str, answer: str):
    hidden = mem.cue_hidden(prompt)
    row = np.asarray(hidden[-1], dtype=np.float64)
    key = (row - np.asarray(mem.query_mean, dtype=np.float64)).astype(np.float32)
    value = np.ones(mem.hidden, dtype=np.float32)
    if float(np.linalg.norm(key)) < 1e-8 or not mem.store.write(key, value, name="Ada", rel=rel, answer=answer):
        raise RuntimeError("probe write failed")
    return mem.store._slots[-1]


def _assign(slot, schedule) -> None:
    slot.strength = float(schedule[0])
    slot.age = int(schedule[1])
    slot.valence = float(schedule[2])


def _phone_gist(surface: str, residue) -> bool:
    parts = phone_parts(surface)
    if parts is None:
        return False
    area, exchange, _line = parts
    return area == int(residue[1]) and exchange == int(residue[2]) and surface.startswith("+1-")


def _birthday_gist(surface: str, residue) -> bool:
    parts = birthday_parts(surface)
    if parts is None:
        return False
    month, _day, year = parts
    return month == int(residue[1]) and year == int(residue[3])


def _peripheral_changed(kind: str, surface: str, residue) -> bool:
    if kind == "phone":
        parts = phone_parts(surface)
        return parts is not None and int(parts[2]) != int(residue[3])
    parts = birthday_parts(surface)
    return parts is not None and int(parts[1]) != int(residue[2])


def _mode_ok(kind: str, surface: str, residue, schedule, expected: str) -> bool:
    mode = grade(*schedule)
    if surface != expected or surface in (LURE_PHONE, LURE_BIRTHDAY):
        return False
    if mode == "exact":
        original = PHONE if kind == "phone" else BIRTHDAY
        return surface == original
    gist = _phone_gist(surface, residue) if kind == "phone" else _birthday_gist(surface, residue)
    return bool(gist and _peripheral_changed(kind, surface, residue))


def _live(mem, prompt: str, rel: str, answer: str, schedule) -> str:
    mem.store.clear()
    slot = _plant(mem, prompt, rel, answer)
    _assign(slot, schedule)
    return mem.generate(prompt)


def score_item(item_id: str) -> dict:
    del item_id
    phone_residue = residue_from_answer(PHONE)
    birthday_residue = residue_from_answer(BIRTHDAY)
    surfaces = {}
    for label, schedule in SCHEDULES:
        surfaces[f"phone_{label}"] = recall_surface(phone_residue, *schedule)
        surfaces[f"birthday_{label}"] = recall_surface(birthday_residue, *schedule)
    if surfaces != {key: canonical_surfaces()[key] for key in surfaces}:
        raise RuntimeError("reconstructed probe does not match the locked surfaces")
    modes = {
        "high": _mode_ok("phone", surfaces["phone_high"], phone_residue, HIGH, surfaces["phone_high"])
        and _mode_ok("birthday", surfaces["birthday_high"], birthday_residue, HIGH, surfaces["birthday_high"])
        and grade(*HIGH) == "exact",
        "low": _mode_ok("phone", surfaces["phone_low"], phone_residue, LOW, surfaces["phone_low"])
        and _mode_ok("birthday", surfaces["birthday_low"], birthday_residue, LOW, surfaces["birthday_low"])
        and grade(*LOW) == "loss"
        and surfaces["phone_low"] != PHONE
        and surfaces["birthday_low"] != BIRTHDAY,
        "long": _mode_ok("phone", surfaces["phone_long"], phone_residue, LONG, surfaces["phone_long"])
        and _mode_ok("birthday", surfaces["birthday_long"], birthday_residue, LONG, surfaces["birthday_long"])
        and grade(*LONG) == "loss"
        and surfaces["phone_long"] != PHONE
        and surfaces["birthday_long"] != BIRTHDAY,
        "emotion": _mode_ok("phone", surfaces["phone_emotion"], phone_residue, EMOTION, surfaces["phone_emotion"])
        and _mode_ok("birthday", surfaces["birthday_emotion"], birthday_residue, EMOTION, surfaces["birthday_emotion"])
        and grade(*EMOTION) == "slip"
        and surfaces["phone_emotion"] != PHONE
        and surfaces["birthday_emotion"] != BIRTHDAY
        and surfaces["phone_emotion"] != surfaces["phone_low"]
        and surfaces["birthday_emotion"] != surfaces["birthday_low"],
        "slip": _mode_ok("phone", surfaces["phone_slip"], phone_residue, SLIP, surfaces["phone_slip"])
        and _mode_ok("birthday", surfaces["birthday_slip"], birthday_residue, SLIP, surfaces["birthday_slip"])
        and grade(*SLIP) == "slip"
        and surfaces["phone_slip"] == surfaces["phone_emotion"]
        and surfaces["birthday_slip"] == surfaces["birthday_emotion"],
    }
    mem = _memory()
    try:
        live_phone_high = _live(mem, PHONE_PROMPT, "phone", PHONE, HIGH)
        live_phone_low = _live(mem, PHONE_PROMPT, "phone", PHONE, LOW)
        live_phone_emotion = _live(mem, PHONE_PROMPT, "phone", PHONE, EMOTION)
        slot = mem.store._slots[-1]
        kept_answer = slot.answer == PHONE
        off = mem.scripted_continuation(OFF_PROMPT)
        fields = mem.cue_fields(PHONE_PROMPT)
        export_ok = int(fields["interface_version"]) == int(CUE_INTERFACE_VERSION) == 7 and set(fields) == EXPORT_KEYS
        live_birthday_high = _live(mem, BIRTHDAY_PROMPT, "birthday", BIRTHDAY, HIGH)
        live_birthday_long = _live(mem, BIRTHDAY_PROMPT, "birthday", BIRTHDAY, LONG)
        kept_date = mem.store._slots[-1].answer == BIRTHDAY
    finally:
        _close(mem)
    live_ok = (
        live_phone_high == surfaces["phone_high"] == PHONE
        and live_phone_low == surfaces["phone_low"]
        and live_phone_low != PHONE
        and live_phone_emotion == surfaces["phone_emotion"]
        and live_birthday_high == surfaces["birthday_high"] == BIRTHDAY
        and live_birthday_long == surfaces["birthday_long"]
        and live_birthday_long != BIRTHDAY
        and kept_answer
        and kept_date
        and export_ok
    )
    intrusion_event = off is not None or any(
        surface in (LURE_PHONE, LURE_BIRTHDAY) for surface in surfaces.values()
    )
    gates = {
        "high": bool(modes["high"]),
        "low": bool(modes["low"]),
        "long": bool(modes["long"]),
        "emotion": bool(modes["emotion"]),
        "slip": bool(modes["slip"]),
        "intrusion": bool(off is None and not intrusion_event),
        "live": bool(live_ok),
    }
    canonical = {
        "phone_high": surfaces["phone_high"],
        "phone_low": surfaces["phone_low"],
        "phone_long": surfaces["phone_long"],
        "phone_emotion": surfaces["phone_emotion"],
        "phone_slip": surfaces["phone_slip"],
        "birthday_high": surfaces["birthday_high"],
        "birthday_low": surfaces["birthday_low"],
        "birthday_long": surfaces["birthday_long"],
        "birthday_emotion": surfaces["birthday_emotion"],
        "birthday_slip": surfaces["birthday_slip"],
        "off_relation": None if off is None else str(off),
        "interface_version": int(CUE_INTERFACE_VERSION),
    }
    return {
        "gates": gates,
        "pass": all(gates.values()),
        "canonical": _roundtrip(canonical),
        "intrusion_event": bool(intrusion_event or off is not None),
        "live": {
            "phone_high": live_phone_high,
            "phone_low": live_phone_low,
            "phone_emotion": live_phone_emotion,
            "birthday_high": live_birthday_high,
            "birthday_long": live_birthday_long,
        },
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


def _intrusion_events(rows: list) -> dict:
    k = int(sum(bool(row["intrusion_event"]) for row in rows))
    n = len(rows)
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else 0.0, "wilson": [low, high], "pass": k == 0 and n > 0}


def _score_ids(ids: tuple) -> list:
    rows = []
    for index, item_id in enumerate(ids):
        print(f"item {index + 1}/{len(ids)} {item_id}", flush=True)
        row = score_item(item_id)
        row["id"] = item_id
        rows.append(row)
    return rows


def _payload_gates(summary: dict) -> dict:
    return {name: [summary[name]["k"], summary[name]["n"]] for name in summary}


def fit() -> dict:
    if FIT_PATH.is_file():
        raise RuntimeError("reconstruct fit report already exists")
    original = _patch_centers()
    started = time.perf_counter()
    try:
        rows = _score_ids(FIT_IDS)
    finally:
        LLMMemory._key_centers = original
    summary = _summarize(rows)
    events = _intrusion_events(rows)
    canonical = rows[0]["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    payload = {
        "kind": "reconstruct-fit",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "ids": list(FIT_IDS),
        "gates": summary,
        "intrusion_events": events,
        "canonical": canonical,
        "canonical_same": bool(same),
        "pass": bool(summary["item"]["pass"] and same and events["pass"]),
        "note": "Fit only. The held bank was not scored.",
    }
    _dump(FIT_PATH, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "gates": _payload_gates(summary),
                "intrusion_events": [events["k"], events["n"]],
                "canonical": canonical,
            },
            indent=2,
        )
    )
    return payload


def held() -> dict:
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The held bank was not scored." not in prereg:
        raise RuntimeError("the held bank is not locked in the preregistration")
    if HELD_PATH.is_file():
        raise RuntimeError("the held bank was already scored")
    if not FIT_PATH.is_file():
        raise RuntimeError("reconstruct fit report is missing")
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
    events = _intrusion_events(rows)
    canonical = fit_report["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    failures = []
    for name, row in summary.items():
        if int(row["n"]) < 50 or int(row["k"]) != int(row["n"]):
            failures.append({"gate": name, "k": row["k"], "n": row["n"]})
    if int(events["n"]) < 50 or int(events["k"]) != 0:
        failures.append({"gate": "intrusion_events", "k": events["k"], "n": events["n"]})
    if not same:
        failures.append({"gate": "canonical", "k": 0, "n": len(rows)})
    payload = {
        "kind": "reconstruct-held",
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "ids": list(HELD_IDS),
        "gates": summary,
        "intrusion_events": events,
        "canonical": rows[0]["canonical"] if rows else None,
        "fit_canonical": canonical,
        "canonical_same": bool(same),
        "failures": failures,
        "pass": not failures,
        "note": "One pass. Constants were not changed after the fit lock.",
    }
    _dump(HELD_PATH, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "failures": failures,
                "gates": _payload_gates(summary),
                "intrusion_events": [events["k"], events["n"], events["wilson"]],
            },
            indent=2,
        )
    )
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


def _require(sentence: str, path: Path, label: str) -> None:
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if sentence not in prereg:
        raise RuntimeError(f"the {label} is not locked in the preregistration")
    if path.is_file():
        raise RuntimeError(f"the {label} was already scored")


def sleep_guard() -> dict:
    """One rescore of the sleep held schedule into a new file."""
    _require("The sleep guard was not scored.", SLEEP_GUARD_PATH, "sleep guard")
    if not SLEEP_RECORDED_PATH.is_file():
        raise RuntimeError("the recorded sleep held report is missing")
    recorded = json.loads(SLEEP_RECORDED_PATH.read_text(encoding="utf-8"))
    mod = _load_module("eval_subconscious_sleep_reconstruct_guard", "eval_subconscious_sleep.py")
    rows, elapsed = _guard_rows(mod, "sleep")
    payload = _guard_payload(
        "reconstruct-sleep-guard",
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
    """One rescore of the arousal held schedule into a new file."""
    _require("The arousal guard was not scored.", AROUSAL_GUARD_PATH, "arousal guard")
    if not AROUSAL_RECORDED_PATH.is_file():
        raise RuntimeError("the recorded arousal held report is missing")
    recorded = json.loads(AROUSAL_RECORDED_PATH.read_text(encoding="utf-8"))
    mod = _load_module("eval_subconscious_arousal_reconstruct_guard", "eval_subconscious_arousal.py")
    rows, elapsed = _guard_rows(mod, "arousal")
    payload = _guard_payload(
        "reconstruct-arousal-guard",
        mod,
        rows,
        elapsed,
        recorded,
        "One pass. Arousal weights were not changed.",
    )
    _dump(AROUSAL_GUARD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": payload["failures"], "gates": _payload_gates(payload["gates"])}, indent=2))
    return payload


def clock_guard() -> dict:
    """One rescore of the clock held schedule into a new file."""
    _require("The clock guard was not scored.", CLOCK_GUARD_PATH, "clock guard")
    if not CLOCK_RECORDED_PATH.is_file():
        raise RuntimeError("the recorded clock held report is missing")
    recorded = json.loads(CLOCK_RECORDED_PATH.read_text(encoding="utf-8"))
    mod = _load_module("eval_subconscious_clock_reconstruct_guard", "eval_subconscious_clock.py")
    rows, elapsed = _guard_rows(mod, "clock")
    payload = _guard_payload(
        "reconstruct-clock-guard",
        mod,
        rows,
        elapsed,
        recorded,
        "One pass. Wall-clock formulas were not changed.",
    )
    _dump(CLOCK_GUARD_PATH, payload)
    print(json.dumps({"pass": payload["pass"], "failures": payload["failures"], "gates": _payload_gates(payload["gates"])}, indent=2))
    return payload


def cue_guard() -> dict:
    """One rescore of the word direction bank and the cue-triggered floors."""
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The cue guard was not scored." not in prereg:
        raise RuntimeError("the cue guard is not locked in the preregistration")
    if CUE_HELD_PATH.is_file() or CUE_FLOORS_PATH.is_file():
        raise RuntimeError("the cue guard was already scored")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    mod = _load_module("eval_subconscious_cue_reconstruct_guard", "eval_subconscious_cue.py")
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
        "kind": "reconstruct-cue-guard",
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
        "note": "One pass. Reconstruction is not an input to theta, the token half-life, or lambda.",
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
    elif mode == "clock-guard":
        clock_guard()
    elif mode == "cue-guard":
        cue_guard()
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
