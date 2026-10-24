"""Measurement of subconscious error patterns. Does not change behavior.

Shape gates use the eye model. Explicit phone and birthday spellings are
reported beside them and are not an input to Cohen's d_z. Implicit margins
use SmolLM2-135M: no memory, the fact in the prompt, and cue-triggered
retrieval. Held refuses unless the preregistration still says the held bank
was not scored, and it refuses if its report already exists.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from error_pattern_data import (  # noqa: E402
    BOTH_LEVELS,
    EXPECTED,
    FIT_IDS,
    FIT_ITEMS,
    HELD_IDS,
    HELD_ITEMS,
    LIVE_INTERNAL_4,
    LURE_BIRTHDAY,
    LURE_PHONE,
    PROTOCOL,
    RIF_COMPETITOR,
    SPONTANEOUS_RATE,
    TOKEN_LEVELS,
    WAKE_PRESSURE_4,
    WALL_LEVELS,
    bank_sha256,
)
from tidal.clock_drift import DRIFT_RATE, WALL_TAU, clock_activation  # noqa: E402
from tidal.cue_inject import FAN_S, CueState, fan_factor, step_cue  # noqa: E402
from tidal.llm_memory import (  # noqa: E402
    ByteTokenizer,
    CUE_A_TRIG,
    CUE_FLOOR,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_TAU,
    CUE_THETA,
    LLMMemory,
    RIF_SPAN,
)
from tidal.reconstruct import birthday_parts, phone_parts, recall_surface, residue_from_answer  # noqa: E402
from tidal.sleep import SLEEP_FLOOR, SPONTANEOUS_CEILING, SPONTANEOUS_PERIOD, wake_after  # noqa: E402
from tidal.subconscious_measure_data import cohens_dz, wilson_interval  # noqa: E402
from tidal.subconscious_reconstruct_data import (  # noqa: E402
    BIRTHDAY,
    BIRTHDAY_PROMPT,
    EMOTION,
    HIGH,
    LONG,
    LOW,
    OFF_PROMPT,
    PHONE,
    PHONE_PROMPT,
    SCHEDULES,
    SLIP,
)

REPORT_DIR = ROOT / "reports"
FIT_PATH = REPORT_DIR / "error_pattern_fit.json"
HELD_PATH = REPORT_DIR / "error_pattern_held.json"
PREREG_PATH = REPORT_DIR / "error_pattern_prereg.md"
SHAPE_GATES = ("rif", "decay", "fan", "false_trigger", "spontaneous", "fresh")
EXPLICIT_GATES = ("exact", "gist", "peripheral", "intrusion", "live")
ABS_TOL = 1e-6


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


def _close(mem) -> None:
    mem.close()


def _git_sha() -> str:
    import subprocess

    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _roundtrip(payload):
    return json.loads(json.dumps(payload))


def _near(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=ABS_TOL)


def _last_key(mem, text: str) -> np.ndarray:
    hidden = mem.cue_hidden(text)
    row = np.asarray(hidden[-1], dtype=np.float64)
    return row - np.asarray(mem.query_mean, dtype=np.float64)


def _rate(flags: list) -> dict:
    k = int(sum(bool(flag) for flag in flags))
    n = len(flags)
    low, high = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n, "wilson": [low, high], "pass": k == n}


def _constants_ok() -> bool:
    return (
        float(CUE_THETA) == 0.32
        and float(CUE_HALF_LIFE) == 30.0
        and CUE_TAU == 30.0 / math.log(2.0)
        and float(CUE_LAMBDA) == 0.3
        and float(CUE_A_TRIG) == 1.0
        and float(CUE_FLOOR) == 0.05
        and float(FAN_S) == 1.45
        and float(RIF_SPAN) == 0.05
        and int(CUE_INTERFACE_VERSION) == 5
        and float(DRIFT_RATE) == 1.25
        and int(SPONTANEOUS_PERIOD) == 16
        and float(SPONTANEOUS_CEILING) == 0.10
    )


def _rif_gate() -> dict:
    mem = _memory()
    try:
        text = "Ab"
        cue = _last_key(mem, text)
        axis = np.zeros_like(cue)
        axis[0] = 1.0
        ortho = axis - cue * (float(np.dot(axis, cue)) / float(np.dot(cue, cue)))
        ortho = ortho / np.linalg.norm(ortho) * np.linalg.norm(cue)
        winner = cue.astype(np.float32)
        competitor = (0.6 * cue + 0.8 * ortho).astype(np.float32)
        outsider = (0.9 * cue).astype(np.float32)
        value = np.ones(mem.hidden, dtype=np.float32)
        wrote = (
            mem.store.write(winner, value, name="w", rel="note", answer="A", text="winner")
            and mem.store.write(competitor, value, name="c", rel="note", answer="B", text="comp")
            and mem.store.write(outsider, value, name="o", rel="other", answer="C", text="out")
        )
        info = mem._rank(text)
        strengths = [float(slot.strength) for slot in mem.store._slots]
        passed = (
            wrote
            and _constants_ok()
            and strengths == [1.0, float(RIF_COMPETITOR), 1.0]
            and float(RIF_COMPETITOR) == 0.95
            and info.get("top") == ("w", "note")
            and int(info.get("memory_id")) == 0
            and mem.last_rif == {"n": 1}
            and float(mem.store._slots[0].strength) == 1.0
            and float(mem.store._slots[2].strength) == 1.0
        )
    finally:
        _close(mem)
    return {"pass": bool(passed), "strengths": strengths}


def _decay_gate() -> dict:
    wall = [clock_activation(1.0, 0, elapsed, CUE_TAU, WALL_TAU) for elapsed in (0.0, 1.0, 4.0, 300.0)]
    token = [clock_activation(1.0, dt, 0.0, CUE_TAU, WALL_TAU) for dt in (0, 1, 4)]
    both = [
        clock_activation(1.0, 1, 30.0, CUE_TAU, WALL_TAU),
        clock_activation(1.0, 8, 1.0, CUE_TAU, WALL_TAU),
    ]
    drifted = clock_activation(1.0, 0, 4.0 * DRIFT_RATE, CUE_TAU, WALL_TAU)
    mem = _memory()
    try:
        mem.cue_state = CueState(
            a=1.0,
            dt=0,
            memory_id=0,
            confidence=1.0,
            strength=1.0,
            fan_factor=1.0,
            vec=np.ones(mem.hidden, dtype=np.float32),
            trigger_wall=0.0,
        )
        mem.advance_wall(4.0)
        live = float(mem.cue_state.a)
        live_wall = float(mem.wall_seconds)
        live_internal = float(mem.internal_seconds)
        live_dt = int(mem.cue_state.dt)
    finally:
        _close(mem)
    cleared = _memory()
    try:
        cleared.cue_state = CueState(
            a=1.0,
            dt=0,
            memory_id=0,
            confidence=1.0,
            strength=1.0,
            fan_factor=1.0,
            vec=np.ones(cleared.hidden, dtype=np.float32),
            trigger_wall=0.0,
        )
        cleared.advance_wall(300.0)
        cleared_id = int(cleared.cue_state.memory_id)
    finally:
        _close(cleared)
    values = np.ones((1, 3), dtype=np.float32)
    stepped, _row = step_cue(
        CueState(a=1.0, dt=0, memory_id=0, confidence=1.0, strength=1.0, fan_factor=1.0, vec=values[0].copy(), trigger_wall=0.0),
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
        _constants_ok()
        and wall == list(WALL_LEVELS)
        and token == list(TOKEN_LEVELS)
        and both == list(BOTH_LEVELS)
        and wall[0] > wall[1] > wall[2] > float(CUE_FLOOR) > wall[3]
        and token[0] > token[1] > token[2]
        and both[0] == min(clock_activation(1.0, 1, 0.0, CUE_TAU, WALL_TAU), clock_activation(1.0, 0, 30.0, CUE_TAU, WALL_TAU))
        and both[1] == min(clock_activation(1.0, 8, 0.0, CUE_TAU, WALL_TAU), clock_activation(1.0, 0, 1.0, CUE_TAU, WALL_TAU))
        and live == wall[2]
        and live != drifted
        and live_dt == 0
        and live_wall == 4.0
        and live_internal == float(LIVE_INTERNAL_4) == 4.0 * DRIFT_RATE
        and cleared_id == -1
        and float(stepped.a) == token[1]
        and int(stepped.dt) == 1
    )
    return {
        "pass": bool(passed),
        "wall": wall,
        "token": token,
        "both": both,
        "live_4": live,
        "live_internal": live_internal,
        "cleared_id": cleared_id,
    }


def _fan_gate() -> dict:
    levels = []
    triggered = []
    for count in (1, 2, 4):
        mem = _memory()
        try:
            cue = _last_key(mem, "Ab").astype(np.float32)
            value = np.ones(mem.hidden, dtype=np.float32)
            for index in range(count):
                assert mem.store.write(cue, value, name=f"n{index}", rel=f"r{index}", answer="A", text=f"t{index}")
            fields = mem.cue_fields("Ab")
            levels.append(float(fields["activation"][0, -1]))
            triggered.append(bool(fields["triggered"][0, -1]))
        finally:
            _close(mem)
    expected = [float(CUE_A_TRIG) * fan_factor(count) for count in (1, 2, 4)]
    passed = (
        _constants_ok()
        and all(flag for flag in triggered)
        and all(_near(got, want) for got, want in zip(levels, expected))
        and levels[0] > levels[1] > levels[2]
        and expected[0] == 1.0
    )
    return {"pass": bool(passed), "activation": levels, "expected": expected}


def _false_gate() -> dict:
    mem = _memory()
    try:
        cue = _last_key(mem, "Ab")
        false = np.zeros_like(cue)
        false[3] = 1.0
        value = np.ones(mem.hidden, dtype=np.float32)
        assert mem.store.write(false.astype(np.float32), value, name="f", rel="note", answer="A", text="false")
        fields = mem.cue_fields("Ab")
        flags = [bool(item) for item in fields["triggered"][0].tolist()]
        ids = [int(item) for item in fields["memory_id"][0].tolist()]
        events = int(sum(flags))
        passed = events == 0 and ids == [-1, -1] and _constants_ok()
    finally:
        _close(mem)
    return {"pass": bool(passed), "events": events}


def _spontaneous_gate() -> dict:
    mem = _memory()
    try:
        mem.wake_for(4.0)
        pressure = float(mem.sleep_pressure)
        cue = _last_key(mem, "Ab").astype(np.float32)
        assert mem.store.write(cue, cue.copy(), name="wander", rel="note", answer="STAY", text="raw-spontaneous")
        ticks = mem.offline_ticks(int(SPONTANEOUS_PERIOD))
        sources = [tick["source"][0][0] for tick in ticks]
        fires = int(sources.count("spontaneous"))
        rate = fires / int(SPONTANEOUS_PERIOD)
        passed = (
            _constants_ok()
            and pressure == wake_after(float(SLEEP_FLOOR), 4.0) == float(WAKE_PRESSURE_4)
            and fires == 1
            and sources[0] == "spontaneous"
            and sources.count("none") == int(SPONTANEOUS_PERIOD) - 1
            and rate == float(SPONTANEOUS_RATE) == 0.0625
            and rate < float(SPONTANEOUS_CEILING)
            and float(mem.store._slots[0].strength) == 1.0
            and "raw-spontaneous" not in json.dumps(mem.sleep_audit())
        )
    finally:
        _close(mem)
    return {"pass": bool(passed), "fires": fires, "rate": rate, "pressure": pressure}


def _fresh_gate() -> dict:
    early = _memory()
    late = _memory()
    try:
        text = "Ab"
        for mem in (early, late):
            cue = _last_key(mem, text).astype(np.float32)
            value = np.zeros(mem.cue_dim, dtype=np.float32)
            value[3] = 4.0
            assert mem.store.write(cue, value, name="ada", rel="note", answer="A", text="fresh-cue")
        before = early.cue_fields(text)
        late.advance_wall(8.0)
        after = late.cue_fields(text)
        forbidden = not any(name in before or name in after for name in ("clock_error", "wall_seconds", "internal_seconds"))
        passed = (
            _constants_ok()
            and int(before["interface_version"]) == int(after["interface_version"]) == 5
            and torch.equal(before["activation"], after["activation"])
            and forbidden
            and float(late.wall_seconds) == 8.0
            and float(late.internal_seconds) == 8.0 * DRIFT_RATE
            and "spontaneous" not in list(before["source"][0])
        )
    finally:
        _close(early)
        _close(late)
    return {"pass": bool(passed), "activation_0": float(before["activation"][0, 0])}


def _plant(mem, prompt: str, rel: str, answer: str):
    key = _last_key(mem, prompt).astype(np.float32)
    value = np.ones(mem.hidden, dtype=np.float32)
    if float(np.linalg.norm(key)) < 1e-8 or not mem.store.write(key, value, name="Ada", rel=rel, answer=answer):
        raise RuntimeError("probe write failed")
    return mem.store._slots[-1]


def _assign(slot, schedule) -> None:
    slot.strength = float(schedule[0])
    slot.age = int(schedule[1])
    slot.valence = float(schedule[2])


def _live(mem, prompt: str, rel: str, answer: str, schedule) -> str:
    mem.store.clear()
    slot = _plant(mem, prompt, rel, answer)
    _assign(slot, schedule)
    return mem.generate(prompt)


def _gist(kind: str, surface: str, residue) -> bool:
    if kind == "phone":
        parts = phone_parts(surface)
        return parts is not None and surface.startswith("+1-") and int(parts[0]) == int(residue[1]) and int(parts[1]) == int(residue[2])
    parts = birthday_parts(surface)
    return parts is not None and int(parts[0]) == int(residue[1]) and int(parts[2]) == int(residue[3])


def _peripheral_same(kind: str, surface: str, residue) -> bool:
    if kind == "phone":
        parts = phone_parts(surface)
        return parts is not None and int(parts[2]) == int(residue[3])
    parts = birthday_parts(surface)
    return parts is not None and int(parts[1]) == int(residue[2])


def _explicit_gate() -> dict:
    phone_residue = residue_from_answer(PHONE)
    birthday_residue = residue_from_answer(BIRTHDAY)
    surfaces = {}
    for label, schedule in SCHEDULES:
        surfaces[f"phone_{label}"] = recall_surface(phone_residue, *schedule)
        surfaces[f"birthday_{label}"] = recall_surface(birthday_residue, *schedule)
    mem = _memory()
    try:
        live = {}
        for label, schedule in SCHEDULES:
            live[f"phone_{label}"] = _live(mem, PHONE_PROMPT, "phone", PHONE, schedule)
            live[f"birthday_{label}"] = _live(mem, BIRTHDAY_PROMPT, "birthday", BIRTHDAY, schedule)
        mem.store.clear()
        slot = _plant(mem, PHONE_PROMPT, "phone", PHONE)
        _assign(slot, EMOTION)
        off = mem.scripted_continuation(OFF_PROMPT)
        kept = slot.answer == PHONE
    finally:
        _close(mem)
    exact = (
        surfaces["phone_high"] == live["phone_high"] == EXPECTED["phone_high"] == PHONE
        and surfaces["birthday_high"] == live["birthday_high"] == EXPECTED["birthday_high"] == BIRTHDAY
    )
    gist = all(
        _gist("phone", surfaces[f"phone_{label}"], phone_residue) and _gist("birthday", surfaces[f"birthday_{label}"], birthday_residue)
        for label, _schedule in SCHEDULES
    )
    peripheral = _peripheral_same("phone", surfaces["phone_high"], phone_residue) and _peripheral_same(
        "birthday", surfaces["birthday_high"], birthday_residue
    )
    for label, _schedule in SCHEDULES:
        if label == "high":
            continue
        peripheral = peripheral and not _peripheral_same("phone", surfaces[f"phone_{label}"], phone_residue)
        peripheral = peripheral and not _peripheral_same("birthday", surfaces[f"birthday_{label}"], birthday_residue)
    intrusion = off is None and all(surface not in (LURE_PHONE, LURE_BIRTHDAY) for surface in list(surfaces.values()) + list(live.values()))
    matched = all(surfaces[key] == live[key] == EXPECTED[key] for key in EXPECTED)
    gates = {
        "exact": bool(exact),
        "gist": bool(gist),
        "peripheral": bool(peripheral),
        "intrusion": bool(intrusion),
        "live": bool(matched and kept and intrusion),
    }
    return {
        "pass": all(gates.values()),
        "gates": gates,
        "surfaces": surfaces,
        "live": live,
        "off": None if off is None else str(off),
        "intrusion_event": not intrusion,
    }


def score_eye(item_id: str) -> dict:
    """Shape gates and the explicit spell report. The id does not change the probe."""
    del item_id
    original = _patch_centers()
    try:
        rif = _rif_gate()
        decay = _decay_gate()
        fan = _fan_gate()
        false = _false_gate()
        spontaneous = _spontaneous_gate()
        fresh = _fresh_gate()
        explicit = _explicit_gate()
    finally:
        LLMMemory._key_centers = original
    shape = {
        "rif": bool(rif["pass"]),
        "decay": bool(decay["pass"]),
        "fan": bool(fan["pass"]),
        "false_trigger": bool(false["pass"]),
        "spontaneous": bool(spontaneous["pass"]),
        "fresh": bool(fresh["pass"]),
    }
    canonical = {
        "rif_strengths": rif["strengths"],
        "wall": decay["wall"],
        "token": decay["token"],
        "both": decay["both"],
        "live_4": decay["live_4"],
        "live_internal": decay["live_internal"],
        "fan": fan["activation"],
        "false_events": false["events"],
        "spontaneous_rate": spontaneous["rate"],
        "spontaneous_fires": spontaneous["fires"],
        "wake_pressure": spontaneous["pressure"],
        "fresh_activation_0": fresh["activation_0"],
        "surfaces": explicit["surfaces"],
        "off": explicit["off"],
    }
    return {
        "shape": shape,
        "shape_pass": all(shape.values()),
        "explicit": explicit["gates"],
        "explicit_pass": bool(explicit["pass"]),
        "intrusion_event": bool(explicit["intrusion_event"]),
        "canonical": _roundtrip(canonical),
    }


def _load_smol():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    return LLMMemory(dtype=torch.float32)


def _margin(mem, prompt: str, good: str, bad: str, name=None, rel=None) -> float:
    from tidal.subconscious_eval import score_pair

    saved = [float(slot.strength) for slot in mem.store._slots]
    try:
        scored = score_pair(mem, prompt, good, bad, name=name, rel=rel)
    finally:
        for slot, value in zip(mem.store._slots, saved):
            slot.strength = float(value)
    return float(scored["margin"])


def score_implicit(mem, item: dict) -> dict:
    """Three margins. Explicit spell strings are not read."""
    mem.store.clear()
    nomem = _margin(mem, item["prompt"], item["good"], item["bad"])
    context = _margin(mem, item["text"] + "\n" + item["prompt"], item["good"], item["bad"])
    mem.store.clear()
    wrote = bool(mem.remember(item["text"], item["answer"], name=item["name"], rel=item["rel"]))
    cue = _margin(mem, item["prompt"], item["good"], item["bad"], item["name"], item["rel"])
    cue_delta = float(cue - nomem)
    context_delta = float(context - nomem)
    return {
        "id": item["id"],
        "wrote": wrote,
        "nomem": nomem,
        "in_context": context,
        "cue": cue,
        "cue_delta": cue_delta,
        "context_delta": context_delta,
        "finite": bool(wrote and math.isfinite(cue_delta) and math.isfinite(context_delta)),
    }


def _implicit_summary(rows: list, expect: int) -> dict:
    cue = [float(row["implicit"]["cue_delta"]) for row in rows]
    context = [float(row["implicit"]["context_delta"]) for row in rows]
    finite = len(rows) == int(expect) and all(bool(row["implicit"]["finite"]) for row in rows)

    def positive(values):
        k = int(sum(value > 0.0 for value in values))
        n = len(values)
        low, high = wilson_interval(k, n)
        return {"k": k, "n": n, "rate": k / n, "wilson": [low, high]}

    def mean(values):
        return float(sum(values) / len(values))

    return {
        "n": len(rows),
        "finite": bool(finite),
        "pass": bool(finite),
        "cue_dz": cohens_dz(cue),
        "context_dz": cohens_dz(context),
        "cue_positive": positive(cue),
        "context_positive": positive(context),
        "cue_mean_delta": mean(cue),
        "context_mean_delta": mean(context),
        "nomem_mean": mean([float(row["implicit"]["nomem"]) for row in rows]),
        "cue_mean": mean([float(row["implicit"]["cue"]) for row in rows]),
        "context_mean": mean([float(row["implicit"]["in_context"]) for row in rows]),
    }


def _shape_summary(rows: list) -> dict:
    summary = {}
    for name in SHAPE_GATES:
        summary[name] = _rate([bool(row["shape"][name]) for row in rows])
    summary["item"] = _rate([bool(row["shape_pass"]) for row in rows])
    return summary


def _explicit_summary(rows: list) -> dict:
    summary = {}
    for name in EXPLICIT_GATES:
        summary[name] = _rate([bool(row["explicit"][name]) for row in rows])
    events = [bool(row["intrusion_event"]) for row in rows]
    k = int(sum(events))
    n = len(events)
    low, high = wilson_interval(k, n)
    summary["intrusion_events"] = {"k": k, "n": n, "rate": k / n, "wilson": [low, high], "pass": k == 0 and n > 0}
    summary["item"] = _rate([bool(row["explicit_pass"]) for row in rows])
    return summary


def _score(ids: tuple, items: tuple) -> list:
    rows = []
    for index, (item_id, item) in enumerate(zip(ids, items)):
        print(f"eye {index + 1}/{len(ids)} {item_id}", flush=True)
        row = score_eye(item_id)
        row["id"] = item_id
        row["item_id"] = item["id"]
        rows.append(row)
    mem = _load_smol()
    try:
        for index, (row, item) in enumerate(zip(rows, items)):
            print(f"implicit {index + 1}/{len(items)} {item['id']}", flush=True)
            row["implicit"] = score_implicit(mem, item)
    finally:
        _close(mem)
    return rows


def _payload(kind: str, ids: tuple, rows: list, started: float, expect: int) -> dict:
    shape = _shape_summary(rows)
    explicit = _explicit_summary(rows)
    implicit = _implicit_summary(rows, expect)
    canonical = rows[0]["canonical"]
    same = all(row["canonical"] == canonical for row in rows)
    false_events = int(sum(int(row["canonical"]["false_events"]) for row in rows))
    return {
        "kind": kind,
        "git": _git_sha(),
        "elapsed_s": time.perf_counter() - started,
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "ids": list(ids),
        "shape": shape,
        "explicit": explicit,
        "implicit": implicit,
        "false_trigger_events": false_events,
        "canonical": canonical,
        "canonical_same": bool(same),
        "shape_pass": bool(shape["item"]["pass"] and same and false_events == 0),
        "explicit_match": bool(explicit["item"]["pass"] and explicit["intrusion_events"]["pass"]),
        "implicit_pass": bool(implicit["pass"]),
        "pass": bool(shape["item"]["pass"] and same and false_events == 0 and implicit["pass"]),
        "rows": [
            {
                "id": row["id"],
                "shape_pass": row["shape_pass"],
                "explicit_pass": row["explicit_pass"],
                "cue_delta": row["implicit"]["cue_delta"],
                "context_delta": row["implicit"]["context_delta"],
                "finite": row["implicit"]["finite"],
            }
            for row in rows
        ],
    }


def fit() -> dict:
    if FIT_PATH.is_file():
        raise RuntimeError("error-pattern fit report already exists")
    if len(FIT_IDS) != 8 or set(FIT_IDS) & set(HELD_IDS):
        raise RuntimeError("fit and held ids are not disjoint")
    started = time.perf_counter()
    rows = _score(FIT_IDS, FIT_ITEMS)
    payload = _payload("error-pattern-fit", FIT_IDS, rows, started, 8)
    payload["note"] = "Fit only. The held bank was not scored."
    _dump(FIT_PATH, payload)
    print(json.dumps({key: payload[key] for key in ("pass", "shape_pass", "explicit_match", "implicit_pass", "implicit", "shape", "explicit")}, indent=2))
    return payload


def held() -> dict:
    prereg = PREREG_PATH.read_text(encoding="utf-8")
    if "The held bank was not scored." not in prereg:
        raise RuntimeError("the held bank is not locked in the preregistration")
    if HELD_PATH.is_file():
        raise RuntimeError("the held bank was already scored")
    if not FIT_PATH.is_file():
        raise RuntimeError("error-pattern fit report is missing")
    fit_report = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    if not fit_report.get("pass"):
        raise RuntimeError("fit did not pass; held is not scored")
    if len(HELD_IDS) < 50:
        raise RuntimeError("held n is below 50")
    started = time.perf_counter()
    rows = _score(HELD_IDS, HELD_ITEMS)
    payload = _payload("error-pattern-held", HELD_IDS, rows, started, 50)
    failures = []
    for name, row in payload["shape"].items():
        if int(row["n"]) < 50 or not row["pass"]:
            failures.append({"gate": "shape:" + name, "k": row["k"], "n": row["n"]})
    if not payload["canonical_same"]:
        failures.append({"gate": "canonical", "k": 0, "n": len(rows)})
    if int(payload["false_trigger_events"]) != 0:
        failures.append({"gate": "false_trigger_events", "k": payload["false_trigger_events"], "n": len(rows)})
    if not payload["implicit_pass"] or int(payload["implicit"]["n"]) < 50:
        failures.append({"gate": "implicit_finite", "k": 0, "n": payload["implicit"]["n"]})
    payload["failures"] = failures
    payload["pass"] = bool(payload["pass"] and not failures)
    payload["note"] = "One held pass. Explicit recall is report-only. Mechanisms were not changed."
    _dump(HELD_PATH, payload)
    print(
        json.dumps(
            {
                "pass": payload["pass"],
                "shape_pass": payload["shape_pass"],
                "explicit_match": payload["explicit_match"],
                "implicit_pass": payload["implicit_pass"],
                "implicit": payload["implicit"],
                "failures": failures,
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
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv)
