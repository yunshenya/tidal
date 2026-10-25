"""Harness for the long-range recall held-out protocol.

Uses main's public ``remember`` / ``cue_fields`` / ``offline_ticks`` /
``wake_for`` / ``sleep_for``. Ablation off-switches are monkeypatches
here only. ``tidal/`` runtime modules do not gain a flag.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Literal

import numpy as np
from torch import nn

from tidal.arousal import AROUSAL_BASELINE
from tidal.llm_memory import ByteTokenizer, CUE_INTERFACE_VERSION, CUE_THETA, LLMMemory
from tidal.long_range_recall_data import (
    BASELINE_TURN_DECAY,
    CONDITIONS,
    FAMILIES,
    ITEMS_N,
    LONG_OFFLINE_TURN,
    N_VALUES,
    OFFLINE_N10,
    OFFLINE_N30,
    OFFLINE_PERIOD,
    PACKET_KEYS,
    PROTOCOL,
    SEEDS,
    SIMILAR_COSINE,
    SLEEP_DURATION,
    SLEEP_WAKE,
    STUB_DIM,
    TARGET_AROUSAL,
    TARGET_EXPERIENCE,
    WAKE_AFTER_PLANT,
    WAKE_EVERY,
    WAKE_STEP,
    WEAK_COSINE,
    Condition,
    Family,
    _unknown_condition,
    _unknown_family,
    distractor_name,
    distractor_text,
    item_by_id,
    mean_std,
    score_hypotheses,
    similar_name,
    similar_text,
    similar_turns,
)
from tidal.subconscious_measure_data import wilson_interval
from tidal.subconscious_reconstruct_data import (
    bank_sha256 as reconstruct_bank_sha256,
    canonical_surfaces,
)
from tidal.weak_cue import WEAK_CEILING, WEAK_FLOOR

Probe = Literal["strong", "weak"]


class _Eye(nn.Module):
    """Tiny CPU stub. Geometry is supplied by the harness, not this embed."""

    def __init__(self, dim: int = STUB_DIM):
        super().__init__()
        vocab = 260
        self.embed = nn.Embedding(vocab, int(dim))
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
        self.lm_head = nn.Linear(int(dim), vocab, bias=False)
        self.config = type("Cfg", (), {"hidden_size": int(dim), "vocab_size": vocab})()

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, **_kwargs):
        del attention_mask, use_cache
        hidden = self.embed(input_ids)
        states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            states.append(hidden)
        logits = self.lm_head(hidden)
        return type("Out", (), {"logits": logits, "hidden_states": tuple(states) if output_hidden_states else None})()


def _unit(index: int, dim: int = STUB_DIM) -> np.ndarray:
    row = np.zeros(int(dim), dtype=np.float64)
    row[int(index) % int(dim)] = 1.0
    return row


def _hash_vec(text: str, dim: int = STUB_DIM, reserved: int = 32) -> np.ndarray:
    rng = np.random.default_rng(int(hashlib_u32(text)))
    row = rng.normal(size=int(dim)).astype(np.float64)
    row[: int(reserved)] = 0.0
    norm = float(np.linalg.norm(row))
    if norm <= 1e-12:
        row[int(reserved)] = 1.0
        return row
    return row / norm


def _mix(src: np.ndarray, overlap: float, lateral: np.ndarray) -> np.ndarray:
    """Unit mix in a chosen plane. Cosine with ``src`` is ``overlap``."""
    source = np.asarray(src, dtype=np.float64).reshape(-1)
    side = np.asarray(lateral, dtype=np.float64).reshape(-1)
    source_norm = float(np.linalg.norm(source))
    if source_norm <= 1e-12:
        raise ValueError("source must be nonzero")
    unit = source / source_norm
    extra = side - float(np.dot(side, unit)) * unit
    extra_norm = float(np.linalg.norm(extra))
    if extra_norm <= 1e-12:
        raise ValueError("lateral must not be parallel")
    extra = extra / extra_norm
    weight = float(overlap)
    mix = weight * unit + float(np.sqrt(max(0.0, 1.0 - weight * weight))) * extra
    mix_norm = float(np.linalg.norm(mix))
    return mix / mix_norm


def hashlib_u32(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


class Geometry:
    """Locked utterance → unit vector map. Harness only."""

    def __init__(self, item: dict[str, Any]):
        self.item = item
        axis = int(item["id"])
        self.target = _unit(axis)
        self.conflict = _unit(10 + axis)
        self.unrelated = _unit(20 + axis)
        # Different laterals so a 0.25 weak cue does not clear theta against
        # a 0.95 neighbor. Cosine with the target stays the locked value.
        self.weak = _mix(self.target, float(WEAK_COSINE), _unit(30))
        self.similar = _mix(self.target, float(SIMILAR_COSINE), _unit(31))

    def vector(self, text: str) -> np.ndarray:
        item = self.item
        if text in (item["text"], item["strong_cue"]):
            return self.target.copy()
        if text == item["weak_cue"]:
            return self.weak.copy()
        if text == item["conflict_text"]:
            return self.conflict.copy()
        if text == item["unrelated"]:
            return self.unrelated.copy()
        if text.startswith("Near "):
            return self.similar.copy()
        if text.startswith("Distractor "):
            return _hash_vec(text)
        raise KeyError(text)


def _make_memory() -> LLMMemory:
    centers = LLMMemory._key_centers

    def zeros(self):
        row = np.zeros(self.hidden, dtype=np.float64)
        return row, row.copy()

    LLMMemory._key_centers = zeros
    try:
        return LLMMemory(_Eye(), ByteTokenizer())
    finally:
        LLMMemory._key_centers = centers


def _attach_geometry(mem: LLMMemory, geom: Geometry) -> None:
    def encode_key(text: str, answer=None):
        del answer
        return geom.vector(text).astype(np.float32, copy=False)

    def cue_hidden(text: str):
        return geom.vector(text).reshape(1, -1)

    mem.encode_key = encode_key
    mem.cue_hidden = cue_hidden
    mem._value_and_surprise = lambda _text, _answer: (
        np.zeros(mem.hidden, dtype=np.float32),
        1.0,
    )


def _noop(*_args, **_kwargs):
    return None


def _zero_retroactive(*_args, **_kwargs) -> int:
    return 0


def _no_weak_winner(*_args, **_kwargs):
    return None


def _keep_one(_mode: str) -> float:
    return 1.0


def _no_reconsolidate(self) -> list:
    del self
    return []


def _gain_one(self, slot, arousal) -> None:
    del self, arousal
    slot.consolidation_gain = 1.0


@contextmanager
def _condition_patches(condition: Condition) -> Iterator[None]:
    import tidal.llm_memory as memory_mod
    import tidal.retroactive as retro_mod

    saved = {
        "weak_winner": memory_mod.weak_winner,
        "write_keep": memory_mod.write_keep,
        "apply_retroactive": retro_mod.apply_retroactive,
    }
    try:
        if condition == "full":
            pass
        elif condition == "off_77":
            pass
        elif condition == "off_79":
            pass
        elif condition == "off_81":
            retro_mod.apply_retroactive = _zero_retroactive
        elif condition == "off_83":
            memory_mod.weak_winner = _no_weak_winner
        elif condition == "baseline":
            retro_mod.apply_retroactive = _zero_retroactive
            memory_mod.weak_winner = _no_weak_winner
            memory_mod.write_keep = _keep_one
        else:
            _unknown_condition(condition)
        yield
    finally:
        memory_mod.weak_winner = saved["weak_winner"]
        memory_mod.write_keep = saved["write_keep"]
        retro_mod.apply_retroactive = saved["apply_retroactive"]


def _apply_instance_ablations(mem: LLMMemory, condition: Condition) -> None:
    if condition == "full":
        return
    if condition == "off_77":
        mem._apply_consolidation_gain = _gain_one.__get__(mem, LLMMemory)
        return
    if condition == "off_79":
        mem._sws_pass = _noop.__get__(mem, LLMMemory)
        mem._rem_pass = _noop.__get__(mem, LLMMemory)
        return
    if condition == "off_81":
        return
    if condition == "off_83":
        return
    if condition == "baseline":
        mem._apply_consolidation_gain = _gain_one.__get__(mem, LLMMemory)
        mem._sws_pass = _noop.__get__(mem, LLMMemory)
        mem._rem_pass = _noop.__get__(mem, LLMMemory)
        mem._reconsolidate_store = _no_reconsolidate.__get__(mem, LLMMemory)
        return
    _unknown_condition(condition)


def _decay_store(mem: LLMMemory) -> None:
    factor = float(BASELINE_TURN_DECAY)
    for slot in mem.store._slots:
        slot.strength = float(slot.strength) * factor


def _slot_index(mem: LLMMemory, name: str, rel: str) -> int:
    for index, slot in enumerate(mem.store._slots):
        if slot.name == name and slot.rel == rel:
            return int(index)
    return -1


def _last_token(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "memory_id": int(fields["memory_id"][0, -1].item()),
        "triggered": bool(fields["triggered"][0, -1].item()),
        "source": str(fields["source"][0][-1]),
        "winner_cosine": float(fields["winner_cosine"][0, -1].item()),
        "keys": tuple(sorted(fields)),
    }


def _remember(
    mem: LLMMemory,
    *,
    text: str,
    answer: str,
    name: str,
    rel: str,
    experience: float | None = None,
    trace: list,
) -> bool:
    payload: dict[str, Any] = {
        "text": text,
        "answer": answer,
        "name": name,
        "rel": rel,
        "kind": "fact",
    }
    if experience is not None:
        payload["experience"] = float(experience)
    wrote = bool(mem.remember(payload))
    trace.append(("remember", name, rel, wrote))
    return wrote


def _run_schedule(
    mem: LLMMemory,
    *,
    item: dict[str, Any],
    seed: int,
    n: int,
    family: Family,
    condition: Condition,
    trace: list,
) -> None:
    depth = int(n)
    neighbors = set(similar_turns(depth)) if family != "conflict" else set()
    for turn in range(1, depth + 1):
        if family != "conflict" and turn in neighbors:
            text = similar_text(seed, int(item["id"]), turn)
            name = similar_name(seed, int(item["id"]), turn)
            rel = "nearnote"
            answer = "same day"
        else:
            text = distractor_text(seed, int(item["id"]), turn)
            name = distractor_name(seed, int(item["id"]), turn)
            rel = "chatnote"
            answer = "mild"
        if not _remember(mem, text=text, answer=answer, name=name, rel=rel, trace=trace):
            raise RuntimeError("distractor write refused")
        if condition == "baseline":
            _decay_store(mem)
        if family == "conflict" and turn == depth // 2:
            if not _remember(
                mem,
                text=str(item["conflict_text"]),
                answer=str(item["conflict_answer"]),
                name=str(item["name"]),
                rel=str(item["rel"]),
                experience=float(TARGET_EXPERIENCE),
                trace=trace,
            ):
                raise RuntimeError("conflict restatement refused")
        if turn % int(WAKE_EVERY) == 0:
            mem.wake_for(float(WAKE_STEP))
            trace.append(("wake_for", float(WAKE_STEP)))
        if depth >= 30 and turn == depth // 2:
            mem.wake_for(float(SLEEP_WAKE))
            trace.append(("wake_for", float(SLEEP_WAKE)))
            mem.sleep_for(float(SLEEP_DURATION))
            trace.append(("sleep_for", float(SLEEP_DURATION)))
        if depth >= 100 and turn == int(LONG_OFFLINE_TURN):
            mem.offline_ticks(int(OFFLINE_PERIOD))
            trace.append(("offline_ticks", int(OFFLINE_PERIOD)))
    if depth == 10:
        mem.offline_ticks(int(OFFLINE_N10))
        trace.append(("offline_ticks", int(OFFLINE_N10)))
        return
    if depth == 30:
        mem.offline_ticks(int(OFFLINE_N30))
        trace.append(("offline_ticks", int(OFFLINE_N30)))
        return
    if depth == 100:
        return
    raise RuntimeError(f"unhandled N: {depth}")


def play_dialogue(
    *,
    seed: int,
    item_id: int,
    n: int,
    family: Family,
    condition: Condition,
) -> tuple[LLMMemory, Geometry, list]:
    if family not in FAMILIES:
        _unknown_family(family)
    if condition not in CONDITIONS:
        _unknown_condition(condition)
    item = item_by_id(item_id)
    geom = Geometry(item)
    mem = _make_memory()
    _attach_geometry(mem, geom)
    _apply_instance_ablations(mem, condition)
    trace: list = []
    mem._arousal = float(TARGET_AROUSAL)
    wrote = _remember(
        mem,
        text=str(item["text"]),
        answer=str(item["answer"]),
        name=str(item["name"]),
        rel=str(item["rel"]),
        experience=float(TARGET_EXPERIENCE),
        trace=trace,
    )
    if not wrote:
        mem.close()
        raise RuntimeError("target write refused")
    mem._arousal = float(AROUSAL_BASELINE)
    mem.wake_for(float(WAKE_AFTER_PLANT))
    trace.append(("wake_for", float(WAKE_AFTER_PLANT)))
    _run_schedule(
        mem,
        item=item,
        seed=int(seed),
        n=int(n),
        family=family,
        condition=condition,
        trace=trace,
    )
    return mem, geom, trace


def score_strong(mem: LLMMemory, item: dict[str, Any]) -> dict[str, Any]:
    fields = mem.cue_fields(str(item["strong_cue"]))
    last = _last_token(fields)
    target = _slot_index(mem, str(item["name"]), str(item["rel"]))
    cosine = last["winner_cosine"]
    hit = (
        last["memory_id"] == target
        and target >= 0
        and last["triggered"] is True
        and last["source"] == "cue"
        and cosine >= float(CUE_THETA)
    )
    return {
        "hit": bool(hit),
        "target": int(target),
        **last,
        "answer": None if target < 0 else str(mem.store._slots[target].answer),
    }


def score_conflict(mem: LLMMemory, item: dict[str, Any]) -> dict[str, Any]:
    fields = mem.cue_fields(str(item["conflict_text"]))
    last = _last_token(fields)
    target = _slot_index(mem, str(item["name"]), str(item["rel"]))
    answer = None if target < 0 else str(mem.store._slots[target].answer)
    newest = (
        last["memory_id"] == target
        and target >= 0
        and last["triggered"] is True
        and last["source"] == "cue"
        and last["winner_cosine"] >= float(CUE_THETA)
        and answer == str(item["conflict_answer"])
        and answer != str(item["answer"])
    )
    return {
        "hit": bool(newest),
        "target": int(target),
        "answer": answer,
        **last,
    }


def score_unrelated(mem: LLMMemory, item: dict[str, Any]) -> dict[str, Any]:
    fields = mem.cue_fields(str(item["unrelated"]))
    last = _last_token(fields)
    false_recall = last["triggered"] is True or last["memory_id"] >= 0
    return {"hit": bool(false_recall), **last}


def score_weak(mem: LLMMemory, item: dict[str, Any]) -> dict[str, Any]:
    presented = _last_token(mem.cue_fields(str(item["weak_cue"])))
    if presented["triggered"] is True or presented["source"] == "cue":
        return {
            "hit": False,
            "reason": "weak presentation took the cue path",
            "presentation": presented,
            "ticks": [],
        }
    target = _slot_index(mem, str(item["name"]), str(item["rel"]))
    ticks = mem.offline_ticks(int(OFFLINE_PERIOD))
    hits = []
    for row in ticks:
        source = str(row["source"][0][0])
        memory_id = int(row["memory_id"][0, 0].item())
        cosine = float(row["winner_cosine"][0, 0].item())
        if (
            source == "spontaneous"
            and memory_id == target
            and target >= 0
            and float(WEAK_FLOOR) <= cosine < float(WEAK_CEILING)
        ):
            hits.append({"memory_id": memory_id, "winner_cosine": cosine, "source": source})
    return {
        "hit": bool(hits),
        "target": int(target),
        "presentation": presented,
        "ticks": hits,
    }


def run_trial(
    *,
    seed: int,
    item_id: int,
    n: int,
    family: Family,
    condition: Condition,
    probe: Probe | None = None,
) -> dict[str, Any]:
    mem = None
    with _condition_patches(condition):
        try:
            mem, _geom, trace = play_dialogue(
                seed=seed,
                item_id=item_id,
                n=n,
                family=family,
                condition=condition,
            )
            item = item_by_id(item_id)
            if family == "recall":
                if probe == "strong":
                    scored = score_strong(mem, item)
                elif probe == "weak":
                    scored = score_weak(mem, item)
                else:
                    raise ValueError("recall trials need a probe")
            elif family == "conflict":
                scored = score_conflict(mem, item)
            elif family == "unrelated":
                scored = score_unrelated(mem, item)
            else:
                _unknown_family(family)
            return {
                "seed": int(seed),
                "item": int(item_id),
                "n": int(n),
                "family": family,
                "condition": condition,
                "probe": probe,
                "trace": list(trace),
                **scored,
            }
        finally:
            if mem is not None:
                mem.close()


def schedule_ok(trace: list, n: int) -> bool:
    sleeps = [row for row in trace if row[0] == "sleep_for"]
    long_offline = [
        row for row in trace if row[0] == "offline_ticks" and row[1] == int(OFFLINE_PERIOD)
    ]
    if int(n) == 10:
        return sleeps == [] and all(row[1] != float(SLEEP_DURATION) for row in sleeps)
    if int(n) == 30:
        return sleeps == [("sleep_for", float(SLEEP_DURATION))]
    if int(n) == 100:
        return sleeps == [("sleep_for", float(SLEEP_DURATION))] and any(
            row[1] == int(OFFLINE_PERIOD) for row in long_offline
        )
    return False


def _rate(hits: list[bool]) -> float:
    if not hits:
        raise ValueError("empty cell")
    return float(sum(bool(hit) for hit in hits) / len(hits))


def run_cell(
    *,
    condition: Condition,
    n: int,
    seeds: tuple[int, ...] = SEEDS,
    items: range | None = None,
) -> dict[str, Any]:
    ids = range(ITEMS_N) if items is None else items
    strong: dict[int, list[bool]] = {int(seed): [] for seed in seeds}
    weak: dict[int, list[bool]] = {int(seed): [] for seed in seeds}
    conflict: dict[int, list[bool]] = {int(seed): [] for seed in seeds}
    unrelated: dict[int, list[bool]] = {int(seed): [] for seed in seeds}
    band_ok = True
    schedule_hits = []
    for seed in seeds:
        for item_id in ids:
            strong_row = run_trial(
                seed=seed,
                item_id=item_id,
                n=n,
                family="recall",
                condition=condition,
                probe="strong",
            )
            weak_row = run_trial(
                seed=seed,
                item_id=item_id,
                n=n,
                family="recall",
                condition=condition,
                probe="weak",
            )
            conflict_row = run_trial(
                seed=seed,
                item_id=item_id,
                n=n,
                family="conflict",
                condition=condition,
            )
            unrelated_row = run_trial(
                seed=seed,
                item_id=item_id,
                n=n,
                family="unrelated",
                condition=condition,
            )
            strong[int(seed)].append(bool(strong_row["hit"]))
            weak[int(seed)].append(bool(weak_row["hit"]))
            conflict[int(seed)].append(bool(conflict_row["hit"]))
            unrelated[int(seed)].append(bool(unrelated_row["hit"]))
            schedule_hits.append(schedule_ok(strong_row["trace"], n))
            if strong_row["hit"]:
                band_ok = band_ok and strong_row["source"] == "cue"
                band_ok = band_ok and float(strong_row["winner_cosine"]) >= float(CUE_THETA)
            if weak_row["hit"]:
                ticks = weak_row["ticks"]
                band_ok = band_ok and bool(ticks)
                for tick in ticks:
                    band_ok = band_ok and tick["source"] == "spontaneous"
                    band_ok = band_ok and float(WEAK_FLOOR) <= float(tick["winner_cosine"]) < float(
                        WEAK_CEILING
                    )
    return {
        "recall_at_1_strong": _seed_metric(strong),
        "hit_rate_weak": _seed_metric(weak),
        "conflict_newest_take": _seed_metric(conflict),
        "false_recall": _seed_metric(unrelated),
        "band_ok": bool(band_ok),
        "schedule_ok": all(schedule_hits),
    }


def _seed_metric(per_seed: dict[int, list[bool]]) -> dict[str, Any]:
    rates = [_rate(per_seed[seed]) for seed in sorted(per_seed)]
    mean, std = mean_std(rates)
    hits = sum(sum(per_seed[seed]) for seed in per_seed)
    n = sum(len(per_seed[seed]) for seed in per_seed)
    low, high = wilson_interval(int(hits), int(n))
    return {
        "mean": float(mean),
        "std": float(std),
        "per_seed": [float(rate) for rate in rates],
        "hits": int(hits),
        "n": int(n),
        "wilson": [float(low), float(high)],
    }


def frozen_interface() -> dict[str, Any]:
    surfaces = canonical_surfaces()
    return {
        "cue_interface_version": int(CUE_INTERFACE_VERSION),
        "reconstruct_interface_version": int(surfaces["interface_version"]),
        "reconstruct_sha256": reconstruct_bank_sha256(),
        "packet_keys": list(PACKET_KEYS),
        "theta": float(CUE_THETA),
    }


def run(
    *,
    seeds: tuple[int, ...] = SEEDS,
    items: range | None = None,
    ns: tuple[int, ...] = N_VALUES,
    conditions: tuple[Condition, ...] = CONDITIONS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    conditions_out: dict[str, Any] = {}
    band_ok = True
    schedule_ok_all = True
    for condition in conditions:
        conditions_out[condition] = {}
        for n in ns:
            if progress is not None:
                progress(f"{condition} N={n}")
            cell = run_cell(condition=condition, n=int(n), seeds=seeds, items=items)
            band_ok = band_ok and bool(cell.pop("band_ok"))
            schedule_ok_all = schedule_ok_all and bool(cell.pop("schedule_ok"))
            conditions_out[condition][str(int(n))] = cell
    summary = {
        "protocol": PROTOCOL,
        "seeds": list(seeds),
        "n_values": list(ns),
        "conditions": conditions_out,
        "interface": frozen_interface(),
        "band_ok": bool(band_ok),
        "schedule_ok": bool(schedule_ok_all),
    }
    if set(conditions) >= {"full", "baseline", "off_83"} and set(ns) >= {10, 100}:
        summary["hypotheses"] = score_hypotheses(summary)
    return summary
