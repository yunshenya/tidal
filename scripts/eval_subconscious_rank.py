"""Bitwise identity of the key-layer stop, then the before/after `_rank` cost.

Identity refuses a mismatch. It does not score a margin and it does not
rewrite a gate. Cost times `_rank` at store sizes 8, 16, 32, and 128.
"""
from __future__ import annotations

import json
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

REPORT = ROOT / "reports" / "subconscious_rank.json"
PROMPT = "Weigh unlabeled cadence.\nTone mqtag0:"
GOOD = " warm"
BAD = " harsh"
SIZES = (8, 16, 32, 128)
WARMUP = 2
REPEATS = 7


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _print(message: str) -> None:
    print(message, flush=True)


def _add(pairs: set, text, answer=None) -> None:
    if not isinstance(text, str) or text == "":
        return
    ans = answer if isinstance(answer, str) and answer != "" else None
    pairs.add((text, ans))


def _walk(obj, pairs: set) -> None:
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            _add(pairs, obj["text"], obj.get("answer"))
        if isinstance(obj.get("prompt"), str):
            _add(pairs, obj["prompt"])
        for value in obj.values():
            if isinstance(value, (dict, list, tuple)):
                _walk(value, pairs)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            if isinstance(value, (dict, list, tuple)):
                _walk(value, pairs)


def _from_bank(bank, pairs: set) -> None:
    for name in dir(bank):
        if name.startswith("_"):
            continue
        value = getattr(bank, name)
        if isinstance(value, str):
            _add(pairs, value)
        elif isinstance(value, (dict, list, tuple)):
            _walk(value, pairs)
        elif callable(value) and name in ("decision_items", "incidental_facts"):
            _walk(value(), pairs)


def _pairs() -> list:
    from tidal.subconscious_data import (
        BIRTHDAY_FACT,
        PHONE_FACT,
        PHONE_QUESTION,
        decision_items,
        incidental_facts,
    )
    from tidal.subconscious_measure_data import (
        avoid_items,
        away_items,
        birthday_items,
        choice_items,
        phone_items,
        prefer_items,
        preference_items,
        rif_quartets,
        surprise_highs,
        surprise_lows,
        surprise_seeds,
        tone_items,
        toward_items,
    )
    from tidal.subconscious_softmix_data import distractors_for

    pairs: set = set()
    lists = (
        tone_items(),
        choice_items(),
        preference_items(),
        phone_items(),
        birthday_items(),
        prefer_items(),
        avoid_items(),
        toward_items(),
        away_items(),
        surprise_seeds(),
        surprise_lows(),
        surprise_highs(),
    )
    for rows in lists:
        _walk(rows, pairs)
    for group in rif_quartets():
        _walk(group, pairs)
    for item in tone_items() + choice_items() + preference_items():
        _walk(distractors_for(item["id"]), pairs)
    _walk(decision_items(), pairs)
    _walk(incidental_facts(), pairs)
    _walk(PHONE_FACT, pairs)
    _walk(BIRTHDAY_FACT, pairs)
    _add(pairs, PHONE_QUESTION)

    modules = (
        "tidal.subconscious_gate_data",
        "tidal.subconscious_direction_data",
        "tidal.subconscious_replay_data",
        "tidal.subconscious_spell_data",
        "tidal.subconscious_valence_data",
        "tidal.subconscious_circadian_data",
        "tidal.subconscious_surprise_rel_data",
        "tidal.subconscious_rif_retry_data",
    )
    import importlib

    for name in modules:
        bank = importlib.import_module(name).held_bank()
        _from_bank(bank, pairs)
    return sorted(pairs, key=lambda row: (row[0], "" if row[1] is None else row[1]))


def _equal_array(left, right) -> bool:
    import numpy as np

    if left is None or right is None:
        return left is None and right is None
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _max_abs(left, right) -> float:
    import numpy as np

    diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    return float(np.max(np.abs(diff)))


def _strengths(mem) -> list:
    return [float(slot.strength) for slot in mem.store._slots]


def _restore(mem, strengths) -> None:
    for slot, strength in zip(mem.store._slots, strengths):
        slot.strength = strength


def _pack(mem, info) -> dict:
    return {
        "top": info.get("top"),
        "maxcos": info.get("maxcos"),
        "gap": info.get("gap"),
        "confidence": info.get("confidence"),
        "weights": info.get("weights"),
        "value": info.get("value"),
        "participation": info.get("participation"),
        "answer": info.get("answer"),
        "n": info.get("n"),
        "rif": None if mem.last_rif is None else dict(mem.last_rif),
    }


def _packs_equal(left, right) -> tuple:
    if left["top"] != right["top"] or left["n"] != right["n"]:
        return False, "top"
    if left["answer"] != right["answer"] or left["rif"] != right["rif"]:
        return False, "answer"
    for key in ("maxcos", "gap", "confidence", "participation"):
        if left[key] != right[key]:
            return False, key
    if not _equal_array(left["weights"], right["weights"]):
        return False, "weights"
    if not _equal_array(left["value"], right["value"]):
        return False, "value"
    return True, ""


def _rank_with(mem, prompt: str, *, full: bool) -> dict:
    original = mem._raw_key_pool

    def pool(text, answer=None, *, full=False, _full=full, _orig=original):
        del full
        return _orig(text, answer, full=_full)

    mem._raw_key_pool = pool
    try:
        return mem._rank(prompt)
    finally:
        mem._raw_key_pool = original
        mem._delta = None


def _delta_of(mem, info):
    import torch

    mem._set_delta(info)
    try:
        if mem._delta is None:
            return None
        return mem._delta.detach().float().cpu().clone()
    finally:
        mem._delta = None


def _last_logits(mem, text: str, delta):
    import torch

    from tidal.llm_memory import _batch_ids, _batch_mask, _logits_of

    mem._delta = None if delta is None else delta
    enc = mem._tokenize(text)
    ids = _batch_ids(enc).to(mem._device)
    attn = _batch_mask(enc, mem._device)
    kwargs = {"input_ids": ids, "use_cache": False}
    if attn is not None:
        kwargs["attention_mask"] = attn
    try:
        with torch.no_grad():
            out = mem.model(**kwargs)
            return _logits_of(out)[0, -1].detach().float().cpu().clone()
    finally:
        mem._delta = None


def _identity(mem) -> dict:
    import torch

    torch.set_num_threads(1)
    from tidal.subconscious_eval import Encoder, write_rows
    from tidal.subconscious_measure_data import choice_items, phone_items, preference_items, tone_items
    from tidal.subconscious_softmix_score import _group_rows

    pairs = _pairs()
    _print(f"identity strings {len(pairs)}")
    sentinel_text, sentinel_answer = pairs[0]
    sentinel_hidden, _, _ = mem._run_hidden(sentinel_text)
    sentinel_hidden = sentinel_hidden.clone()
    sentinel_pool = mem._raw_key_pool(sentinel_text, sentinel_answer, full=True)
    mismatches = []
    for index, (text, answer) in enumerate(pairs):
        if index % 50 == 0:
            _print(f"pool {index}/{len(pairs)}")
        slow_h, _, _ = mem._run_hidden(text)
        fast_h, _ = mem._run_key_hidden(text)
        if slow_h.shape != fast_h.shape or not torch.equal(slow_h, fast_h):
            max_abs = None
            if slow_h.shape == fast_h.shape:
                max_abs = float((slow_h - fast_h).abs().max())
            mismatches.append(
                {"text": text[:80], "answer": answer, "kind": "hidden", "max_abs": max_abs}
            )
            break
        slow = mem._raw_key_pool(text, answer, full=True)
        fast = mem._raw_key_pool(text, answer)
        if not _equal_array(slow, fast):
            mismatches.append(
                {"text": text[:80], "answer": answer, "kind": "pool", "max_abs": _max_abs(slow, fast)}
            )
            break
    sentinel_again, _, _ = mem._run_hidden(sentinel_text)
    sentinel_pool_again = mem._raw_key_pool(sentinel_text, sentinel_answer, full=True)
    sentinel_ok = bool(torch.equal(sentinel_hidden, sentinel_again) and _equal_array(sentinel_pool, sentinel_pool_again))
    if not sentinel_ok:
        hidden_gap = float((sentinel_hidden - sentinel_again).abs().max())
        mismatches.append(
            {
                "text": "sentinel",
                "max_abs": max(hidden_gap, _max_abs(sentinel_pool, sentinel_pool_again)),
            }
        )

    probes = []
    for family, items in (
        ("tone", tone_items()),
        ("choice", choice_items()),
        ("preference", preference_items()),
    ):
        for item in items:
            row = dict(item)
            row["family"] = family
            probes.append(row)
    if len(probes) != 150:
        raise RuntimeError(f"implicit probes {len(probes)}")
    encoder = Encoder(mem)
    rank_mismatch = None
    residual_mismatch = None
    checked = 0
    logit_control = None
    logit_match = None
    if not mismatches:
        for index, probe in enumerate(probes):
            if index % 25 == 0:
                _print(f"rank {index}/{len(probes)} {probe['id']}")
            encoded = encoder.many(_group_rows(probe))
            written = write_rows(mem, encoded)
            if int(written["slots"]) != 4:
                rank_mismatch = {"id": probe["id"], "slots": int(written["slots"])}
                break
            base = _strengths(mem)
            fast_info = _rank_with(mem, probe["prompt"], full=False)
            fast_pack = _pack(mem, fast_info)
            _restore(mem, base)
            slow_info = _rank_with(mem, probe["prompt"], full=True)
            slow_pack = _pack(mem, slow_info)
            _restore(mem, base)
            ok, why = _packs_equal(fast_pack, slow_pack)
            if not ok:
                rank_mismatch = {"id": probe["id"], "field": why}
                break
            fast_delta = _delta_of(mem, fast_info)
            slow_delta = _delta_of(mem, slow_info)
            if fast_delta is None or slow_delta is None or not torch.equal(fast_delta, slow_delta):
                residual_mismatch = {"id": probe["id"]}
                break
            if index == 0:
                control_a = _last_logits(mem, probe["prompt"], fast_delta)
                control_b = _last_logits(mem, probe["prompt"], fast_delta)
                armed = _last_logits(mem, probe["prompt"], slow_delta)
                logit_control = bool(torch.equal(control_a, control_b))
                logit_match = bool(torch.equal(control_a, armed)) if logit_control else None
            checked += 1
        mem.store.clear()
        mem._delta = None

    phone = phone_items()[0]
    mem.store.clear()
    wrote = mem.remember(phone["text"], phone["answer"], name=phone["name"], rel=phone["rel"])
    spelled_fast = mem.generate(phone["prompt"], max_new_tokens=32) if wrote else None
    spelled_slow = None
    if wrote:
        original = mem._raw_key_pool

        def full_pool(text, answer=None, *, full=False):
            del full
            return original(text, answer, full=True)

        mem._raw_key_pool = full_pool
        try:
            spelled_slow = mem.generate(phone["prompt"], max_new_tokens=32)
        finally:
            mem._raw_key_pool = original
            mem._delta = None
    mem.store.clear()
    bare_fast = mem.generate(PROMPT, max_new_tokens=4)
    bare_slow_model = mem.generate(PROMPT, max_new_tokens=4)

    return {
        "strings": len(pairs),
        "pool_mismatch": mismatches,
        "sentinel_ok": sentinel_ok,
        "probes": len(probes),
        "ranks_checked": checked,
        "rank_mismatch": rank_mismatch,
        "residual_mismatch": residual_mismatch,
        "logit_control_equal": logit_control,
        "armed_logits_equal": logit_match,
        "phone_written": bool(wrote),
        "phone_fast": spelled_fast,
        "phone_slow": spelled_slow,
        "phone_equal": spelled_fast == spelled_slow,
        "bare_generate_equal": bare_fast == bare_slow_model,
        "threads": int(__import__("torch").get_num_threads()),
        "ok": (
            not mismatches
            and sentinel_ok
            and rank_mismatch is None
            and residual_mismatch is None
            and logit_control is True
            and logit_match is True
            and bool(wrote)
            and spelled_fast == spelled_slow
            and bare_fast == bare_slow_model
            and checked == 150
        ),
    }


def _bench(fn) -> dict:
    for _ in range(WARMUP):
        fn()
    samples = []
    for _ in range(REPEATS):
        started = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - started)
    samples.sort()
    return {
        "median_s": float(samples[len(samples) // 2]),
        "min_s": float(samples[0]),
        "max_s": float(samples[-1]),
        "n": len(samples),
    }


def _fill(mem, n: int) -> None:
    import numpy as np

    rng = np.random.default_rng(0)
    mem.store.clear()
    for index in range(n):
        key = rng.normal(size=mem.hidden).astype(np.float32)
        value = rng.normal(size=mem.hidden).astype(np.float32)
        mem.store.write(
            key,
            value,
            name=f"rank{index}",
            rel="rankcost",
            answer=f"a{index}",
            text=f"rankcost fact {index} token{index}",
        )
    if len(mem.store) != n:
        raise RuntimeError(f"store size {len(mem.store)} != {n}")


def _pool_spread(rows) -> float:
    import numpy as np

    if len(rows) < 2:
        return 0.0
    spread = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            diff = np.asarray(rows[i], dtype=np.float64) - np.asarray(rows[j], dtype=np.float64)
            spread = max(spread, float(np.max(np.abs(diff))))
    return spread


def _noise(mem) -> dict:
    """Max abs pooled-key movement at the cost thread count. Not a pass rule."""
    fulls = [mem._raw_key_pool(PROMPT, full=True) for _ in range(REPEATS)]
    fasts = [mem._raw_key_pool(PROMPT) for _ in range(REPEATS)]
    cross = 0.0
    import numpy as np

    for left in fulls:
        for right in fasts:
            diff = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
            cross = max(cross, float(np.max(np.abs(diff))))
    return {
        "prompt": PROMPT,
        "repeats": REPEATS,
        "full_max_abs": _pool_spread(fulls),
        "prefix_max_abs": _pool_spread(fasts),
        "cross_max_abs": cross,
    }


def _cost(mem) -> dict:
    import torch

    from tidal.subconscious_eval import token_ids

    torch.set_num_threads(os.cpu_count() or 1)
    prompt_n = len(token_ids(mem.tokenizer, PROMPT))
    good_n = len(token_ids(mem.tokenizer, PROMPT + GOOD)) - prompt_n
    bad_n = len(token_ids(mem.tokenizer, PROMPT + BAD)) - prompt_n
    cont_n = int(good_n + bad_n)
    if cont_n <= 0:
        raise RuntimeError("continuation token count is empty")
    rows = []
    original = mem._raw_key_pool

    def full_pool(text, answer=None, *, full=False):
        del full
        return original(text, answer, full=True)

    for size in SIZES:
        _print(f"cost size {size}")
        _fill(mem, size)
        mem._raw_key_pool = full_pool
        try:
            before = _bench(lambda: mem._rank(PROMPT))
        finally:
            mem._raw_key_pool = original
            mem._delta = None
        after = _bench(lambda: mem._rank(PROMPT))
        mem._delta = None
        query = mem._query_vector(PROMPT)

        def scan(_query=query):
            saved = mem._query_vector
            mem._query_vector = lambda _prompt: _query
            try:
                mem._rank(PROMPT)
            finally:
                mem._query_vector = saved

        scanned = _bench(scan)
        mem._delta = None
        rows.append(
            {
                "size": int(size),
                "before_median_s": before["median_s"],
                "after_median_s": after["median_s"],
                "before_per_continuation_token_s": before["median_s"] / float(cont_n),
                "after_per_continuation_token_s": after["median_s"] / float(cont_n),
                "scan_median_s": scanned["median_s"],
                "before": before,
                "after": after,
                "scan": scanned,
            }
        )
    mem.store.clear()
    return {
        "prompt": PROMPT,
        "good": GOOD,
        "bad": BAD,
        "prompt_tokens": int(prompt_n),
        "continuation_tokens": cont_n,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "threads": int(__import__("torch").get_num_threads()),
        "prompt_pool_noise": _noise(mem),
        "rows": rows,
    }


def main() -> None:
    import torch

    torch.set_num_threads(1)
    started = time.perf_counter()
    from tidal.llm_memory import LLMMemory

    mem = LLMMemory()
    try:
        identity = _identity(mem)
        _print(f"identity ok {identity['ok']}")
        if not identity["ok"]:
            payload = {
                "ok": False,
                "git_sha": _git_sha(),
                "identity": identity,
                "elapsed_s": time.perf_counter() - started,
            }
            REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise SystemExit(1)
        cost = _cost(mem)
    finally:
        mem.close()
    payload = {
        "ok": True,
        "git_sha": _git_sha(),
        "model": "HuggingFaceTB/SmolLM2-135M",
        "identity": identity,
        "cost": cost,
        "elapsed_s": time.perf_counter() - started,
    }
    REPORT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print(f"wrote {REPORT}")
    for row in cost["rows"]:
        _print(
            f"size {row['size']} before {row['before_per_continuation_token_s']:.6f} "
            f"after {row['after_per_continuation_token_s']:.6f} "
            f"scan {row['scan_median_s']:.6f}"
        )


if __name__ == "__main__":
    main()
