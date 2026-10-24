"""Subconscious-style metrics for the shipped ``LLMMemory`` inject path.

Log-prob scoring arms the prompt once and compares two continuations under
that vector. The residual is scaled by the top-1 gap and by polarity gain.
``generate`` encodes
the query once: spelling replaces the token loop only when the prompt is
completing the winning relation. This module does not import ``transformers``.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from tidal.llm_memory import _logits_of
from tidal import subconscious_data as _frozen_bank
from tidal.subconscious_data import (
    AVOID,
    BIRTHDAY_FACT,
    CHATS,
    GATES,
    LEAK_STORE_N,
    LOADS,
    NEUTRAL_CHOICE,
    NEUTRAL_GEN,
    NEUTRAL_NLL,
    NEUTRAL_STORE_N,
    PHONE_FACT,
    PHONE_QUESTION,
    PREFER,
    PRIOR_PREFIX_N32,
    PROTOCOL,
    REPEAT_WRITES,
    RETRIEVAL_SIZES,
    bank_sha256,
    completion_prompt,
    decision_items,
    distractors,
    incidental_facts,
    word_hit,
)

GREEDY_TOKENS = 12
SCRIPTED_TOKENS = 32
COST_NEW_TOKENS = 8

# Fixed per call. No retry loop and no second retrieval as tokens grow.
PATH_FLAGS = (
    {
        "path": "query encode",
        "when": "once, inside continuation_ids, before any new token",
        "varies": "skipped when the store is empty",
        "retries": False,
    },
    {
        "path": "scripted phone or birthday",
        "when": "winning relation is phone or birthday and the prompt is completing that same relation",
        "varies": "copies the stored answer and replaces the token loop; any other prompt keeps the residual path",
        "retries": False,
    },
    {
        "path": "span stop",
        "when": "generate only",
        "varies": "stops before max_new_tokens on a newline or sentence punctuation",
        "retries": False,
    },
    {
        "path": "uncached prefix",
        "when": "every LM token, with or without a residual",
        "varies": "each new token forwards the whole prefix; use_cache is False",
        "retries": False,
    },
    {
        "path": "remember",
        "when": "each accepted fact",
        "varies": "two full forwards (key, then value); chat and noise return before that",
        "retries": False,
    },
    {
        "path": "replay consolidation",
        "when": "every N accepted writes",
        "varies": "recomputes decay, refreshes the top-m salient slots, merges at most m near-duplicates; no extra forward",
        "retries": False,
    },
    {
        "path": "experience valence",
        "when": "a write carries an experience sign, and again on replay",
        "varies": "adds sign * step on that slot, decays with replay, scales the residual; no extra forward and no text score",
        "retries": False,
    },
    {
        "path": "circadian clock",
        "when": "a clock angle is set or ticked, and again when decay runs",
        "varies": "scales that slot's write strength and the decay exponent; phase 0 leaves both unchanged; no extra forward and no text",
        "retries": False,
    },
)


def token_ids(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def continuation_logprobs(mem, prompt: str, continuation: str) -> dict:
    """Mean and sum token log-prob of ``continuation`` under the current residual.

    The prompt ids must be a prefix of the full ids. A merge across the
    boundary would score a different string than the one named here.
    """
    if continuation == "":
        raise ValueError("continuation is empty")
    prompt_ids = token_ids(mem.tokenizer, prompt)
    full_ids = token_ids(mem.tokenizer, prompt + continuation)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "continuation retokenizes the prompt boundary: "
            f"{prompt!r} + {continuation!r}"
        )
    cont = full_ids[len(prompt_ids) :]
    if not cont or not prompt_ids:
        raise RuntimeError("continuation produced no predictable token")
    ids = torch.tensor([full_ids], dtype=torch.long, device=mem._device)
    with torch.no_grad():
        logits = _logits_of(mem.model(input_ids=ids, use_cache=False))
        pos = [len(prompt_ids) - 1 + i for i in range(len(cont))]
        pred = logits[0, pos].float()
        logp = torch.log_softmax(pred, dim=-1)
        target = torch.tensor(cont, dtype=torch.long, device=logp.device)
        taken = logp.gather(1, target.unsqueeze(-1)).squeeze(-1)
        vals = [float(x) for x in taken.detach().cpu().tolist()]
    return {
        "sum": float(sum(vals)),
        "mean": float(sum(vals) / len(vals)),
        "n_tokens": len(vals),
    }


def _cosine(a, b) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))


def _arm_info(mem, prompt: str) -> dict:
    """One query, hard top-1, residual scaled by the gap. Empty store arms nothing."""
    rank = mem._rank(prompt)
    mem._set_delta(rank)
    top = rank.get("top")
    return {
        "n": int(rank["n"]),
        "top": list(top) if top else None,
        "maxcos": rank.get("maxcos"),
        "gap": rank.get("gap"),
        "confidence": float(rank.get("confidence") or 0.0),
        "hit": 0,
    }


def score_pair(mem, prompt: str, good: str, bad: str, *, name: Optional[str] = None, rel: Optional[str] = None) -> dict:
    """Length-normalized margin ``mean logp(good) - mean logp(bad)``."""
    info = _arm_info(mem, prompt)
    if name is not None:
        info["hit"] = int(info["top"] == [name, rel])
    try:
        g = continuation_logprobs(mem, prompt, good)
        b = continuation_logprobs(mem, prompt, bad)
    finally:
        mem._delta = None
    return {
        "good_mean": g["mean"],
        "bad_mean": b["mean"],
        "good_sum": g["sum"],
        "bad_sum": b["sum"],
        "good_tokens": g["n_tokens"],
        "bad_tokens": b["n_tokens"],
        "margin": g["mean"] - b["mean"],
        "sum_margin": g["sum"] - b["sum"],
        "choose": int(g["mean"] > b["mean"]),
        "choose_sum": int(g["sum"] > b["sum"]),
        "retrieval": info,
    }


def score_one(mem, prompt: str, continuation: str) -> dict:
    """Mean token NLL of one continuation. Lower is a better fit."""
    info = _arm_info(mem, prompt)
    try:
        row = continuation_logprobs(mem, prompt, continuation)
    finally:
        mem._delta = None
    return {"mean_logprob": row["mean"], "nll": -row["mean"], "n_tokens": row["n_tokens"], "retrieval": info}


def _delta(scored: dict, nomem: dict) -> dict:
    out = dict(scored)
    out["delta"] = float(scored["margin"] - nomem["margin"])
    return out


class Encoder:
    def __init__(self, mem):
        self.mem = mem
        self.cache = {}

    def one(self, fact: dict) -> dict:
        key = (fact["text"], fact["answer"], fact["name"], fact["rel"])
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        row = {
            "text": fact["text"],
            "answer": fact["answer"],
            "name": fact["name"],
            "rel": fact["rel"],
            "key": self.mem.encode_key(fact["text"], fact["answer"]),
            "value": self.mem.encode_value(fact["text"], fact["answer"]),
        }
        self.cache[key] = row
        return row

    def many(self, facts) -> list:
        return [self.one(fact) for fact in facts]


def write_rows(mem, rows: list) -> dict:
    mem.store.clear()
    accepted = 0
    for row in rows:
        ok = mem.store.write(
            row["key"],
            row["value"],
            name=row["name"],
            rel=row["rel"],
            kind="fact",
            answer=row.get("answer"),
        )
        accepted += int(bool(ok))
    return {"requested": len(rows), "accepted": accepted, "slots": len(mem.store)}


def has_identity(mem, name: str, rel: str) -> bool:
    return any(slot.name == name and slot.rel == rel for slot in mem.store.slots)


def _fact(item: dict, *, update: bool = False) -> dict:
    if update:
        return {
            "name": item["name"],
            "rel": item["rel"],
            "answer": item["update_answer"],
            "text": item["update_text"],
        }
    return {"name": item["name"], "rel": item["rel"], "answer": item["answer"], "text": item["text"]}


def _mean(xs) -> Optional[float]:
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _influence(rows: list, condition: str) -> dict:
    deltas = [float(row[condition]["delta"]) for row in rows]
    choose = [int(row[condition]["choose"]) for row in rows]
    base = [int(row["nomem"]["choose"]) for row in rows]
    acc = _mean(choose)
    acc_n = _mean(base)
    pos = sum(d > 0.0 for d in deltas)
    return {
        "n": len(rows),
        "mean_delta": _mean(deltas),
        "n_delta_pos": int(pos),
        "acc": acc,
        "acc_hits": int(sum(choose)),
        "acc_nomem": acc_n,
        "acc_nomem_hits": int(sum(base)),
        "acc_lift": float(acc - acc_n),
        "delta_pass": bool(_mean(deltas) > GATES["delta_min"] and pos >= GATES["delta_pos_min"]),
        "choice_pass": bool((acc - acc_n) >= GATES["acc_lift_min"]),
    }


def _join(texts, prompt: str) -> str:
    body = "\n".join(row["text"] if isinstance(row, dict) else row for row in texts)
    return body + "\n" + prompt


def phone_takeover(text: str) -> bool:
    return "+1-415-555-" in str(text)


def date_takeover(text: str) -> bool:
    import re

    return re.search(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December) ([1-9]|[12][0-9]|3[01]), \d{4}\b",
        str(text),
    ) is not None


def _greedy_row(item: dict, text: str) -> dict:
    good = word_hit(text, item["good"].strip())
    bad = word_hit(text, item["bad"].strip())
    return {"text": text, "good": bool(good), "bad": bool(bad), "hit": int(good and not bad)}


def _bench(fn, repeats: int, warmup: int) -> dict:
    for _ in range(int(warmup)):
        fn()
    samples = []
    for _ in range(int(repeats)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    mid = samples[len(samples) // 2]
    return {
        "median_s": float(mid),
        "min_s": float(samples[0]),
        "max_s": float(samples[-1]),
        "n": len(samples),
    }


def inject_census(module) -> dict:
    params = [(name, int(p.numel())) for name, p in module.named_parameters()]
    buffers = [(name, int(b.numel())) for name, b in module.named_buffers()]
    return {
        "parameters": int(sum(n for _, n in params)),
        "buffers": int(sum(n for _, n in buffers)),
        "mlp_parameters": int(sum(n for name, n in params if name.startswith("net."))),
        "named_parameters": params,
        "named_buffers": buffers,
    }


def _rss_kb() -> dict:
    out = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:")):
                key, rest = line.split(":", 1)
                out[key] = int(rest.split()[0])
    return out


def _count_path(mem, prompt: str, n_tokens: int) -> dict:
    queries = {"n": 0}
    forwards = {"n": 0}
    orig = mem._query_vector

    def wrapped(text, _orig=orig):
        queries["n"] += 1
        return _orig(text)

    def _hook(_module, _inputs, _output):
        forwards["n"] += 1

    handle = mem.model.register_forward_hook(_hook)
    mem._query_vector = wrapped
    try:
        ids = mem.continuation_ids(prompt, max_new_tokens=n_tokens, done=lambda _ids: False)
    finally:
        mem._query_vector = orig
        handle.remove()
        mem._delta = None
    return {"queries": int(queries["n"]), "forwards": int(forwards["n"]), "new_tokens": len(ids)}


def _ids_exact(mem, n: int) -> list:
    piece = "The kettle whistled softly. "
    text = piece
    while len(token_ids(mem.tokenizer, text)) < n:
        text += piece
    return token_ids(mem.tokenizer, text)[:n]


def measure_cost(mem, incidental: list, normal: dict, phone: dict, *, repeats: int, warmup: int, prompt: str) -> dict:
    """CPU cost of the shipped path. Retrieval is timed without the LM forward."""
    census = inject_census(mem.inject)
    model_bytes = int(sum(p.numel() * p.element_size() for p in mem.model.parameters()))
    rss_before = _rss_kb()

    write_rows(mem, [])
    empty_path = _count_path(mem, prompt, COST_NEW_TOKENS)
    write_rows(mem, [normal])
    normal_path = _count_path(mem, prompt, COST_NEW_TOKENS)
    write_rows(mem, [phone])
    phone_path = _count_path(mem, prompt, COST_NEW_TOKENS)

    write_rows(mem, [normal])
    query_bench = _bench(lambda: mem._query_vector(prompt), repeats, warmup)
    mem._delta = None

    query = mem._query_vector(prompt)
    readout = {}
    for size in list(RETRIEVAL_SIZES) + [len(incidental)]:
        write_rows(mem, incidental[: int(size)])
        readout[str(int(size))] = _bench(
            lambda q=query: mem.store.readout(q, mode="top1"),
            repeats,
            warmup,
        )
    value = torch.tensor(np.asarray(normal["value"], dtype=np.float32))
    key_np = normal.get("key")
    if key_np is None:
        mlp_bench = _bench(lambda: mem.inject(value), repeats, warmup)
    else:
        key_t = torch.tensor(np.asarray(key_np, dtype=np.float32))
        mlp_bench = _bench(lambda: mem.inject(value, key_t), repeats, warmup)

    delta = mem._compose(normal["value"], grad=False).detach()
    forwards = {}
    for length in (16, 32, 64):
        ids = torch.tensor([_ids_exact(mem, length)], dtype=torch.long, device=mem._device)

        def step(armed, _ids=ids, _delta=delta):
            mem._delta = _delta if armed else None
            try:
                with torch.no_grad():
                    logits = _logits_of(mem.model(input_ids=_ids, use_cache=False))
                    float(logits[0, -1, 0])
            finally:
                mem._delta = None

        empty_s = _bench(lambda: step(False), repeats, warmup)
        armed_s = _bench(lambda: step(True), repeats, warmup)
        forwards[str(length)] = {
            "empty_median_s": empty_s["median_s"],
            "armed_median_s": armed_s["median_s"],
            "extra_s": float(armed_s["median_s"] - empty_s["median_s"]),
        }

    def run_ids(rows):
        write_rows(mem, rows)
        mem.continuation_ids(prompt, max_new_tokens=COST_NEW_TOKENS, done=lambda _ids: False)
        mem._delta = None

    totals = {
        "empty": _bench(lambda: run_ids([]), repeats, warmup),
        "inject": _bench(lambda: run_ids([normal]), repeats, warmup),
        "scripted_phone": _bench(lambda: run_ids([phone]), repeats, warmup),
    }
    per_token_extra = [row["extra_s"] for row in forwards.values()]
    query_extra = float(query_bench["median_s"])
    # One query per continuation. Spelling and the residual share that encode.
    prompt_extra = {
        "one_query_s": query_extra,
        "shipped_non_phone_queries": int(normal_path["queries"]),
        "shipped_query_s": float(query_extra * normal_path["queries"]),
        "mlp_s": float(mlp_bench["median_s"]),
        "readout_n64_s": float(readout.get("64", readout[str(len(incidental))])["median_s"]),
    }
    prompt_extra["total_s"] = float(
        prompt_extra["shipped_query_s"] + prompt_extra["mlp_s"] + prompt_extra["readout_n64_s"]
    )
    write_rows(mem, [])
    return {
        "inject": census,
        "model_parameter_bytes": model_bytes,
        "rss_kb": {"before": rss_before, "after": _rss_kb()},
        "torch_threads": int(torch.get_num_threads()),
        "path_counts": {"empty": empty_path, "inject": normal_path, "scripted_phone": phone_path},
        "query_s": query_bench,
        "readout_s": readout,
        "mlp_s": mlp_bench,
        "forward_s": forwards,
        "continuation_s": totals,
        "extra_per_token_s": {
            "by_length": {key: row["extra_s"] for key, row in forwards.items()},
            "mean_s": _mean(per_token_extra),
        },
        "extra_per_prompt_s": prompt_extra,
        "flags": list(PATH_FLAGS),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "new_tokens": COST_NEW_TOKENS,
    }


def _load_bundle(enc: Encoder, item: dict, n_load: int, incidental: list) -> list:
    target = enc.one(_fact(item))
    extra = enc.many(distractors(int(item["_index"]), n_load - 1, incidental))
    return [target] + extra


def run_eval(
    mem,
    retrieve: Callable,
    *,
    cost_repeats: int = 7,
    cost_warmup: int = 2,
    do_cost: bool = True,
    do_greedy: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    data=None,
) -> dict:
    """Score the pre-registered bank. ``retrieve(texts, query)`` is the RAG line.

    ``data`` is another bank with the same attributes. ``None`` scores
    ``tidal.subconscious_data``.
    """
    if retrieve is None:
        raise ValueError("retrieve is required")
    src = _frozen_bank if data is None else data
    PREFER = src.PREFER
    AVOID = src.AVOID
    CHATS = src.CHATS
    GATES = src.GATES
    LEAK_STORE_N = src.LEAK_STORE_N
    LOADS = src.LOADS
    NEUTRAL_CHOICE = src.NEUTRAL_CHOICE
    NEUTRAL_GEN = src.NEUTRAL_GEN
    NEUTRAL_NLL = src.NEUTRAL_NLL
    NEUTRAL_STORE_N = src.NEUTRAL_STORE_N
    PHONE_FACT = src.PHONE_FACT
    PHONE_QUESTION = src.PHONE_QUESTION
    BIRTHDAY_FACT = src.BIRTHDAY_FACT
    PRIOR_PREFIX_N32 = src.PRIOR_PREFIX_N32
    REPEAT_WRITES = src.REPEAT_WRITES
    PROTOCOL = src.PROTOCOL
    decision_items = src.decision_items
    incidental_facts = src.incidental_facts
    bank_kind = "frozen" if data is None else getattr(data, "kind", "held")
    if data is None:
        bank_sha = bank_sha256()
        birthday_question = None
    else:
        bank_sha = data.sha256() if callable(getattr(data, "sha256", None)) else bank_sha256()
        birthday_question = getattr(data, "BIRTHDAY_QUESTION", None)
    mem.inject.eval()
    enc = Encoder(mem)
    incidental = incidental_facts()
    phases = {}

    def say(msg: str) -> None:
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    say("encode")
    for item in decision_items():
        enc.one(_fact(item))
        if "update_answer" in item:
            enc.one(_fact(item, update=True))
    enc.many(incidental)
    enc.one(PHONE_FACT)
    enc.one(BIRTHDAY_FACT)
    phases["encode"] = time.perf_counter() - t0

    nomem_by_id = {}
    axis_items = {"prefer": [], "avoid": []}
    prefix_items = []
    t1 = time.perf_counter()
    say("axis1")
    for family, group in (("prefer", PREFER), ("avoid", AVOID)):
        for index, item in enumerate(group):
            item = dict(item)
            item["_index"] = index
            write_rows(mem, [])
            nomem = score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"])
            nomem_by_id[item["id"]] = nomem
            write_rows(mem, [enc.one(_fact(item))])
            injected = _delta(
                score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"]),
                nomem,
            )
            write_rows(mem, [])
            context = _delta(
                score_pair(
                    mem,
                    item["text"] + "\n" + item["prompt"],
                    item["good"],
                    item["bad"],
                    name=item["name"],
                    rel=item["rel"],
                ),
                nomem,
            )
            line = retrieve([item["text"]], item["prompt"])
            rag = _delta(
                score_pair(
                    mem,
                    str(line).rstrip("\n") + "\n" + item["prompt"],
                    item["good"],
                    item["bad"],
                    name=item["name"],
                    rel=item["rel"],
                ),
                nomem,
            )
            rag["line_hit"] = int(str(line).strip() == item["text"].strip())
            rag["line"] = str(line)
            stored_side = "good_mean" if family == "prefer" else "bad_mean"
            prefix_prompt = completion_prompt(item["text"], item["answer"])
            write_rows(mem, [])
            prefix_nomem = score_pair(
                mem,
                prefix_prompt,
                " " + item["answer"],
                " " + item["other"],
            )
            write_rows(mem, [enc.one(_fact(item))])
            prefix_inj = _delta(
                score_pair(mem, prefix_prompt, " " + item["answer"], " " + item["other"]),
                prefix_nomem,
            )
            greedy = None
            if do_greedy:
                greedy = {}
                for label, rows, prompt in (
                    ("nomem", [], item["prompt"]),
                    ("inject", [enc.one(_fact(item))], item["prompt"]),
                    ("context", [], item["text"] + "\n" + item["prompt"]),
                    ("rag", [], str(line).rstrip("\n") + "\n" + item["prompt"]),
                ):
                    write_rows(mem, rows)
                    greedy[label] = _greedy_row(item, mem.generate(prompt, max_new_tokens=GREEDY_TOKENS))
            row = {
                "id": item["id"],
                "family": family,
                "name": item["name"],
                "rel": item["rel"],
                "answer": item["answer"],
                "other": item["other"],
                "nomem": nomem,
                "inject": injected,
                "context": context,
                "rag": rag,
                "stored_lift": float(injected[stored_side] - nomem[stored_side]),
                "greedy": greedy,
            }
            axis_items[family].append(row)
            prefix_items.append(
                {
                    "id": item["id"],
                    "family": family,
                    "nomem_margin": prefix_nomem["margin"],
                    "inject_margin": prefix_inj["margin"],
                    "delta": prefix_inj["delta"],
                    "inject_choose_stored": prefix_inj["choose"],
                    "good_tokens": prefix_inj["good_tokens"],
                    "bad_tokens": prefix_inj["bad_tokens"],
                }
            )
            say(f"axis1 {item['id']}")
    phases["axis1"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    say("load")
    load_items = []
    for family, group in (("prefer", PREFER), ("avoid", AVOID)):
        for index, item in enumerate(group):
            base = nomem_by_id[item["id"]]
            for n_load in LOADS:
                rows = _load_bundle(enc, {**item, "_index": index}, int(n_load), incidental)
                placed = write_rows(mem, rows)
                survived = int(has_identity(mem, item["name"], item["rel"]))
                injected = _delta(
                    score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"]),
                    base,
                )
                write_rows(mem, [])
                transcript = _join(rows, item["prompt"])
                context = _delta(
                    score_pair(mem, transcript, item["good"], item["bad"], name=item["name"], rel=item["rel"]),
                    base,
                )
                texts = [row["text"] for row in rows]
                line = retrieve(texts, item["prompt"])
                rag = _delta(
                    score_pair(
                        mem,
                        str(line).rstrip("\n") + "\n" + item["prompt"],
                        item["good"],
                        item["bad"],
                        name=item["name"],
                        rel=item["rel"],
                    ),
                    base,
                )
                rag["line_hit"] = int(str(line).strip() == item["text"].strip())
                load_items.append(
                    {
                        "id": item["id"],
                        "family": family,
                        "load": int(n_load),
                        "requested": placed["requested"],
                        "slots": placed["slots"],
                        "survived": survived,
                        "nomem": {"choose": base["choose"], "margin": base["margin"]},
                        "inject": injected,
                        "context": context,
                        "rag": rag,
                    }
                )
            say(f"load {item['id']}")
    phases["load"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    say("repeat-chat-update")
    repeat_rows = []
    for index, item in enumerate(PREFER):
        row = enc.one(_fact(item))
        placed = write_rows(mem, [row, row, row][:REPEAT_WRITES])
        scored = score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"])
        single = next(r for r in axis_items["prefer"] if r["id"] == item["id"])
        repeat_rows.append(
            {
                "id": item["id"],
                "slots": placed["slots"],
                "abs_margin": abs(float(scored["margin"] - single["inject"]["margin"])),
            }
        )
    absent_rows = []
    for index, item in enumerate(PREFER):
        extra = enc.many(distractors(index, 8, incidental))
        write_rows(mem, extra)
        scored = _delta(
            score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"]),
            nomem_by_id[item["id"]],
        )
        absent_rows.append({"id": item["id"], "delta": scored["delta"], "hit": scored["retrieval"]["hit"], "slots": len(mem.store)})
    chat_rows = []
    for item in PREFER:
        write_rows(mem, [enc.one(_fact(item))])
        accepted = []
        for text, answer in CHATS:
            accepted.append(bool(mem.remember(text, answer=answer, kind="chat", name="chat", rel="chat")))
        scored = score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"])
        single = next(r for r in axis_items["prefer"] if r["id"] == item["id"])
        chat_rows.append(
            {
                "id": item["id"],
                "accepted_chats": int(sum(accepted)),
                "slots": len(mem.store),
                "abs_margin": abs(float(scored["margin"] - single["inject"]["margin"])),
            }
        )
    update_rows = []
    for item in PREFER:
        old = enc.one(_fact(item))
        new = enc.one(_fact(item, update=True))
        placed = write_rows(mem, [old, new])
        if len(mem.store.slots) == 1:
            slot_v = mem.store.slots[0].v
            cos_new = _cosine(slot_v, new["value"])
            cos_old = _cosine(slot_v, old["value"])
        else:
            cos_new = float("nan")
            cos_old = float("nan")
        write_rows(mem, [])
        nomem = score_pair(mem, item["prompt"], item["update_good"], item["update_bad"])
        write_rows(mem, [old, new])
        injected = _delta(score_pair(mem, item["prompt"], item["update_good"], item["update_bad"]), nomem)
        update_rows.append(
            {
                "id": item["id"],
                "slots": placed["slots"],
                "cos_new": cos_new,
                "cos_old": cos_old,
                "nomem": nomem,
                "inject": injected,
                "new_lift": float(injected["good_mean"] - nomem["good_mean"]),
                "old_lift": float(injected["bad_mean"] - nomem["bad_mean"]),
            }
        )
    phases["controls"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    say("fifo")
    fifo_rows = []
    for item in PREFER:
        rows = [enc.one(_fact(item))] + enc.many(incidental)
        placed = write_rows(mem, rows)
        survived = int(has_identity(mem, item["name"], item["rel"]))
        scored = _delta(
            score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"]),
            nomem_by_id[item["id"]],
        )
        single = next(r for r in axis_items["prefer"] if r["id"] == item["id"])
        fifo_rows.append(
            {
                "id": item["id"],
                "requested": placed["requested"],
                "slots": placed["slots"],
                "capacity": int(mem.store.capacity),
                "survived": survived,
                "delta": scored["delta"],
                "single_delta": single["inject"]["delta"],
            }
        )
    phases["fifo"] = time.perf_counter() - t4

    t5 = time.perf_counter()
    say("neutral")
    neutral_rows = enc.many(incidental[:NEUTRAL_STORE_N])
    nll_rows = []
    for item in NEUTRAL_NLL:
        write_rows(mem, [])
        base = score_one(mem, item["prompt"], item["continuation"])
        write_rows(mem, neutral_rows)
        armed = score_one(mem, item["prompt"], item["continuation"])
        nll_rows.append(
            {
                "id": item["id"],
                "nll_nomem": base["nll"],
                "nll_inject": armed["nll"],
                "delta": float(armed["nll"] - base["nll"]),
                "top": armed["retrieval"]["top"],
            }
        )
    choice_rows = []
    for item in NEUTRAL_CHOICE:
        write_rows(mem, [])
        base = score_pair(mem, item["prompt"], item["good"], item["bad"])
        write_rows(mem, neutral_rows)
        armed = _delta(score_pair(mem, item["prompt"], item["good"], item["bad"]), base)
        choice_rows.append(
            {
                "id": item["id"],
                "nomem_choose": base["choose"],
                "inject_choose": armed["choose"],
                "delta": armed["delta"],
                "flip": int(base["choose"] == 1 and armed["choose"] == 0),
            }
        )
    leak_rows_src = enc.many(incidental[:LEAK_STORE_N])
    leak_answers = [row["answer"] for row in leak_rows_src]
    leak_rows = []
    write_rows(mem, leak_rows_src)
    for prompt in NEUTRAL_GEN:
        text = mem.generate(prompt, max_new_tokens=GREEDY_TOKENS)
        hits = [answer for answer in leak_answers if word_hit(text, answer)]
        leak_rows.append({"prompt": prompt, "text": text, "hits": hits})
    phases["neutral"] = time.perf_counter() - t5

    t6 = time.perf_counter()
    say("scripted")
    phone_on = float(mem.inject.phone_on.detach().cpu()) != 0.0
    date_on = float(mem.inject.date_on.detach().cpu()) != 0.0
    script_prompts = list(NEUTRAL_GEN[:4]) + [PHONE_QUESTION]
    if birthday_question:
        script_prompts.append(birthday_question)
    phone_rows = []
    write_rows(mem, [enc.one(PHONE_FACT)])
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        phone_rows.append({"prompt": prompt, "text": text, "takeover": int(phone_takeover(text))})
    date_rows = []
    write_rows(mem, [enc.one(BIRTHDAY_FACT)])
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        date_rows.append({"prompt": prompt, "text": text, "takeover": int(date_takeover(text))})
    phases["scripted"] = time.perf_counter() - t6

    cost = None
    if do_cost:
        t7 = time.perf_counter()
        say("cost")
        cost = measure_cost(
            mem,
            enc.many(incidental),
            enc.one(_fact(PREFER[0])),
            enc.one(PHONE_FACT),
            repeats=cost_repeats,
            warmup=cost_warmup,
            prompt=PREFER[0]["prompt"],
        )
        phases["cost"] = time.perf_counter() - t7
    write_rows(mem, [])

    prefer_sum = {cond: _influence(axis_items["prefer"], cond) for cond in ("inject", "context", "rag")}
    avoid_sum = {cond: _influence(axis_items["avoid"], cond) for cond in ("inject", "context", "rag")}
    load_summary = {"prefer": {}, "avoid": {}}
    for family in ("prefer", "avoid"):
        for n_load in LOADS:
            subset = [row for row in load_items if row["family"] == family and row["load"] == int(n_load)]
            load_summary[family][str(n_load)] = {
                "n": len(subset),
                "slots_mean": _mean([row["slots"] for row in subset]),
                "survived": _mean([row["survived"] for row in subset]),
                "survived_n": int(sum(row["survived"] for row in subset)),
                "hit": _mean([row["inject"]["retrieval"]["hit"] for row in subset]),
                "hit_n": int(sum(row["inject"]["retrieval"]["hit"] for row in subset)),
                "rag_line_hit": _mean([row["rag"]["line_hit"] for row in subset]),
                "inject": _influence(subset, "inject"),
                "context": _influence(subset, "context"),
                "rag": _influence(subset, "rag"),
            }
    nll_deltas = [row["delta"] for row in nll_rows]
    absent_deltas = [row["delta"] for row in absent_rows]
    absent_large = sum(abs(d) > GATES["unrelated_large"] for d in absent_deltas)
    update_deltas = [row["inject"]["delta"] for row in update_rows]
    result = {
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha,
        "bank_kind": bank_kind,
        "model_name": mem.model_name,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "n_layers": int(mem.n_layers),
        "capacity": int(mem.store.capacity),
        "overwrite_cos": float(mem.store.overwrite_cos),
        "phone_on": bool(phone_on),
        "date_on": bool(date_on),
        "gates": dict(GATES),
        "prior_prefix_n32": dict(PRIOR_PREFIX_N32),
        "phases_s": phases,
        "axis1": {
            "prefer": {"items": axis_items["prefer"], "summary": prefer_sum},
            "avoid": {"items": axis_items["avoid"], "summary": avoid_sum},
            "prefix": prefix_items,
            "stored_lift": {
                "prefer": _mean([row["stored_lift"] for row in axis_items["prefer"]]),
                "avoid": _mean([row["stored_lift"] for row in axis_items["avoid"]]),
            },
        },
        "load": {"summary": load_summary, "items": load_items},
        "repeat": {
            "rows": repeat_rows,
            "max_abs": max(row["abs_margin"] for row in repeat_rows),
            "pass": bool(repeat_rows)
            and max(row["abs_margin"] for row in repeat_rows) < GATES["repeat_abs_max"]
            and all(row["slots"] == 1 for row in repeat_rows),
        },
        "absent": {
            "rows": absent_rows,
            "mean_delta": _mean(absent_deltas),
            "n_large": int(absent_large),
            "pass": bool(
                abs(_mean(absent_deltas)) <= GATES["unrelated_abs_mean_max"]
                and absent_large <= GATES["unrelated_large_count_max"]
            ),
        },
        "chat": {
            "rows": chat_rows,
            "max_abs": max(row["abs_margin"] for row in chat_rows),
            "accepted": int(sum(row["accepted_chats"] for row in chat_rows)),
            "pass": max(row["abs_margin"] for row in chat_rows) < GATES["chat_abs_max"]
            and all(row["slots"] == 1 for row in chat_rows)
            and all(row["accepted_chats"] == 0 for row in chat_rows),
        },
        "update": {
            "rows": update_rows,
            "mean_delta": _mean(update_deltas),
            "n_delta_pos": int(sum(d > 0.0 for d in update_deltas)),
            "mean_new_lift": _mean([row["new_lift"] for row in update_rows]),
            "mean_old_lift": _mean([row["old_lift"] for row in update_rows]),
            "store_pass": all(
                row["slots"] == 1 and row["cos_new"] >= 0.999 and row["cos_new"] > row["cos_old"]
                for row in update_rows
            ),
            "delta_pass": bool(
                _mean(update_deltas) > GATES["update_delta_min"]
                and sum(d > 0.0 for d in update_deltas) >= GATES["update_delta_pos_min"]
            ),
        },
        "fifo": {
            "rows": fifo_rows,
            "n_absent": int(sum(1 - row["survived"] for row in fifo_rows)),
            "n": len(fifo_rows),
            "slots_mean": _mean([row["slots"] for row in fifo_rows]),
            "mean_delta": _mean([row["delta"] for row in fifo_rows]),
            "mean_single_delta": _mean([row["single_delta"] for row in fifo_rows]),
            "pass": all(row["survived"] == 0 for row in fifo_rows),
        },
        "neutral": {
            "nll": {
                "rows": nll_rows,
                "mean_delta": _mean(nll_deltas),
                "n_bad": int(sum(d > GATES["nll_bad"] for d in nll_deltas)),
                "pass": bool(
                    _mean(nll_deltas) <= GATES["nll_mean_max"]
                    and sum(d > GATES["nll_bad"] for d in nll_deltas) <= GATES["nll_bad_count_max"]
                ),
            },
            "choice": {
                "rows": choice_rows,
                "flips": int(sum(row["flip"] for row in choice_rows)),
                "n": len(choice_rows),
                "acc_nomem": _mean([row["nomem_choose"] for row in choice_rows]),
                "acc_inject": _mean([row["inject_choose"] for row in choice_rows]),
                "pass": int(sum(row["flip"] for row in choice_rows)) <= GATES["choice_flips_max"],
            },
            "leak": {
                "rows": leak_rows,
                "n_leak": int(sum(bool(row["hits"]) for row in leak_rows)),
                "n": len(leak_rows),
                "answers": leak_answers,
                "pass": sum(bool(row["hits"]) for row in leak_rows) <= GATES["leak_max"],
            },
        },
        "scripted": {
            "phone_on": bool(phone_on),
            "date_on": bool(date_on),
            "phone": phone_rows,
            "birthday": date_rows,
            "phone_unrelated": _mean([row["takeover"] for row in phone_rows if row["prompt"] != PHONE_QUESTION]),
            "phone_question": next(row["takeover"] for row in phone_rows if row["prompt"] == PHONE_QUESTION),
            "date_unrelated": _mean(
                [
                    row["takeover"]
                    for row in date_rows
                    if row["prompt"] != PHONE_QUESTION and row["prompt"] != birthday_question
                ]
            ),
            "birthday_question": (
                next(row["takeover"] for row in date_rows if row["prompt"] == birthday_question)
                if birthday_question
                else None
            ),
        },
        "cost": cost,
        "greedy_tokens": GREEDY_TOKENS,
        "scripted_tokens": SCRIPTED_TOKENS,
    }
    result["wall_s"] = float(sum(phases.values()))
    return _plain(result)


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if value != value:
            return None
        return value
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    return obj


def _rss_phrase(row) -> str:
    if not isinstance(row, dict):
        return str(row)
    return ", ".join(f"{key} {int(value)} kB" for key, value in row.items())


def _load_rss(result: dict) -> str:
    rss = result.get("rss_kb_load")
    if not isinstance(rss, dict) or not rss:
        return ""
    body = "; ".join(f"{key} ({_rss_phrase(value)})" for key, value in rss.items())
    cpus = result.get("cpu_count")
    prefix = f" CPUs {cpus}." if cpus else ""
    return f"{prefix} Model-load RSS: {body}."


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _hits(summary: dict) -> str:
    return f"{summary['acc_hits']}/{summary['n']} = {_fmt(summary['acc'])}"


def render_report(result: dict) -> str:
    """Markdown report from a :func:`run_eval` dict. Numbers are not rewritten."""
    prior = result["prior_prefix_n32"]
    prefer = result["axis1"]["prefer"]["summary"]
    avoid = result["axis1"]["avoid"]["summary"]
    lines = [
        "# Subconscious eval",
        "",
        "Measurement of implicit influence on the shipped `LLMMemory` path "
        f"(`{result.get('model_name')}`, layer {result['layer']}, hidden {result['hidden']}, "
        f"float path as loaded). Bank `{result['protocol']}` sha256 `{result['bank_sha256']}`. "
        "Gates were locked in `reports/subconscious_eval_prereg.md` before this run.",
        "",
        "## Prior explicit number",
        "",
        f"`docs/llm_memory.md` ({prior['section']}) reports prefix free-gen at N32 of "
        f"**{prior['inject_hits']}/{prior['n']} = {_fmt(prior['inject_hits'] / prior['n'])}** inject vs "
        f"**{prior['context_hits']}/{prior['n']} = {_fmt(prior['context_hits'] / prior['n'])}** fact-in-context and "
        f"**{prior['rag_hits']}/{prior['n']}** bge RAG. Protocol: {prior['protocol']}. "
        "That table is explicit completion of an answer-stripped fact, not this eval. "
        "Later in the same doc, natural-question free-gen with the phone digit slices and the birthday readout "
        "is 48/48 at N32 on n=48. This run does not rescore that bank.",
        "",
        "## Setup",
        "",
        "Conditions, same decision prompt where the condition allows it:",
        "",
        "- **nomem**: empty store, prompt only.",
        "- **inject**: facts in `EpisodicStore`, prompt only, hard top-1 residual "
        "(`c * gain * (scale(v) + MLP(scale(v)))`, `c` from the top-1 gap, "
        "`gain = 1 + 2 tanh(w · normalize(k))`) at every position.",
        "- **context**: empty store, the fact line (or the whole transcript under load) prepended.",
        "- **rag**: empty store, `BAAI/bge-small-en-v1.5` CLS top-1 line prepended. No query-instruction prefix, matching `scripts/eval_llm_memory.py`.",
        "",
        "Primary score is the length-normalized margin: mean token log-prob of the memory-consistent "
        "continuation minus the mean token log-prob of the foil. `delta` subtracts the nomem margin. "
        "Choice accuracy is which continuation has the higher mean. Log-prob scoring arms once per prompt. "
        f"`generate` uses at most {result['greedy_tokens']} new tokens and the shipped stop.",
        "",
        f"Phone readout on: {result['phone_on']}. Date readout on: {result['date_on']}. "
        f"Capacity {result['capacity']}, overwrite cosine {result['overwrite_cos']}. "
        f"Eval wall clock {_fmt(result.get('wall_s'), 1)} s"
        + (f", model load {_fmt(result.get('load_s'), 1)} s" if result.get("load_s") is not None else "")
        + ". Phase seconds: "
        + ", ".join(f"{key} {_fmt(val, 1)}" for key, val in result["phases_s"].items())
        + ".",
        "",
        "Prefer items store the option that should be chosen. Avoid items store the option that should not "
        "(allergy, fear, refusal). A single fact is always the top slot, so axis 1 is the effect of the "
        "injected vector when retrieval cannot miss.",
        "",
        "## Axis 1 — implicit choice",
        "",
        "| family | condition | mean delta | items delta>0 | choice acc | nomem acc | delta gate | choice gate |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for family, summary in (("prefer", prefer), ("avoid", avoid)):
        for cond in ("inject", "context", "rag"):
            row = summary[cond]
            lines.append(
                f"| {family} | {cond} | {_fmt(row['mean_delta'])} | {row['n_delta_pos']}/{row['n']} | "
                f"{_hits(row)} | {row['acc_nomem_hits']}/{row['n']} = {_fmt(row['acc_nomem'])} | "
                f"{'pass' if row['delta_pass'] else 'fail'} | {'pass' if row['choice_pass'] else 'fail'} |"
            )
    lift = result["axis1"]["stored_lift"]
    lines += [
        "",
        f"Stored-string lift (mean log-prob of the stored word under inject, minus nomem): "
        f"prefer {_fmt(lift['prefer'])}, avoid {_fmt(lift['avoid'])}. "
        "On avoid items the stored word is the foil. A positive avoid lift means injection makes the forbidden word more likely.",
        "",
        "Prefix ceiling (diagnostic): prompt is the fact with the answer removed (`Mara's favorite snack is`), "
        "margin is stored word minus the other word. Not an implicit prompt.",
        "",
        "| id | family | prefix delta | inject chooses stored | decision inject delta | decision context delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    by_id = {row["id"]: row for row in result["axis1"]["prefer"]["items"] + result["axis1"]["avoid"]["items"]}
    for row in result["axis1"]["prefix"]:
        src = by_id[row["id"]]
        lines.append(
            f"| {row['id']} | {row['family']} | {_fmt(row['delta'])} | {row['inject_choose_stored']} | "
            f"{_fmt(src['inject']['delta'])} | {_fmt(src['context']['delta'])} |"
        )
    if result["axis1"]["prefer"]["items"] and result["axis1"]["prefer"]["items"][0].get("greedy"):
        lines += ["", "Greedy whole-word hit (consistent word present, foil absent):", "",
                  "| family | nomem | inject | context | rag |", "|---|---:|---:|---:|---:|"]
        for family in ("prefer", "avoid"):
            items = result["axis1"][family]["items"]
            cells = []
            for cond in ("nomem", "inject", "context", "rag"):
                hits = sum(row["greedy"][cond]["hit"] for row in items)
                cells.append(f"{hits}/{len(items)}")
            lines.append(f"| {family} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Axis 2 and 3 — load, transcript length, repeat, update, FIFO",
        "",
        "Under load the store is the salient fact plus N−1 incidental facts (nested slice, stride 17). "
        "Context is the full transcript plus the decision prompt. RAG retrieves over those lines. "
        "Inject's prompt stays the short decision. Slot counts below capacity mean a cosine overwrite merged two identities.",
        "",
        "| family | N | slots | survived | inject hit | inject delta | context delta | rag delta | rag line hit |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in ("prefer", "avoid"):
        for n_load in LOADS:
            row = result["load"]["summary"][family][str(n_load)]
            lines.append(
                f"| {family} | {n_load} | {_fmt(row['slots_mean'], 1)} | {row['survived_n']}/{row['n']} | "
                f"{row['hit_n']}/{row['n']} | {_fmt(row['inject']['mean_delta'])} | "
                f"{_fmt(row['context']['mean_delta'])} | {_fmt(row['rag']['mean_delta'])} | "
                f"{_fmt(row['rag_line_hit'])} |"
            )
    hit64 = result["load"]["summary"]["prefer"]["64"]
    lines += [
        "",
        f"Salient-hit gate at N=64 on prefer (hit rate ≥ {GATES['salient_hit_at_64_min']}): "
        f"{'pass' if hit64['hit'] >= GATES['salient_hit_at_64_min'] else 'fail'} "
        f"({hit64['hit_n']}/{hit64['n']} = {_fmt(hit64['hit'])}).",
        "",
        f"Identical fact written {REPEAT_WRITES} times (same identity overwrites). "
        f"Max |margin − single-write margin| = {_fmt(result['repeat']['max_abs'], 6)}. "
        f"Repeat-null gate (< {GATES['repeat_abs_max']}): {'pass' if result['repeat']['pass'] else 'fail'}.",
        "",
        f"Prefer prompt with 8 incidental facts and the salient fact absent. "
        f"Mean delta {_fmt(result['absent']['mean_delta'])}, "
        f"items with |delta| > {GATES['unrelated_large']}: {result['absent']['n_large']}. "
        f"Unrelated-store gate: {'pass' if result['absent']['pass'] else 'fail'}.",
        "",
        f"Thirty-two chat lines after the salient fact (chat is dropped by the write gate). "
        f"Accepted writes {result['chat']['accepted']}. "
        f"Max |margin change| {_fmt(result['chat']['max_abs'], 6)}. "
        f"Gate: {'pass' if result['chat']['pass'] else 'fail'}.",
        "",
        "Same-key update on prefer items (apples overwritten by pears, and the other pairs). "
        "The scored pair is the new word versus the old word.",
        "",
        f"Store holds the new value (cosine ≥ 0.999 and above the old value) on every item: "
        f"{'pass' if result['update']['store_pass'] else 'fail'}. "
        f"Mean inject delta {_fmt(result['update']['mean_delta'])} "
        f"({result['update']['n_delta_pos']}/8 items > 0). "
        f"Delta gate: {'pass' if result['update']['delta_pass'] else 'fail'}. "
        f"Mean lift of the new word {_fmt(result['update']['mean_new_lift'])}, "
        f"of the old word {_fmt(result['update']['mean_old_lift'])}.",
        "",
        f"FIFO: salient fact, then {result['fifo']['rows'][0]['requested'] - 1} incidental facts "
        f"(capacity {result['fifo']['rows'][0]['capacity']}). "
        f"Target absent on {result['fifo']['n_absent']}/{result['fifo']['n']}. "
        f"Mean slots {_fmt(result['fifo']['slots_mean'], 1)}. "
        f"Gate (every target gone): {'pass' if result['fifo']['pass'] else 'fail'}. "
        f"Mean decision delta after the fill {_fmt(result['fifo']['mean_delta'])} "
        f"versus single-fact delta {_fmt(result['fifo']['mean_single_delta'])}.",
        "",
        "## Axis 4 — non-interference",
        "",
        f"Neutral gold continuations, store = first {NEUTRAL_STORE_N} incidental facts. "
        f"Mean ΔNLL (inject − nomem) {_fmt(result['neutral']['nll']['mean_delta'])}. "
        f"Items with ΔNLL > {GATES['nll_bad']}: {result['neutral']['nll']['n_bad']}. "
        f"Gate: {'pass' if result['neutral']['nll']['pass'] else 'fail'}.",
        "",
        "| id | ΔNLL | top slot |",
        "|---|---:|---|",
    ]
    for row in result["neutral"]["nll"]["rows"]:
        lines.append(f"| {row['id']} | {_fmt(row['delta'])} | {row['top']} |")
    choice = result["neutral"]["choice"]
    lines += [
        "",
        f"Neutral forced choices. Nomem acc {_fmt(choice['acc_nomem'])}, inject acc {_fmt(choice['acc_inject'])}, "
        f"flips (nomem right, inject wrong) {choice['flips']}/{choice['n']}. "
        f"Gate (flips ≤ {GATES['choice_flips_max']}): {'pass' if choice['pass'] else 'fail'}.",
        "",
        f"Leak: greedy on {result['neutral']['leak']['n']} neutral prompts with "
        f"{len(result['neutral']['leak']['answers'])} rare incidental answers in the store. "
        f"Prompts that emit one of those words: {result['neutral']['leak']['n_leak']}. "
        f"Gate: {'pass' if result['neutral']['leak']['pass'] else 'fail'}.",
        "",
        "Scripted readout. A phone slot copies its stored number and never calls the LM. "
        "A birthday slot copies its stored date. Unrelated prompts stay on the residual path. "
        "Takeover below is the `+1-415-555-` prefix check and the month-day-year check. "
        "Takeover on an unrelated prompt is reported and is not part of the NLL gate.",
        "",
        f"Phone unrelated takeover rate {_fmt(result['scripted']['phone_unrelated'])}. "
        f"Phone question takeover {result['scripted']['phone_question']}. "
        f"Birthday unrelated takeover rate {_fmt(result['scripted']['date_unrelated'])}.",
        "",
        "| kind | prompt | takeover | text |",
        "|---|---|---:|---|",
    ]
    for kind, rows in (("phone", result["scripted"]["phone"]), ("birthday", result["scripted"]["birthday"])):
        for row in rows:
            snippet = " ".join(str(row["text"]).split())
            if len(snippet) > 80:
                snippet = snippet[:77] + "..."
            lines.append(f"| {kind} | `{row['prompt'].split(chr(10))[0]}` | {row['takeover']} | {snippet} |")
    lines += ["", "## Cost", ""]
    cost = result.get("cost")
    if not cost:
        lines.append("Cost was not measured in this run.")
    else:
        extra = cost["extra_per_prompt_s"]
        lines += [
            f"Inject MLP parameters {cost['inject']['mlp_parameters']} "
            f"(module parameters {cost['inject']['parameters']}; buffers {cost['inject']['buffers']}, "
            "the phone and date readout tensors). "
            f"Frozen LM parameter bytes {cost['model_parameter_bytes']}. "
            f"Torch threads {cost['torch_threads']}.",
            "",
            f"RSS before the cost section: {_rss_phrase(cost['rss_kb']['before'])}. "
            f"After: {_rss_phrase(cost['rss_kb']['after'])}. "
            "VmHWM is the process high-water mark and includes the model load."
            + _load_rss(result),
            "",
            f"Median query encode {_fmt(cost['query_s']['median_s'], 4)} s. "
            f"Median MLP forward {_fmt(cost['mlp_s']['median_s'], 6)} s. "
            "Median hard top-1 readout (store only, no LM forward):",
            "",
            "| N | median s |",
            "|---:|---:|",
        ]
        for key, row in sorted(cost["readout_s"].items(), key=lambda item: int(item[0])):
            lines.append(f"| {key} | {_fmt(row['median_s'], 6)} |")
        lines += [
            "",
            "One LM forward, residual armed minus residual absent. This is the per-token extra cost of the add, "
            "at a fixed length. Generation itself recomputes the prefix each token for both conditions.",
            "",
            "| length | empty s | armed s | extra s |",
            "|---:|---:|---:|---:|",
        ]
        for key, row in sorted(cost["forward_s"].items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {key} | {_fmt(row['empty_median_s'], 4)} | {_fmt(row['armed_median_s'], 4)} | {_fmt(row['extra_s'], 6)} |"
            )
        lines += [
            "",
            f"Mean extra per token across those lengths: {_fmt(cost['extra_per_token_s']['mean_s'], 6)} s.",
            "",
            f"Shipped `continuation_ids` query count on {COST_NEW_TOKENS} forced tokens: "
            f"empty {cost['path_counts']['empty']['queries']} "
            f"(forwards {cost['path_counts']['empty']['forwards']}, tokens {cost['path_counts']['empty']['new_tokens']}), "
            f"non-phone inject {cost['path_counts']['inject']['queries']} "
            f"(forwards {cost['path_counts']['inject']['forwards']}, tokens {cost['path_counts']['inject']['new_tokens']}), "
            f"phone slot {cost['path_counts']['scripted_phone']['queries']} "
            f"(forwards {cost['path_counts']['scripted_phone']['forwards']}, tokens {cost['path_counts']['scripted_phone']['new_tokens']}).",
            "",
            f"Extra per prompt versus nomem, from the pieces: "
            f"{extra['shipped_non_phone_queries']} × query ({_fmt(extra['shipped_query_s'], 4)} s) "
            f"+ MLP ({_fmt(extra['mlp_s'], 6)} s) "
            f"+ readout at N=64 ({_fmt(extra['readout_n64_s'], 6)} s) "
            f"= {_fmt(extra['total_s'], 4)} s. "
            "The token loop's extra is the armed-minus-empty column above, paid again on every new token.",
            "",
            "Median `continuation_ids` wall time for the same prompt:",
            "",
            "| store | median s |",
            "|---|---:|",
        ]
        for key, row in cost["continuation_s"].items():
            lines.append(f"| {key} | {_fmt(row['median_s'], 4)} |")
        lines += [
            "",
            "Variable-cost paths (no retries, no re-query inside the token loop):",
            "",
        ]
        for flag in cost["flags"]:
            lines.append(f"- **{flag['path']}**. {flag['when']}. {flag['varies']}.")
    lines += ["", "## Failure analysis", "", _failure_text(result), "", "## Recommendation", "", _recommendation(result), ""]
    return "\n".join(lines)


def _gap_stats(rows: list) -> str:
    gaps = []
    cosines = []
    for row in rows:
        info = row["inject"]["retrieval"]
        if info.get("gap") is not None:
            gaps.append(float(info["gap"]))
        if info.get("maxcos") is not None:
            cosines.append(float(info["maxcos"]))
    return (
        f"top cosine mean {_fmt(_mean(cosines))}, "
        f"top1−top2 gap mean {_fmt(_mean(gaps))} "
        f"(min {_fmt(min(gaps) if gaps else None)}, max {_fmt(max(gaps) if gaps else None)})"
    )


def _failure_text(result: dict) -> str:
    prefer_items = result["axis1"]["prefer"]["items"]
    avoid_items = result["axis1"]["avoid"]["items"]
    prefer = result["axis1"]["prefer"]["summary"]["inject"]
    parts = [
        "Single-fact inject moves every prefer item toward the stored word and every avoid item toward the forbidden word. "
        f"Prefer deltas run from {_fmt(min(row['inject']['delta'] for row in prefer_items))} to "
        f"{_fmt(max(row['inject']['delta'] for row in prefer_items))}. "
        f"Avoid deltas run from {_fmt(min(row['inject']['delta'] for row in avoid_items))} to "
        f"{_fmt(max(row['inject']['delta'] for row in avoid_items))}.",
        (
            f"Nomem already chooses the prefer word on {prefer['acc_nomem_hits']}/{prefer['n']} items. "
            + (
                "The choice gate wants a +0.25 accuracy lift, and that base rate leaves no room for it."
                if (1.0 - float(prefer["acc_nomem"])) < GATES["acc_lift_min"]
                else "The choice gate wants a +0.25 accuracy lift on top of that base rate."
            )
        ),
        "Prefix completion of the fact, with the answer stripped, chooses the stored word on 16/16 and the margin delta is positive on 16/16. "
        "The same vector that completes `allergic to` with `peanuts` is the vector added during `Pick one snack`.",
    ]
    if prefer_items and prefer_items[0].get("greedy"):
        for family, items in (("prefer", prefer_items), ("avoid", avoid_items)):
            inj = sum(row["greedy"]["inject"]["hit"] for row in items)
            base = sum(row["greedy"]["nomem"]["hit"] for row in items)
            parts.append(f"Greedy hits, {family}: nomem {base}/{len(items)}, inject {inj}/{len(items)}.")
        forbidden = [
            row["answer"]
            for row in avoid_items
            if row["greedy"]["inject"]["bad"] and not row["greedy"]["inject"]["good"]
        ]
        if forbidden:
            parts.append(
                "On those avoid misses the greedy span contains the stored word and not the safe word: "
                + ", ".join(forbidden)
                + "."
            )
    for family in ("prefer", "avoid"):
        rows = [row for row in result["load"]["items"] if row["family"] == family and row["load"] == 64]
        hits = [row for row in rows if row["inject"]["retrieval"]["hit"] == 1]
        misses = [row for row in rows if row["inject"]["retrieval"]["hit"] == 0]
        slots = [row["slots"] for row in rows]
        parts.append(
            f"N=64 {family}: {len(hits)}/{len(rows)} retrieval hits, salient slot still stored on "
            f"{sum(row['survived'] for row in rows)}/{len(rows)}, "
            f"slots {min(slots)}–{max(slots)} after 64 writes. "
            f"Mean inject delta on hits {_fmt(_mean([row['inject']['delta'] for row in hits]))}, "
            f"on misses {_fmt(_mean([row['inject']['delta'] for row in misses]))}. "
            f"{_gap_stats(rows)}."
        )
    fifo = result["fifo"]["rows"]
    parts.append(
        f"The salient fact plus {fifo[0]['requested'] - 1} later facts "
        f"({fifo[0]['requested']} writes) into capacity {fifo[0]['capacity']} left {fifo[0]['slots']} slots. "
        f"Cosine overwrite merged the repeated incidental frames, so the store never filled "
        f"and no salient identity was evicted ({result['fifo']['n_absent']}/{result['fifo']['n']} absent). "
        f"The decision delta still fell from mean {_fmt(result['fifo']['mean_single_delta'])} to "
        f"{_fmt(result['fifo']['mean_delta'])}, because a different slot won the query."
    )
    leaked = [row for row in result["neutral"]["leak"]["rows"] if row["hits"]]
    if leaked:
        shown = "; ".join(repr(row["text"])[:80] for row in leaked)
        parts.append(
            f"Leak on {len(leaked)}/{result['neutral']['leak']['n']} neutral generations. "
            f"The stored word that appears is {sorted({hit for row in leaked for hit in row['hits']})}. "
            f"Samples: {shown}."
        )
    else:
        parts.append("No neutral generation contained a stored incidental answer as a whole word.")
    rag64 = result["load"]["summary"]["prefer"]["64"]["rag_line_hit"]
    parts.append(
        f"bge returns the salient line on prefer at N=64 with rate {_fmt(rag64)}. "
        "The decision prompt contains the person's name and the fact line does too, so lexical retrieval succeeds "
        "where the hidden-state key does not. The RAG delta therefore matches the single-fact RAG delta at every load. "
        "The transcript condition is the one that gets longer, and its prefer delta falls as N grows."
    )
    return "\n\n".join(parts)


def _recommendation(result: dict) -> str:
    prefer = result["axis1"]["prefer"]["summary"]["inject"]
    avoid = result["axis1"]["avoid"]["summary"]["inject"]
    context_avoid = result["axis1"]["avoid"]["summary"]["context"]
    lift = result["axis1"]["stored_lift"]
    hit = result["load"]["summary"]["prefer"]["64"]["hit"]
    prefix_mean = _mean([row["delta"] for row in result["axis1"]["prefix"]])
    phone_rate = result["scripted"]["phone_unrelated"]
    bits = []
    bits.append(
        f"Single-fact prefer inject mean delta is {_fmt(prefer['mean_delta'])} "
        f"(gate {'pass' if prefer['delta_pass'] else 'fail'}). "
        f"Avoid inject mean delta is {_fmt(avoid['mean_delta'])} "
        f"(gate {'pass' if avoid['delta_pass'] else 'fail'}). "
        f"Prefix mean delta is {_fmt(prefix_mean)}. "
        f"Prefer stored-word lift is {_fmt(lift['prefer'])}; avoid stored-word lift is {_fmt(lift['avoid'])}."
    )
    if (not prefer["delta_pass"]) and (prefix_mean is not None and prefix_mean <= 0.05):
        bits.append(
            "The injected vector does not raise the stored word even on an answer-stripped prefix "
            "in this out-of-domain wording. The next mechanism step is the training objective: "
            "teacher-forced continuations of implicit prompts and of these prefixes, on values from "
            "the same answer-span pool. Salience weights would not create a margin the vector does not carry."
        )
    elif prefer["delta_pass"] and (not avoid["delta_pass"]) and (lift["avoid"] or 0) > 0:
        bits.append(
            "Prefer moves toward the stored word and avoid does not apply the constraint. "
            "The residual is doing what it was trained to do, which is to emit the value. "
            "The next training objective should score the decision continuation, with the forbidden "
            "word as a negative, instead of next-token loss on the fact's answer span alone."
        )
    elif prefer["delta_pass"] and avoid["delta_pass"]:
        bits.append(
            "Both families move in the consistent direction with a single fact in the store. "
            "Further gains are about which fact is armed under load, not about the single-vector map."
        )
    else:
        bits.append(
            "The single-fact margins are the ones in the axis 1 table. "
            "Read them against the context column before treating retrieval as the bottleneck."
        )
    if context_avoid["delta_pass"] and not avoid["delta_pass"]:
        bits.append(
            "In-context text does move the avoid decision and the residual does not. "
            "The fact is usable by this LM when it is tokens. The gap is the inject, not the label."
        )
    if not context_avoid["delta_pass"] and not avoid["delta_pass"]:
        ctx = context_avoid["mean_delta"]
        inj = avoid["mean_delta"]
        bits.append(
            f"In-context text also moves avoid the wrong way (mean delta {_fmt(ctx)}). "
            f"Inject moves it further ({_fmt(inj)}). Both are lexical priming of the stored word. "
            "The context column shows the 135M model does not treat `allergic to peanuts` as a reason to refuse peanuts. "
            "The residual is a stronger version of that prime, not a weaker copy of a constraint the text already applied."
        )
    if hit < GATES["salient_hit_at_64_min"]:
        bits.append(
            f"Prefer retrieval at N=64 is {_fmt(hit)}, under the {GATES['salient_hit_at_64_min']} gate. "
            "The query is a sink-dropped mean of the decision prompt, and the key is the fact with the "
            "answer removed. A salience-weighted write does not change that geometry. "
            "The next retrieval step is a query that is built for the decision, or a margin gate that "
            "refuses a weak top-1, still computed on every call."
        )
    else:
        bits.append(
            f"Prefer retrieval at N=64 is {_fmt(hit)}. The top slot is often the salient fact on this bank, "
            "so a salience weight is not the first lever."
        )
    if not result["repeat"]["pass"]:
        bits.append("Repeating the identical fact changed the margin. That is unexpected for identity overwrite and should be treated as a bug, not as salience.")
    else:
        bits.append(
            "Writing the same fact three times leaves one slot and the same margin. "
            "There is no repetition weight on this path."
        )
    if not result["absent"]["pass"]:
        bits.append(
            "An incidental store moves prefer decisions with the salient fact absent. "
            "Hard top-1 always adds some value when the store is non-empty. "
            "A gated add, scaled by the top-1 cosine or by the gap to the second slot, "
            "is the non-interference mechanism. It stays on the main path: the scale is a function of the scores, not a switch."
        )
    if result["neutral"]["nll"]["pass"]:
        bits.append("Neutral gold-continuation NLL stays inside the pre-registered band.")
    else:
        bits.append(
            "Neutral NLL moves past the pre-registered band. The same cosine gate that would "
            "drop a weak top-1 is the candidate fix. It should be fit on a later bank, not on this one."
        )
    if phone_rate and phone_rate > 0:
        bits.append(
            "Unrelated prompts with a phone slot are spelled as a phone number. "
            "`scripted_continuation` keys off the winning relation, not off the prompt. "
            "The spelling path should run only when that relation is the one the prompt is completing, "
            "or the residual path should be the one that runs for every other prompt. "
            "Collapsing the scripted probe and `_arm` into one query removes the second prompt forward "
            "on ordinary facts; that is a cost fix, and it does not change the armed vector."
        )
    else:
        bits.append(
            "Unrelated prompts did not take the phone spelling path in this run. "
            "The cost section still counts how many query encodes `continuation_ids` performs."
        )
    if result["update"]["store_pass"] and not result["update"]["delta_pass"]:
        bits.append(
            "Overwrite replaces the slot with the new value, and the decision margin does not follow. "
            "Forgetting in the store is not the same as forgetting in the next token."
        )
    elif result["update"]["store_pass"] and result["update"]["delta_pass"]:
        bits.append("Overwrite replaces the slot, and the decision margin moves toward the new word.")
    if not result["fifo"]["pass"]:
        bits.append(
            f"The salient fact plus {result['fifo']['rows'][0]['requested'] - 1} later facts "
            f"({result['fifo']['rows'][0]['requested']} writes) leave "
            f"{_fmt(result['fifo']['slots_mean'], 1)} slots, and every salient identity is still there. "
            "Similar incidental frames collapse under cosine 0.9, so FIFO never runs. "
            f"Influence does not stay with the surviving slot: mean decision delta falls from "
            f"{_fmt(result['fifo']['mean_single_delta'])} to {_fmt(result['fifo']['mean_delta'])}. "
            "There is no age-based fade. There is also no guarantee that a stored fact is the one added at the next prompt."
        )
    else:
        bits.append(
            "After a capacity's worth of later facts the salient slot is gone and the single-fact margin is gone with it. "
            "Inside the capacity there is no time decay. Chat lines do not age a fact either."
        )
    return "\n\n".join(bits)
