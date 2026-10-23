"""Score one surprise bank through ``LLMMemory.remember``.

The write gate is the production threshold. This module does not choose it
and does not import the held-out bank.
"""
from __future__ import annotations

import time

from tidal.llm_memory import answer_surprise
from tidal.subconscious_data import word_hit
from tidal.subconscious_eval import (
    GREEDY_TOKENS,
    SCRIPTED_TOKENS,
    date_takeover,
    inject_census,
    phone_takeover,
    score_one,
    score_pair,
)


def _mean(xs):
    vals = [float(x) for x in xs]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _forward_count(mem, fn):
    calls = {"n": 0}

    def _hook(_module, _inputs, _output):
        calls["n"] += 1

    handle = mem.model.register_forward_hook(_hook)
    try:
        result = fn()
    finally:
        handle.remove()
        mem._delta = None
    return result, int(calls["n"])


def measure_gate_cost(mem, bank) -> dict:
    """Forward counts and the cost of reading logprobs the value forward already returned."""
    census = inject_census(mem.inject)
    low = bank.LOW[0]
    high = bank.HIGH[0]
    chat = bank.CHATS[0]
    hidden, enc, logits = mem._run_hidden(low["text"])
    mask = mem._answer_mask(low["text"], enc, low["answer"], int(hidden.shape[0]))
    ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]

    def once():
        return answer_surprise(logits, ids, mask)

    once()
    samples = []
    for _ in range(30):
        t0 = time.perf_counter()
        once()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    chat_ok, chat_n = _forward_count(
        mem,
        lambda: mem.remember(chat["text"], answer=chat["answer"], kind="chat", name="chat", rel="chat"),
    )
    mem.store.clear()
    low_ok, low_n = _forward_count(mem, lambda: mem.remember(low))
    low_surprise = mem.last_surprise
    mem.store.clear()
    high_ok, high_n = _forward_count(mem, lambda: mem.remember(high))
    high_surprise = mem.last_surprise
    mem.store.clear()
    return {
        "parameters_new": 0,
        "mlp_parameters": census["mlp_parameters"],
        "polarity_parameters": int(dict(census["named_parameters"]).get("polarity", 0)),
        "parameters": census["parameters"],
        "buffers": census["buffers"],
        "answer_surprise_median_s": float(samples[len(samples) // 2]),
        "answer_surprise_rows": int(sum(1 for flag in mask if flag)),
        "chat_written": bool(chat_ok),
        "chat_forwards": chat_n,
        "below_threshold_written": bool(low_ok),
        "below_threshold_forwards": low_n,
        "below_threshold_surprise": low_surprise,
        "above_threshold_written": bool(high_ok),
        "above_threshold_forwards": high_n,
        "above_threshold_surprise": high_surprise,
    }


def _item_score(mem, item) -> dict:
    mem.store.clear()
    nomem = score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"])
    mem.store.clear()
    written = bool(mem.remember(item))
    surprise = mem.last_surprise
    if not written:
        return {
            "id": item["id"],
            "family": item["family"],
            "written": False,
            "surprise": surprise,
            "delta": 0.0,
            "stored_lift": 0.0,
            "choose": nomem["choose"],
            "nomem_choose": nomem["choose"],
            "nomem_margin": nomem["margin"],
            "inject_margin": nomem["margin"],
        }
    injected = score_pair(mem, item["prompt"], item["good"], item["bad"], name=item["name"], rel=item["rel"])
    stored_side = "bad_mean" if item["family"] == "avoid" else "good_mean"
    return {
        "id": item["id"],
        "family": item["family"],
        "written": True,
        "surprise": surprise,
        "delta": float(injected["margin"] - nomem["margin"]),
        "stored_lift": float(injected[stored_side] - nomem[stored_side]),
        "choose": injected["choose"],
        "nomem_choose": nomem["choose"],
        "nomem_margin": nomem["margin"],
        "inject_margin": injected["margin"],
    }


def _family(rows) -> dict:
    deltas = [float(row["delta"]) for row in rows]
    return {
        "n": len(rows),
        "written": int(sum(1 for row in rows if row["written"])),
        "mean_delta": _mean(deltas),
        "n_delta_pos": int(sum(d > 0.0 for d in deltas)),
        "mean_surprise": _mean([row["surprise"] for row in rows if row["surprise"] is not None]),
        "min_surprise": min((float(row["surprise"]) for row in rows if row["surprise"] is not None), default=None),
        "max_surprise": max((float(row["surprise"]) for row in rows if row["surprise"] is not None), default=None),
    }


def score_bank(mem, bank, *, progress=None) -> dict:
    """One pass. Writes go through ``remember``, so a low score adds nothing."""

    def say(msg):
        if progress is not None:
            progress(msg)

    t0 = time.perf_counter()
    phases = {}
    say("influence")
    low = [_item_score(mem, item) for item in bank.LOW]
    high = [_item_score(mem, item) for item in bank.HIGH]
    prefer = [_item_score(mem, item) for item in bank.PREFER]
    avoid = [_item_score(mem, item) for item in bank.AVOID]
    phases["influence"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    say("neutral")
    mem.store.clear()
    nll_base = [score_one(mem, item["prompt"], item["continuation"]) for item in bank.NEUTRAL_NLL]
    choice_base = [score_pair(mem, item["prompt"], item["good"], item["bad"]) for item in bank.NEUTRAL_CHOICE]
    incidental = bank.incidental_facts()[: int(bank.NEUTRAL_STORE_N)]
    inc_written = 0
    inc_surprise = []
    mem.store.clear()
    for fact in incidental:
        inc_written += int(bool(mem.remember(fact)))
        inc_surprise.append(mem.last_surprise)
    inc_slots = len(mem.store)
    nll_rows = []
    for item, base in zip(bank.NEUTRAL_NLL, nll_base):
        armed = score_one(mem, item["prompt"], item["continuation"])
        nll_rows.append(
            {
                "id": item["id"],
                "delta": float(armed["nll"] - base["nll"]),
                "nll_nomem": base["nll"],
                "nll_inject": armed["nll"],
            }
        )
    choice_rows = []
    for item, base in zip(bank.NEUTRAL_CHOICE, choice_base):
        armed = score_pair(mem, item["prompt"], item["good"], item["bad"])
        choice_rows.append(
            {
                "id": item["id"],
                "nomem_choose": base["choose"],
                "inject_choose": armed["choose"],
                "delta": float(armed["margin"] - base["margin"]),
                "flip": int(base["choose"] == 1 and armed["choose"] == 0),
            }
        )
    phases["neutral"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    say("leak")
    leak_src = bank.incidental_facts()[: int(bank.LEAK_STORE_N)]
    mem.store.clear()
    leak_written = 0
    for fact in leak_src:
        leak_written += int(bool(mem.remember(fact)))
    leak_answers = [row["answer"] for row in leak_src]
    leak_rows = []
    for prompt in bank.NEUTRAL_GEN:
        text = mem.generate(prompt, max_new_tokens=GREEDY_TOKENS)
        hits = [answer for answer in leak_answers if word_hit(text, answer)]
        leak_rows.append({"prompt": prompt, "text": text, "hits": hits})
    phases["leak"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    say("scripted")
    script_prompts = list(bank.NEUTRAL_GEN[:4]) + [bank.PHONE_QUESTION, bank.BIRTHDAY_QUESTION]
    mem.store.clear()
    phone_written = bool(mem.remember(bank.PHONE_FACT))
    phone_surprise = mem.last_surprise
    phone_rows = []
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        phone_rows.append({"prompt": prompt, "text": text, "takeover": int(phone_takeover(text))})
    mem.store.clear()
    birthday_written = bool(mem.remember(bank.BIRTHDAY_FACT))
    birthday_surprise = mem.last_surprise
    date_rows = []
    for prompt in script_prompts:
        text = mem.generate(prompt, max_new_tokens=SCRIPTED_TOKENS)
        date_rows.append({"prompt": prompt, "text": text, "takeover": int(date_takeover(text))})
    phases["scripted"] = time.perf_counter() - t3
    mem.store.clear()

    nll_deltas = [row["delta"] for row in nll_rows]
    phone_unrelated = _mean([row["takeover"] for row in phone_rows if row["prompt"] != bank.PHONE_QUESTION])
    phone_question = next(row["takeover"] for row in phone_rows if row["prompt"] == bank.PHONE_QUESTION)
    date_unrelated = _mean(
        [
            row["takeover"]
            for row in date_rows
            if row["prompt"] != bank.PHONE_QUESTION and row["prompt"] != bank.BIRTHDAY_QUESTION
        ]
    )
    birthday_question = next(row["takeover"] for row in date_rows if row["prompt"] == bank.BIRTHDAY_QUESTION)
    prefer_sum = _family(prefer)
    avoid_sum = _family(avoid)
    high_sum = _family(high)
    low_sum = _family(low)
    forbidden_down = int(sum(float(row["stored_lift"]) < 0.0 for row in avoid))
    return {
        "protocol": bank.PROTOCOL,
        "bank_sha256": bank.sha256(),
        "bank_kind": bank.kind,
        "layer": int(mem.layer),
        "hidden": int(mem.hidden),
        "model_name": mem.model_name,
        "phases_s": phases,
        "low": {"items": low, "summary": low_sum},
        "high": {"items": high, "summary": high_sum},
        "prefer": {"items": prefer, "summary": prefer_sum},
        "avoid": {"items": avoid, "summary": avoid_sum},
        "forbidden_down": forbidden_down,
        "forbidden_lift": _mean([row["stored_lift"] for row in avoid]),
        "prefer_lift": _mean([row["stored_lift"] for row in prefer]),
        "neutral": {
            "nll": {
                "rows": nll_rows,
                "mean_delta": _mean(nll_deltas),
                "n_bad": int(sum(d > 0.5 for d in nll_deltas)),
            },
            "choice": {
                "rows": choice_rows,
                "flips": int(sum(row["flip"] for row in choice_rows)),
                "n": len(choice_rows),
            },
            "incidental_written": inc_written,
            "incidental_n": len(incidental),
            "incidental_slots": inc_slots,
            "incidental_surprise_min": min(inc_surprise) if inc_surprise else None,
            "leak_written": leak_written,
            "leak": {
                "rows": leak_rows,
                "n_leak": int(sum(bool(row["hits"]) for row in leak_rows)),
                "n": len(leak_rows),
            },
        },
        "scripted": {
            "phone_written": phone_written,
            "phone_surprise": phone_surprise,
            "birthday_written": birthday_written,
            "birthday_surprise": birthday_surprise,
            "phone_unrelated": phone_unrelated,
            "phone_question": phone_question,
            "date_unrelated": date_unrelated,
            "birthday_question": birthday_question,
            "phone": phone_rows,
            "birthday": date_rows,
        },
        "wall_s": float(sum(phases.values())),
    }


def gate_rows(result, gates, *, use_means: bool):
    """Pass/fail rows. Means are held-out gates; fit selection leaves them off."""
    high = result["high"]["summary"]
    low = result["low"]["summary"]
    prefer = result["prefer"]["summary"]
    avoid = result["avoid"]["summary"]
    nll = result["neutral"]["nll"]
    choice = result["neutral"]["choice"]
    leak = result["neutral"]["leak"]
    scripted = result["scripted"]
    rows = [
        (
            "high written",
            int(high["written"]) >= gates["high_written_min"],
            f"{high['written']}/{high['n']}",
        ),
        (
            "low written",
            int(low["written"]) <= gates["low_written_max"],
            f"{low['written']}/{low['n']}",
        ),
        (
            "high influence",
            float(high["mean_delta"]) > float(low["mean_delta"]),
            f"high {_fmt(high['mean_delta'])} vs low {_fmt(low['mean_delta'])}",
        ),
        (
            "prefer margin",
            int(prefer["n_delta_pos"]) >= gates["prefer_delta_pos_min"]
            and (not use_means or float(prefer["mean_delta"]) >= gates["prefer_mean_min"]),
            f"{prefer['n_delta_pos']}/{prefer['n']} > 0, mean {_fmt(prefer['mean_delta'])}",
        ),
        (
            "avoid margin",
            int(avoid["n_delta_pos"]) >= gates["avoid_delta_pos_min"]
            and (not use_means or float(avoid["mean_delta"]) >= gates["avoid_mean_min"]),
            f"{avoid['n_delta_pos']}/{avoid['n']} > 0, mean {_fmt(avoid['mean_delta'])}",
        ),
        (
            "forbidden logprob down",
            (not use_means) or int(result["forbidden_down"]) >= gates["forbidden_down_min"],
            f"{result['forbidden_down']}/{avoid['n']} down, mean stored lift {_fmt(result['forbidden_lift'])}",
        ),
        (
            "neutral NLL",
            abs(float(nll["mean_delta"])) <= gates["nll_abs_mean_max"]
            and int(nll["n_bad"]) <= gates["nll_bad_count_max"],
            f"mean {_fmt(nll['mean_delta'])}, above 0.5: {nll['n_bad']}",
        ),
        (
            "choice flips",
            int(choice["flips"]) <= gates["choice_flips_max"],
            f"{choice['flips']}/{choice['n']}",
        ),
        ("leak", int(leak["n_leak"]) <= gates["leak_max"], f"{leak['n_leak']}/{leak['n']}"),
        (
            "phone unrelated",
            float(scripted["phone_unrelated"]) <= gates["phone_unrelated_max"]
            and scripted["phone_written"],
            _fmt(scripted["phone_unrelated"]),
        ),
        (
            "birthday unrelated",
            float(scripted["date_unrelated"]) <= gates["birthday_unrelated_max"]
            and scripted["birthday_written"],
            _fmt(scripted["date_unrelated"]),
        ),
        (
            "phone question",
            scripted["phone_question"] == gates["phone_question"] and scripted["phone_written"],
            str(scripted["phone_question"]),
        ),
        (
            "birthday question",
            scripted["birthday_question"] == gates["birthday_question"] and scripted["birthday_written"],
            str(scripted["birthday_question"]),
        ),
        (
            "incidental written",
            int(result["neutral"]["incidental_written"]) == int(result["neutral"]["incidental_n"])
            and int(result["neutral"]["incidental_slots"]) == int(result["neutral"]["incidental_n"]),
            f"{result['neutral']['incidental_written']}/{result['neutral']['incidental_n']} "
            f"slots {result['neutral']['incidental_slots']}",
        ),
    ]
    return rows


def _fmt(value, digits=3):
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def passes(rows) -> bool:
    return all(ok for _name, ok, _detail in rows)
