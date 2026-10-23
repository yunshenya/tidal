"""Choose the surprise threshold on the fit bank only.

Does not score the held-out bank. Writes the preregistration file only when
the source constant already equals the chosen threshold and the fit gates pass.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import tidal.llm_memory as llm_memory
from tidal.llm_memory import LLMMemory
from tidal.llm_memory_data import names
from tidal.subconscious_surprise_data import (
    DIRECTION_FIT_SHA,
    DIRECTION_HELD_SHA,
    DIRECTION_TRAIN_SHA,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    SURPRISE_GATES,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_surprise_score import gate_rows, measure_gate_cost, passes, score_bank


def _surprises(mem, rows):
    out = []
    for row in rows:
        _value, surprise = mem._value_and_surprise(row["text"], row["answer"])
        out.append(float(surprise))
    return out


def _green_floor(mem):
    vals = []
    for name in names("train"):
        text = f"{name}'s favorite color is green."
        _value, surprise = mem._value_and_surprise(text, "green")
        vals.append(float(surprise))
    return {"min": min(vals), "max": max(vals), "n": len(vals)}


def _eligible(measured, green_min):
    """Highest tau on the 0.25 grid that still writes the facts the floors need."""
    found = []
    for step in range(1, 40):
        tau = step / 4
        if tau > float(green_min) + 1e-9:
            continue
        low_written = sum(value >= tau for value in measured["low"])
        high_written = sum(value >= tau for value in measured["high"])
        prefer_written = sum(value >= tau for value in measured["prefer"])
        avoid_written = sum(value >= tau for value in measured["avoid"])
        if low_written > 2 or high_written < 6 or prefer_written < 6 or avoid_written < 6:
            continue
        if measured["phone"] < tau or measured["birthday"] < tau:
            continue
        if any(value < tau for value in measured["incidental"]):
            continue
        found.append(tau)
    return found


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_prereg(result, cost, green, tau):
    rows = gate_rows(result, SURPRISE_GATES, use_means=False)
    lines = [
        "# Surprise preregistration",
        "",
        "Surprise-gated writes. The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run "
        "only when the winning relation is the one the prompt is completing. "
        "A continuation still encodes the query once.",
        "",
        "## Banks",
        "",
        f"- fit `{bank_sha256('fit')}`",
        f"- held `{bank_sha256('held')}`",
        f"- frozen `{FROZEN_SHA}`",
        f"- #39 fit `{FIT39_SHA}`",
        f"- #39 held `{HELD39_SHA}`",
        f"- #40 held `{GATE_SHA}`",
        f"- #41 train `{DIRECTION_TRAIN_SHA}`",
        f"- #41 fit `{DIRECTION_FIT_SHA}`",
        f"- #41 held `{DIRECTION_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Write score",
        "",
        "Salience is 1 for a fact and 0 for chat and noise. "
        "Surprise is the total NLL, in nats, of the answer tokens under the logits "
        "the value forward already returned. "
        "`write_score = salience * max(0, surprise)`. "
        "A fact is written only when that score is at least `surprise_tau`. "
        "Chat and noise return before any forward. "
        "A fact below the threshold pays the value forward and skips the key forward. "
        "There is no third forward and no second memory layer.",
        "",
        f"- surprise_tau: {_fmt(tau, 2)}",
        f"- green-color floor on the memory train names: min {_fmt(green['min'])}, "
        f"max {_fmt(green['max'])}, n {green['n']}. Tau is at most that minimum.",
        f"- new parameters: {cost['parameters_new']}",
        f"- MLP parameters: {cost['mlp_parameters']}",
        f"- polarity parameters: {cost['polarity_parameters']}",
        f"- answer-token log_softmax median: {_fmt(cost['answer_surprise_median_s'], 6)} s "
        f"over {cost['answer_surprise_rows']} answer row(s) of logits already computed",
        f"- forwards: chat {cost['chat_forwards']} (written {cost['chat_written']}), "
        f"below threshold {cost['below_threshold_forwards']} "
        f"(written {cost['below_threshold_written']}, surprise {_fmt(cost['below_threshold_surprise'])}), "
        f"above threshold {cost['above_threshold_forwards']} "
        f"(written {cost['above_threshold_written']}, surprise {_fmt(cost['above_threshold_surprise'])})",
        "",
        "## Fit selection",
        "",
        "The grid is `i/4` for `i` from 1 to 39. A point is eligible when it writes "
        "at most 2 fit lows, at least 6 fit highs, at least 6 prefer and 6 avoid facts, "
        "the fit phone and birthday, and all 32 neutral incidentals, and it does not "
        "exceed the green-color floor. The highest eligible point is scored. "
        "A lower point is tried only when a higher one fails because a required fact "
        "was not written. The frozen bank and the held-out bank have not been scored.",
        "",
        f"Fit bank `{result['bank_sha256']}`.",
        "",
        "| gate | pass | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in rows:
        if name == "forbidden logprob down":
            lines.append(f"| {name} | reported | {detail} |")
        else:
            lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(
        [
            "",
            "Fit does not require the held-out mean floors (prefer mean 8, avoid mean 0.6) "
            "or forbidden-down. Those are held-out gates.",
            "",
            "## Held-out gates",
            "",
            "The held-out bank has not been scored. It is merge-critical.",
            "",
            "- high-surprise facts written on at least 6/8, low-surprise facts written on at most 2/8",
            "- mean margin delta of the high facts greater than the low facts",
            "- prefer margin delta > 0 on at least 6/8 and mean at least 8.0 "
            "(#41 held was 10.905 on 8/8)",
            "- avoid margin delta > 0 on at least 6/8 and mean at least 0.6 "
            "(#41 held was 1.277 on 7/8)",
            "- forbidden logprob down on at least 6/8 "
            "(#41 held was 8/8, mean stored lift -1.300)",
            "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
            "- choice flips 0",
            "- leak 0",
            "- phone and birthday takeover 0 off-relation",
            "- phone question takeover 1 and birthday question takeover 1",
            "",
            "The stored phone's full digit string is not a gate. "
            "The readout still spells the `+1-415-555-` prefix it was trained on. "
            "A mismatch between that prefix and the stored number is out of scope here.",
            "",
        ]
    )
    return "\n".join(lines)


def _qualitative_locked(result) -> bool:
    """True when every required fact is already written, so a lower tau cannot help."""
    prefer = result["prefer"]["summary"]
    avoid = result["avoid"]["summary"]
    neutral = result["neutral"]
    scripted = result["scripted"]
    return (
        int(prefer["written"]) == int(prefer["n"])
        and int(avoid["written"]) == int(avoid["n"])
        and int(neutral["incidental_written"]) == int(neutral["incidental_n"])
        and bool(scripted["phone_written"])
        and bool(scripted["birthday_written"])
    )


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    bank = fit_bank()
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    print("measuring fit surprises", flush=True)
    measured = {
        "low": _surprises(mem, bank.LOW),
        "high": _surprises(mem, bank.HIGH),
        "prefer": _surprises(mem, bank.PREFER),
        "avoid": _surprises(mem, bank.AVOID),
        "incidental": _surprises(mem, bank.incidental_facts()),
        "phone": _surprises(mem, [bank.PHONE_FACT])[0],
        "birthday": _surprises(mem, [bank.BIRTHDAY_FACT])[0],
    }
    green = _green_floor(mem)
    print(
        f"green min {_fmt(green['min'])} max {_fmt(green['max'])} "
        f"low {_fmt(min(measured['low']))}..{_fmt(max(measured['low']))} "
        f"high {_fmt(min(measured['high']))}..{_fmt(max(measured['high']))} "
        f"phone {_fmt(measured['phone'])} birthday {_fmt(measured['birthday'])} "
        f"incidental min {_fmt(min(measured['incidental']))}",
        flush=True,
    )
    eligible = _eligible(measured, green["min"])
    print("eligible", eligible, flush=True)
    if not eligible:
        print("no eligible tau", flush=True)
        mem.close()
        return 2
    source_tau = float(llm_memory.DEFAULT_SURPRISE_TAU)
    chosen = None
    result = None
    for tau in sorted(eligible, reverse=True):
        llm_memory.DEFAULT_SURPRISE_TAU = float(tau)
        print(f"scoring fit at tau {tau}", flush=True)
        result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
        result["tau"] = float(tau)
        result["measured"] = measured
        result["green"] = green
        rows = gate_rows(result, SURPRISE_GATES, use_means=False)
        for name, ok, detail in rows:
            print(f"  {'pass' if ok else 'FAIL'} {name}: {detail}", flush=True)
        if passes(rows):
            chosen = float(tau)
            break
        if _qualitative_locked(result):
            print("fit gates failed with every required fact already written; not walking down", flush=True)
            mem.close()
            return 2
        print("walking down", flush=True)
    if chosen is None or result is None:
        print("no tau passed the fit gates", flush=True)
        mem.close()
        return 2
    llm_memory.DEFAULT_SURPRISE_TAU = chosen
    cost = measure_gate_cost(mem, bank)
    mem.close()
    llm_memory.DEFAULT_SURPRISE_TAU = source_tau
    print(
        f"chosen tau {chosen:.2f}; source DEFAULT_SURPRISE_TAU is {source_tau}",
        flush=True,
    )
    if abs(source_tau - chosen) > 1e-9:
        print(f"set DEFAULT_SURPRISE_TAU = {chosen:.2f} and rerun to write the preregistration", flush=True)
        return 2
    payload = {"tau": chosen, "green": green, "measured": measured, "fit": result, "cost": cost}
    out_json = ROOT / "reports" / "subconscious_surprise_fit.json"
    out_md = ROOT / "reports" / "subconscious_surprise_prereg.md"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(_render_prereg(result, cost, green, chosen), encoding="utf-8")
    print(f"wrote {out_md} and {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
