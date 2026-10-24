"""Fit replay N and m on the fit bank, then score that bank once.

Does not import the held-out bank and does not score it. Prints the constants
to set when the source does not already match the chosen budget, and exits
before the full fit score in that case.
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
from tidal.llm_memory import LLMMemory, REPLAY_DECAY, REPLAY_EVERY, REPLAY_M
from tidal.subconscious_replay_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    REPLAY_GATES,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_replay_score import gate_rows, passes, score_bank, score_replay

# Largest N first, then smallest m. The keep is the smallest budget that
# clears the replay floors: salient gain at least 0.25, or a slot drop of
# at least 1. Decay is not on the grid.
GRID = tuple((n, m) for n in (32, 16, 8, 4) for m in (1, 2, 4, 8))


def _probe(mem, bank, every, width):
    llm_memory.REPLAY_EVERY = int(every)
    llm_memory.REPLAY_M = int(width)
    return score_replay(mem, bank)


def _choose(rows):
    """Smallest m/N, then largest N, then smallest m."""
    passing = [row for row in rows if row["ok"]]
    if not passing:
        return None
    passing.sort(key=lambda row: (row["m"] / row["n"], -row["n"], row["m"]))
    return passing[0]


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_prereg(result, grid_rows, chosen):
    cost = result["cost"]
    prefer = result["direction"]["prefer"]["summary"]
    avoid = result["direction"]["avoid"]["summary"]
    lines = [
        "# Replay consolidation preregistration",
        "",
        "Fixed-budget replay. The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run "
        "only when the winning relation is the one the prompt is completing. "
        "A continuation still encodes the query once. Readout stays hard top-1. "
        "The surprise gate and the retrieval-induced-forgetting gate are not in this change.",
        "",
        "Every N accepted writes, decay is recomputed as `0.5 ** age`, the top-m "
        "slots by salience are refreshed to strength 1, and at most m same-group "
        "near-duplicates are merged. A near-duplicate is a key cosine strictly "
        "above the store overwrite cosine, 0.9. Salience is "
        "`|tanh(w · normalize(k))|` on the polarity vector already in the checkpoint. "
        "The add is `c * s * gain * (scale(v) + MLP(scale(v)))`. `s` is 1 until that replay.",
        "",
        "## Banks",
        "",
        f"- fit `{bank_sha256('fit')}`",
        f"- held `{bank_sha256('held')}`",
        "- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`",
        "- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`",
        "- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`",
        "- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`",
        "- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`",
        "- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`",
        "- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`",
        f"- #42 fit `{SURPRISE_FIT_SHA}`",
        f"- #42 held `{SURPRISE_HELD_SHA}`",
        f"- #43 fit `{RIF_FIT_SHA}`",
        f"- #43 held `{RIF_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Budget",
        "",
        f"- replay every: {int(chosen['n'])}",
        f"- replay m: {int(chosen['m'])}",
        f"- decay: {REPLAY_DECAY:.2f}",
        f"- new parameters: {cost['parameters_new']}",
        f"- MLP parameters: {cost['mlp_parameters']}",
        f"- polarity parameters: {cost['polarity_parameters']}",
        f"- total parameters: {cost['parameters']}",
        f"- buffers: {cost['buffers']}",
        f"- forwards inside one remember that consolidates: {cost['remember_forwards']}",
        f"- forwards inside consolidate alone: {cost['consolidate_forwards']}",
        f"- forwards inside one armed query: {cost['query_forwards']}",
        f"- consolidate median: {cost['consolidate_median_s']:.3e} s",
        "",
        "## Fit selection",
        "",
        "The grid is N in {4, 8, 16, 32} and m in {1, 2, 4, 8}. "
        "A point passes when the fit probe's salient-slot gain is at least 0.25 "
        "or the slot count drops by at least 1, against the decay-only prefix of "
        "that same replay. The smallest budget (smallest m/N, then the largest N, "
        "then the smallest m) is the one scored below. The full fit score then "
        "runs once. It is not repeated on a larger budget if a non-interference "
        "floor fails. The held-out bank has not been scored.",
        "",
        "| N | m | pass | salient gain | slot drop | near-dup drop |",
        "|---|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {row['n']} | {row['m']} | {'yes' if row['ok'] else 'no'} | "
            f"{_fmt(row['salient_gain'])} | {row['slot_drop']} | {row['near_dup_drop']} |"
        )
    lines.extend(
        [
            "",
            f"Fit bank `{result['bank_sha256']}`.",
            "",
            "| gate | pass | detail |",
            "|---|---|---|",
        ]
    )
    for name, ok, detail in gate_rows(result, REPLAY_GATES, held=False):
        label = "reported" if name == "forbidden logprob down" else ("yes" if ok else "no")
        lines.append(f"| {name} | {label} | {detail} |")
    lines.extend(
        [
            "",
            "Fit does not require the held-out mean floors (prefer mean 8, avoid mean 0.6) "
            "or forbidden-down. Those are held-out gates.",
            "",
            f"Fit prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0). "
            f"Fit avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0). "
            f"Fit forbidden down {avoid['n_stored_down']}/{avoid['n']}, "
            f"mean stored lift {_fmt(avoid['mean_stored_lift'])}.",
            "",
            "## Held-out gates",
            "",
            "The held-out bank has not been scored. It is merge-critical.",
            "",
            "- after replay, salient mean strength exceeds the decay-only control by at least "
            "0.25, or the slot count drops by at least 1",
            "- prefer margin delta > 0 on at least 6/8 and mean at least 8.0 "
            f"(#41 held was {DIRECTION_HELD_PREFER_MEAN:.3f})",
            "- avoid margin delta > 0 on at least 6/8 and mean at least 0.6 "
            f"(#41 held was {DIRECTION_HELD_AVOID_MEAN:.3f})",
            "- forbidden logprob down on at least 6/8 "
            f"(#41 held mean stored lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f})",
            "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
            "- choice flips 0",
            "- leak 0",
            "- phone and birthday takeover 0 off-relation",
            "- phone question takeover 1 and birthday question takeover 1",
            "- incidental store keeps 32 slots",
            "",
            "Report the prefer, avoid, and forbidden deltas against the #41 held numbers above. "
            "Do not retune N or m on this bank.",
            "",
            "The stored phone's full digit string is not a gate. The readout still spells the "
            "`+1-415-555-` prefix it was trained on. A mismatch between that prefix and the "
            "stored number is out of scope here.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source_every = int(REPLAY_EVERY)
    source_m = int(REPLAY_M)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
    bank = fit_bank()
    rows = []
    try:
        for every, width in GRID:
            print(f"replay N={every} m={width}", flush=True)
            info = _probe(mem, bank, every, width)
            ok = (float(info["salient_gain"] or 0.0) >= float(REPLAY_GATES["salient_gain_min"])) or (
                int(info["slot_drop"]) >= int(REPLAY_GATES["near_dup_drop_min"])
            )
            row = {
                "n": int(every),
                "m": int(width),
                "ok": bool(ok),
                "salient_gain": info["salient_gain"],
                "slot_drop": int(info["slot_drop"]),
                "near_dup_drop": int(info["near_dup_drop"]),
            }
            rows.append(row)
            print(
                f"  ok {ok} gain {row['salient_gain']} drop {row['slot_drop']} "
                f"dups {row['near_dup_drop']}",
                flush=True,
            )
    finally:
        llm_memory.REPLAY_EVERY = source_every
        llm_memory.REPLAY_M = source_m
    chosen = _choose(rows)
    if chosen is None:
        raise SystemExit("no grid point cleared the replay floors")
    if int(chosen["n"]) != source_every or int(chosen["m"]) != source_m:
        print(f"set REPLAY_EVERY = {chosen['n']} and REPLAY_M = {chosen['m']}", flush=True)
        mem.close()
        raise SystemExit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    out = ROOT / "reports" / "subconscious_replay_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"grid": rows, "chosen": {"n": chosen["n"], "m": chosen["m"]}, "result": result}
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    if not passes(result, REPLAY_GATES, held=False):
        print("fit gates failed", flush=True)
        for name, ok, detail in gate_rows(result, REPLAY_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        mem.close()
        raise SystemExit(1)
    prereg = ROOT / "reports" / "subconscious_replay_prereg.md"
    prereg.write_text(_render_prereg(result, rows, chosen), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
