"""Choose the competitor keep on the fit bank only.

Does not score the held-out bank. Writes the preregistration file only when
the source constant already equals the chosen keep and the fit gates pass.
A milder keep that fails the weaken gate is not a reason to try a stronger
one after the full score: the full score runs once, at the keep already in
the source.
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
from tidal.llm_memory import LLMMemory, RIF_COMPETITOR_KEEP, RIF_OVERWRITE_KEEP
from tidal.subconscious_rif_data import (
    DIRECTION_FIT_SHA,
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    DIRECTION_HELD_SHA,
    DIRECTION_TRAIN_SHA,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    RIF_GATES,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_rif_score import gate_rows, passes, score_bank, score_quartets

# Largest keep first. The first point that weakens competitors is the choice.
KEEP_GRID = (0.99, 0.97, 0.95, 0.93, 0.90, 0.85)


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _weaken_ok(quartets) -> bool:
    if not quartets["geometry_ok"]:
        return False
    return (
        float(quartets["competitor_drop_min"]) >= float(RIF_GATES["competitor_drop_min"])
        and float(quartets["competitor_drop_mean"]) >= float(RIF_GATES["competitor_mean_drop_min"])
        and float(quartets["winner_abs_drop_max"]) <= float(RIF_GATES["winner_abs_drop_max"])
    )


def _render_prereg(result, keep):
    rows = gate_rows(result, RIF_GATES, held=False)
    cost = result["cost"]
    lines = [
        "# Retrieval-induced forgetting preregistration",
        "",
        "Retrieval-induced forgetting. The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run "
        "only when the winning relation is the one the prompt is completing. "
        "A continuation still encodes the query once. Readout stays hard top-1. "
        "The surprise gate is not in this change.",
        "",
        "When a strict top-1 wins inside a store of at least two slots, the other "
        "members of the soft top-4 have their strength multiplied by a fixed keep. "
        "The winner is not decayed by that query. A tie adds nothing and decays nothing. "
        "Strength multiplies the residual after the gap scale and the polarity gain. "
        "A same-key overwrite stores the new value and keeps none of the old one.",
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
        f"- #42 fit `{SURPRISE_FIT_SHA}`",
        f"- #42 held `{SURPRISE_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Decay",
        "",
        "The competitive set is the store's soft top-4. The injected value is still "
        "the hard top-1. `s` in `c * s * gain * (scale(v) + MLP(scale(v)))` is the "
        "winning slot's strength. Strength starts at 1. There is no extra forward: "
        "the top-4 are an argpartition of the cosines the gap already computed.",
        "",
        f"- competitor keep: {_fmt(keep, 2)}",
        f"- overwrite keep: {_fmt(RIF_OVERWRITE_KEEP, 2)}",
        f"- new parameters: {cost['parameters_new']}",
        f"- MLP parameters: {cost['mlp_parameters']}",
        f"- polarity parameters: {cost['polarity_parameters']}",
        f"- total parameters: {cost['parameters']}",
        f"- buffers: {cost['buffers']}",
        f"- forwards inside one armed query: {cost['rank_forwards']}",
        f"- scale_strength median: {cost['scale_strength_median_s']:.3e} s "
        f"over {cost['losers']} loser slots",
        "",
        "## Fit selection",
        "",
        "The grid is 0.99, 0.97, 0.95, 0.93, 0.90, 0.85. "
        "The largest keep whose fit quartets have minimum competitor drop at least 0.10, "
        "mean competitor drop at least 0.20, and absolute winner drop at most 0.05 is the one "
        "scored below. The full fit score then runs once. It is not repeated at a stronger decay "
        "if a non-interference floor fails. The frozen bank and the held-out bank have not been scored.",
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
    prefer = result["direction"]["prefer"]["summary"]
    avoid = result["direction"]["avoid"]["summary"]
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
            "- competitors weaken after one target hit, relative to a strength-1 control: "
            "minimum drop at least 0.10, mean drop at least 0.20, absolute winner drop at most 0.05",
            "- same-key overwrite follows the new value, and the new word beats the old word on at least 6/8",
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
            "Do not retune the keep on this bank.",
            "",
            "The stored phone's full digit string is not a gate. "
            "The readout still spells the `+1-415-555-` prefix it was trained on. "
            "A mismatch between that prefix and the stored number is out of scope here.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source = float(RIF_COMPETITOR_KEEP)
    if float(RIF_OVERWRITE_KEEP) != 0.0:
        raise SystemExit("overwrite keep is fixed at 0")
    bank = fit_bank()
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    chosen = None
    for keep in KEEP_GRID:
        llm_memory.RIF_COMPETITOR_KEEP = float(keep)
        print(f"quartet keep {keep:.2f}", flush=True)
        quartets = score_quartets(mem, bank)
        print(
            "  "
            f"geometry {quartets['geometry_ok']} "
            f"min {_fmt(quartets['competitor_drop_min'])} "
            f"mean {_fmt(quartets['competitor_drop_mean'])} "
            f"winner {_fmt(quartets['winner_abs_drop_max'])}",
            flush=True,
        )
        if _weaken_ok(quartets):
            chosen = float(keep)
            break
    llm_memory.RIF_COMPETITOR_KEEP = source
    if chosen is None:
        print("no keep on the grid passed the weaken gate", flush=True)
        sys.exit(2)
    if abs(chosen - source) > 1e-9:
        print(f"set RIF_COMPETITOR_KEEP = {chosen:.2f}", flush=True)
        sys.exit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    out = ROOT / "reports" / "subconscious_rif_fit.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    if not passes(result, RIF_GATES, held=False):
        print("fit gates failed; preregistration was not written", flush=True)
        for name, ok, detail in gate_rows(result, RIF_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        sys.exit(1)
    prereg = ROOT / "reports" / "subconscious_rif_prereg.md"
    prereg.write_text(_render_prereg(result, source), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
