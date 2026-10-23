"""Score the held-out retrieval-induced-forgetting bank once.

Refuses a second pass. Does not edit the preregistration and does not retune.
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

from tidal.llm_memory import DEFAULT_GAP_TAU, LLMMemory, POLARITY_SPAN, RIF_COMPETITOR_KEEP, RIF_OVERWRITE_KEEP
from tidal.subconscious_rif_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    RIF_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_rif_score import gate_rows, passes, score_bank


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _load_fit():
    path = ROOT / "reports" / "subconscious_rif_fit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(held, fit):
    held_rows = gate_rows(held, RIF_GATES, held=True)
    fit_rows = gate_rows(fit, RIF_GATES, held=False)
    prefer = held["direction"]["prefer"]["summary"]
    avoid = held["direction"]["avoid"]["summary"]
    cost = held["cost"]
    lines = [
        "# Subconscious retrieval-induced forgetting",
        "",
        "When soft top-4 retrieves and top-1 wins, the other competitors in that "
        "batch are scaled by a fixed keep. A same-key overwrite keeps none of the "
        "old value. SmolLM2-135M, "
        f"layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        "The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, "
        "`gain = 1 + 2 tanh(w · normalize(k))`, "
        f"`s` kept at {_fmt(RIF_COMPETITOR_KEEP, 2)} for a loser and "
        f"{_fmt(RIF_OVERWRITE_KEEP, 2)} for a value that was corrected. "
        "Phone and birthday spelling still runs only when the prompt is completing that relation. "
        "One query per continuation. The surprise gate is not in this model.",
        "",
        f"Fit bank `{fit['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "The keep was chosen on the fit bank before the held-out bank was scored. "
        "Held-out was scored once.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for name, ok, detail in fit_rows:
        label = "reported" if name == "forbidden logprob down" else ("yes" if ok else "no")
        lines.append(f"| fit | {name} | {label} | {detail} |")
    for name, ok, detail in held_rows:
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    prefer_delta = float(prefer["mean_delta"]) - float(DIRECTION_HELD_PREFER_MEAN)
    avoid_delta = float(avoid["mean_delta"]) - float(DIRECTION_HELD_AVOID_MEAN)
    forbid_delta = float(avoid["mean_stored_lift"]) - float(DIRECTION_HELD_FORBIDDEN_LIFT)
    lines.extend(
        [
            "",
            "## Direction against #41 held",
            "",
            f"- prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0), "
            f"delta {_fmt(prefer_delta)} against {DIRECTION_HELD_PREFER_MEAN:.3f}",
            f"- avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0), "
            f"delta {_fmt(avoid_delta)} against {DIRECTION_HELD_AVOID_MEAN:.3f}",
            f"- forbidden mean stored lift {_fmt(avoid['mean_stored_lift'])} "
            f"({avoid['n_stored_down']}/{avoid['n']} down), "
            f"delta {_fmt(forbid_delta)} against {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- forwards inside one armed query: {cost['rank_forwards']}",
            f"- scale_strength median: {cost['scale_strength_median_s']:.3e} s "
            f"over {cost['losers']} loser slots",
            "- The top-4 are the cosines already computed for the gap. "
            "There is no second query and no feature flag.",
            "",
            "The stored phone's full digit string is not a gate. "
            "The readout still spells the `+1-415-555-` prefix it was trained on.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. Held-out pass: "
            + ("yes" if passes(held, RIF_GATES, held=True) else "no")
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_rif.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_rif_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    held_sha = bank_sha256("held")
    if held_sha not in prereg:
        raise SystemExit("preregistration does not lock the held-out bank")
    keep_text = f"{float(RIF_COMPETITOR_KEEP):.2f}"
    if f"competitor keep: {keep_text}" not in prereg:
        raise SystemExit("preregistration keep does not match the source constant")
    if abs(float(RIF_OVERWRITE_KEEP)) > 0.0:
        raise SystemExit("overwrite keep is fixed at 0")
    fit = _load_fit()
    if fit["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = held_bank()
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    scored.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_rif.md"
    report.write_text(render_report(result, fit), encoding="utf-8")
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, RIF_GATES, held=True):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
