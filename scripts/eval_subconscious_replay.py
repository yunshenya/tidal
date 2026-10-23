"""Score the held-out replay bank once.

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

from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    LLMMemory,
    POLARITY_SPAN,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
)
from tidal.subconscious_replay_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    REPLAY_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_replay_score import gate_rows, passes, score_bank


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _load_fit():
    path = ROOT / "reports" / "subconscious_replay_fit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(held, fit):
    fit_result = fit["result"]
    held_rows = gate_rows(held, REPLAY_GATES, held=True)
    fit_rows = gate_rows(fit_result, REPLAY_GATES, held=False)
    prefer = held["direction"]["prefer"]["summary"]
    avoid = held["direction"]["avoid"]["summary"]
    cost = held["cost"]
    lines = [
        "# Subconscious replay consolidation",
        "",
        "Every N accepted writes, decay is recomputed, the top-m salient slots "
        "are refreshed, and at most m near-duplicates are merged. SmolLM2-135M, "
        f"layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        "The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, "
        "`gain = 1 + 2 tanh(w · normalize(k))`, "
        f"`s` refreshed for the top {int(REPLAY_M)} slot"
        f"{'s' if int(REPLAY_M) != 1 else ''} every {int(REPLAY_EVERY)} accepted writes. "
        f"Decay is {REPLAY_DECAY:.2f} per cycle. "
        "Phone and birthday spelling still runs only when the prompt is completing that relation. "
        "One query per continuation. The surprise gate and the retrieval-induced-forgetting "
        "gate are not in this model.",
        "",
        f"Fit bank `{fit_result['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "N and m were chosen on the fit bank before the held-out bank was scored. "
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
            f"- forwards inside one remember that consolidates: {cost['remember_forwards']}",
            f"- forwards inside consolidate alone: {cost['consolidate_forwards']}",
            f"- forwards inside one armed query: {cost['query_forwards']}",
            f"- consolidate median: {cost['consolidate_median_s']:.3e} s",
            "- The scan is the slots already in the store. "
            "There is no second query and no feature flag.",
            "",
            "The stored phone's full digit string is not a gate. "
            "The readout still spells the `+1-415-555-` prefix it was trained on.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. Held-out pass: "
            + ("yes" if passes(held, REPLAY_GATES, held=True) else "no")
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_replay.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_replay_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    held_sha = bank_sha256("held")
    if held_sha not in prereg:
        raise SystemExit("preregistration does not lock the held-out bank")
    if f"replay every: {int(REPLAY_EVERY)}" not in prereg:
        raise SystemExit("preregistration N does not match the source constant")
    if f"replay m: {int(REPLAY_M)}" not in prereg:
        raise SystemExit("preregistration m does not match the source constant")
    fit = _load_fit()
    if fit["result"]["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = held_bank()
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    scored.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_replay.md"
    report.write_text(render_report(result, fit), encoding="utf-8")
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, REPLAY_GATES, held=True):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
