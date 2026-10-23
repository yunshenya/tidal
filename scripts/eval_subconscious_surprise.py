"""Score the held-out surprise bank once. Does not retune the threshold.

    python scripts/eval_subconscious_surprise.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tidal.llm_memory import DEFAULT_GAP_TAU, DEFAULT_SURPRISE_TAU, LLMMemory, POLARITY_SPAN
from tidal.subconscious_surprise_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    SURPRISE_GATES,
    held_bank,
)
from tidal.subconscious_surprise_score import gate_rows, measure_gate_cost, score_bank


def _fmt(value, digits=3):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _rss_kb():
    out = {}
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("VmRSS:", "VmHWM:", "VmPeak:")):
                key, rest = line.split(":", 1)
                out[key] = int(rest.split()[0])
    return out


def render_report(result, cost, rss) -> str:
    gates = gate_rows(result, SURPRISE_GATES, use_means=True)
    prefer = result["prefer"]["summary"]
    avoid = result["avoid"]["summary"]
    high = result["high"]["summary"]
    low = result["low"]["summary"]
    prefer_delta = float(prefer["mean_delta"]) - DIRECTION_HELD_PREFER_MEAN
    avoid_delta = float(avoid["mean_delta"]) - DIRECTION_HELD_AVOID_MEAN
    lift_delta = float(result["forbidden_lift"]) - DIRECTION_HELD_FORBIDDEN_LIFT
    lines = [
        "# Subconscious surprise",
        "",
        "Surprise-gated writes on the frozen SmolLM2 path. "
        f"Layer {result['layer']}, hidden {result['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. surprise_tau {_fmt(DEFAULT_SURPRISE_TAU, 2)}. "
        "The add is `c * gain * (scale(v) + MLP(scale(v)))`, "
        f"`gain = 1 + {POLARITY_SPAN:.0f} tanh(w · normalize(k))`. "
        "A fact is written only when salience times the answer-token NLL of the value "
        "forward clears surprise_tau. Chat and noise stay at salience 0. "
        "Phone and birthday spelling still runs only when the prompt is completing that relation. "
        "One query per continuation.",
        "",
        f"Held-out bank `{result['bank_sha256']}`. "
        "The threshold was chosen on the fit bank before this bank was scored. "
        "This is the one held-out pass. The threshold was not changed after it.",
        "",
        "## Gates",
        "",
        "| gate | pass | detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in gates:
        lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(
        [
            "",
            "## High surprise versus low surprise",
            "",
            f"- high written {high['written']}/{high['n']}, "
            f"mean margin delta {_fmt(high['mean_delta'])}, "
            f"surprise {_fmt(high['min_surprise'])}..{_fmt(high['max_surprise'])}",
            f"- low written {low['written']}/{low['n']}, "
            f"mean margin delta {_fmt(low['mean_delta'])}, "
            f"surprise {_fmt(low['min_surprise'])}..{_fmt(low['max_surprise'])}",
            "",
            "## Prefer and avoid",
            "",
            f"- prefer: {prefer['n_delta_pos']}/{prefer['n']} > 0, "
            f"mean {_fmt(prefer['mean_delta'])} "
            f"(delta vs #41 held {DIRECTION_HELD_PREFER_MEAN:.3f}: {_fmt(prefer_delta)})",
            f"- avoid: {avoid['n_delta_pos']}/{avoid['n']} > 0, "
            f"mean {_fmt(avoid['mean_delta'])} "
            f"(delta vs #41 held {DIRECTION_HELD_AVOID_MEAN:.3f}: {_fmt(avoid_delta)})",
            f"- forbidden logprob down: {result['forbidden_down']}/{avoid['n']}, "
            f"mean stored lift {_fmt(result['forbidden_lift'])} "
            f"(delta vs #41 held {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}: {_fmt(lift_delta)})",
            f"- prefer stored-word lift: {_fmt(result['prefer_lift'])}",
            "",
            "## Non-interference",
            "",
            f"- neutral NLL mean {_fmt(result['neutral']['nll']['mean_delta'])}, "
            f"above 0.5: {result['neutral']['nll']['n_bad']}",
            f"- choice flips {result['neutral']['choice']['flips']}/{result['neutral']['choice']['n']}",
            f"- leak {result['neutral']['leak']['n_leak']}/{result['neutral']['leak']['n']}",
            f"- incidental slots {result['neutral']['incidental_slots']} "
            f"of {result['neutral']['incidental_n']} written {result['neutral']['incidental_written']}",
            f"- phone unrelated {_fmt(result['scripted']['phone_unrelated'])}, "
            f"question {result['scripted']['phone_question']}",
            f"- birthday unrelated {_fmt(result['scripted']['date_unrelated'])}, "
            f"question {result['scripted']['birthday_question']}",
            "",
            "The phone question gate looks for the trained `+1-415-555-` prefix. "
            "The stored held-out number is a different prefix. Matching every stored digit "
            "is the known readout miss from #41 and is not fixed here.",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- answer-token log_softmax, median of 30 calls on logits already returned: "
            f"{_fmt(cost['answer_surprise_median_s'], 6)} s "
            f"({cost['answer_surprise_rows']} answer row(s))",
            f"- chat forwards {cost['chat_forwards']}, written {str(cost['chat_written']).lower()}",
            f"- below-threshold forwards {cost['below_threshold_forwards']}, "
            f"written {str(cost['below_threshold_written']).lower()}, "
            f"surprise {_fmt(cost['below_threshold_surprise'])}",
            f"- above-threshold forwards {cost['above_threshold_forwards']}, "
            f"written {str(cost['above_threshold_written']).lower()}, "
            f"surprise {_fmt(cost['above_threshold_surprise'])}",
            "- No extra model forward. The key forward is skipped when the score is below tau.",
            f"- RSS after the score: VmRSS {rss.get('VmRSS')} kB, VmHWM {rss.get('VmHWM')} kB",
            "",
            f"Wall clock {_fmt(result.get('wall_s'), 1)} s. Phase seconds: "
            + ", ".join(f"{key} {_fmt(val, 1)}" for key, val in result["phases_s"].items())
            + ".",
            "",
            "## Misses",
            "",
        ]
    )
    missed = [name for name, ok, _detail in gates if not ok]
    if missed:
        lines.append("- held failed: " + ", ".join(missed) + ".")
    else:
        lines.append("- held met every surprise gate.")
    lines.extend(
        [
            "",
            "Numbers above are read from the JSON written by this script. "
            "surprise_tau was not changed after this pass.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    out_json = ROOT / "reports" / "subconscious_surprise.json"
    out_md = ROOT / "reports" / "subconscious_surprise.md"
    if out_json.exists():
        raise SystemExit(f"{out_json} already exists; refusing a second held-out pass")
    prereg = (ROOT / "reports" / "subconscious_surprise_prereg.md").read_text(encoding="utf-8")
    bank = held_bank()
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    if bank.sha256() not in prereg:
        raise SystemExit("preregistration does not cite this held-out bank")
    if f"{DEFAULT_SURPRISE_TAU:.2f}" not in prereg:
        raise SystemExit("preregistration does not cite the source surprise_tau")
    print(
        f"loading SmolLM2-135M float32 tau={DEFAULT_SURPRISE_TAU:.2f} gap={DEFAULT_GAP_TAU}",
        flush=True,
    )
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    mem = LLMMemory(dtype=torch.float32)
    print(f"loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    result["tau"] = float(DEFAULT_SURPRISE_TAU)
    result["gap_tau"] = float(DEFAULT_GAP_TAU)
    cost = measure_gate_cost(mem, bank)
    mem.close()
    rss = _rss_kb()
    text = render_report(result, cost, rss)
    payload = {
        "tau": float(DEFAULT_SURPRISE_TAU),
        "gap_tau": float(DEFAULT_GAP_TAU),
        "rss_kb": {"before": rss_before, "after": rss},
        "held": result,
        "cost": cost,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
