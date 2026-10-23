"""Score non-interference of the gap-scaled inject. Does not train.

    python scripts/eval_subconscious_gate.py
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

from tidal.subconscious_eval import _rss_kb, run_eval
from tidal.subconscious_gate_data import GATE_SPEC


class BGE:
    """CLS top-1. Same pooling as ``scripts/eval_subconscious.py``."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    @torch.no_grad()
    def __call__(self, texts, query: str) -> str:
        rows = [str(text) for text in texts]
        if not rows:
            raise ValueError("RAG corpus is empty")
        batch = self.tok(rows + [str(query)], padding=True, truncation=True, return_tensors="pt")
        hidden = self.model(**batch).last_hidden_state[:, 0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        scores = hidden[:-1] @ hidden[-1]
        return rows[int(scores.argmax())]


def _fmt(value, digits=3):
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


def _family_line(summary):
    return (
        f"{_fmt(summary['mean_delta'])} "
        f"({summary['n_delta_pos']}/{summary['n']} > 0), "
        f"choice {summary['acc_hits']}/{summary['n']} "
        f"vs nomem {summary['acc_nomem_hits']}/{summary['n']}"
    )


def _gate_rows(result):
    spec = GATE_SPEC
    nll = result["neutral"]["nll"]
    choice = result["neutral"]["choice"]
    leak = result["neutral"]["leak"]
    scripted = result["scripted"]
    rows = [
        (
            "neutral NLL",
            abs(float(nll["mean_delta"])) <= spec["nll_abs_mean_max"]
            and int(nll["n_bad"]) <= spec["nll_bad_count_max"],
            f"mean {_fmt(nll['mean_delta'])}, |mean| {_fmt(abs(float(nll['mean_delta'])))}, "
            f"above {spec['nll_bad']} : {nll['n_bad']}",
        ),
        (
            "choice flips",
            int(choice["flips"]) <= spec["choice_flips_max"],
            f"{choice['flips']}/{choice['n']}",
        ),
        ("leak", int(leak["n_leak"]) <= spec["leak_max"], f"{leak['n_leak']}/{leak['n']}"),
        ("phone unrelated", float(scripted["phone_unrelated"]) == 0.0, _fmt(scripted["phone_unrelated"])),
        ("birthday unrelated", float(scripted["date_unrelated"]) == 0.0, _fmt(scripted["date_unrelated"])),
        ("phone question", scripted["phone_question"] == 1, str(scripted["phone_question"])),
    ]
    if scripted.get("birthday_question") is not None:
        rows.append(
            ("birthday question", scripted["birthday_question"] == 1, str(scripted["birthday_question"]))
        )
    return rows


def _cost_section(result):
    cost = result.get("cost")
    if not cost:
        return ["Cost was not measured on this bank.", ""]
    inject = cost["inject"]
    readout = cost["readout_s"]
    rss = cost["rss_kb"]["after"]
    lines = [
        f"- MLP parameters: {inject['mlp_parameters']} (the #38 checkpoint; no key projection)",
        f"- buffers: {inject['buffers']}",
        f"- extra per token, mean over lengths 16/32/64: {_fmt(cost['extra_per_token_s']['mean_s'], 6)} s",
    ]
    for length, row in cost["extra_per_token_s"]["by_length"].items():
        lines.append(f"  - length {length}: {_fmt(row, 6)} s")
    lines.append("- readout median, hard top-1:")
    for size, row in readout.items():
        lines.append(f"  - N={size}: {_fmt(row['median_s'], 6)} s")
    lines.append(
        f"- query median: {_fmt(cost['query_s']['median_s'], 6)} s; "
        f"MLP median: {_fmt(cost['mlp_s']['median_s'], 6)} s"
    )
    counts = cost["path_counts"]
    lines.append(
        f"- queries per continuation: empty {counts['empty']['queries']}, "
        f"inject {counts['inject']['queries']}, "
        f"phone slot on the decision prompt {counts['scripted_phone']['queries']} "
        f"(forwards {counts['scripted_phone']['forwards']}; spelling stays off because "
        f"that prompt is not completing phone)"
    )
    lines.append(
        f"- extra per prompt (queries × query + MLP + readout N=64): "
        f"{_fmt(cost['extra_per_prompt_s']['total_s'], 6)} s"
    )
    lines.append(
        f"- continuation wall, {cost['new_tokens']} new tokens: "
        f"empty {_fmt(cost['continuation_s']['empty']['median_s'], 4)} s, "
        f"inject {_fmt(cost['continuation_s']['inject']['median_s'], 4)} s, "
        f"phone slot on that prompt {_fmt(cost['continuation_s']['scripted_phone']['median_s'], 4)} s"
    )
    lines.append(f"- RSS after the cost pass: VmRSS {rss.get('VmRSS')} kB, VmHWM {rss.get('VmHWM')} kB")
    lines.append("- Variable-cost paths (no retries, no second query on the main path):")
    for flag in cost["flags"]:
        lines.append(f"  - {flag['path']}: {flag['varies']}")
    lines.append("")
    return lines


def render_gate_report(payload):
    frozen = payload["frozen"]
    held = payload["held"]
    lines = [
        "# Subconscious gate",
        "",
        "Confidence scale and relation-gated spelling on the #38 inject MLP. "
        f"SmolLM2-135M, layer {frozen['layer']}, hidden {frozen['hidden']}, float32. "
        f"gap_tau {_fmt(payload['gap_tau'], 2)}. "
        "The add is `c * (scale(v) + MLP(scale(v)))`. "
        "`c` is 1 for one slot and `clip((top1-top2)/gap_tau, 0, 1)` otherwise. "
        "Phone and birthday spelling runs only when the winning relation is the one the prompt is completing. "
        "One query per continuation. The MLP weights are the #38 checkpoint.",
        "",
        f"Frozen bank `{frozen['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "gap_tau was not fit on either bank. Prefer and avoid deltas are measurements, not gates.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for label, result in (("frozen", frozen), ("held", held)):
        for name, ok, detail in _gate_rows(result):
            lines.append(f"| {label} | {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(["", "## Prefer and avoid (not gated)", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        lines.append(f"### {label}")
        lines.append("")
        for family in ("prefer", "avoid"):
            summary = result["axis1"][family]["summary"]
            lines.append(f"- {family} inject: {_family_line(summary['inject'])}")
            lines.append(f"- {family} context: {_family_line(summary['context'])}")
            lines.append(f"- {family} rag: {_family_line(summary['rag'])}")
        absent = result["absent"]
        lines.append(
            f"- unrelated N=8: mean {_fmt(absent['mean_delta'])}, "
            f"large {absent['n_large']}"
        )
        prefer64 = result["load"]["summary"]["prefer"]["64"]
        lines.append(
            f"- prefer top-1 at N=64: {prefer64['hit_n']}/{prefer64['n']} "
            f"(reported, not a gate)"
        )
        lines.append("")
    lines.extend(["## Non-interference detail", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        nll = result["neutral"]["nll"]
        choice = result["neutral"]["choice"]
        leak = result["neutral"]["leak"]
        scripted = result["scripted"]
        lines.append(
            f"- {label}: NLL mean {_fmt(nll['mean_delta'])} ({nll['n_bad']} > 0.5), "
            f"flips {choice['flips']}/{choice['n']}, leak {leak['n_leak']}/{leak['n']}, "
            f"phone unrelated {_fmt(scripted['phone_unrelated'])}, "
            f"birthday unrelated {_fmt(scripted['date_unrelated'])}, "
            f"phone question {scripted['phone_question']}, "
            f"birthday question {scripted.get('birthday_question')}"
        )
    lines.extend(["", "## Cost", ""])
    lines.extend(_cost_section(frozen))
    if payload.get("rss_kb_load"):
        rss = payload["rss_kb_load"]
        lines.append(
            f"Process RSS around the run: before {_rss_phrase(rss.get('before'))}; "
            f"after load {_rss_phrase(rss.get('after_load'))}; "
            f"after eval {_rss_phrase(rss.get('after_eval'))}."
        )
        lines.append("")
    lines.extend(["## Misses", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        missed = [name for name, ok, _detail in _gate_rows(result) if not ok]
        if missed:
            lines.append(f"- {label} failed: {', '.join(missed)}.")
        else:
            lines.append(f"- {label} met every non-interference gate.")
    lines.extend(
        [
            "",
            "Numbers above are read from the JSON written by this script. "
            "Weights and gap_tau were not changed after these two banks were scored.",
            "",
        ]
    )
    return "\n".join(lines)


def _rss_phrase(row):
    if not isinstance(row, dict):
        return str(row)
    return ", ".join(f"{key} {int(value)} kB" for key, value in row.items())


def main():
    parser = __import__("argparse").ArgumentParser(description="score the gap-scale gates")
    parser.add_argument("--json", default="reports/subconscious_gate.json")
    parser.add_argument("--md", default="reports/subconscious_gate.md")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    from tidal.llm_memory import DEFAULT_GAP_TAU, LLMMemory
    from tidal.subconscious_gate_data import held_bank

    print("loading SmolLM2-135M float32", flush=True)
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    mem = LLMMemory(dtype=torch.float32)
    load_s = time.perf_counter() - t0
    rss_after = _rss_kb()
    print(f"loaded in {load_s:.1f}s gap_tau={DEFAULT_GAP_TAU}", flush=True)
    bge = BGE()

    def progress(msg):
        print(msg, flush=True)

    frozen = run_eval(mem, bge, cost_repeats=args.repeats, cost_warmup=args.warmup, progress=progress)
    held = run_eval(
        mem,
        bge,
        cost_repeats=args.repeats,
        cost_warmup=args.warmup,
        do_cost=False,
        progress=progress,
        data=held_bank(),
    )
    payload = {
        "load_s": float(load_s),
        "rss_kb_load": {"before": rss_before, "after_load": rss_after, "after_eval": _rss_kb()},
        "cpu_count": int(os.cpu_count() or 0),
        "gap_tau": float(DEFAULT_GAP_TAU),
        "frozen": frozen,
        "held": held,
    }
    text = render_gate_report(payload)
    out_json = Path(args.json)
    out_md = Path(args.md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")
    print(f"wrote {out_json} and {out_md}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
