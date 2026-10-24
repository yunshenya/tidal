"""Score prefer/avoid direction once. Does not train.

    python scripts/eval_subconscious_direction.py
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

from tidal.subconscious_direction_data import DIRECTION_GATES, held_bank
from tidal.subconscious_eval import _rss_kb, run_eval


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


def _forbidden_down(result):
    rows = result["axis1"]["avoid"]["items"]
    return int(sum(float(row["stored_lift"]) < 0.0 for row in rows)), len(rows)


def direction_gate_rows(result):
    spec = DIRECTION_GATES
    prefer = result["axis1"]["prefer"]["summary"]["inject"]
    avoid = result["axis1"]["avoid"]["summary"]["inject"]
    down, n_avoid = _forbidden_down(result)
    nll = result["neutral"]["nll"]
    choice = result["neutral"]["choice"]
    leak = result["neutral"]["leak"]
    scripted = result["scripted"]
    rows = [
        (
            "prefer margin",
            int(prefer["n_delta_pos"]) >= spec["prefer_delta_pos_min"],
            f"{prefer['n_delta_pos']}/{prefer['n']} > 0, mean {_fmt(prefer['mean_delta'])}",
        ),
        (
            "avoid margin",
            int(avoid["n_delta_pos"]) >= spec["avoid_delta_pos_min"],
            f"{avoid['n_delta_pos']}/{avoid['n']} > 0, mean {_fmt(avoid['mean_delta'])}",
        ),
        (
            "forbidden logprob down",
            down >= spec["forbidden_down_min"],
            f"{down}/{n_avoid} down, mean stored lift {_fmt(result['axis1']['stored_lift']['avoid'])}",
        ),
        (
            "neutral NLL",
            abs(float(nll["mean_delta"])) <= spec["nll_abs_mean_max"]
            and int(nll["n_bad"]) <= spec["nll_bad_count_max"],
            f"mean {_fmt(nll['mean_delta'])}, above {spec['nll_bad']}: {nll['n_bad']}",
        ),
        (
            "choice flips",
            int(choice["flips"]) <= spec["choice_flips_max"],
            f"{choice['flips']}/{choice['n']}",
        ),
        ("leak", int(leak["n_leak"]) <= spec["leak_max"], f"{leak['n_leak']}/{leak['n']}"),
        (
            "phone unrelated",
            float(scripted["phone_unrelated"]) <= spec["phone_unrelated_max"],
            _fmt(scripted["phone_unrelated"]),
        ),
        (
            "birthday unrelated",
            float(scripted["date_unrelated"]) <= spec["birthday_unrelated_max"],
            _fmt(scripted["date_unrelated"]),
        ),
        ("phone question", scripted["phone_question"] == spec["phone_question"], str(scripted["phone_question"])),
    ]
    if scripted.get("birthday_question") is not None:
        rows.append(
            (
                "birthday question",
                scripted["birthday_question"] == spec["birthday_question"],
                str(scripted["birthday_question"]),
            )
        )
    return rows


def _cost_section(result):
    cost = result.get("cost")
    if not cost:
        return ["Cost was not measured on this bank.", ""]
    inject = cost["inject"]
    named = dict(inject["named_parameters"])
    readout = cost["readout_s"]
    rss = cost["rss_kb"]["after"]
    lines = [
        f"- MLP parameters (`net.*`): {inject['mlp_parameters']}",
        f"- polarity parameters: {named.get('polarity', 0)}",
        f"- total parameters: {inject['parameters']}",
        f"- buffers: {inject['buffers']}",
        f"- extra per token, mean over lengths 16/32/64: {_fmt(cost['extra_per_token_s']['mean_s'], 6)} s",
    ]
    for length, row in cost["extra_per_token_s"]["by_length"].items():
        lines.append(f"  - length {length}: {_fmt(row, 6)} s")
    lines.append("- readout median, hard top-1, versus store size N:")
    for size, row in readout.items():
        lines.append(f"  - N={size}: {_fmt(row['median_s'], 6)} s")
    lines.append(
        f"- query median: {_fmt(cost['query_s']['median_s'], 6)} s; "
        f"MLP median (includes the polarity dot): {_fmt(cost['mlp_s']['median_s'], 6)} s"
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
    lines.append(f"- RSS after the cost pass: VmRSS {rss.get('VmRSS')} kB, VmHWM {rss.get('VmHWM')} kB")
    lines.append("- Variable-cost paths still in the runtime (no retries, no second query):")
    for flag in cost["flags"]:
        lines.append(f"  - {flag['path']}: {flag['varies']}")
    lines.append(
        "- Polarity is a fixed 576-d dot product inside that one MLP call. "
        "It does not add a retrieval, a second query, or a per-token branch."
    )
    lines.append("")
    return lines


def render_direction_report(payload):
    frozen = payload["frozen"]
    held = payload["held"]
    lines = [
        "# Subconscious direction",
        "",
        "Prefer versus avoid polarity on the frozen #38 MLP. "
        f"SmolLM2-135M, layer {frozen['layer']}, hidden {frozen['hidden']}, float32. "
        f"gap_tau {_fmt(payload['gap_tau'], 2)}. "
        "The add is `c * gain * (scale(v) + MLP(scale(v)))`, "
        "`gain = 1 + 2 tanh(w · normalize(k))`. "
        "Phone and birthday spelling still runs only when the prompt is completing that relation. "
        "One query per continuation.",
        "",
        f"Frozen bank `{frozen['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "The checkpoint was chosen on the fit bank before either of these was scored. "
        "Held-out is the merge-critical gate. Frozen is a measurement under the same gates.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for label, result in (("frozen", frozen), ("held", held)):
        for name, ok, detail in direction_gate_rows(result):
            lines.append(f"| {label} | {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(["", "## Prefer and avoid", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        lines.append(f"### {label}")
        lines.append("")
        for family in ("prefer", "avoid"):
            summary = result["axis1"][family]["summary"]
            lines.append(f"- {family} inject: {_family_line(summary['inject'])}")
            lines.append(f"- {family} context: {_family_line(summary['context'])}")
            lines.append(f"- {family} rag: {_family_line(summary['rag'])}")
        down, n_avoid = _forbidden_down(result)
        lines.append(
            f"- forbidden logprob down: {down}/{n_avoid}, "
            f"mean stored lift {_fmt(result['axis1']['stored_lift']['avoid'])}"
        )
        absent = result["absent"]
        lines.append(
            f"- unrelated N=8: mean {_fmt(absent['mean_delta'])}, large {absent['n_large']}"
        )
        prefer64 = result["load"]["summary"]["prefer"]["64"]
        lines.append(
            f"- prefer top-1 at N=64: {prefer64['hit_n']}/{prefer64['n']} (reported, not a gate)"
        )
        misses = []
        for family in ("prefer", "avoid"):
            for row in result["axis1"][family]["items"]:
                delta = float(row["inject"]["delta"])
                if family == "avoid":
                    bad = float(row["stored_lift"]) >= 0.0
                    if delta <= 0.0 or bad:
                        misses.append(
                            f"{row['id']} delta {_fmt(delta)} stored_lift {_fmt(row['stored_lift'])}"
                        )
                elif delta <= 0.0:
                    misses.append(f"{row['id']} delta {_fmt(delta)}")
        if misses:
            lines.append("- misses: " + "; ".join(misses))
        lines.append("")
    lines.extend(["## Cost", ""])
    lines.extend(_cost_section(frozen))
    if payload.get("rss_kb_load"):
        rss = payload["rss_kb_load"]
        lines.append(
            "Process RSS around the run: "
            + "; ".join(
                f"{key} "
                + ", ".join(f"{name} {int(value)} kB" for name, value in row.items())
                for key, row in rss.items()
                if isinstance(row, dict)
            )
            + "."
        )
        lines.append("")
    lines.extend(["## Misses", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        missed = [name for name, ok, _detail in direction_gate_rows(result) if not ok]
        if missed:
            lines.append(f"- {label} failed: {', '.join(missed)}.")
        else:
            lines.append(f"- {label} met every direction gate.")
    lines.extend(
        [
            "",
            "Numbers above are read from the JSON written by this script. "
            "Weights were not changed after these two banks were scored.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    parser = __import__("argparse").ArgumentParser(description="score prefer versus avoid direction")
    parser.add_argument("--json", default="reports/subconscious_direction.json")
    parser.add_argument("--md", default="reports/subconscious_direction.md")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    from tidal.llm_memory import DEFAULT_GAP_TAU, LLMMemory

    print("loading SmolLM2-135M float32", flush=True)
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
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
    text = render_direction_report(payload)
    out_json = Path(args.json)
    out_md = Path(args.md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"wrote {out_json} and {out_md}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
