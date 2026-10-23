"""Rescore the frozen bank and the held-out bank. Does not train.

    python scripts/eval_subconscious_fix.py
"""
from __future__ import annotations

import argparse
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


def _mean(xs):
    vals = [float(x) for x in xs if x is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _family_line(summary):
    return (
        f"{_fmt(summary['mean_delta'])} "
        f"({summary['n_delta_pos']}/{summary['n']} > 0), "
        f"choice {summary['acc_hits']}/{summary['n']} "
        f"vs nomem {summary['acc_nomem_hits']}/{summary['n']}, "
        f"lift {_fmt(summary['acc_lift'])}"
    )


def _gate_rows(result):
    gates = result["gates"]
    prefer = result["axis1"]["prefer"]["summary"]["inject"]
    avoid = result["axis1"]["avoid"]["summary"]["inject"]
    nll = result["neutral"]["nll"]
    choice = result["neutral"]["choice"]
    leak = result["neutral"]["leak"]
    absent = result["absent"]
    update = result["update"]
    repeat = result["repeat"]
    chat = result["chat"]
    scripted = result["scripted"]
    rows = [
        ("prefer delta", prefer["mean_delta"] > gates["delta_min"] and prefer["n_delta_pos"] >= gates["delta_pos_min"],
         f"mean {_fmt(prefer['mean_delta'])}, pos {prefer['n_delta_pos']}/{prefer['n']}"),
        ("avoid delta", avoid["mean_delta"] > gates["delta_min"] and avoid["n_delta_pos"] >= gates["delta_pos_min"],
         f"mean {_fmt(avoid['mean_delta'])}, pos {avoid['n_delta_pos']}/{avoid['n']}"),
        ("neutral NLL", nll["pass"], f"mean {_fmt(nll['mean_delta'])}, bad {nll['n_bad']}"),
        ("choice flips", choice["pass"], f"{choice['flips']}/{choice['n']}"),
        ("leak", leak["pass"], f"{leak['n_leak']}/{leak['n']}"),
        ("unrelated N=8", absent["pass"], f"mean {_fmt(absent['mean_delta'])}, large {absent['n_large']}"),
        ("repeat", repeat["pass"], f"max abs {_fmt(repeat['max_abs'], 6)}"),
        ("chat", chat["pass"], f"max abs {_fmt(chat['max_abs'], 6)}"),
        ("update delta", update["delta_pass"], f"mean {_fmt(update['mean_delta'])}, pos {update['n_delta_pos']}"),
        ("update store", update["store_pass"], "one slot, new value closer"),
        ("phone unrelated", scripted["phone_unrelated"] == 0, _fmt(scripted["phone_unrelated"])),
        ("birthday unrelated", scripted["date_unrelated"] == 0, _fmt(scripted["date_unrelated"])),
        ("phone question", scripted["phone_question"] == 1, str(scripted["phone_question"])),
    ]
    if scripted["birthday_question"] is not None:
        rows.append(
            ("birthday question", scripted["birthday_question"] == 1, str(scripted["birthday_question"]))
        )
    return rows


def _load_line(result, family, n):
    row = result["load"]["summary"][family][str(n)]
    gaps = [
        item["inject"]["retrieval"].get("gap")
        for item in result["load"]["items"]
        if item["family"] == family and item["load"] == int(n)
    ]
    conf = [
        item["inject"]["retrieval"].get("confidence")
        for item in result["load"]["items"]
        if item["family"] == family and item["load"] == int(n)
    ]
    return (
        f"hit {row['hit_n']}/{row['n']}, survived {row['survived_n']}/{row['n']}, "
        f"slots {_fmt(row['slots_mean'], 1)}, gap {_fmt(_mean(gaps), 4)}, "
        f"conf {_fmt(_mean(conf), 3)}, delta {_fmt(row['inject']['mean_delta'])}"
    )


def _cost_section(result):
    cost = result.get("cost")
    if not cost:
        return ["Cost was not measured on this bank.", ""]
    inject = cost["inject"]
    readout = cost["readout_s"]
    rss = cost["rss_kb"]["after"]
    lines = [
        f"- MLP parameters (net + key_proj): {inject['mlp_parameters']} "
        f"(net {inject['net_parameters']}, key_proj {inject['key_parameters']})",
        f"- buffers: {inject['buffers']}",
        f"- gap_tau: {_fmt(cost.get('gap_tau'), 4)}",
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
        f"phone question {counts['scripted_phone']['queries']}"
    )
    if "phone_on_decision" in counts:
        lines.append(
            f"- phone slot on the decision prompt queries {counts['phone_on_decision']['queries']}, "
            f"forwards {counts['phone_on_decision']['forwards']} "
            f"(the token loop still runs)"
        )
    lines.append(
        f"- extra per prompt (one query + MLP + readout N=64): "
        f"{_fmt(cost['extra_per_prompt_s']['total_s'], 6)} s"
    )
    lines.append(
        f"- continuation wall, {cost['new_tokens']} new tokens: "
        f"empty {_fmt(cost['continuation_s']['empty']['median_s'], 4)} s, "
        f"inject {_fmt(cost['continuation_s']['inject']['median_s'], 4)} s, "
        f"phone question {_fmt(cost['continuation_s']['scripted_phone']['median_s'], 4)} s"
    )
    lines.append(f"- RSS after the cost pass: VmRSS {rss.get('VmRSS')} kB, VmHWM {rss.get('VmHWM')} kB")
    lines.append("- Variable-cost paths (no retries, no second query on the main path):")
    for flag in cost["flags"]:
        lines.append(f"  - {flag['path']}: {flag['varies']}")
    lines.append("")
    return lines


def render_fix_report(payload):
    frozen = payload["frozen"]
    held = payload["held"]
    lines = [
        "# Subconscious fix",
        "",
        "Direction and interference of the soft inject on the main path. "
        f"SmolLM2-135M, layer {frozen['layer']}, hidden {frozen['hidden']}, float32. "
        f"gap_tau {_fmt(frozen.get('gap_tau'), 4)}. "
        "The add is `c * (scale(v) + MLP(scale(v) + W k))`. "
        "`c` is 1 for one slot and `clip((top1-top2)/gap_tau, 0, 1)` otherwise. "
        "Phone and birthday spelling runs only when the winning relation is the one the prompt is completing. "
        "One query per continuation.",
        "",
        f"Frozen bank `{frozen['bank_sha256']}` (`{frozen['bank_kind']}`). "
        f"Held-out bank `{held['bank_sha256']}` (`{held['bank_kind']}`). "
        "Neither bank was used to train the MLP or to choose gap_tau. "
        f"Train log: selected step {payload['train']['selected_step']}, "
        f"fit-bank sha `{payload['train']['fit_bank_sha256']}`.",
        "",
        "Load top-1 and FIFO eviction are reported and are not gates.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for label, result in (("frozen", frozen), ("held", held)):
        for name, ok, detail in _gate_rows(result):
            lines.append(f"| {label} | {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(["", "## Influence", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        lines.append(f"### {label}")
        lines.append("")
        for family in ("prefer", "avoid"):
            summary = result["axis1"][family]["summary"]
            lift = result["axis1"]["stored_lift"][family]
            lines.append(f"- {family} inject: {_family_line(summary['inject'])}; stored-side lift {_fmt(lift)}")
            lines.append(f"- {family} context: {_family_line(summary['context'])}")
            lines.append(f"- {family} rag: {_family_line(summary['rag'])}")
        lines.append(
            f"- update mean delta {_fmt(result['update']['mean_delta'])}, "
            f"new lift {_fmt(result['update']['mean_new_lift'])}, "
            f"old lift {_fmt(result['update']['mean_old_lift'])}"
        )
        lines.append("")
    lines.extend(["## Non-interference", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        nll = result["neutral"]["nll"]
        choice = result["neutral"]["choice"]
        leak = result["neutral"]["leak"]
        scripted = result["scripted"]
        lines.append(
            f"- {label}: NLL mean {_fmt(nll['mean_delta'])} ({nll['n_bad']} > 0.5), "
            f"flips {choice['flips']}/{choice['n']}, leak {leak['n_leak']}/{leak['n']}, "
            f"unrelated mean {_fmt(result['absent']['mean_delta'])} "
            f"({result['absent']['n_large']} |delta| > 0.1), "
            f"phone unrelated {_fmt(scripted['phone_unrelated'])}, "
            f"birthday unrelated {_fmt(scripted['date_unrelated'])}, "
            f"phone question {scripted['phone_question']}, "
            f"birthday question {scripted['birthday_question']}"
        )
    lines.extend(["", "## Load and persistence (not gated)", ""])
    for label, result in (("frozen", frozen), ("held", held)):
        lines.append(f"### {label}")
        lines.append("")
        for family in ("prefer", "avoid"):
            for n in (8, 32, 64):
                lines.append(f"- {family} N={n}: {_load_line(result, family, n)}")
        fifo = result["fifo"]
        lines.append(
            f"- FIFO: absent {fifo['n_absent']}/{fifo['n']}, "
            f"slots {_fmt(fifo['slots_mean'], 1)}, "
            f"delta {_fmt(fifo['mean_delta'])} (single-slot delta {_fmt(fifo['mean_single_delta'])})"
        )
        lines.append("")
    lines.extend(["## Cost", ""])
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
            lines.append(f"- {label} met every gate in the table above.")
        prefer_hit = result["load"]["summary"]["prefer"]["64"]["hit_n"]
        prefer_n = result["load"]["summary"]["prefer"]["64"]["n"]
        lines.append(
            f"- {label} prefer top-1 at N=64 is {prefer_hit}/{prefer_n}. "
            "That retrieval gap is out of scope and was not tuned."
        )
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
    parser = argparse.ArgumentParser(description="score frozen and held subconscious banks")
    parser.add_argument("--json", default="reports/subconscious_fix.json")
    parser.add_argument("--md", default="reports/subconscious_fix.md")
    parser.add_argument("--train-log", default="reports/subconscious_fix_train.json")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_fix_data import held_bank

    train = json.loads(Path(args.train_log).read_text(encoding="utf-8"))
    print("loading SmolLM2-135M float32", flush=True)
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    mem = LLMMemory(dtype=torch.float32)
    load_s = time.perf_counter() - t0
    rss_after = _rss_kb()
    print(f"loaded in {load_s:.1f}s gap_tau={float(mem.inject.gap_tau)}", flush=True)
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
        "train": {
            "selected_step": train.get("selected_step"),
            "selected_dev": train.get("selected_dev"),
            "tau": train.get("tau"),
            "tau_rule": train.get("tau_rule"),
            "fit_bank_sha256": train.get("fit_bank_sha256"),
            "mlp_parameters": train.get("mlp_parameters"),
            "key_parameters": train.get("key_parameters"),
        },
        "frozen": frozen,
        "held": held,
    }
    text = render_fix_report(payload)
    out_json = Path(args.json)
    out_md = Path(args.md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(text, encoding="utf-8")
    print(f"wrote {out_json} and {out_md}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
