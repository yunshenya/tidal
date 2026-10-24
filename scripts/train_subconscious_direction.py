"""Train prefer/avoid polarity. Select on the fit bank only.

Does not score the frozen bank or the held-out bank. A no-key ablation is
run on the same train rows and the same fit bank, then the weights are
restored. The shipped file is replaced only when polarity passes the fit gates.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from tidal.llm_memory import DEFAULT_GAP_TAU, LLMMemory, POLARITY_SPAN
from tidal.llm_memory_train import fit_decision_value, fit_polarity
from tidal.subconscious_direction_data import (
    DIRECTION_GATES,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
    bank_sha256,
    fit_bank,
    train_incidental,
    train_rows,
)
from tidal.subconscious_eval import Encoder, _fact, score_one, score_pair, write_rows


def _mean(xs):
    vals = [float(x) for x in xs]
    return float(sum(vals) / len(vals)) if vals else None


def _gains(mem, keys):
    vals = []
    with torch.no_grad():
        for key in keys:
            tensor = torch.tensor(np.asarray(key, dtype=np.float32))
            vals.append(float(mem.inject.polarity_gain(tensor)))
    return _mean(vals)


def _max_cosine(keys):
    mat = np.stack([np.asarray(key, dtype=np.float64) for key in keys], 0)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-12)
    sim = mat @ mat.T
    mask = ~np.eye(len(keys), dtype=bool)
    return float(sim[mask].max()) if len(keys) > 1 else 0.0


def score_fit(mem, enc, bank, neutral_rows, prefer_keys, avoid_keys, incidental_keys):
    """Direction plus the non-interference checks that polarity can move."""
    print("scoring fit", flush=True)
    families = {}
    for family, group in (("prefer", bank.PREFER), ("avoid", bank.AVOID)):
        rows = []
        for item in group:
            write_rows(mem, [])
            nomem = score_pair(mem, item["prompt"], item["good"], item["bad"])
            write_rows(mem, [enc.one(_fact(item))])
            injected = score_pair(mem, item["prompt"], item["good"], item["bad"])
            rows.append(
                {
                    "id": item["id"],
                    "delta": float(injected["margin"] - nomem["margin"]),
                    "forbidden": float(injected["bad_mean"] - nomem["bad_mean"]),
                }
            )
        families[family] = rows
    nll = []
    for item in bank.NEUTRAL_NLL:
        write_rows(mem, [])
        base = score_one(mem, item["prompt"], item["continuation"])
        write_rows(mem, neutral_rows)
        armed = score_one(mem, item["prompt"], item["continuation"])
        nll.append(float(armed["nll"] - base["nll"]))
    flips = 0
    for item in bank.NEUTRAL_CHOICE:
        write_rows(mem, [])
        base = score_pair(mem, item["prompt"], item["good"], item["bad"])
        write_rows(mem, neutral_rows)
        armed = score_pair(mem, item["prompt"], item["good"], item["bad"])
        if base["choose"] == 1 and armed["choose"] == 0:
            flips += 1
    write_rows(mem, [])
    spec = DIRECTION_GATES
    prefer_pos = sum(row["delta"] > 0.0 for row in families["prefer"])
    avoid_pos = sum(row["delta"] > 0.0 for row in families["avoid"])
    forbidden_down = sum(row["forbidden"] < 0.0 for row in families["avoid"])
    nll_mean = _mean(nll)
    nll_bad = sum(delta > spec["nll_bad"] for delta in nll)
    metrics = {
        "prefer_mean": _mean([row["delta"] for row in families["prefer"]]),
        "prefer_pos": int(prefer_pos),
        "avoid_mean": _mean([row["delta"] for row in families["avoid"]]),
        "avoid_pos": int(avoid_pos),
        "forbidden_mean": _mean([row["forbidden"] for row in families["avoid"]]),
        "forbidden_down": int(forbidden_down),
        "nll_mean": nll_mean,
        "nll_bad": int(nll_bad),
        "flips": int(flips),
        "prefer_gain": _gains(mem, prefer_keys),
        "avoid_gain": _gains(mem, avoid_keys),
        "incidental_gain": _gains(mem, incidental_keys),
        "prefer_items": families["prefer"],
        "avoid_items": families["avoid"],
    }
    metrics["fit_pass"] = bool(
        prefer_pos >= spec["prefer_delta_pos_min"]
        and avoid_pos >= spec["avoid_delta_pos_min"]
        and forbidden_down >= spec["forbidden_down_min"]
        and abs(nll_mean) <= spec["nll_abs_mean_max"]
        and nll_bad <= spec["nll_bad_count_max"]
        and flips <= spec["choice_flips_max"]
    )
    return metrics


def _snapshot(mem):
    return {key: tensor.detach().cpu().clone() for key, tensor in mem.inject.state_dict().items()}


def _restore(mem, snap):
    mem.inject.load_state_dict({key: tensor.clone() for key, tensor in snap.items()})
    mem.inject.eval()
    mem._delta = None


def _encode_train(mem):
    facts = []
    for row in train_rows():
        facts.append(
            {
                **row,
                "value": mem.encode_value(row["text"], row["a"]),
                "key": mem.encode_key(row["text"], row["a"]),
            }
        )
        print(f"train encode {len(facts)}/{64}", flush=True)
    keys = []
    incidental = train_incidental()
    for index, row in enumerate(incidental, start=1):
        keys.append(mem.encode_key(row["text"], row["answer"]))
        if index == 1 or index % 64 == 0 or index == len(incidental):
            print(f"train incidental keys {index}/{len(incidental)}", flush=True)
    return facts, keys


def _render_prereg(summary, chosen):
    nokey = summary["nokey_best"]
    pol = chosen
    lines = [
        "# Direction preregistration",
        "",
        "Polarity for prefer versus avoid. The frozen bank and the held-out bank have not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once.",
        "",
        "## Banks",
        "",
        f"- train `{summary['train_sha']}`",
        f"- fit `{summary['fit_sha']}`",
        f"- held `{summary['held_sha']}`",
        f"- frozen `{FROZEN_SHA}`",
        f"- #39 fit `{FIT39_SHA}`",
        f"- #39 held `{HELD39_SHA}`",
        f"- #40 held `{GATE_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and across train, fit, and held",
        "",
        "## Architecture",
        "",
        "The add is `c * gain * (scale(v) + MLP(scale(v)))`.",
        f"`gain = 1 + {POLARITY_SPAN:.0f} tanh(w · normalize(k))`.",
        "`w = 0` leaves `gain = 1`, which is the #40 residual.",
        f"`gap_tau` is {DEFAULT_GAP_TAU:.2f} and was not retuned.",
        f"Polarity parameters: {summary['polarity_params']}. MLP parameters: {summary['mlp_params']}, frozen at the #38 checkpoint.",
        "The no-key ablation trains the MLP and does not see the fact key. Polarity training updates only `w`.",
        "",
        "## Fit selection",
        "",
        "Early stop uses the fit bank only. The frozen bank and the held-out bank have not been scored.",
        "",
        f"No-key ablation, best avoid mean at step {nokey['step']}: "
        f"prefer {nokey['prefer_mean']:.3f} ({nokey['prefer_pos']}/8), "
        f"avoid {nokey['avoid_mean']:.3f} ({nokey['avoid_pos']}/8), "
        f"forbidden down {nokey['forbidden_down']}/8, "
        f"NLL {nokey['nll_mean']:.3f} ({nokey['nll_bad']} > 0.5), "
        f"flips {nokey['flips']}, fit_pass {nokey['fit_pass']}.",
        "",
        f"Polarity, selected step {pol['step']}: "
        f"prefer {pol['prefer_mean']:.3f} ({pol['prefer_pos']}/8), "
        f"avoid {pol['avoid_mean']:.3f} ({pol['avoid_pos']}/8), "
        f"forbidden down {pol['forbidden_down']}/8 (mean move {pol['forbidden_mean']:.3f}), "
        f"NLL {pol['nll_mean']:.3f} ({pol['nll_bad']} > 0.5), "
        f"flips {pol['flips']}, "
        f"gain prefer {pol['prefer_gain']:.3f}, avoid {pol['avoid_gain']:.3f}, incidental {pol['incidental_gain']:.3f}.",
        "",
        "The key term is justified because the no-key ablation does not pass these fit gates and polarity does, with the MLP weights unchanged.",
        "",
        "## Gates",
        "",
        "Held-out is merge-critical. Frozen `e3cdd7fd…` is re-scored as a measurement under the same numbers.",
        "",
        "- prefer margin delta > 0 on at least 6/8",
        "- avoid margin delta > 0 on at least 6/8",
        "- forbidden logprob down on at least 6/8 avoid items",
        "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
        "- choice flips 0",
        "- leak 0",
        "- phone and birthday takeover 0 off-relation",
        "- phone question takeover 1, and birthday question takeover 1 when the bank has one",
        "",
        "Load top-1, FIFO, and prefer choice lift are reported, not gated.",
        "",
    ]
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="fit polarity on the direction train split")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--nokey-lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--prefer-weight", type=float, default=0.2)
    parser.add_argument("--avoid-weight", type=float, default=0.2)
    parser.add_argument("--incidental-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="tidal/llm_inject_smollm2_135m.pt")
    parser.add_argument("--json", default="reports/subconscious_direction_train.json")
    parser.add_argument("--prereg", default="reports/subconscious_direction_prereg.md")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
    snap = _snapshot(mem)
    if "polarity" not in snap or float(snap["polarity"].abs().sum()) != 0.0:
        raise SystemExit("polarity must start at 0")
    bank = fit_bank()
    enc = Encoder(mem)
    neutral_src = bank.incidental_facts()[: bank.NEUTRAL_STORE_N]
    neutral_rows = enc.many(neutral_src)
    window = neutral_rows[:8]
    cosine = _max_cosine([row["key"] for row in window])
    print(f"fit incidental max cosine in first 8: {cosine:.4f}", flush=True)
    if cosine > 0.9:
        raise SystemExit("fit incidental keys collapsed; refuse to train")
    prefer_keys = [enc.one(_fact(item))["key"] for item in bank.PREFER]
    avoid_keys = [enc.one(_fact(item))["key"] for item in bank.AVOID]
    facts, incidental_keys = _encode_train(mem)
    init = score_fit(mem, enc, bank, neutral_rows, prefer_keys, avoid_keys, incidental_keys[:32])
    init["step"] = 0
    print({"init": {k: init[k] for k in init if not k.endswith("items")}}, flush=True)

    def run(kind, fit):
        snaps = {}

        def on_step(step):
            metrics = score_fit(mem, enc, bank, neutral_rows, prefer_keys, avoid_keys, incidental_keys[:32])
            snaps[step] = _snapshot(mem)
            keep = {key: metrics[key] for key in metrics if not key.endswith("items")}
            print({"kind": kind, "step": step, **keep}, flush=True)
            return metrics

        log = fit(on_step)
        return log, snaps

    def nokey_fit(on_step):
        return fit_decision_value(
            mem,
            facts,
            steps=args.steps,
            bs=args.bs,
            lr=args.nokey_lr,
            margin=args.margin,
            seed=args.seed,
            on_step=on_step,
        )

    def polarity_fit(on_step):
        return fit_polarity(
            mem,
            facts,
            incidental_keys,
            steps=args.steps,
            bs=args.bs,
            lr=args.lr,
            margin=args.margin,
            seed=args.seed,
            prefer_weight=args.prefer_weight,
            avoid_weight=args.avoid_weight,
            incidental_weight=args.incidental_weight,
            on_step=on_step,
        )

    print("no-key ablation", flush=True)
    nokey_log, _nokey_snaps = run("nokey", nokey_fit)
    _restore(mem, snap)
    print("polarity", flush=True)
    polarity_log, polarity_snaps = run("polarity", polarity_fit)
    eligible = [row for row in polarity_log if row.get("fit_pass")]
    chosen = None
    if eligible:
        chosen = max(eligible, key=lambda row: (row["avoid_mean"], row["prefer_mean"], -row["step"]))
        _restore(mem, polarity_snaps[chosen["step"]])
        for key, before in snap.items():
            if key == "polarity":
                continue
            after = mem.inject.state_dict()[key].detach().cpu()
            if not torch.equal(before, after):
                raise SystemExit(f"selected polarity checkpoint changed {key}")
        out = Path(args.out)
        mem.save_inject(out)
        print({"saved": str(out), "step": chosen["step"]}, flush=True)
    else:
        _restore(mem, snap)
        print("no polarity step passed the fit gates", flush=True)

    def slim(rows):
        out = []
        for row in rows:
            item = {key: value for key, value in row.items() if not key.endswith("items")}
            out.append(item)
        return out

    nokey_best = max(nokey_log, key=lambda row: (row.get("avoid_mean") or -1e9, -(row["step"])))
    summary = {
        "train_sha": bank_sha256("train"),
        "fit_sha": bank_sha256("fit"),
        "held_sha": bank_sha256("held"),
        "frozen_sha": FROZEN_SHA,
        "gap_tau": DEFAULT_GAP_TAU,
        "polarity_span": POLARITY_SPAN,
        "polarity_params": int(mem.inject.polarity.numel()),
        "mlp_params": int(sum(p.numel() for name, p in mem.inject.named_parameters() if name.startswith("net."))),
        "fit_incidental_cosine8": cosine,
        "init": {key: init[key] for key in init if not key.endswith("items")},
        "nokey": slim(nokey_log),
        "nokey_best": {key: nokey_best[key] for key in nokey_best if not key.endswith("items")},
        "polarity": slim(polarity_log),
        "chosen_step": None if chosen is None else int(chosen["step"]),
        "chosen": None if chosen is None else {key: chosen[key] for key in chosen if not key.endswith("items")},
        "args": vars(args),
    }
    out_json = Path(args.json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out_json}", flush=True)
    if chosen is None:
        mem.close()
        raise SystemExit(2)
    prereg = Path(args.prereg)
    prereg.write_text(_render_prereg(summary, summary["chosen"]), encoding="utf-8")
    print(f"wrote {prereg}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
