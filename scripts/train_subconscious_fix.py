"""Train the decision inject, then fit gap_tau on the scale bank.

Does not score ``tidal.subconscious_data`` and does not load the held-out bank.
Dev rows choose the checkpoint. The fit bank chooses ``gap_tau`` only.

    python scripts/train_subconscious_fix.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

TAUS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20)
# Same numeric budget as the unrelated-store gate. Applied at N=8 and N=32
# on the fit bank only. The smallest tau that meets both is kept.
ABS_MEAN_MAX = 0.05
LARGE = 0.1
LARGE_MAX = 2


def _dev_cache(mem, rows):
    cached = []
    for fact in rows:
        cached.append(
            {
                "family": fact["family"],
                "name": fact["name"],
                "rel": fact["rel"],
                "text": fact["text"],
                "prompt": fact["q"],
                "good": fact["good"],
                "bad": fact["bad"],
                "key": mem.encode_key(fact["text"], fact["a"]),
                "value": mem.encode_value(fact["text"], fact["a"]),
            }
        )
    return cached


def _probe(mem, cached):
    from tidal.subconscious_eval import score_pair

    buckets = {"prefer": [], "avoid": []}
    with torch.no_grad():
        for ex in cached:
            mem.store.clear()
            base = score_pair(mem, ex["prompt"], ex["good"], ex["bad"])
            mem.store.write(ex["key"], ex["value"], name=ex["name"], rel=ex["rel"])
            armed = score_pair(mem, ex["prompt"], ex["good"], ex["bad"])
            buckets[ex["family"]].append(float(armed["margin"] - base["margin"]))
    mem.store.clear()
    out = {}
    for family, deltas in buckets.items():
        out[f"{family}_mean"] = float(sum(deltas) / len(deltas))
        out[f"{family}_pos"] = int(sum(delta > 0.0 for delta in deltas))
        out[f"{family}_n"] = len(deltas)
    return out


def _eligible(stats):
    return (
        stats["prefer_mean"] > ABS_MEAN_MAX
        and stats["prefer_pos"] >= 6
        and stats["avoid_mean"] > ABS_MEAN_MAX
        and stats["avoid_pos"] >= 6
    )


def _select(snapshots):
    eligible = [row for row in snapshots if _eligible(row["stats"])]
    if eligible:
        pool = eligible
        rule = "eligible dev: prefer and avoid mean > 0.05 with at least 6 positive items; max avoid mean"
    else:
        prefer_ok = [row for row in snapshots if row["stats"]["prefer_mean"] > 0.0]
        pool = prefer_ok or snapshots
        rule = "no eligible dev step; max avoid mean among prefer-positive steps"
    chosen = max(pool, key=lambda row: (row["stats"]["avoid_mean"], row["step"]))
    return chosen, rule


def _encode_map(mem, facts):
    found = {}
    for fact in facts:
        key = (fact["text"], fact["answer"], fact["name"], fact["rel"])
        if key in found:
            continue
        found[key] = {
            "key": mem.encode_key(fact["text"], fact["answer"]),
            "value": mem.encode_value(fact["text"], fact["answer"]),
            "name": fact["name"],
            "rel": fact["rel"],
        }
    return found


def _fit_gap(mem):
    """Smallest tau whose unrelated stores on the fit bank meet the budget."""
    from tidal.subconscious_data import distractors
    from tidal.subconscious_eval import score_pair
    from tidal.subconscious_fix_data import fit_bank

    bank = fit_bank()
    items = list(bank.PREFER)
    incidental = bank.incidental_facts()
    encoded = _encode_map(mem, list(items) + incidental)
    mem.store.clear()
    nomem = [
        score_pair(mem, item["prompt"], item["good"], item["bad"])
        for item in items
    ]
    trials = []
    for tau in TAUS:
        mem.inject.gap_tau.fill_(float(tau))
        for n_load in (8, 32):
            deltas = []
            for index, item in enumerate(items):
                mem.store.clear()
                for fact in distractors(index, n_load, incidental):
                    row = encoded[(fact["text"], fact["answer"], fact["name"], fact["rel"])]
                    mem.store.write(row["key"], row["value"], name=row["name"], rel=row["rel"])
                armed = score_pair(mem, item["prompt"], item["good"], item["bad"])
                deltas.append(float(armed["margin"] - nomem[index]["margin"]))
            mean = float(sum(deltas) / len(deltas))
            n_large = int(sum(abs(delta) > LARGE for delta in deltas))
            trials.append(
                {
                    "tau": float(tau),
                    "n": int(n_load),
                    "mean_delta": mean,
                    "abs_mean": abs(mean),
                    "n_large": n_large,
                    "n_items": len(deltas),
                    "pass": bool(abs(mean) <= ABS_MEAN_MAX and n_large <= LARGE_MAX),
                }
            )
            print(trials[-1], flush=True)
    by_tau = {}
    for row in trials:
        by_tau.setdefault(row["tau"], {})[row["n"]] = row
    passing = [
        tau
        for tau, rows in by_tau.items()
        if rows[8]["pass"] and rows[32]["pass"]
    ]
    if passing:
        chosen = float(min(passing))
        rule = "smallest tau that passes the unrelated budget at N=8 and N=32"
    else:
        chosen = float(
            min(
                by_tau,
                key=lambda tau: (
                    max(by_tau[tau][8]["abs_mean"], by_tau[tau][32]["abs_mean"]),
                    by_tau[tau][8]["n_large"] + by_tau[tau][32]["n_large"],
                    -tau,
                ),
            )
        )
        rule = "no tau passed both budgets; smallest worst-case absolute mean, then fewer large items, then larger tau"
    mem.store.clear()
    return chosen, rule, trials, bank.sha256()


def main():
    parser = argparse.ArgumentParser(description="decision inject and gap_tau fit")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--out", default="tidal/llm_inject_smollm2_135m.pt")
    parser.add_argument("--log", default="reports/subconscious_fix_train.json")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)

    from tidal.llm_memory import LLMMemory
    from tidal.llm_memory_train import fit_decision
    from tidal.subconscious_fix_data import dev_rows, train_rows

    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    phone_w = mem.inject.phone_w.detach().cpu().clone()
    date_w = mem.inject.date_digit_w.detach().cpu().clone()
    phone_on = float(mem.inject.phone_on.detach().cpu())
    date_on = float(mem.inject.date_on.detach().cpu())
    if phone_on != 1.0 or date_on != 1.0:
        raise SystemExit(f"readouts must stay on, phone_on={phone_on} date_on={date_on}")
    dev = _dev_cache(mem, dev_rows())
    snapshots = []

    def on_step(step):
        stats = _probe(mem, dev)
        snapshots.append(
            {
                "step": int(step),
                "stats": stats,
                "state": {key: tensor.detach().cpu().clone() for key, tensor in mem.inject.state_dict().items()},
            }
        )
        return stats

    log = fit_decision(
        mem,
        train_rows(),
        steps=args.steps,
        bs=args.bs,
        lr=args.lr,
        margin=args.margin,
        seed=args.seed,
        weight_decay=args.weight_decay,
        on_step=on_step,
    )
    chosen, select_rule = _select(snapshots)
    mem.inject.load_state_dict(chosen["state"])
    mem.inject.eval()
    print({"selected_step": chosen["step"], "rule": select_rule, **chosen["stats"]}, flush=True)
    tau, tau_rule, trials, fit_sha = _fit_gap(mem)
    mem.inject.gap_tau.fill_(float(tau))
    if not torch.equal(mem.inject.phone_w.detach().cpu(), phone_w):
        raise SystemExit("decision fit changed phone_w")
    if not torch.equal(mem.inject.date_digit_w.detach().cpu(), date_w):
        raise SystemExit("decision fit changed date_digit_w")
    if float(mem.inject.phone_on) != 1.0 or float(mem.inject.date_on) != 1.0:
        raise SystemExit("decision fit turned a readout off")
    out = Path(args.out)
    mem.save_inject(out)
    params = sum(p.numel() for name, p in mem.inject.named_parameters() if name.startswith(("net.", "key_proj.")))
    key_params = sum(p.numel() for name, p in mem.inject.named_parameters() if name.startswith("key_proj."))
    record = {
        "steps": int(args.steps),
        "bs": int(args.bs),
        "lr": float(args.lr),
        "margin": float(args.margin),
        "seed": int(args.seed),
        "weight_decay": float(args.weight_decay),
        "select_rule": select_rule,
        "selected_step": int(chosen["step"]),
        "selected_dev": chosen["stats"],
        "dev_trace": [{"step": row["step"], "stats": row["stats"]} for row in snapshots],
        "loss_log": log,
        "tau": float(tau),
        "tau_rule": tau_rule,
        "tau_candidates": list(TAUS),
        "tau_trials": trials,
        "fit_bank_sha256": fit_sha,
        "frozen_bank_sha256": "e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66",
        "mlp_parameters": int(params),
        "key_parameters": int(key_params),
        "phone_on": True,
        "date_on": True,
        "scored_frozen": False,
        "scored_held": False,
    }
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print({"saved": str(out), "tau": tau, "log": str(log_path)}, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
