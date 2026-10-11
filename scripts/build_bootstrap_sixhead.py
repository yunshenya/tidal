#!/usr/bin/env python3
"""Build a six-head bootstrap dataset from synthetic streams + private replay labels.

The private replay rows only supervise labels that are actually observable (address/action/recheck).
The remaining heads come from script-synthetic conversations, so the resulting checkpoint is a
bootstrap baseline and must not be treated as a validated real-chat model.
"""
import csv, hashlib, hmac, json
from pathlib import Path
import numpy as np, pandas as pd
from tidal import scenarios, features
from tidal.labels import compute_labels
from tidal.dataset import HEADS

def pseudo(salt, value): return "p_" + hmac.new(salt, str(value).encode(), hashlib.sha256).hexdigest()[:16]
def parse_ts(v):
    from datetime import datetime
    import re
    v = v.replace(" ", "T")
    if re.search(r"[+-]\d{2}$", v): v += ":00"
    v = re.sub(r"\.(\d{1,6})(?=[+-])", lambda m: "." + m.group(1).ljust(6, "0"), v)
    return datetime.fromisoformat(v).timestamp()

def private_rows(path, salt):
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("scope_key") or (r.get("scope_kind", "global") + ":" + (r.get("event_id") or r["id"]))
            conv = pseudo(salt, key)
            action = r.get("chosen_action") or None
            if action not in ("speak", "silent"): action = None
            role = "other" if r.get("event_type") == "message_received" else "self"
            rows.append(dict(conv=conv, conv_type="group" if r.get("is_private") != "t" else "private",
                             ts=parse_ts(r["occurred_at"]), role=role, speaker=None, speaker_known=False,
                             text=None, addr_src=float(r.get("mentioned_bot") == "t" or r.get("reply_to_bot") == "t"),
                             bot_act_src=action, msg_id=pseudo(salt, r["id"]), source="real",
                             split="train", n_participants=np.nan, modality="text"))
    return pd.DataFrame(rows)

def main():
    root = Path(__file__).resolve().parents[1]; salt = (root / ".secrets/pseudo_salt").read_bytes().strip()
    priv = private_rows(root / "data/private/turn_replays_joined.csv", salt)
    syn = scenarios.build(n_live=8, n_1on1=60, seed=17)
    syn = compute_labels(syn); syn["source"] = "synth"
    syn["split"] = syn["split"].map({"syn_train":"synth", "syn_val":"synth_val", "syn_test":"synth_test"})
    priv = compute_labels(priv)
    d = pd.concat([priv, syn], ignore_index=True, sort=False)
    for c in ("n_participants", "modality", "msg_id", "responds_to", "initiate"):
        if c not in d: d[c] = None
    d = d.sort_values(["source", "conv", "ts"], kind="stable").reset_index(drop=True)
    X = features.compute(d)
    Y = d[HEADS].to_numpy(np.float32)
    # The synthetic split is already per conversation; private replay rows are training only.
    meta = d[["source", "conv", "conv_type", "ts", "role", "split", "speaker_known"]].copy()
    meta["split"] = meta["split"].fillna("train")
    out = root / "data/proc"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "dataset.npz", X=X, Y=Y, eidx=np.full(len(d), -1, np.int64))
    meta.to_parquet(out / "meta.parquet", index=False); np.save(out / "emb.npy", np.zeros((0,512), np.float16))
    stats = {"private_rows": int(len(priv)), "synthetic_rows": int(len(syn)),
             "private_label_counts": {h: int(np.isfinite(priv[h]).sum()) for h in HEADS},
             "synthetic_label_counts": {h: int(np.isfinite(syn[h]).sum()) for h in HEADS},
             "warning": "bootstrap only; private replay lacks original sender/message event stream"}
    (out / "bootstrap_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
if __name__ == "__main__": main()
