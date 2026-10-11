#!/usr/bin/env python3
"""Assemble private replay + InteractSpeech + script data for six-head training."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from tidal import scenarios, features
from tidal.labels import compute_labels
from tidal.dataset import HEADS
from scripts.build_bootstrap_sixhead import private_rows

def main():
    root = Path(__file__).resolve().parents[1]; salt = (root / ".secrets/pseudo_salt").read_bytes().strip()
    private = private_rows(root / "data/private/turn_replays_joined.csv", salt)
    public = pd.read_parquet(root / "data/public/proc/pub_interactspeech.parquet")
    public["split"] = public["split"].replace({"pub_train":"pub_train", "pub_val":"pub_val", "pub_test":"pub_test"})
    syn = compute_labels(scenarios.build(n_live=8, n_1on1=60, seed=17)); syn["source"] = "synth"
    syn["split"] = syn["split"].map({"syn_train":"synth", "syn_val":"synth_val", "syn_test":"synth_test"})
    private = compute_labels(private)
    d = pd.concat([private, public, syn], ignore_index=True, sort=False)
    for c in ("n_participants", "modality", "msg_id", "responds_to", "initiate"):
        if c not in d: d[c] = None
    d = d.sort_values(["source", "conv", "ts"], kind="stable").reset_index(drop=True)
    X = features.compute(d); Y = d[HEADS].to_numpy(np.float32)
    meta = d[["source", "conv", "conv_type", "ts", "role", "split", "speaker_known"]].copy()
    meta["split"] = meta["split"].fillna("train")
    out = root / "data/proc"; out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "dataset.npz", X=X, Y=Y, eidx=np.full(len(d), -1, np.int64))
    meta.to_parquet(out / "meta.parquet", index=False); np.save(out / "emb.npy", np.zeros((0,512), np.float16))
    stats = {"private_rows": int(len(private)), "interactspeech_rows": int(len(public)), "synthetic_rows": int(len(syn)),
             "labels": {h: int(np.isfinite(d[h]).sum()) for h in HEADS},
             "warning": "public timing corpus has no group addressee labels; y_addr is weak/private or perspective-derived"}
    (out / "full_public_sixhead_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
if __name__ == "__main__": main()
