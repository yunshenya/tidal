#!/usr/bin/env python3
"""Assemble private replay + InteractSpeech + script data for six-head training."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from tidal import scenarios, features, embed
from tidal.labels import compute_labels
from tidal.dataset import HEADS
from scripts.build_bootstrap_sixhead import private_rows

def main():
    root = Path(__file__).resolve().parents[1]; salt = (root / ".secrets/pseudo_salt").read_bytes().strip()
    private = private_rows(root / "data/private/turn_replays_joined.csv", salt)
    public_frames = [pd.read_parquet(root / "data/public/proc/pub_interactspeech.parquet"),
                     pd.read_parquet(root / "data/public/proc/pub_magicdata_ms.parquet")]
    turngate = root / "data/private/processed/turngate_group.parquet"
    if turngate.exists(): public_frames.append(pd.read_parquet(turngate))
    public = pd.concat(public_frames, ignore_index=True, sort=False)
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
    texts = [t for t in d.text.tolist() if isinstance(t, str) and t]
    # Singleton quantized inference matches interactive inference; a dedicated
    # cache avoids mixing vectors produced by batch-dependent int8 activations.
    k2i, emb, embed_stats = embed.ensure(texts, threads=int(os.environ.get("TIDAL_EMBED_THREADS", "1")),
                                       bs=1, workers=int(os.environ.get("TIDAL_EMBED_WORKERS", "4")),
                                       cache="data/emb/singleton_cache.npz")
    eidx = np.array([k2i[embed.key(t)] if isinstance(t, str) and t else -1 for t in d.text], np.int64)
    np.savez(out / "dataset.npz", X=X, Y=Y, eidx=eidx)
    meta["has_text"] = eidx >= 0
    meta.to_parquet(out / "meta.parquet", index=False)
    stats = {"private_rows": int(len(private)), "public_rows": int(len(public)), "synthetic_rows": int(len(syn)),
             "labels": {h: int(np.isfinite(d[h]).sum()) for h in HEADS},
             "embedding": embed_stats, "unique_embedded_texts": int(len(emb)),
             "warning": "public timing corpus has no group addressee labels; y_addr is weak/private or perspective-derived"}
    np.save(out / "emb.npy", emb.astype(np.float16))
    (out / "full_public_sixhead_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
if __name__ == "__main__": main()
