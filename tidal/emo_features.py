"""Phase 4A integration: per-event emotion features for the event encoder, computed LOCALLY from the cached bge
embeddings of each message (nothing leaves the box). 11 columns: 8 calibrated class probabilities, valence, arousal,
has_emotion flag. Rows without text (and all public rows) are 0 (== unknown, the FG convention).
Ensemble = mean over the 3 text-head seeds. -> data/proc/p3_emo.npy (row-aligned with data/proc/p3.npz)."""
import numpy as np, torch
from tidal.emotion import EmoHead

def heads():
    hs = []
    for s in range(3):
        ck = torch.load(f"models/emo_text_s{s}.pt", weights_only=False); h = EmoHead(); h.load_state_dict(ck["state"]); hs.append(h.eval())
    return hs

def main():
    z = np.load("data/proc/dataset.npz"); eidx = z["eidx"]; emb = np.load("data/proc/emb.npy").astype(np.float32)
    n = len(np.load("data/proc/p3.npz", mmap_mode="r")["Y"]) if False else None
    import pandas as pd
    n = len(pd.read_parquet("data/proc/p3_meta.parquet", columns=["source"]))
    out = np.zeros((n, 11), np.float32); has = np.flatnonzero(eidx >= 0)
    f = np.mean([h.features(emb[eidx[has]]) for h in heads()], 0)
    out[has, :10] = f; out[has, 10] = 1.0
    np.save("data/proc/p3_emo.npy", out); print("rows with emotion features:", len(has), "of", n)

if __name__ == "__main__": main()
