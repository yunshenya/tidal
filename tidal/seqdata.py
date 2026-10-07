"""Windowing of per-conversation event streams for a regime (T: all events; TS: text-available events)."""
import numpy as np, pandas as pd
from tidal.features import cols
from tidal.dataset import HEADS
CTX = 64
def load(regime):
    z = np.load("data/proc/dataset.npz"); X, Y, eidx = z["X"], z["Y"], z["eidx"]
    meta = pd.read_parquet("data/proc/meta.parquet"); emb = np.load("data/proc/emb.npy")
    keep = np.ones(len(X), bool) if regime == "T" else (eidx >= 0)
    Xr = X[:, cols(regime)]
    tr = keep & (meta.source == "real").to_numpy() & (meta.split == "train").to_numpy()
    mu, sd = Xr[tr].mean(0), Xr[tr].std(0) + 1e-6
    Xr = ((Xr - mu) / sd).astype(np.float32)
    return dict(X=Xr, Y=Y, eidx=eidx, emb=emb, meta=meta, keep=keep, mu=mu, sd=sd)
def conv_index(D):
    """list of arrays of global row indices (kept rows) per conversation, time-ordered."""
    m = D["meta"]; idx = np.flatnonzero(D["keep"]); conv = m.conv.to_numpy()[idx]
    out = []; starts = np.r_[0, np.flatnonzero(conv[1:] != conv[:-1]) + 1, len(idx)]
    for a, b in zip(starts[:-1], starts[1:]): out.append(idx[a:b])
    return out
def train_windows(convs, stride=32):
    """[(rows<=64, loss_from)]: every position gets loss exactly once, with >= 64-stride context after the first window."""
    W = []
    for rows in convs:
        n = len(rows)
        W.append((rows[:CTX], 0))
        prev = min(n, CTX)
        while prev < n:
            end = min(prev + stride, n)
            W.append((rows[end - CTX:end], CTX - (end - prev))); prev = end
    return W
def eval_windows(convs, targets):
    """one window per target row: the last <=64 events of its conversation ending at the target."""
    pos = {}
    for rows in convs:
        for k, r in enumerate(rows): pos[r] = (rows, k)
    return [(pos[t][0][max(0, pos[t][1] - CTX + 1): pos[t][1] + 1]) for t in targets]
def batchify(D, wins, use_text, loss_from=None):
    B = len(wins); T = max(len(w) for w in wins)
    F = np.zeros((B, T, D["X"].shape[1]), np.float32); E = np.zeros((B, T, 512), np.float32) if use_text else None
    Yb = np.full((B, T, len(HEADS)), np.nan, np.float32); valid = np.zeros((B, T), bool)
    for i, w in enumerate(wins):
        L = len(w); o = T - L                                # left-pad so the newest event is at position T-1
        F[i, o:] = D["X"][w]; Yb[i, o:] = D["Y"][w]; valid[i, o:] = True
        if loss_from is not None: Yb[i, : o + loss_from[i]] = np.nan
        if use_text:
            e = D["eidx"][w]; ok = e >= 0
            E[i, o:][ok] = D["emb"][e[ok]].astype(np.float32)
    return F, E, Yb, valid
