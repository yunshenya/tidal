"""Cache frozen body + text-emotion hidden states for phase-5 rows.

The window is the same 64-event causal window training uses. Default body is
P4emo_mamba3_siso_s0 (the pre-registered phase-5 run, cache data/proc/p5_h.npy). The
shadow winner's cache uses P4emo_m3_ablate_m2_s0 -> data/proc/p5_h_m2.npy. Body weights
are not updated. Resume-safe: one memmap, a row is skipped once its slot is marked done.
usage: python -m tidal.phase5_encode [labelled|live|all] [BODY_TAG OUT_NAME]
"""
import os, json
import numpy as np, pandas as pd, torch
from tidal.vap import VAPModel
from tidal.seqdata import CTX

PROC = "data/proc"


def _windows(X):
    """X [L, F] -> feats [L, 64, F], valid [L, 64] (left-padded causal windows)."""
    L, F = X.shape
    full = np.concatenate([np.zeros((CTX - 1, F), np.float32), X], 0)
    from numpy.lib.stride_tricks import sliding_window_view
    W = sliding_window_view(full, CTX, axis=0)          # [L, F, 64]
    feats = np.moveaxis(W, -1, 1)
    ev = np.arange(L)[:, None] - (CTX - 1) + np.arange(CTX)[None, :]
    valid = ev >= 0
    return feats, valid


def encode(which="labelled", bs=128, body="P4emo_mamba3_siso_s0", out="p5_h"):
    torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", "1")))
    rows = pd.read_parquet(PROC + "/p5_rows.parquet", columns=["source", "conv", "ts", "p3_index", "livestream", "y_interrupt", "y_addr", "y_topic", "split"])
    X = np.load(PROC + "/p5_X.npy", mmap_mode="r")
    n, F = X.shape
    assert len(rows) == n
    HP, DONE = f"{PROC}/{out}.npy", f"{PROC}/{out}_done.npy"
    h = np.lib.format.open_memmap(HP, mode="w+" if not os.path.exists(HP) else "r+", dtype=np.float16, shape=(n, 128))
    if os.path.exists(DONE) and np.load(DONE).shape == (n,):
        done = np.load(DONE)
    else:
        done = np.zeros(n, np.uint8)
    lab = rows.y_interrupt.notna().to_numpy() | rows.y_addr.notna().to_numpy() | rows.y_topic.notna().to_numpy()
    if which == "labelled":
        want = lab & (done == 0)
    elif which == "live":
        want = (rows.livestream.to_numpy() == 1) & (done == 0)
    else:
        want = done == 0
    ck = torch.load(f"models/{body}.pt", map_location="cpu", weights_only=False)
    model = VAPModel(ck["n_feat"], kind=ck["kind"])
    model.load_state_dict(ck["state"])
    model.eval()
    print(f"encode {which} body={body} kind={ck['kind']} -> {HP}: todo {int(want.sum())} of {n}", flush=True)
    # group by conversation, but only conversations that still have wanted rows
    conv = rows.conv.to_numpy()
    src = rows.source.to_numpy()
    ts = rows.ts.to_numpy()
    # group by (source, conv) and keep time order inside the conversation
    key = np.array([f"{a}\x00{b}" for a, b in zip(src.astype(str), conv.astype(str))])
    order = np.lexsort((ts, key))
    key_s = key[order]
    cuts = np.flatnonzero(key_s[1:] != key_s[:-1]) + 1
    cuts = np.r_[0, cuts, len(order)]
    nwin = 0
    last_save = 0
    import time
    t0 = time.time()
    with torch.no_grad():
        for a, b in zip(cuts[:-1], cuts[1:]):
            ix = order[a:b]
            if not want[ix].any():
                continue
            # rows of one conv are stored in time order already (build sorted by ts within conv for new
            # frames; p3 meta is sorted by conv, ts). Re-sort by the original frame order, which is time.
            feats, valid = _windows(np.asarray(X[ix], np.float32))
            # encode every position in the conv so the window context is the real stream, then keep wanted
            H = np.empty((len(ix), 128), np.float16)
            for s in range(0, len(ix), bs):
                f = np.ascontiguousarray(feats[s:s + bs])
                v = np.ascontiguousarray(valid[s:s + bs])
                o = model(torch.from_numpy(f), torch.from_numpy(v))
                H[s:s + bs] = o["h"][:, -1].numpy().astype(np.float16)
                nwin += len(f)
            h[ix] = H
            done[ix] = 1
            if nwin - last_save >= 8000:
                np.save(DONE, done)
                h.flush()
                last_save = nwin
                print(f"encoded_windows {nwin} marked {int(done.sum())} {time.time()-t0:.0f}s", flush=True)
    np.save(DONE, done)
    h.flush()
    print(f"encode {which} done windows {nwin} in {time.time()-t0:.0f}s marked {int(done.sum())}", flush=True)


if __name__ == "__main__":
    import sys
    a = sys.argv[1:]
    encode(a[0] if a else "labelled", **(dict(body=a[1], out=a[2]) if len(a) >= 3 else {}))
