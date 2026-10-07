"""Assemble real + synthetic frames: labels, splits, causal features, embedding indices -> data/proc/*.npz"""
import json, os, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from tidal.adapters import real_frame, synth_frame
from tidal.labels import compute_labels
from tidal import features, embed
TZ = timezone(timedelta(hours=8))
VAL_START = datetime(2026, 10, 1, tzinfo=TZ).timestamp()
TEST_START = datetime(2026, 10, 4, tzinfo=TZ).timestamp()
EMBARGO = 600.0   # label windows reach <=300 s ahead; drop labels of events within 10 min before a split boundary
from tidal.config import HOLDOUT_GROUPS   # unseen-group test (never in training, any time); set in config/local.json
HEADS = ["y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply"]
def assign_split(d):
    s = np.where(d.ts < VAL_START, "train", np.where(d.ts < TEST_START, "val", "test_time")).astype(object)
    emb = ((d.ts >= VAL_START - EMBARGO) & (d.ts < VAL_START)) | ((d.ts >= TEST_START - EMBARGO) & (d.ts < TEST_START))
    s[emb.to_numpy()] = "embargo"
    s[d.conv.isin(HOLDOUT_GROUPS).to_numpy()] = "test_group"
    return s
def build(with_synth=True):
    real = compute_labels(real_frame()); real["split"] = assign_split(real)
    frames = [real]
    if with_synth and os.path.exists("data/synth/clean.jsonl"):
        syn = compute_labels(synth_frame()); syn["split"] = "synth"; frames.append(syn)
    d = pd.concat(frames, ignore_index=True).sort_values(["source", "conv", "ts"], kind="stable").reset_index(drop=True)
    X = features.compute(d)
    k2i, emb, st = embed.ensure(d.text.dropna().tolist())
    eidx = np.array([k2i[embed.key(t)] if isinstance(t, str) and t else -1 for t in d.text], np.int64)
    Y = d[HEADS].to_numpy(np.float32)
    meta = d[["source", "conv", "conv_type", "ts", "role", "split", "speaker_known"]].copy()
    meta["has_text"] = d.text.notna()
    meta["has_name"] = X[:, features.FEATURES.index("has_name")] > 0
    os.makedirs("data/proc", exist_ok=True)
    np.savez("data/proc/dataset.npz", X=X, Y=Y, eidx=eidx)
    meta.to_parquet("data/proc/meta.parquet")
    np.save("data/proc/emb.npy", emb.astype(np.float16))
    return d, X, Y, eidx, meta, st
def stats(d):
    out = {}
    for src, g in d.groupby("source"):
        s = dict(messages=len(g), conversations=g.conv.nunique(), bot_messages=int((g.role == "self").sum()),
                 with_text=int(g.text.notna().sum()), speaker_known=int(g.speaker_known.sum()))
        if src == "real":
            s["groups"] = int(g[g.conv_type == "group"].conv.nunique()); s["private_chats"] = int(g[g.conv_type == "private"].conv.nunique())
            s["date_range"] = [datetime.fromtimestamp(g.ts.min(), TZ).isoformat(timespec="minutes"), datetime.fromtimestamp(g.ts.max(), TZ).isoformat(timespec="minutes")]
            s["by_split"] = {}
            for sp, gg in g.groupby("split"):
                s["by_split"][sp] = dict(messages=len(gg), **{h: label_rate(gg[h]) for h in HEADS})
        s["labels"] = {h: label_rate(g[h]) for h in HEADS}
        out[src] = s
    return out
def label_rate(v):
    v = v.dropna()
    if len(v) == 0: return dict(n=0)
    vc = v.value_counts(normalize=True).sort_index()
    return dict(n=int(len(v)), dist={str(int(k)): round(float(x), 4) for k, x in vc.items()})
if __name__ == "__main__":
    d, X, Y, eidx, meta, st = build()
    s = stats(d); s["embedding_new"] = st
    json.dump(s, open("data/proc/dataset_stats.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(s, indent=1, ensure_ascii=False)[:6000])
