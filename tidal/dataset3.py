"""Phase-3 dataset: the phase-2 arrays (identical rows/order, so phase-2 evaluations align by index) + public
converted streams (tidal/public_data/convert.py) + optionally a private AI-streamer turn-order set.
Outputs data/proc/p3.npz (G, Y, V, W = per-row reliability weight) and data/proc/p3_meta.parquet."""
import os, json, numpy as np, pandas as pd
from tidal import features, features_g, vap_targets
from tidal.dataset import HEADS
from tidal.public_data.manifest import DATA

PUB = ["pub_tg", "pub_irc", "pub_twitch", "pub_artemis", "pub_candor", "pub_aishell4", "pub_danmaku"]
PRIVATE_EXTRA = "data/private/ai_stream/events.parquet"      # optional, never published

def _ai_stream():
    if not os.path.exists(PRIVATE_EXTRA): return None
    d = pd.read_parquet(PRIVATE_EXTRA)
    rng = np.random.default_rng(7); convs = sorted(d.conv.unique()); u = dict(zip(convs, rng.random(len(convs))))
    d["split"] = d.conv.map(lambda c: "ai_train" if u[c] < 0.5 else "ai_test")
    # turn-order labels: "speak" = the AI host takes the next turn after this human turn
    nxt_role = d.groupby("conv").role.shift(-1); gap_next = d.groupby("conv").gap_before.shift(-1).fillna(False).astype(bool)
    d["bot_act_src"] = np.where(d.role == "other", np.where((nxt_role == "self") & ~gap_next, "speak", np.where(nxt_role.isna() | gap_next, None, "silent")), None)
    d["addr_src"] = np.nan
    from tidal.public_data.fastlabels import compute_labels_fast
    return compute_labels_fast(d)

def build():
    z = np.load("data/proc/p2.npz"); m2 = pd.read_parquet("data/proc/p2_meta.parquet")
    frames = [pd.read_parquet(DATA / "proc" / f"{s}.parquet") for s in PUB if (DATA / "proc" / f"{s}.parquet").exists()]
    ai = _ai_stream()
    if ai is not None: frames.append(ai)
    d = pd.concat(frames, ignore_index=True)
    d["weight"] = pd.to_numeric(d.get("weight"), errors="coerce").fillna(1.0)
    X = features.compute(d); G = features_g.compute(d, X); V = vap_targets.compute(d); Y = d[HEADS].to_numpy(np.float32)
    assert len(d) == len(G)
    meta = d[["source", "conv", "conv_type", "ts", "role", "split", "scenario", "speaker_known", "msg_id", "responds_to", "initiate"]].copy()
    meta["has_text"] = d.text.notna(); meta["weight"] = d.weight.to_numpy(float)
    meta["partner_kind"] = d["partner_kind"] if "partner_kind" in d else None
    m2 = m2.assign(weight=1.0, partner_kind=None)
    M = pd.concat([m2, meta], ignore_index=True)
    np.savez("data/proc/p3.tmp.npz", G=np.concatenate([z["G"], G]), Y=np.concatenate([z["Y"], Y]), V=np.concatenate([z["V"], V]),
             W=M.weight.to_numpy(np.float32), n_p2=len(m2))
    M.to_parquet("data/proc/p3_meta.tmp.parquet")
    for a, b in (("data/proc/p3.tmp.npz", "data/proc/p3.npz"), ("data/proc/p3_meta.tmp.parquet", "data/proc/p3_meta.parquet")):
        os.chmod(a, 0o600); os.replace(a, b)                     # atomic swap: running jobs keep their loaded copy
    st = {s: dict(events=int((meta.source == s).sum()), convs=int(meta[meta.source == s].conv.nunique()),
                  splits=meta[meta.source == s].split.value_counts().to_dict(),
                  label_n={h: int(np.sum(~np.isnan(Y[(meta.source == s).to_numpy(), i]))) for i, h in enumerate(HEADS)},
                  addr_pos=int(np.nansum(Y[(meta.source == s).to_numpy(), HEADS.index("y_addr")])),
                  vap_defined=float(np.mean(~np.isnan(V[(meta.source == s).to_numpy()])))) for s in meta.source.unique()}
    json.dump(st, open("data/proc/p3_stats.json", "w"), indent=1); print(json.dumps(st, indent=1))

if __name__ == "__main__": build()
