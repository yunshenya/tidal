"""Phase-2 dataset: real group/private chat + LLM-synthetic group chat (phase 1) + script-synthetic livestream and 1:1
streams (tidal/scenarios.py). Outputs data/proc/p2.npz (scenario-general inputs, downstream labels, VAP targets) and
data/proc/p2_meta.parquet. Real rows come first, in exactly the phase-1 order, so phase-1 baseline predictions align."""
import json, os, numpy as np, pandas as pd
from tidal.adapters import real_frame, synth_frame
from tidal.labels import compute_labels
from tidal.dataset import assign_split, HEADS
from tidal import features, features_g, vap_targets, scenarios
from tidal.privacy import pseudo

def participants_real():
    """observed participants per real conversation (+1 for the bot), keyed by pseudonymous conv id. Supplied by the
    private real-data adapter (tidal/build_real.py, not published); without it the participant count stays unknown."""
    try: from tidal.build_real import participants
    except ImportError: return {}
    return participants()

def build():
    real = compute_labels(real_frame()); real["split"] = assign_split(real)
    real["n_participants"] = real.conv.map(participants_real()); real["modality"] = None; real["scenario"] = "real_chat"
    syn = compute_labels(synth_frame()); syn["split"] = "synth"; syn["scenario"] = "llm_synth_group"
    syn["n_participants"] = syn.groupby("conv").speaker.transform("nunique"); syn["modality"] = None
    sc = scenarios.build(); spl = sc[["msg_id", "split"]]; sc = compute_labels(sc.drop(columns="split"))
    sc = sc.merge(spl, on="msg_id", how="left")
    cols = ["source", "conv", "conv_type", "ts", "role", "speaker", "speaker_known", "text", "split", "scenario",
            "n_participants", "modality", "msg_id", "responds_to", "initiate"] + HEADS
    for f in (real, syn, sc):
        for c in cols:
            if c not in f: f[c] = None
    d = pd.concat([real[cols], syn[cols], sc[cols]], ignore_index=True)
    order = {"real": 0, "synth": 1, "syn_live": 2, "syn_1on1": 3}
    d["_o"] = d.source.map(order)
    d = d.sort_values(["_o", "conv", "ts"], kind="stable").reset_index(drop=True).drop(columns="_o")
    # phase-1 alignment check for the real block
    m1 = pd.read_parquet("data/proc/meta.parquet"); r1 = m1[m1.source == "real"].reset_index(drop=True)
    rr = d[d.source == "real"].reset_index(drop=True)
    assert len(r1) == len(rr) and (r1.conv.values == rr.conv.values).all() and np.allclose(r1.ts.values, rr.ts.values)
    X = features.compute(d); G = features_g.compute(d, X)
    end = {c: real.ts.max() for c in real.conv.unique()}            # real streams are complete up to the export
    V = vap_targets.compute(d, end)
    Y = d[HEADS].to_numpy(np.float32)
    meta = d[["source", "conv", "conv_type", "ts", "role", "split", "scenario", "speaker_known", "msg_id", "responds_to", "initiate"]].copy()
    meta["has_text"] = d.text.notna()
    np.savez("data/proc/p2.npz", G=G, Y=Y, V=V); meta.to_parquet("data/proc/p2_meta.parquet")
    for p in ("data/proc/p2.npz", "data/proc/p2_meta.parquet"): os.chmod(p, 0o600)
    st = {s: dict(events=int((meta.scenario == s).sum()), convs=int(meta[meta.scenario == s].conv.nunique()),
                  vap_defined=float(np.mean(~np.isnan(V[meta.scenario.values == s]))))
          for s in meta.scenario.unique()}
    json.dump(st, open("data/proc/p2_stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
    return d

if __name__ == "__main__":
    build()
