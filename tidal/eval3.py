"""Phase 3 evaluation on tidal's REAL held-out data (same rows, splits and conv x hour blocks as phases 1-2).
Systems: phase-1 baselines (val-chosen strongest + best-on-test), phase-2 primary (P2a_s1, all P2a seeds),
phase-3 models: P3ft (pretrained on public data, fine-tuned on real + LLM-synth) and P3mix (joint training on real +
LLM-synth + public). Pre-registered selection: config with lowest mean real-val loss, then its best-val seed.
usage: python -m tidal.eval3 -> reports/eval_phase3.json (private: real-data metrics)"""
import json, os, numpy as np
from sklearn.metrics import average_precision_score, f1_score
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal import vap as VP, metrics as M

def probs(lg, h):
    l = lg[h]
    if HEAD_DIMS[h] == 1: return 1 / (1 + np.exp(-l[:, 0]))
    e = np.exp(l - l.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def prim(y, p, b):
    if b: return float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else np.nan
    return float(f1_score(y.astype(int), p.argmax(1), average="macro", labels=list(range(p.shape[1])), zero_division=0))

def main(boot=1000):
    D = VP.load("p3"); meta = D["meta"]; Y = D["Y"]; split = meta.split.to_numpy(); real = (meta.source == "real").to_numpy()
    rows = np.flatnonzero(real & np.isin(split, ["val", "test_time", "test_group"]) & ~np.all(np.isnan(Y), 1))
    hour = (meta.ts.to_numpy() // 3600).astype(int); blocks = np.array([f"{c}:{h}" for c, h in zip(meta.conv.to_numpy(), hour)])[rows]
    sp = split[rows]; Yr = Y[rows]
    D2 = VP.load("p2"); assert np.array_equal(np.isnan(D2["Y"][rows]), np.isnan(Yr)) and np.allclose(D2["X"][rows], D["X"][rows])
    systems = {}
    for k, v in dict(np.load("data/proc/baseline_preds_T.npz")).items():
        h, name = k.split("|"); systems.setdefault(name, {})[h] = v[rows]
    base = list(systems)
    tags = {c: [f"{c}_s{s}" for s in range(3) if os.path.exists(f"models/{c}_s{s}.pt")] for c in ("P2a", "P3ft", "P3mix")}
    ck = {}
    for c, ts in tags.items():
        for t in ts:
            lg, ck[t] = VP.predict(t, D, rows, conv_mask=real); systems[t] = {h: probs(lg, h) for h in HEADS}
    cfg_val = {c: float(np.mean([ck[t]["best_val"] for t in ts])) for c, ts in tags.items() if ts and c != "P2a"}
    best_cfg = min(cfg_val, key=cfg_val.get); p3 = min(tags[best_cfg], key=lambda t: ck[t]["best_val"]); p2 = "P2a_s1"
    rep = dict(selection=dict(config_val_loss=cfg_val, p2a_val_loss=float(np.mean([ck[t]["best_val"] for t in tags["P2a"]])), best_config=best_cfg, primary=p3),
               models={t: dict(best_val=float(ck[t]["best_val"]), data=ck[t]["data"], init=ck[t].get("init"), train_seconds=ck[t]["train_seconds"]) for t in ck},
               heads={})
    for hi, h in enumerate(HEADS):
        b = HEAD_DIMS[h] == 1; y = Yr[:, hi]; lab = ~np.isnan(y); va = lab & (sp == "val")
        thr = {n: (M.best_threshold(y[va], s[h][va]) if b else .5) for n, s in systems.items() if h in s}
        valsc = {n: prim(y[va], systems[n][h][va], b) for n in base if h in systems[n]}
        strongest = max([n for n in valsc if not np.isnan(valsc[n])], key=lambda n: valsc[n])
        H = dict(strongest_baseline=strongest)
        for spn in ("test_time", "test_group"):
            te = lab & (sp == spn)
            if te.sum() < 5: continue
            sc = {n: prim(y[te], s[h][te], b) for n, s in systems.items() if h in s}
            bt = max([n for n in base if h in systems[n] and not np.isnan(sc[n])], key=lambda n: sc[n])
            R = dict(n=int(te.sum()), pos=int(y[te].sum()) if b else np.bincount(y[te].astype(int), minlength=HEAD_DIMS[h]).tolist(), scores=sc, best_on_test=bt)
            kind = "binary" if b else "multi"
            for lab_, ref in (("p3_vs_p2", p2), ("p3_vs_strongest", strongest), ("p3_vs_best_on_test", bt), ("p2_vs_strongest", strongest)):
                a = p3 if lab_.startswith("p3") else p2
                R[lab_] = M.paired_delta(y[te], systems[a][h][te], systems[ref][h][te], thr[a], thr[ref], blocks[te], kind, n_boot=boot)
            for c in ("P3ft", "P3mix", "P2a"):
                if tags.get(c): R[f"seedmean_{c}"] = float(np.nanmean([sc[t] for t in tags[c]]))
            H[spn] = R
        rep["heads"][h] = H
    json.dump(rep, open("reports/eval_phase3.json", "w"), indent=1, default=float)
    print(json.dumps(rep["selection"])); 
    for h, H in rep["heads"].items():
        for spn in ("test_time", "test_group"):
            if spn not in H: continue
            R = H[spn]; f = lambda k: f"{R[k]['d_primary_mean']:+.3f}[{R[k]['d_primary_ci'][0]:+.3f},{R[k]['d_primary_ci'][1]:+.3f}]"
            print(f"{h:9s} {spn:10s} n={R['n']:5d} p3={R['scores'][p3]:.3f} p2={R['scores'][p2]:.3f} strongest={R['scores'][H['strongest_baseline']]:.3f} "
                  f"Δp3-p2={f('p3_vs_p2')} Δp3-strong={f('p3_vs_strongest')} Δp3-bestTest={f('p3_vs_best_on_test')} " +
                  " ".join(f"mean{c}={R.get('seedmean_' + c, float('nan')):.3f}" for c in ("P3ft", "P3mix", "P2a")))

if __name__ == "__main__": main()
