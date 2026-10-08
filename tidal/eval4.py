"""Phase 4 evaluation (A: emotion features, B: backbones) on tidal's REAL held-out rows (same rows/splits/conv x hour
blocks as phases 1-3) -> reports/eval_phase4.json (private), and on PUBLIC held-out streams -> reports/eval_phase4_pub.json.
Comparisons use the 3-seed ensemble (mean probability) of each config vs the reference config, paired, block bootstrap.
Pre-registered rules (reports/phase4_progress.md):
  B: winner = lowest mean real-VAL loss over 3 fine-tune seeds; within 0.005 of best -> lower streaming p95 latency.
  A: emotion adopted only if mean real-VAL loss < the no-emotion control's."""
import json, os, sys, numpy as np, torch
import torch.nn.functional as Fn
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal import vap as VP, metrics as M
from tidal.eval3 import probs, prim

def ens(tags, D, rows, mask):
    P, V, ck = [], [], []
    for t in tags:
        lg, c = VP.predict(t, D, rows, conv_mask=mask); P.append({h: probs(lg, h) for h in HEADS}); V.append(lg["vap"]); ck.append(c)
    return {h: np.mean([p[h] for p in P], 0) for h in HEADS}, np.mean(V, 0), ck, P

def vap_rowloss(vlogit, V):
    m = ~np.isnan(V); l = np.where(m, np.logaddexp(0, vlogit) - np.nan_to_num(V) * vlogit, 0.0)
    return l.sum(1) / np.maximum(m.sum(1), 1), m.any(1)

def block_delta(la, lb, blocks, n_boot=1000, seed=0):
    ub, inv = np.unique(blocks, return_inverse=True); rng = np.random.default_rng(seed)
    sa = np.bincount(inv, la); sb = np.bincount(inv, lb); n = np.bincount(inv); out = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)); out.append(((k * sa).sum() - (k * sb).sum()) / (k * n).sum())
    return dict(d_mean=float((la.sum() - lb.sum()) / len(la)), ci=M.ci(np.array(out)))

def compare(name_a, A, name_b, B, Yr, Vr, sp, blocks, boot):
    out = {}
    for hi, h in enumerate(HEADS):
        b = HEAD_DIMS[h] == 1; y = Yr[:, hi]; lab = ~np.isnan(y); va = lab & (sp == "val")
        thr_a = M.best_threshold(y[va], A[0][h][va]) if b else .5; thr_b = M.best_threshold(y[va], B[0][h][va]) if b else .5
        H = {}
        for spn in ("test_time", "test_group"):
            te = lab & (sp == spn)
            if te.sum() < 5: continue
            R = dict(n=int(te.sum()), a=prim(y[te], A[0][h][te], b), b=prim(y[te], B[0][h][te], b))
            R["delta"] = M.paired_delta(y[te], A[0][h][te], B[0][h][te], thr_a, thr_b, blocks[te], "binary" if b else "multi", n_boot=boot)
            if b: R["ece_a"] = M.ece(y[te].astype(int), A[0][h][te]); R["ece_b"] = M.ece(y[te].astype(int), B[0][h][te])
            R["seed_scores_a"] = [prim(y[te], p[h][te], b) for p in A[3]]; R["seed_scores_b"] = [prim(y[te], p[h][te], b) for p in B[3]]
            H[spn] = R
        out[h] = H
    for spn in ("test_time", "test_group"):
        la, ma = vap_rowloss(A[1], Vr); lb, _ = vap_rowloss(B[1], Vr); s = (sp == spn) & ma
        out[f"vap_loss_{spn}"] = dict(a=float(la[s].mean()), b=float(lb[s].mean()), **block_delta(la[s], lb[s], blocks[s], boot))
    return out

def main(boot=1000):
    configs = {"GRU(P3ft)": ([f"P3ft_s{s}" for s in range(3)], None), "GRU-ctl(P4ctl)": ([f"P4ctl_s{s}" for s in range(3)], None),
               "GRU+emo(P4emo)": ([f"P4emo_s{s}" for s in range(3)], "emo"), "GRU+flag(P4emoflag)": ([f"P4emoflag_s{s}" for s in range(3)], "emoflag")}
    for k in ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2"): configs[k] = ([f"P4ft_{k}_s{s}" for s in range(3)], None)
    for k in ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2"): configs[f"{k}+emo"] = ([f"P4emo_{k}_s{s}" for s in range(3)], "emo")
    configs = {n: c for n, c in configs.items() if all(os.path.exists(f"models/{t}.pt") for t in c[0])}
    Ds = {None: VP.load("p3")}
    for ex in {c[1] for c in configs.values()} - {None}: Ds[ex] = VP.load("p3", ex)
    D = Ds[None]; meta = D["meta"]; Y = D["Y"]; split = meta.split.to_numpy(); real = (meta.source == "real").to_numpy()
    rows = np.flatnonzero(real & np.isin(split, ["val", "test_time", "test_group"]) & ~np.all(np.isnan(Y), 1))
    hour = (meta.ts.to_numpy() // 3600).astype(int); blocks = np.array([f"{c}:{h}" for c, h in zip(meta.conv.to_numpy(), hour)])[rows]
    sp = split[rows]; Yr = Y[rows]; Vr = D["V"][rows]
    E = {n: ens(c[0], Ds[c[1]], rows, real) for n, c in configs.items()}
    val = {n: float(np.mean([c["best_val"] for c in E[n][2]])) for n in E}
    rep = dict(val_loss=val, params={n: E[n][2][0]["params"] for n in E}, comparisons={})
    ref = "GRU(P3ft)"
    for n in E:
        if n != ref: rep["comparisons"][f"{n} vs {ref}"] = compare(n, E[n], ref, E[ref], Yr, Vr, sp, blocks, boot)
    if "GRU+emo(P4emo)" in E and "GRU-ctl(P4ctl)" in E:
        rep["comparisons"]["GRU+emo(P4emo) vs GRU-ctl(P4ctl)"] = compare("emo", E["GRU+emo(P4emo)"], "ctl", E["GRU-ctl(P4ctl)"], Yr, Vr, sp, blocks, boot)
    if "GRU+emo(P4emo)" in E and "GRU+flag(P4emoflag)" in E:
        rep["comparisons"]["GRU+emo(P4emo) vs GRU+flag(P4emoflag)"] = compare("emo", E["GRU+emo(P4emo)"], "flag", E["GRU+flag(P4emoflag)"], Yr, Vr, sp, blocks, boot)
    for k in ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2"):
        if f"{k}+emo" in E and k in E: rep["comparisons"][f"{k}+emo vs {k}"] = compare("emo", E[f"{k}+emo"], k, E[k], Yr, Vr, sp, blocks, boot)
    json.dump(rep, open("reports/eval_phase4.json", "w"), indent=1, default=float); print(json.dumps(val, indent=1))
    for k, C in rep["comparisons"].items():
        print("==", k)
        for h in HEADS:
            for spn, R in C[h].items():
                d = R["delta"]; print(f"  {h:9s} {spn:10s} n={R['n']:5d} a={R['a']:.3f} b={R['b']:.3f} d={d['d_primary_mean']:+.3f} [{d['d_primary_ci'][0]:+.3f},{d['d_primary_ci'][1]:+.3f}]")
        for spn in ("test_time", "test_group"):
            R = C[f"vap_loss_{spn}"]; print(f"  vap_loss {spn}: a={R['a']:.4f} b={R['b']:.4f} d={R['d_mean']:+.4f} [{R['ci'][0]:+.4f},{R['ci'][1]:+.4f}]")

def pub_main(n_rows=40000, boot=1000):
    """public held-out streams (pub_test split of the 5 pretraining sources): VAP loss + EOT/self heads of the public
    pretrained backbones P3pub_s0 (GRU) / P4pub_<kind>. Public-safe numbers."""
    D = VP.load("p3"); meta = D["meta"]; src = meta.source.to_numpy(); split = meta.split.to_numpy()
    pubs = ["pub_tg", "pub_irc", "pub_twitch", "pub_candor", "pub_aishell4"]; m = np.isin(src, pubs) & (split == "pub_test")
    rows = np.flatnonzero(m & ~np.all(np.isnan(D["V"]), 1)); rows = np.sort(np.random.default_rng(0).permutation(rows)[:n_rows])
    blocks = np.array([f"{c}:{h}" for c, h in zip(meta.conv.to_numpy()[rows], (meta.ts.to_numpy()[rows] // 3600).astype(int))])
    tags = {"gru": "P3pub_s0", **{k: f"P4pub_{k}" for k in ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2")}}
    tags = {k: t for k, t in tags.items() if os.path.exists(f"models/{t}.pt")}; out = {}; Vr = D["V"][rows]; Yr = D["Y"][rows]; L = {}
    for k, t in tags.items():
        lg, ck = VP.predict(t, D, rows, conv_mask=m, stage="pre" if False else "ft"); l, mk = vap_rowloss(lg["vap"], Vr); L[k] = (l, mk, lg)
        out[k] = dict(params=ck["params"], vap_loss=float(l[mk].mean()), pre_best_val_vap=float(min(h["val_vap"] for h in ck["hist"] if h["stage"] == "pre")), train_seconds=ck["train_seconds"])
        for hi, h in enumerate(HEADS):
            if HEAD_DIMS[h] != 1: continue
            y = Yr[:, hi]; ok = ~np.isnan(y)
            if ok.sum() > 50 and 0 < y[ok].sum() < ok.sum():
                p = probs(lg, h); out[k][f"{h}_prauc"] = prim(y[ok], p[ok], True); out[k][f"{h}_ece"] = M.ece(y[ok].astype(int), p[ok])
    for k in L:
        if k == "gru": continue
        s = L[k][1]; out[k]["vap_loss_delta_vs_gru"] = block_delta(L[k][0][s], L["gru"][0][s], blocks[s], boot)
    out["n_rows"] = len(rows); json.dump(out, open("reports/eval_phase4_pub.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))

if __name__ == "__main__":
    (pub_main if len(sys.argv) > 1 and sys.argv[1] == "pub" else main)()
