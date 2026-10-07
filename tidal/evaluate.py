"""Evaluate models vs baselines on REAL held-out data only (val for calibration/thresholds; test_time, test_group).
usage: python -m tidal.evaluate REGIME model_tag [model_tag ...]  -> reports/eval_REGIME.json"""
import sys, json, numpy as np, torch, pandas as pd
from tidal.seqdata import load, conv_index, eval_windows, batchify
from tidal.model import TurnModel, HEAD_DIMS
from tidal.dataset import HEADS
from tidal import metrics as M
from sklearn.metrics import average_precision_score, f1_score
BIN = [h for h in HEADS if HEAD_DIMS[h] == 1]
def predict_logits(tag, D, rows):
    ck = torch.load(f"models/{tag}.pt", weights_only=False)
    m = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"]); m.load_state_dict(ck["state"]); m.eval()
    W = eval_windows(conv_index(D), rows); out = {h: [] for h in HEADS}
    with torch.no_grad():
        for b in range(0, len(W), 256):
            F, E, _, valid = batchify(D, W[b:b + 256], ck["use_text"])
            o = m(torch.from_numpy(F), torch.from_numpy(valid), torch.from_numpy(E) if ck["use_text"] else None)
            for h in HEADS: out[h].append(o[h][:, -1].numpy())
    return {h: np.concatenate(v) for h, v in out.items()}, ck
def fit_temperature(logit, y, binary):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.25), np.log(4), 41)):
        if binary: p = 1 / (1 + np.exp(-logit / T)); l = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        else:
            z = logit / T; z = z - z.max(1, keepdims=True); P = np.exp(z); P /= P.sum(1, keepdims=True)
            l = -np.mean(np.log(P[np.arange(len(y)), y.astype(int)] + 1e-9))
        if l < best[0]: best = (l, T)
    return best[1]
def to_prob(logit, T, binary):
    if binary: return 1 / (1 + np.exp(-logit / T))
    z = logit / T; z = z - z.max(1, keepdims=True); P = np.exp(z); return P / P.sum(1, keepdims=True)
def main(regime, tags):
    D = load(regime); meta = D["meta"]; Y = D["Y"]
    real = D["keep"] & (meta.source == "real").to_numpy(); split = meta.split.to_numpy()
    rows = np.flatnonzero(real & np.isin(split, ["val", "test_time", "test_group"]) & ~np.all(np.isnan(Y), 1))
    hour = (meta.ts.to_numpy() // 3600).astype(int)
    blocks = np.array([f"{c}:{h}" for c, h in zip(meta.conv.to_numpy(), hour)])
    bp = dict(np.load(f"data/proc/baseline_preds_{regime}.npz"))
    systems = {}
    for k, v in bp.items():
        h, name = k.split("|"); systems.setdefault(name, {})[h] = v[rows]
    report = dict(regime=regime, models={}, heads={})
    for tag in tags:
        lg, ck = predict_logits(tag, D, rows); probs = {}; temps = {}
        for hi, h in enumerate(HEADS):
            va = (split[rows] == "val") & ~np.isnan(Y[rows, hi]); binary = HEAD_DIMS[h] == 1
            l = lg[h][:, 0] if binary else lg[h]
            T = fit_temperature(l[va], Y[rows[va], hi], binary) if va.sum() > 10 else 1.0
            temps[h] = float(T); probs[h] = to_prob(l, T, binary)
        systems["model:" + tag] = probs
        report["models"][tag] = dict(params=int(ck["params"]), best_ep=int(ck["best_ep"]), data=ck["data"], kind=ck["kind"], temps=temps)
    for hi, h in enumerate(HEADS):
        binary = HEAD_DIMS[h] == 1; y = Y[rows, hi]; lab = ~np.isnan(y)
        va = lab & (split[rows] == "val"); hres = dict(systems={}, val={})
        thr = {}
        for name, pr in systems.items():
            if h not in pr: continue
            p = pr[h]
            if binary:
                thr[name] = M.best_threshold(y[va], p[va]) if va.sum() else 0.5
                hres["val"][name] = float(average_precision_score(y[va], p[va])) if 0 < y[va].sum() < va.sum() else None
            else:
                hres["val"][name] = float(f1_score(y[va].astype(int), p[va].argmax(1), average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0))
        base_names = [n for n in hres["val"] if not n.startswith("model:") and hres["val"][n] is not None]
        strongest = max(base_names, key=lambda n: hres["val"][n]); hres["strongest_baseline"] = strongest
        for sp in ["test_time", "test_group"]:
            te = lab & (split[rows] == sp)
            if te.sum() == 0: continue
            hres["systems"].setdefault(sp, {})
            for name, pr in systems.items():
                if h not in pr: continue
                p = pr[h]
                if binary: r = M.binary_report(y[te], p[te], thr[name], blocks[rows][te], n_boot=500)
                else: r = M.multiclass_report(y[te], p[te], blocks[rows][te], n_boot=500)
                hres["systems"][sp][name] = r
            key = "pr_auc" if binary else "macro_f1"
            bt = max([n for n in base_names if n in hres["systems"][sp]], key=lambda n: np.nan_to_num(hres["systems"][sp][n][key], nan=-1))
            hres.setdefault("best_baseline_on_test", {})[sp] = bt
            for name in systems:
                if name.startswith("model:") and h in systems[name]:
                    hres["systems"][sp][name]["vs_best_on_test"] = M.paired_delta(y[te], systems[name][h][te], systems[bt][h][te], thr.get(name, .5), thr.get(bt, .5),
                                                                                  blocks[rows][te], "binary" if binary else "multi", n_boot=500)
                    pa, pb = systems[name][h][te], systems[strongest][h][te]
                    hres["systems"][sp][name]["vs_strongest"] = M.paired_delta(y[te], pa, pb, thr.get(name, .5), thr.get(strongest, .5),
                                                                               blocks[rows][te], "binary" if binary else "multi", n_boot=500)
        report["heads"][h] = hres
    json.dump(report, open(f"reports/eval_{regime}.json", "w"), indent=1, default=float)
    summarize(report)
def summarize(rep):
    for h, r in rep["heads"].items():
        print(f"== {rep['regime']} {h}  strongest baseline (val): {r['strongest_baseline']}")
        for sp, ss in r["systems"].items():
            for name, m in ss.items():
                key = "pr_auc" if "pr_auc" in m else "macro_f1"
                extra = f" Δ={m['vs_strongest']['d_primary_mean']:+.3f} CI{tuple(round(x,3) for x in m['vs_strongest']['d_primary_ci'])}" if "vs_strongest" in m else ""
                f1 = m.get("f1_macro", m.get("macro_f1"))
                print(f"  {sp:10s} {name:28s} n={m['n']:5d} {key}={m[key]:.3f} f1={f1:.3f} ece={m.get('ece', m.get('ece_top')):.3f}{extra}")
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
