"""Phase-2 extra analyses on REAL held-out data -> reports/phase2_extra.json
  1. VAP future-event projection quality: model (fine-tuned), pretrain-only (zero-shot) and no-pretrain ablation vs
     non-sequential baselines on the same scenario-general features (LR, GBDT; fit on real train), mean ROC-AUC and
     mean BCE over the 20 outputs, paired conv x hour block bootstrap.
  2. ECE before / after per-head temperature scaling (temperatures fitted on real val) for the primary model.
  3. Robustness: participant count unknown (as in shadow mode) vs known.
usage: python -m tidal.report2 [--boot 500]"""
import argparse, json, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal import vap as VP, vap_targets as VT, metrics as M, features_g as FG
warnings.filterwarnings("ignore")
sig = lambda z: 1 / (1 + np.exp(-z))
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def vap_scores(P, T, idx):
    a, l = [], []
    for j in range(VT.NV):
        m = idx[~np.isnan(T[idx, j])]
        if len(m) < 20 or not (0 < T[m, j].sum() < len(m)): continue
        a.append(roc_auc_score(T[m, j], P[m, j])); p = np.clip(P[m, j], 1e-6, 1 - 1e-6)
        l.append(-np.mean(T[m, j] * np.log(p) + (1 - T[m, j]) * np.log(1 - p)))
    return (float(np.mean(a)) if a else np.nan, float(np.mean(l)) if l else np.nan)

def main(boot=500):
    ev = json.load(open("reports/eval_phase2.json")); primary = ev["selection"]["primary"].split(":", 1)[1]
    seed = primary.rsplit("_s", 1)[1]; ablation = f"P2n_s{seed}"
    D = VP.load(); meta = D["meta"]; sp = meta.split.to_numpy(); real = (meta.source == "real").to_numpy()
    rows = np.flatnonzero(real & np.isin(sp, ["val", "test_time", "test_group"])); spr = sp[rows]
    blocks = np.array([f"{c}:{int(t // 3600)}" for c, t in zip(meta.conv.to_numpy()[rows], meta.ts.to_numpy()[rows])])
    out = dict(primary=primary, ablation=ablation, vap={}, ece={}, participants_unknown={})
    # ---------------- 1. VAP projection
    lg, _ = VP.predict(primary, D, rows, conv_mask=real); lz, _ = VP.predict(primary, D, rows, conv_mask=real, stage="pre")
    ln, _ = VP.predict(ablation, D, rows, conv_mask=real)
    P = {"model_ft": sig(lg["vap"]), "pretrain_only": sig(lz["vap"]), "no_pretrain": sig(ln["vap"])}
    tr = np.flatnonzero(real & (sp == "train")); X = D["X"]; V = D["V"]
    for name, mk in (("lr_G", lambda: LogisticRegression(max_iter=500)), ("gbdt_G", lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05))):
        Pb = np.full((len(rows), VT.NV), 0.5)
        for j in range(VT.NV):
            m = tr[~np.isnan(V[tr, j])]
            if len(np.unique(V[m, j])) == 2: Pb[:, j] = mk().fit(X[m], V[m, j].astype(int)).predict_proba(X[rows])[:, 1]
        P[name] = Pb
    T = V[rows]
    for spn in ["test_time", "test_group"]:
        idx = np.flatnonzero(spr == spn); r = {}
        for n, p in P.items():
            auc, bce = vap_scores(p, T, idx); r[n] = dict(mean_auc=auc, mean_bce=bce)
            r[n]["per_channel_auc"] = {}
            for c in VT.CHANNELS:
                a = []
                for b in range(len(VT.BINS)):
                    j = VT.idx(c, b); m = idx[~np.isnan(T[idx, j])]
                    if len(m) >= 20 and 0 < T[m, j].sum() < len(m): a.append(roc_auc_score(T[m, j], p[m, j]))
                r[n]["per_channel_auc"][c] = float(np.mean(a)) if a else None
        r["n_events"] = int(len(idx)); r["defined_frac"] = float(np.mean(~np.isnan(T[idx])))
        loc = {k: v for k, v in zip(range(len(idx)), idx)}
        for a_, b_ in [("model_ft", "lr_G"), ("model_ft", "gbdt_G"), ("pretrain_only", "gbdt_G"), ("model_ft", "no_pretrain"), ("model_ft", "pretrain_only")]:
            def fn(ii, a_=a_, b_=b_):
                sel = idx[ii]; x = vap_scores(P[a_], T, sel); y = vap_scores(P[b_], T, sel); return (x[0] - y[0], x[1] - y[1])
            bs = M.boot(fn, blocks[idx], boot, 0)
            r[f"{a_}-{b_}"] = dict(d_auc=float(np.nanmean(bs[:, 0])), d_auc_ci=M.ci(bs[:, 0]), d_bce=float(np.nanmean(bs[:, 1])), d_bce_ci=M.ci(bs[:, 1]))
        out["vap"][spn] = r
    # ---------------- 2. ECE before / after temperature (primary)
    temps = ev["temps"]["p2:" + primary]; Y = D["Y"][rows]
    for spn in ["test_time", "test_group"]:
        idx = spr == spn; r = {}
        for hi, h in enumerate(HEADS):
            lab = idx & ~np.isnan(Y[:, hi]); y = Y[lab, hi].astype(int); z = lg[h][lab]
            if HEAD_DIMS[h] == 1:
                pr, pc = sig(z[:, 0]), sig(z[:, 0] / temps[h])
                r[h] = dict(n=int(lab.sum()), T=temps[h], ece_raw=M.ece(y, pr), ece_cal=M.ece(y, pc),
                            brier_raw=float(np.mean((pr - y) ** 2)), brier_cal=float(np.mean((pc - y) ** 2)))
                bs = M.boot(lambda ii: (M.ece(y[ii], pc[ii]) - M.ece(y[ii], pr[ii]),), blocks[lab], boot, 0)
            else:
                pr, pc = softmax(z), softmax(z / temps[h]); cr = (pr.argmax(1) == y).astype(int)
                r[h] = dict(n=int(lab.sum()), T=temps[h], ece_raw=M.ece(cr, pr.max(1)), ece_cal=M.ece(cr, pc.max(1)),
                            nll_raw=float(-np.mean(np.log(pr[np.arange(len(y)), y]))), nll_cal=float(-np.mean(np.log(pc[np.arange(len(y)), y]))))
                bs = M.boot(lambda ii: (M.ece(cr[ii], pc[ii].max(1)) - M.ece(cr[ii], pr[ii].max(1)),), blocks[lab], boot, 0)
            r[h]["d_ece_cal_minus_raw"] = float(np.nanmean(bs[:, 0])); r[h]["d_ece_ci"] = M.ci(bs[:, 0])
        out["ece"][spn] = r
    # ---------------- 3. participants unknown (shadow mode setting)
    D2 = dict(D); X2 = D["X"].copy(); X2[real, FG.CTX_SL] = 0.0; D2["X"] = X2
    lu, _ = VP.predict(primary, D2, rows, conv_mask=real)
    for spn in ["test_time", "test_group"]:
        idx = spr == spn; r = {}
        for hi, h in enumerate(HEADS):
            lab = idx & ~np.isnan(Y[:, hi]); y = Y[lab, hi].astype(int)
            if HEAD_DIMS[h] == 1:
                if not (0 < y.sum() < len(y)): continue
                r[h] = dict(known=float(average_precision_score(y, lg[h][lab, 0])), unknown=float(average_precision_score(y, lu[h][lab, 0])))
            else:
                r[h] = dict(known=float(f1_score(y, lg[h][lab].argmax(1), average="macro", zero_division=0)), unknown=float(f1_score(y, lu[h][lab].argmax(1), average="macro", zero_division=0)))
        out["participants_unknown"][spn] = r
    json.dump(out, open("reports/phase2_extra.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:12000])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=500); main(ap.parse_args().boot)
