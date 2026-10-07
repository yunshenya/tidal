"""Baselines per head, trained on REAL train split only; thresholds/hyper-params picked on REAL val.
Outputs data/proc/baseline_preds.npz (per-event probabilities for every baseline) + reports/baselines.json"""
import json, numpy as np, pandas as pd, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score
from tidal.features import FEATURES, cols, FEAT_T, FEAT_TS
import sys
from tidal.dataset import HEADS
warnings.filterwarnings("ignore")
BIN = {"y_eot", "y_self", "y_addr", "y_hreply"}; K = {"y_act": 3, "y_recheck": 7}
GAP_FEATS = ["log_gap_prev", "log_gap_same_spk", "log_burst", "log_since_bot", "log_act_60s", "log_act_600s"]
def score(h, y, P):
    if h in BIN: return average_precision_score(y, P) if 0 < y.sum() < len(y) else np.nan
    return f1_score(y, P.argmax(1), average="macro", labels=list(range(K[h])), zero_division=0)
def fit_lr(X, y, h, C):
    m = LogisticRegression(C=C, max_iter=2000); m.fit(X, y.astype(int)); return m
def proba(m, X, h):
    P = m.predict_proba(X)
    if h in BIN: return P[:, list(m.classes_).index(1)] if 1 in m.classes_ else np.zeros(len(X))
    out = np.zeros((len(X), K[h])); out[:, m.classes_.astype(int)] = P; return out
def main(regime="TS", save_models=None, write_preds=True):
    """save_models: optional path -> joblib bundle of the fitted baselines (used by shadow mode).
    write_preds=False leaves the phase-1 artifacts (preds npz, choices json) untouched."""
    z = np.load("data/proc/dataset.npz"); X, Y, eidx = z["X"], z["Y"], z["eidx"]
    meta = pd.read_parquet("data/proc/meta.parquet"); emb = np.load("data/proc/emb.npy").astype(np.float32)
    real = (meta.source == "real").to_numpy(); split = meta.split.to_numpy()
    fc = cols(regime); X = X[:, fc]; FEATS = FEAT_T if regime == "T" else FEAT_TS
    if regime == "TS": real = real & (eidx >= 0)     # TS regime: text-available events only
    preds = {}; chosen = {}; fitted = {}
    sc = StandardScaler().fit(X[real & (split == "train")])
    Xs = sc.transform(X)
    E = np.where(eidx[:, None] >= 0, emb[np.clip(eidx, 0, None)], 0.0)
    for hi, h in enumerate(HEADS):
        y = Y[:, hi]; lab = ~np.isnan(y)
        tr = real & lab & (split == "train"); va = real & lab & (split == "val")
        cands = {}
        # majority / prior
        if h in BIN: prior = np.full(len(X), y[tr].mean())
        else: prior = np.tile(np.bincount(y[tr].astype(int), minlength=K[h]) / tr.sum(), (len(X), 1))
        cands["majority"] = prior; fm = {"majority": dict(kind="prior", p=prior[0].copy())}
        # gap-time single feature (1-D logistic = monotone threshold family); feature picked on val
        best = None
        for f in [g for g in GAP_FEATS if g in FEATS]:
            j = FEATS.index(f); m = fit_lr(Xs[tr][:, [j]], y[tr], h, 1.0); P = proba(m, Xs[:, [j]], h)
            s = score(h, y[va].astype(int), P[va])
            if best is None or s > best[0]: best = (s, f, P, m, j)
        cands["gap"] = best[2]; chosen[h + ":gap_feature"] = best[1]; fm["gap"] = dict(kind="lr", m=best[3], cols=[best[4]], scaled=True)
        # LR on hand features (C on val)
        best = None
        for C in [0.01, 0.1, 1.0, 10.0]:
            m = fit_lr(Xs[tr], y[tr], h, C); P = proba(m, Xs, h); s = score(h, y[va].astype(int), P[va])
            if best is None or s > best[0]: best = (s, C, P, m)
        cands["lr_hand"] = best[2]; chosen[h + ":lr_C"] = best[1]; fm["lr_hand"] = dict(kind="lr", m=best[3], cols=None, scaled=True)
        # gradient-boosted trees on hand features
        best = None
        for lr_, leaves in [(0.05, 7), (0.05, 15)]:
            m = HistGradientBoostingClassifier(learning_rate=lr_, max_leaf_nodes=leaves, max_iter=200, l2_regularization=1.0, random_state=0)
            m.fit(X[tr], y[tr].astype(int)); P = proba(m, X, h); s = score(h, y[va].astype(int), P[va])
            if best is None or s > best[0]: best = (s, (lr_, leaves), P, m)
        cands["hgb_hand"] = best[2]; chosen[h + ":hgb"] = best[1]; fm["hgb_hand"] = dict(kind="hgb", m=best[3])
        # text-only on the single current message (bge embedding); prior where no text
        ht = eidx >= 0
        if regime == "TS":
            best = None
            for C in [0.1, 1.0]:
                m = fit_lr(E[tr & ht], y[tr & ht], h, C); P = prior.copy(); P[ht] = proba(m, E[ht], h)
                s = score(h, y[va].astype(int), P[va])
                if best is None or s > best[0]: best = (s, C, P, m)
            cands["text_only"] = best[2]; chosen[h + ":text_C"] = best[1]; fm["text_only"] = dict(kind="text", m=best[3], p=prior[0].copy())
        for k, v in cands.items(): preds[f"{h}|{k}"] = v.astype(np.float32)
        fitted[h] = fm
        print(h, {k: round(score(h, y[va].astype(int), v[va]), 4) for k, v in cands.items()}, flush=True)
    if write_preds:
        np.savez(f"data/proc/baseline_preds_{regime}.npz", **preds)
        json.dump(chosen, open(f"reports/baseline_choices_{regime}.json", "w"), indent=1)
    if save_models:
        import joblib
        joblib.dump(dict(regime=regime, feats=FEATS, scaler=sc, heads=fitted, chosen=chosen), save_models)
    return preds
def predict_bundle(bundle, X, E=None):
    """Apply a saved baseline bundle. X: [N, len(FEATURES)] raw features (all columns); E: [N,512] current-message
    embeddings (zeros/None where no text). Returns {"head|name": probs}."""
    Xr = X[:, cols(bundle["regime"])]; Xs = bundle["scaler"].transform(Xr); out = {}
    for h, fm in bundle["heads"].items():
        for name, o in fm.items():
            if o["kind"] == "prior": P = np.tile(o["p"], (len(X), 1)) if np.ndim(o["p"]) else np.full(len(X), float(o["p"]))
            elif o["kind"] == "lr": P = proba(o["m"], Xs if o["cols"] is None else Xs[:, o["cols"]], h)
            elif o["kind"] == "hgb": P = proba(o["m"], Xr, h)
            else:
                P = np.tile(o["p"], (len(X), 1)) if np.ndim(o["p"]) else np.full(len(X), float(o["p"]))
                if E is not None:
                    ht = np.abs(E).sum(1) > 0
                    if ht.any(): P[ht] = proba(o["m"], E[ht], h)
            out[f"{h}|{name}"] = np.asarray(P, np.float32)
    return out
if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) > 1 and a[1] == "--save":   # python -m tidal.baselines T --save models/baselines_T.joblib
        main(a[0], save_models=a[2], write_preds=False)
    else:
        main(a[0] if a else "TS")
