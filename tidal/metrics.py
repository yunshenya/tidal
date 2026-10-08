import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score
def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)
def ece(y, p, bins=15):
    y = np.asarray(y); p = np.asarray(p); e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any(): e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)
def best_threshold(y, p):
    ts = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    f = [f1_score(y, p >= t, average="macro", labels=[0, 1], zero_division=0) for t in ts]
    return float(ts[int(np.argmax(f))]) if len(ts) else 0.5
def binary_report(y, p, thr, blocks=None, n_boot=1000, seed=0):
    y = np.asarray(y).astype(int); p = np.asarray(p, float); yh = p >= thr
    tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum())
    r = dict(n=int(len(y)), pos=int(y.sum()), prevalence=float(y.mean()),
             f1=float(f1_score(y, yh, zero_division=0)), f1_macro=float(f1_score(y, yh, average="macro", labels=[0, 1], zero_division=0)), precision=float(tp / max(1, tp + fp)), recall=float(tp / max(1, tp + fn)),
             precision_ci=wilson(tp, tp + fp), recall_ci=wilson(tp, tp + fn),
             pr_auc=float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
             roc_auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
             brier=float(np.mean((p - y) ** 2)), ece=ece(y, p), threshold=thr)
    if blocks is not None and 0 < y.sum() < len(y):
        bs = boot(lambda idx: (f1_score(y[idx], yh[idx], average="macro", labels=[0, 1], zero_division=0), average_precision_score(y[idx], p[idx]) if 0 < y[idx].sum() < len(idx) else np.nan), blocks, n_boot, seed)
        r["f1_macro_ci"] = ci(bs[:, 0]); r["pr_auc_ci"] = ci(bs[:, 1])
    return r
def multiclass_report(y, P, blocks=None, n_boot=1000, seed=0, speak_idx=0):
    y = np.asarray(y).astype(int); P = np.asarray(P, float); yh = P.argmax(1)
    K = P.shape[1]; conf = P.max(1)
    r = dict(n=int(len(y)), dist=np.bincount(y, minlength=K).tolist(), macro_f1=float(f1_score(y, yh, average="macro", labels=list(range(K)), zero_division=0)),
             accuracy=float((yh == y).mean()), accuracy_ci=wilson(int((yh == y).sum()), len(y)), ece_top=ece((yh == y).astype(int), conf),
             nll=float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)))))
    ys = (y == speak_idx).astype(int)
    if 0 < ys.sum() < len(ys): r["c0_pr_auc"] = float(average_precision_score(ys, P[:, speak_idx]))
    r["per_class_f1"] = f1_score(y, yh, average=None, labels=list(range(K)), zero_division=0).round(4).tolist()
    if blocks is not None:
        bs = boot(lambda idx: (f1_score(y[idx], yh[idx], average="macro", labels=list(range(K)), zero_division=0),), blocks, n_boot, seed)
        r["macro_f1_ci"] = ci(bs[:, 0])
    return r
def boot(fn, blocks, n_boot, seed):
    rng = np.random.default_rng(seed); blocks = np.asarray(blocks); ub = np.unique(blocks)
    groups = [np.flatnonzero(blocks == b) for b in ub]; out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups)); idx = np.concatenate([groups[k] for k in pick])
        out.append(fn(idx))
    return np.array(out, float)
def ci(v): v = v[~np.isnan(v)]; return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if len(v) else (np.nan, np.nan)
def paired_delta(y, pa, pb, thr_a, thr_b, blocks, kind="binary", n_boot=1000, seed=0):
    """Block-bootstrap CI of (model a - baseline b) for PR-AUC/F1 (binary) or macro-F1 (multiclass)."""
    y = np.asarray(y).astype(int)
    if kind == "binary":
        def fn(idx):
            yy = y[idx]
            if not (0 < yy.sum() < len(yy)): return (np.nan, np.nan)
            return (average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx]),
                    f1_score(yy, pa[idx] >= thr_a, average="macro", labels=[0, 1], zero_division=0) - f1_score(yy, pb[idx] >= thr_b, average="macro", labels=[0, 1], zero_division=0))
    else:
        K = pa.shape[1]
        def fn(idx):
            yy = y[idx]
            return (f1_score(yy, pa[idx].argmax(1), average="macro", labels=list(range(K)), zero_division=0) -
                    f1_score(yy, pb[idx].argmax(1), average="macro", labels=list(range(K)), zero_division=0), np.nan)
    bs = boot(fn, blocks, n_boot, seed)
    primary = bs[:, 0][np.isfinite(bs[:, 0])]
    return dict(d_primary_ci=ci(primary), d_primary_mean=float(primary.mean()) if len(primary) else np.nan,
                p_le0=float((primary <= 0).mean()) if len(primary) else np.nan, n_boot_valid=int(len(primary)),
                d_f1_ci=ci(bs[:, 1]) if kind == "binary" else None)


def strat_auc(y, p, strata, kind="roc"):
    """n-weighted mean of per-stratum AUCs (e.g. per CV fold, whose models have different calibration); strata
    without both classes are skipped."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    f = roc_auc_score if kind == "roc" else average_precision_score; a, w = [], []
    for s in np.unique(strata):
        m = strata == s
        if 0 < y[m].sum() < m.sum(): a.append(f(y[m], p[m])); w.append(m.sum())
    return float(np.average(a, weights=w)) if a else np.nan
