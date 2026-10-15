#!/usr/bin/env python3
"""Model-based weak labels for pending turngate rows (no hand-written word lists).

A small local classifier is trained ONLY on human-consensus rows. Quality and calibration
are measured out-of-fold with GroupKFold over group/scope ids (no row is ever scored by a
model that saw it or its group). Pending rows are then labeled by the model fit on all human
rows with sigmoid-calibrated probabilities; rows below --min-confidence are left unlabeled.
Output is training-only provenance ``assistant_model_v2`` and stamped with an input hash.
Nothing is sent over the network.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from tidal.turngate import AUTO, group_of, is_human, read_jsonl

TASKS = ("completion", "response")
_HASH = HashingVectorizer(analyzer="char_wb", ngram_range=(1, 3), n_features=2**16, alternate_sign=False, norm="l2")


def featurize(rows):
    cur = [str(r.get("current_text") or "") for r in rows]
    prev = [str(((r.get("context") or {}).get("recent_turns") or [{}])[-1].get("text") or "") for r in rows]
    ctx = [[float(bool((r.get("context") or {}).get(k))) for k in ("addressed_to_agent", "replies_to_agent", "bot_last_asked_question")]
           for r in rows]
    return hstack([_HASH.transform(cur), 0.5 * _HASH.transform(prev), csr_matrix(np.asarray(ctx, np.float32))]).tocsr()


def model():
    return CalibratedClassifierCV(LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"), method="sigmoid", cv=3)


def ece(p, y, bins=10):
    conf = p.max(1); hit = p.argmax(1) == y; e = 0.
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf >= lo) & (conf < lo + 1 / bins)
        if m.any(): e += m.mean() * abs(conf[m].mean() - hit[m].mean())
    return float(e)


def crossfit(X, y, groups, folds):
    classes = np.unique(y); oof = np.zeros((len(y), len(classes)))
    for tr, te in GroupKFold(n_splits=folds).split(X, y, groups):
        if len(np.unique(y[tr])) < len(classes): raise ValueError("a fold lacks a class; need more human rows")
        oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])
    return classes, oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path, help="review JSONL files")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--min-confidence", type=float, default=0.8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-human", type=int, default=200)
    a = ap.parse_args()
    digest = hashlib.sha256(); rows = []
    for p in sorted(a.inputs):
        digest.update(p.read_bytes()); rows += read_jsonl(p)
    human = [r for r in rows if is_human(r)]; humankeys = {str(r["source_key"]) for r in human}
    pending = [r for r in rows if not is_human(r) and str(r["source_key"]) not in humankeys]
    if len(human) < a.min_human:
        sys.exit(f"only {len(human)} human-consensus rows (< --min-human {a.min_human}); not labeling. "
                 "Keep auto rows out of training.")
    Xh, Xp = featurize(human), featurize(pending)
    groups = np.asarray([group_of(r) for r in human])
    labels = {k: {} for k in range(len(pending))}; report = {}
    for task in TASKS:
        keep = np.asarray([bool((r.get("labels") or {}).get(task)) for r in human])
        y = np.asarray([(r.get("labels") or {}).get(task) or "" for r in human])[keep]
        classes, oof = crossfit(Xh[keep], y, groups[keep], a.folds)
        yi = np.searchsorted(classes, y); sel = oof.max(1) >= a.min_confidence
        report[task] = dict(n=int(keep.sum()), oof_accuracy=float((oof.argmax(1) == yi).mean()), oof_ece=ece(oof, yi),
                            coverage_at_threshold=float(sel.mean()),
                            accuracy_at_threshold=float((oof.argmax(1) == yi)[sel].mean()) if sel.any() else None)
        if pending:
            pp = model().fit(Xh[keep], y).predict_proba(Xp)
            for i in np.flatnonzero(pp.max(1) >= a.min_confidence):
                labels[i][task] = (str(classes[pp[i].argmax()]), round(float(pp[i].max()), 4))
    stamp = digest.hexdigest(); n = 0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", encoding="utf-8") as fo:
        for i, r in enumerate(pending):
            if not labels[i]: continue
            r = dict(r, labels={t: v[0] for t, v in labels[i].items()}, review_status="auto_labeled",
                     label_provenance={"source": AUTO, "method": "crossfit_char_ngram_logreg_sigmoid",
                                       "confidence": {t: v[1] for t, v in labels[i].items()},
                                       "input_sha256": stamp, "human_rows": len(human)})
            fo.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    print(json.dumps({"output": str(a.output), "input_sha256": stamp, "human_rows": len(human),
                      "pending_rows": len(pending), "auto_rows": n, "oof": report}, indent=2))


if __name__ == "__main__": main()
