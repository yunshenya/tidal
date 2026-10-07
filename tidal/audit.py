"""Leakage audit -> reports/leakage_audit.json"""
import json, inspect, numpy as np, pandas as pd, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from tidal import features
from tidal.adapters import real_frame
from tidal.labels import compute_labels
from tidal.dataset import assign_split, HEADS, HOLDOUT_GROUPS, VAL_START, TEST_START, EMBARGO
from tidal.model import TurnModel
def main(model_tags=()):
    R = {}
    d = compute_labels(real_frame()); d["split"] = assign_split(d)
    X = features.compute(d)
    # 1. causality of features: truncate each sampled conversation right after event i -> identical features for i
    rng = np.random.default_rng(0); idx = rng.choice(len(d), 300, replace=False); bad = 0
    for i in idx:
        c = d.conv.iat[i]; sub = d[(d.conv == c) & (d.index <= i)]
        Xi = features.compute(sub)[-1]
        bad += int(not np.allclose(Xi, X[i], atol=1e-6))
    R["feature_future_truncation_test"] = dict(sampled=len(idx), mismatches=bad)
    # 2. label-source columns are not feature inputs: permute them -> identical features
    d2 = d.copy()
    for col in ["addr_src", "bot_act_src"] + [h for h in HEADS]:
        d2[col] = rng.permutation(d2[col].to_numpy())
    R["feature_label_permutation_test"] = dict(identical=bool(np.allclose(features.compute(d2), X)))
    src = inspect.getsource(features.compute)
    forbidden = ["attention", "chosen_action", "delivered", "cur_addressed", "cur_reply", "addr_src", "bot_act_src", "y_", "tags"]
    R["forbidden_columns_read_by_features"] = [f for f in forbidden if f in src]
    R["excluded_availability_features"] = features.EXCLUDED
    # 3. single-feature AUC scan (train split), per regime feature set
    tr = (d.split == "train").to_numpy(); scan = {}
    for h in ["y_eot", "y_self", "y_addr", "y_hreply"]:
        y = d[h].to_numpy(); m = tr & ~np.isnan(y)
        for f in features.FEAT_TS:
            j = features.FEATURES.index(f); v = X[m, j]
            if len(np.unique(v)) < 2 or len(np.unique(y[m])) < 2: continue
            a = roc_auc_score(y[m], v); a = max(a, 1 - a)
            if a > 0.85: scan[f"{h}:{f}"] = round(float(a), 3)
    R["single_feature_auc_gt_0.85"] = scan
    # 4. availability probe: can the *pattern of text availability* predict labels? (motivates T vs TS regimes)
    ht = d.text.notna().to_numpy(); conv = d.conv.to_numpy(); ts = d.ts.to_numpy()
    P = np.zeros((len(d), 3)); last_text_t = {}; run = 0
    for i in range(len(d)):
        if i == 0 or conv[i] != conv[i - 1]: run = 0
        lo = max(0, i - 16); same = conv[lo:i] == conv[i]
        P[i, 0] = ht[lo:i][same].mean() if same.any() else 0
        P[i, 1] = ht[i]
        P[i, 2] = np.log1p(ts[i] - last_text_t[conv[i]]) if conv[i] in last_text_t else 10
        if ht[i]: last_text_t[conv[i]] = ts[i]
    probe = {}
    te = np.isin(d.split.to_numpy(), ["test_time", "test_group"])
    for h in ["y_act", "y_addr", "y_eot", "y_self"]:
        y = d[h].to_numpy(); yy = (y == 0) if h == "y_act" else (y == 1)  # y_act: 'speak' vs rest
        for name, cols, sub in [("all_events_availability_pattern", [0, 1, 2], np.ones(len(d), bool)),
                                ("TS_subset_history_pattern", [0, 2], ht)]:
            mtr = tr & ~np.isnan(y) & sub; mte = te & ~np.isnan(y) & sub
            if yy[mtr].sum() < 5 or yy[mte].sum() < 3: continue
            lr = LogisticRegression(max_iter=1000).fit(P[mtr][:, cols], yy[mtr])
            p = lr.predict_proba(P[mte][:, cols])[:, 1]
            probe[f"{h}:{name}"] = dict(test_pr_auc=round(float(average_precision_score(yy[mte], p)), 3), prior=round(float(yy[mte].mean()), 3),
                                        roc_auc=round(float(roc_auc_score(yy[mte], p)), 3))
    R["availability_probe"] = probe
    # 5. split integrity
    tr_rows = d[d.split == "train"]
    R["split_integrity"] = dict(holdout_groups_in_train=int(tr_rows.conv.isin(HOLDOUT_GROUPS).sum()),
                                max_train_ts_le_val_start_minus_embargo=bool(tr_rows.ts.max() < VAL_START - EMBARGO),
                                max_val_ts_le_test_start_minus_embargo=bool(d[d.split == "val"].ts.max() < TEST_START - EMBARGO),
                                n_embargo=int((d.split == "embargo").sum()))
    # 6. synthetic vs real text overlap (exact, normalized, len>=6) + eval sets contain no synthetic rows
    meta = pd.read_parquet("data/proc/meta.parquet")
    R["synthetic_rows_in_eval_splits"] = int(((meta.source == "synth") & meta.split.isin(["val", "test_time", "test_group"])).sum())
    try:
        from tidal.adapters import synth_frame
        s = synth_frame(); norm = lambda t: "".join(ch for ch in t if ch.isalnum())
        rs = {norm(t) for t in d.text.dropna() if len(norm(t)) >= 6}; ss = {norm(t) for t in s.text if len(norm(t)) >= 6}
        R["synthetic_real_exact_overlap_len6"] = dict(real_unique=len(rs), synth_unique=len(ss), overlap=len(rs & ss), examples_count_only=True)
    except FileNotFoundError: pass
    # 7. real text duplicates across train/test (generic short phrases are expected)
    trt = set(d[d.split == "train"].text.dropna()); tet = d[d.split.isin(["test_time", "test_group"])].text.dropna()
    R["test_texts_seen_in_train"] = dict(test_texts=int(len(tet)), seen=int(tet.isin(trt).sum()), seen_len_ge_8=int((tet.isin(trt) & (tet.str.len() >= 8)).sum()))
    # 8. model causality: outputs at position t unchanged when inputs after t are randomized
    for tag in model_tags:
        ck = torch.load(f"models/{tag}.pt", weights_only=False)
        m = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"]); m.load_state_dict(ck["state"]); m.eval()
        g = torch.Generator().manual_seed(0)
        F = torch.randn(8, 64, ck["n_feat"], generator=g); V = torch.ones(8, 64, dtype=torch.bool); E = torch.randn(8, 64, 512, generator=g) if ck["use_text"] else None
        with torch.no_grad():
            o1 = m(F, V, E); F2 = F.clone(); F2[:, 40:] = torch.randn(8, 24, ck["n_feat"], generator=g)
            E2 = None
            if E is not None: E2 = E.clone(); E2[:, 40:] = torch.randn(8, 24, 512, generator=g)
            o2 = m(F2, V, E2)
        R.setdefault("model_causality", {})[tag] = float(max((o1[h][:, :40] - o2[h][:, :40]).abs().max() for h in o1))
    json.dump(R, open("reports/leakage_audit.json", "w"), indent=1, ensure_ascii=False); print(json.dumps(R, indent=1, ensure_ascii=False))
if __name__ == "__main__":
    import sys; main(sys.argv[1:])
