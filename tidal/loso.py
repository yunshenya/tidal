"""Leave-one-scenario-out evaluation (场景通用). ALL livestream / 1:1 streams here are SCRIPT-SYNTHETIC (tidal/scenarios.py).
Direction A: models trained without a scenario are tested on that scenario's held-out synthetic streams.
Direction B: a model trained on synthetic only (LLM synth group + script livestream + script 1:1) is tested on REAL data.
Baselines are fitted on exactly the same training rows as the model: per-output prior, and logistic regression on the
same scenario-general event features (no sequence context). Paired block bootstrap over conv x 5-min blocks
(synthetic) / conv x hour (real).  usage: python -m tidal.loso -> reports/loso_phase2.json"""
import json, numpy as np, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal import vap as VP, vap_targets as VT, metrics as M
warnings.filterwarnings("ignore")

def train_mask(meta, data):
    src = meta.source.to_numpy(); sp = meta.split.to_numpy(); m = np.zeros(len(meta), bool)
    for k in data: s, sps = VP.SRC[k]; m |= (src == s) & np.isin(sp, sps)
    return m

def fit_baselines(D, tm, rng_seed=0):
    X, Y, V = D["X"], D["Y"], D["V"]; out = {}
    rs = np.random.default_rng(rng_seed)
    for hi, h in enumerate(HEADS):
        m = tm & ~np.isnan(Y[:, hi])
        if m.sum() < 50: continue
        if HEAD_DIMS[h] == 1 and h in VP.NO_SYNTH_HEADS: pass
        idx = np.flatnonzero(m); idx = idx if len(idx) <= 60000 else rs.choice(idx, 60000, replace=False)
        y = Y[idx, hi].astype(int)
        if len(np.unique(y)) < 2: continue
        lr = LogisticRegression(max_iter=300, C=1.0).fit(X[idx], y)
        prior = np.bincount(y, minlength=HEAD_DIMS[h] if HEAD_DIMS[h] > 1 else 2) / len(y)
        out[h] = (lr, prior)
    vout = {}
    for j in range(VT.NV):
        m = tm & ~np.isnan(V[:, j]); idx = np.flatnonzero(m); idx = idx if len(idx) <= 60000 else rs.choice(idx, 60000, replace=False)
        y = V[idx, j].astype(int)
        if len(np.unique(y)) == 2: vout[j] = LogisticRegression(max_iter=300).fit(X[idx], y)
    return out, vout

def bl_predict(bl, h, X):
    lr, prior = bl[h]; P = np.zeros((len(X), len(prior))); P[:, lr.classes_] = lr.predict_proba(X)
    pr = np.tile(prior, (len(X), 1))
    return (P[:, 1], pr[:, 1]) if HEAD_DIMS[h] == 1 else (P, pr)

def evaluate_set(D, rows, blocks, model_tags, bl, vbl, conv_mask, boot=300, vap_boot=0):
    Y = D["Y"][rows]; X = D["X"][rows]; V = D["V"][rows]; res = {"n": int(len(rows)), "heads": {}, "vap": {}}
    preds = {}
    for t in model_tags:
        lg, _ = VP.predict(t, D, rows, conv_mask=conv_mask); preds[t] = lg
    for hi, h in enumerate(HEADS):
        lab = ~np.isnan(Y[:, hi]); y = Y[lab, hi].astype(int)
        if h not in bl or lab.sum() < 30 or len(np.unique(y)) < 2: continue
        b = HEAD_DIMS[h] == 1; plr, ppr = bl_predict(bl, h, X[lab]); sys = {"lr_G": plr, "prior": ppr}
        for t in model_tags:
            l = preds[t][h][lab]; sys[t] = 1 / (1 + np.exp(-l[:, 0])) if b else np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True)
        hr = {"n": int(lab.sum()), "pos_rate": float(y.mean()) if b else np.bincount(y, minlength=HEAD_DIMS[h]).tolist()}
        for n, p in sys.items():
            if b: hr[n] = dict(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)) if n != "prior" else 0.5)
            else: hr[n] = dict(macro_f1=float(f1_score(y, p.argmax(1), average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0)))
        for t in model_tags:
            hr[t]["vs_lr_G"] = M.paired_delta(y, sys[t], plr, .5, .5, blocks[lab], "binary" if b else "multi", n_boot=boot)
        res["heads"][h] = hr
    # projection quality: mean AUC over the 20 outputs (masked), model vs LR
    for n in model_tags + ["lr_G"]:
        a = []
        for j in range(VT.NV):
            m = ~np.isnan(V[:, j])
            if m.sum() < 30 or not (0 < V[m, j].sum() < m.sum()) or j not in vbl: continue
            p = (1 / (1 + np.exp(-preds[n]["vap"][m, j]))) if n != "lr_G" else vbl[j].predict_proba(X[m])[:, 1]
            a.append(roc_auc_score(V[m, j], p))
        res["vap"][n] = dict(mean_auc=float(np.mean(a)) if a else None, n_outputs=len(a))
    if vap_boot:                                   # block-bootstrap CI of (model - LR) mean projection AUC
        outs = [j for j in range(VT.NV) if j in vbl and (~np.isnan(V[:, j])).sum() >= 30 and 0 < np.nansum(V[:, j]) < (~np.isnan(V[:, j])).sum()]
        P = {n: {j: ((1 / (1 + np.exp(-preds[n]["vap"][:, j]))) if n != "lr_G" else _vproba(vbl[j], X)) for j in outs} for n in model_tags + ["lr_G"]}
        def mauc(idx, n):
            a = []
            for j in outs:
                v = V[idx, j]; m = ~np.isnan(v)
                if m.sum() >= 30 and 0 < v[m].sum() < m.sum(): a.append(roc_auc_score(v[m], P[n][j][idx][m]))
            return np.mean(a) if a else np.nan
        for t in model_tags:
            d = M.boot(lambda idx: mauc(idx, t) - mauc(idx, "lr_G"), blocks, vap_boot, 0)
            res["vap"][t]["minus_lr_G"] = dict(mean=float(np.nanmean(d)), ci=M.ci(np.asarray(d).ravel()))
    return res, preds

def _vproba(clf, X): return clf.predict_proba(X)[:, 1]

def respond_to_hit1(D, rows_conv_mask, preds_by_tag, rows):
    """which-event-to-respond-to on synthetic livestream: candidates = non-self events in the 30 s before a host reply."""
    meta = D["meta"]; ts = meta.ts.to_numpy(); role = meta.role.to_numpy(); mid = meta.msg_id.to_numpy(); conv = meta.conv.to_numpy()
    pos = {r: k for k, r in enumerate(rows)}; out = {}
    host = [r for r in rows if role[r] == "self" and isinstance(meta.responds_to.iat[r], str)]
    for name in list(preds_by_tag) + ["most_recent", "random"]:
        hits = []; nc = []
        for r in host:
            c = [q for q in range(max(0, r - 400), r) if conv[q] == conv[r] and role[q] != "self" and ts[r] - ts[q] <= 30 and q in pos]
            if not c: continue
            nc.append(len(c)); tgt = meta.responds_to.iat[r]
            if name == "most_recent": pick = c[-1]
            elif name == "random": hits.append(float(tgt in set(mid[c])) / len(c)); continue
            else:
                lg = preds_by_tag[name]["y_act"][[pos[q] for q in c]]; ps = np.exp(lg[:, 0]) / np.exp(lg).sum(1)
                sc = ps * np.exp(-(ts[r] - ts[c]) / 20.0); pick = c[int(np.argmax(sc))]
            hits.append(float(mid[pick] == tgt))
        out[name] = dict(hit1=float(np.mean(hits)) if hits else None, n=len(hits), mean_candidates=float(np.mean(nc)) if nc else None)
    return out

def main():
    D = VP.load(); meta = D["meta"]; sc = meta.scenario.to_numpy(); sp = meta.split.to_numpy(); conv = meta.conv.to_numpy()
    ts = meta.ts.to_numpy()
    blk5 = np.array([f"{c}:{int(t // 300)}" for c, t in zip(conv, ts)]); blkh = np.array([f"{c}:{int(t // 3600)}" for c, t in zip(conv, ts)])
    has = lambda r: r[~np.all(np.isnan(D["Y"][r]), 1)]
    sets = {"syn_livestream_test": has(np.flatnonzero((sc == "livestream") & (sp == "syn_test"))),
            "syn_1on1_test": has(np.flatnonzero((sc == "one_on_one") & (sp == "syn_test"))),
            "real_test": has(np.flatnonzero((sc == "real_chat") & np.isin(sp, ["test_time", "test_group"]))),
            "real_group_test": has(np.flatnonzero((sc == "real_chat") & (meta.conv_type.to_numpy() == "group") & np.isin(sp, ["test_time", "test_group"])))}
    import os
    ok = lambda t: os.path.exists(f"models/{t}.pt")
    plan = [  # (name, test set, training data of the OOD model, model tags, in-domain reference tags)
        ("A1: train real+LLM-synth -> test synthetic livestream (unseen)", "syn_livestream_test", ["real", "llm"], ["P2a_s0", "P2a_s1", "P2a_s2"]),
        ("A2: train real+LLM-synth+synthetic 1:1 -> test synthetic livestream (unseen)", "syn_livestream_test", ["real", "llm", "1on1"], ["P2L_nolive_s0"]),
        ("A3: train real+LLM-synth -> test synthetic 1:1 (unseen)", "syn_1on1_test", ["real", "llm"], ["P2a_s0", "P2a_s1", "P2a_s2"]),
        ("A4: train real+LLM-synth+synthetic livestream -> test synthetic 1:1 (unseen)", "syn_1on1_test", ["real", "llm", "live"], ["P2L_no1on1_s0"]),
        ("B: no real data (LLM-synth group + synthetic livestream + synthetic 1:1) -> test REAL chat", "real_test", ["llm", "live", "1on1"], ["P2L_synonly_s0"]),
        ("C: no group data at all (synthetic livestream + synthetic 1:1 only) -> test REAL group chats (unseen scenario)", "real_group_test", ["live", "1on1"], ["P2L_nogroup_s0"]),
        ("ref: all sources -> REAL group chats", "real_group_test", ["real", "llm", "live", "1on1"], ["P2b_s0", "P2b_s1", "P2b_s2"]),
        ("ref: real+LLM-synth -> REAL group chats", "real_group_test", ["real", "llm"], ["P2a_s0", "P2a_s1", "P2a_s2"]),
        ("ref-in-domain: all sources -> synthetic livestream", "syn_livestream_test", ["real", "llm", "live", "1on1"], ["P2b_s0", "P2b_s1", "P2b_s2"]),
        ("ref-in-domain: all sources -> synthetic 1:1", "syn_1on1_test", ["real", "llm", "live", "1on1"], ["P2b_s0", "P2b_s1", "P2b_s2"]),
        ("ref-in-domain: all sources -> real", "real_test", ["real", "llm", "live", "1on1"], ["P2b_s0"]),
    ]
    rep = dict(note="livestream and 1:1 streams are SCRIPT-SYNTHETIC; LLM synth group chats are LLM-generated; only real_test is real data",
               sets={k: int(len(v)) for k, v in sets.items()}, experiments=[]); bl_cache = {}
    for name, set_name, data, tags in plan:
        tags = [t for t in tags if ok(t)]
        if not tags: continue
        key = ",".join(data)
        if key not in bl_cache: bl_cache[key] = fit_baselines(D, train_mask(meta, data))
        bl, vbl = bl_cache[key]; rows = sets[set_name]
        cm = np.isin(sc, ["real_chat"]) if set_name.startswith("real") else (sc == ("livestream" if "live" in set_name else "one_on_one"))
        r, preds = evaluate_set(D, rows, (blkh if set_name.startswith("real") else blk5)[rows], tags, bl, vbl, cm)
        r.update(name=name, test=set_name, train=data, models=tags)
        if set_name == "syn_livestream_test": r["respond_to_hit1"] = respond_to_hit1(D, cm, preds, rows)
        rep["experiments"].append(r); print(name)
        for h, hr in r["heads"].items():
            k = "pr_auc" if "pr_auc" in hr["lr_G"] else "macro_f1"
            print(f"  {h:10s} n={hr['n']:5d} prior={hr['prior'][k]:.3f} lr_G={hr['lr_G'][k]:.3f} " + " ".join(
                f"{t}={hr[t][k]:.3f}(Δ{hr[t]['vs_lr_G']['d_primary_mean']:+.3f}[{hr[t]['vs_lr_G']['d_primary_ci'][0]:+.3f},{hr[t]['vs_lr_G']['d_primary_ci'][1]:+.3f}])" for t in tags))
        print("  vap", {k: round(v["mean_auc"], 3) if v["mean_auc"] else None for k, v in r["vap"].items()}, r.get("respond_to_hit1", ""))
    # also: does the real-trained 'real' rows need the real rows only in real_test? (blocks conv x hour)
    json.dump(rep, open("reports/loso_phase2.json", "w"), indent=1, default=float)
if __name__ == "__main__": main()
