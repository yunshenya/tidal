"""Phase-2 evaluation on REAL held-out data (val: temperature/thresholds/abstention; test_time & test_group: report).
Compares phase-2 VAP models vs the phase-1 shipped model (T_rs_gru_s1) and all phase-1 baselines (identical rows),
with the same paired conv x hour block bootstrap. Adds per-head temperature scaling, ECE, risk-coverage/AURC and a
selective-abstention ('wait and recheck after X s') policy; plus zero-shot heads derived from the projection alone.
usage: python -m tidal.evaluate2 [--boot 500] -> reports/eval_phase2.json"""
import argparse, json, numpy as np, torch
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal import metrics as M, vap as VP, vap_targets as VT, seqdata as SQ
from tidal.evaluate import predict_logits, fit_temperature, to_prob
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
BIN = [h for h in HEADS if HEAD_DIMS[h] == 1]
RC_MID = np.array([1, 3.5, 7.5, 15, 40, 180, 300.0])      # y_recheck bucket midpoints (s)
CONFIGS = {"P2a": "VAP pretrain (real+LLM synth) -> multitask FT + VAP aux",
           "P2b": "same + script synthetic livestream/1:1 streams",
           "P2n": "no pretrain (ablation): multitask + VAP aux from scratch"}
PH1 = "T_rs_gru_s1"

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def any_of(P, ch, bins):   # P(at least one event of channel ch in the union of bins), bins treated independent
    q = np.ones(len(P))
    for b in bins: q *= 1 - P[:, VT.idx(ch, b)]
    return 1 - q
def derive(vp):
    """zero-shot head scores from projection probabilities [n,20] (no downstream supervision)."""
    return {"y_self": any_of(vp, "current", range(5)),            # same author returns within 180 s
            "y_hreply": any_of(vp, "others", range(4)),           # another human within 60 s
            "y_eot": 1 - any_of(vp, "current", range(3)),         # author does not continue within 15 s
            "speak": any_of(vp, "self", range(3)),                # bot posts within 15 s (-> y_act == speak)
            "y_addr": any_of(vp, "self", range(4))}               # bot posts within 60 s (weak proxy)

def risk_cov(correct, conf):
    o = np.argsort(-conf, kind="stable"); c = correct[o].astype(float); k = np.arange(1, len(c) + 1)
    risk = 1 - np.cumsum(c) / k; return float(risk.mean()), risk
def sel_points(correct, conf, covs=(0.5, 0.7, 0.8, 0.9, 1.0)):
    _, risk = risk_cov(correct, conf); n = len(risk)
    return {f"{c:.1f}": float(risk[max(0, int(np.ceil(c * n)) - 1)]) for c in covs}
def decision(h, p):
    if HEAD_DIMS[h] == 1: return (p >= .5).astype(int), np.maximum(p, 1 - p)
    return p.argmax(1), p.max(1)

def main(boot=500):
    D = VP.load(); meta = D["meta"]; Y = D["Y"]; split = meta.split.to_numpy()
    real = (meta.source == "real").to_numpy()
    rows = np.flatnonzero(real & np.isin(split, ["val", "test_time", "test_group"]) & ~np.all(np.isnan(Y), 1))
    hour = (meta.ts.to_numpy() // 3600).astype(int)
    blocks = np.array([f"{c}:{h}" for c, h in zip(meta.conv.to_numpy(), hour)])[rows]
    sp = split[rows]; Yr = Y[rows]
    # ---- systems: phase-1 baselines (same row order as phase-1 dataset), phase-1 model, phase-2 models
    systems, raw = {}, {}
    for k, v in dict(np.load("data/proc/baseline_preds_T.npz")).items():
        h, name = k.split("|"); raw.setdefault(name, {})[h] = v[rows]
    rep = dict(rows=dict(val=int((sp == "val").sum()), test_time=int((sp == "test_time").sum()), test_group=int((sp == "test_group").sum())),
               configs=CONFIGS, models={}, heads={}, temps={})
    D1 = SQ.load("T"); assert np.array_equal(np.isnan(D1["Y"][rows]), np.isnan(Yr))
    lg1, ck1 = predict_logits(PH1, D1, rows)
    logits = {"phase1:" + PH1: {h: (lg1[h][:, 0] if HEAD_DIMS[h] == 1 else lg1[h]) for h in HEADS}}
    rep["models"]["phase1:" + PH1] = dict(params=int(ck1["params"]), data=ck1["data"])
    tags = []
    for cfg in CONFIGS:
        for s in range(3):
            tag = f"{cfg}_s{s}"
            try: lg, ck = VP.predict(tag, D, rows, conv_mask=real)
            except FileNotFoundError: continue
            tags.append(tag); logits["p2:" + tag] = {h: (lg[h][:, 0] if HEAD_DIMS[h] == 1 else lg[h]) for h in HEADS}
            vp = 1 / (1 + np.exp(-lg["vap"])); logits["p2:" + tag]["_vap"] = vp
            rep["models"]["p2:" + tag] = dict(params=int(ck["params"]), data=ck["data"], pretrain=ck["pretrain"], best_val=float(ck["best_val"]),
                                              best_ep=int(ck["best_ep"]), train_seconds=ck["train_seconds"], pre_seconds=ck.get("pre_seconds", 0))
            if ck["pretrain"]:
                lz, _ = VP.predict(tag, D, rows, conv_mask=real, stage="pre")
                logits["zs:" + tag] = {"_vap": 1 / (1 + np.exp(-lz["vap"]))}
    # pre-registered selection: config with lowest mean real-val downstream loss, then its best-val seed
    cv = {c: np.mean([rep["models"]["p2:" + t]["best_val"] for t in tags if t.startswith(c)]) for c in CONFIGS if any(t.startswith(c) for t in tags)}
    best_cfg = min(cv, key=cv.get); primary = min([t for t in tags if t.startswith(best_cfg)], key=lambda t: rep["models"]["p2:" + t]["best_val"])
    rep["selection"] = dict(config_val_loss=cv, best_config=best_cfg, primary="p2:" + primary, zero_shot="zs:" + primary)
    va_all = sp == "val"
    # temperature scaling per head per system (fit on val), applied to models AND baselines (on logit(p))
    for name, lg in logits.items():
        if name.startswith("zs:"): continue
        systems[name] = {}
        for hi, h in enumerate(HEADS):
            b = HEAD_DIMS[h] == 1; va = va_all & ~np.isnan(Yr[:, hi])
            T = fit_temperature(lg[h][va], Yr[va, hi], b); rep["temps"].setdefault(name, {})[h] = float(T)
            systems[name][h] = to_prob(lg[h], T, b)
    for name, pr in raw.items():
        systems[name] = {}; rep["temps"].setdefault(name, {})
        for hi, h in enumerate(HEADS):
            if h not in pr: continue
            b = HEAD_DIMS[h] == 1; va = va_all & ~np.isnan(Yr[:, hi])
            l = logit(pr[h]) if b else np.log(np.clip(pr[h], 1e-6, 1))
            T = fit_temperature(l[va], Yr[va, hi], b); rep["temps"][name][h] = float(T); systems[name][h] = to_prob(l, T, b)
            systems.setdefault(name + "(raw)", {})[h] = pr[h]
    # zero-shot projection-derived systems (no temperature: they are not fitted to downstream labels at all)
    zsn = "zs:" + primary
    if zsn in logits:
        d = derive(logits[zsn]["_vap"]); systems[zsn] = {h: d[h] for h in ["y_self", "y_hreply", "y_eot", "y_addr"]}
        systems[zsn]["speak"] = d["speak"]
    base_names = [n for n in raw]
    P2 = "p2:" + primary; P1 = "phase1:" + PH1
    # ---- per-head discriminative + calibration metrics
    for hi, h in enumerate(HEADS):
        b = HEAD_DIMS[h] == 1; y = Yr[:, hi]; lab = ~np.isnan(y); va = lab & va_all
        hres = dict(val={}, thr={}, systems={}, risk_coverage={}, abstention={})
        for name, pr in systems.items():
            if h not in pr: continue
            p = pr[h]
            if b:
                hres["thr"][name] = M.best_threshold(y[va], p[va])
                hres["val"][name] = float(average_precision_score(y[va], p[va])) if 0 < y[va].sum() < va.sum() else None
            else:
                hres["val"][name] = float(f1_score(y[va].astype(int), p[va].argmax(1), average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0))
        strongest = max([n for n in base_names if hres["val"].get(n) is not None], key=lambda n: hres["val"][n]); hres["strongest_baseline"] = strongest
        for spn in ["test_time", "test_group"]:
            te = lab & (sp == spn)
            if te.sum() < 5: continue
            S = hres["systems"].setdefault(spn, {}); RC = hres["risk_coverage"].setdefault(spn, {})
            for name, pr in systems.items():
                if h not in pr or name.endswith("(raw)") and h in raw.get(name[:-5], {}) and False: continue
                if h not in pr: continue
                p = pr[h][te]
                r = M.binary_report(y[te], p, hres["thr"].get(name, .5), blocks[te], n_boot=boot) if b else M.multiclass_report(y[te], p, blocks[te], n_boot=boot)
                S[name] = r
                if not name.startswith("zs:"):
                    dec, conf = decision(h, p); a, _ = risk_cov(dec == y[te].astype(int), conf)
                    RC[name] = dict(aurc=a, sel_risk=sel_points(dec == y[te].astype(int), conf))
            key = "pr_auc" if b else "macro_f1"
            bt = max([n for n in base_names if n in S], key=lambda n: np.nan_to_num(S[n][key], nan=-1)); hres.setdefault("best_baseline_on_test", {})[spn] = bt
            kind = "binary" if b else "multi"
            for name in [n for n in systems if (n.startswith("p2:") or n.startswith("phase1:") or n.startswith("zs:")) and h in systems[n]]:
                pa = systems[name][h][te]
                for lab_, ref in [("vs_strongest", strongest), ("vs_best_on_test", bt)] + ([("vs_phase1", P1)] if name != P1 else []):
                    S[name][lab_] = M.paired_delta(y[te], pa, systems[ref][h][te], hres["thr"].get(name, .5), hres["thr"].get(ref, .5), blocks[te], kind, n_boot=boot)
            # paired delta AURC (primary & phase-1 vs strongest baseline; primary vs phase-1)
            yy = y[te].astype(int)
            def d_aurc(na, nb):
                da, ca = decision(h, systems[na][h][te]); db, cb = decision(h, systems[nb][h][te])
                bs = M.boot(lambda idx: (risk_cov(da[idx] == yy[idx], ca[idx])[0] - risk_cov(db[idx] == yy[idx], cb[idx])[0],), blocks[te], boot, 0)
                return dict(mean=float(np.nanmean(bs[:, 0])), ci=M.ci(bs[:, 0]))
            RC["d_aurc"] = {f"{P2}-{strongest}": d_aurc(P2, strongest), f"{P1}-{strongest}": d_aurc(P1, strongest), f"{P2}-{P1}": d_aurc(P2, P1)}
            # abstention policy: abstain on the least-confident 20% (threshold fitted on val); recheck after E[time-to-next-event]
            for name in [P2, P1, strongest]:
                dv, cv_ = decision(h, systems[name][h][va]); tau = float(np.quantile(cv_, 0.2))
                dt, ct = decision(h, systems[name][h][te]); ans = ct >= tau; corr = dt == yy
                rc = systems[name]["y_recheck"][te] if "y_recheck" in systems[name] else None
                delay = np.clip(rc @ RC_MID, 2, 300) if rc is not None else np.full(te.sum(), 60.0)
                nxt = Yr[te, HEADS.index("y_recheck")]; nxt_s = np.where(np.isnan(nxt), np.nan, RC_MID[np.nan_to_num(nxt).astype(int)])
                ab = ~ans
                ent = dict(tau=tau, coverage=float(ans.mean()), acc_all=float(corr.mean()), acc_answered=float(corr[ans].mean()) if ans.any() else None,
                           acc_abstained=float(corr[ab].mean()) if ab.any() else None,
                           recheck_delay_median_s=float(np.median(delay[ab])) if ab.any() else None,
                           new_event_before_recheck=float(np.nanmean(nxt_s[ab] <= delay[ab])) if ab.any() else None)
                if not b:
                    ent["macro_f1_all"] = float(f1_score(yy, dt, average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0))
                    ent["macro_f1_answered"] = float(f1_score(yy[ans], dt[ans], average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0)) if ans.any() else None
                hres["abstention"].setdefault(spn, {})[name] = ent
        rep["heads"][h] = hres
    # speak-now (y_act == speak) as a binary view, incl. the projection-derived zero-shot score
    hi = HEADS.index("y_act"); y = Yr[:, hi]; lab = ~np.isnan(y); sv = {}
    for spn in ["test_time", "test_group"]:
        te = lab & (sp == spn); ys = (y[te] == 0).astype(int)
        if not (0 < ys.sum() < len(ys)): continue
        sv[spn] = {n: dict(pr_auc=float(average_precision_score(ys, systems[n]["y_act"][te][:, 0] if n != zsn else systems[n]["speak"][te])),
                           roc_auc=float(roc_auc_score(ys, systems[n]["y_act"][te][:, 0] if n != zsn else systems[n]["speak"][te])), n=int(len(ys)), pos=int(ys.sum()))
                   for n in [P2, P1, zsn] + base_names if n in systems and ("y_act" in systems[n] or n == zsn)}
    rep["speak_now_binary"] = sv
    # intrinsic projection quality on real test (AUC per channel, averaged over bins)
    vp_true = D["V"][rows]; intr = {}
    for spn in ["test_time", "test_group"]:
        te = sp == spn; out = {}
        for n in [P2, zsn]:
            if n not in logits: continue
            P = logits[n]["_vap"][te]; T = vp_true[te]; ch = {}
            for c in VT.CHANNELS:
                a = []
                for bb in range(len(VT.BINS)):
                    j = VT.idx(c, bb); m = ~np.isnan(T[:, j])
                    if m.sum() > 20 and 0 < T[m, j].sum() < m.sum(): a.append(roc_auc_score(T[m, j], P[m, j]))
                ch[c] = float(np.mean(a)) if a else None
            out[n] = ch
        intr[spn] = out
    rep["vap_intrinsic_auc"] = intr
    json.dump(rep, open("reports/eval_phase2.json", "w"), indent=1, default=float)
    summarize(rep)

def summarize(rep):
    sel = rep["selection"]; P2 = sel["primary"]; print("selection", sel)
    for h, r in rep["heads"].items():
        print(f"== {h} strongest(val)={r['strongest_baseline']}")
        for spn, S in r["systems"].items():
            for n, m in S.items():
                if not (n.startswith("p2:") and n != P2) and not n.endswith("(raw)"):
                    key = "pr_auc" if "pr_auc" in m else "macro_f1"
                    ex = "".join(f" {k[3:]}Δ={m[k]['d_primary_mean']:+.3f}[{m[k]['d_primary_ci'][0]:+.3f},{m[k]['d_primary_ci'][1]:+.3f}]" for k in ("vs_strongest", "vs_phase1") if k in m)
                    print(f"  {spn:10s} {n:26s} n={m['n']:5d} {key}={m[key]:.3f} ece={m.get('ece', m.get('ece_top')):.3f}{ex}")
            print("   AURC", {k: round(v['aurc'], 3) for k, v in r["risk_coverage"][spn].items() if isinstance(v, dict) and 'aurc' in v and not k.startswith('p2:') or k == P2})
            print("   dAURC", {k: (round(v['mean'], 3), tuple(round(x, 3) for x in v['ci'])) for k, v in r["risk_coverage"][spn]["d_aurc"].items()})
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--boot", type=int, default=500); a = ap.parse_args(); main(a.boot)
