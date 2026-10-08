"""Identity-free roles: cold-start + per-conversation online adaptation, on HELD-OUT groups (never in training).

Roles are relative only (self = the bot/host, other = anyone else; 'current' = the author of the event in the VAP
targets) -- no user identity, no per-user embedding, nothing that has to be learned per person. So a new conversation
needs no enrolment; what can still go wrong is the per-conversation *rate* (how chatty the group is, how often the bot
is addressed). We test that directly:

  join points  : tidal 'joins' a held-out conversation with NO history every JOIN events (fresh features + context),
                 and is scored on the first N = 20 / 50 / 100 events after joining.
  systems      : warm   = same model, same events, full conversation history (reference upper bound)
                 cold   = model with only the events since joining (features recomputed by the streaming featurizer)
                 cold+adapt = cold + per-conversation online Platt/bias adaptation, updated only from labels that are
                              already REVEALED at prediction time (label horizon per head), lr chosen on real val
                 base_online = per-conversation running base rate of revealed labels (shrunk to the train prior)
                 lr_G   = logistic regression on the same cold, scenario-general event features (fit on real train)
                 prior  = train base rate
Blocks for the paired bootstrap = join segments.  usage: python -m tidal.coldstart [TAG] -> reports/coldstart_phase2.json"""
import sys, json, numpy as np, torch, warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal.seqdata import CTX
from tidal import vap as VP, metrics as M, features_g as FG
from tidal.duplex import EventFeaturizer, Event
warnings.filterwarnings("ignore")
JOIN = 120; NS = (20, 50, 100)
HORIZON = {"y_eot": 20, "y_self": 200, "y_addr": 60, "y_act": 60, "y_recheck": 300, "y_hreply": 60}   # label known after (s)
PRIOR_N = 10.0

def segments(meta, rows_mask, join=JOIN):
    """[(conv, global rows of one join segment)] in time order; a segment = events from a join point to the next."""
    out = []
    for rows in VP.conv_rows(meta, rows_mask):
        for a in range(0, len(rows), join): out.append(rows[a:a + join])
    return out

def cold_features(meta, seg, mu, sd):
    f = EventFeaturizer(mu, sd); ts = meta.ts.to_numpy(); role = meta.role.to_numpy()
    X = np.stack([f(Event(str(r), float(ts[r]), "self" if role[r] == "self" else "other")) for r in seg])
    return X

def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

def model_logits(m, X_seg):
    """logits at every position of a segment from <=64-event windows that never reach before the join point."""
    n = len(X_seg); F = np.zeros((n, CTX, X_seg.shape[1]), np.float32); V = np.zeros((n, CTX), bool)
    for k in range(n):
        lo = max(0, k - CTX + 1); w = X_seg[lo:k + 1]; F[k, CTX - len(w):] = w; V[k, CTX - len(w):] = True
    with torch.no_grad(): o = m(torch.from_numpy(F), torch.from_numpy(V))
    return {h: o[h][:, -1].numpy() for h in HEADS}

class Adapter:
    """per-conversation online recalibration of one head: binary -> sigmoid(a*z+b); multiclass -> softmax(z+b)."""
    def __init__(self, k, lr): self.k = k; self.lr = lr; self.a = 1.0; self.b = np.zeros(max(1, k) if k > 1 else 1)
    def prob(self, z):
        return 1 / (1 + np.exp(-(self.a * z + self.b[0]))) if self.k == 1 else softmax(z + self.b)
    def update(self, z, y):
        p = self.prob(z)
        if self.k == 1:
            g = p - y; self.b[0] -= self.lr * g; self.a -= self.lr * g * z
            self.b[0] *= 0.995; self.a = 1 + 0.995 * (self.a - 1)          # weak pull back to the identity
        else:
            t = np.zeros(self.k); t[int(y)] = 1; self.b -= self.lr * (p - t); self.b *= 0.995

def run_segment(meta, Y, seg, lz, temps, lr):
    """online pass: at each event, predict with the adapter state built ONLY from labels revealed before it."""
    ts = meta.ts.to_numpy()[seg]; out = {h: [] for h in HEADS}; base = {h: [] for h in HEADS}
    for hi, h in enumerate(HEADS):
        k = HEAD_DIMS[h]; ad = Adapter(k, lr[h] if isinstance(lr, dict) else lr); z_all = lz[h] / temps.get(h, 1.0); z_all = z_all[:, 0] if k == 1 else z_all
        done = 0; order = np.argsort(ts + HORIZON[h], kind="stable"); rev_t = (ts + HORIZON[h])[order]
        cnt = np.zeros(max(2, k)); ptr = 0
        for i in range(len(seg)):
            while ptr < len(order) and rev_t[ptr] <= ts[i]:
                j = order[ptr]; y = Y[seg[j], hi]
                if not np.isnan(y): ad.update(z_all[j], y); cnt[int(y)] += 1
                ptr += 1
            out[h].append(ad.prob(z_all[i])); base[h].append(cnt.copy())
        out[h] = np.array(out[h]); base[h] = np.array(base[h])
    return out, base

def main(tag=None, boot=500):
    ev = json.load(open("reports/eval_phase2.json")); tag = tag or ev["selection"]["primary"].split(":", 1)[1]
    temps = ev["temps"]["p2:" + tag]
    D = VP.load(); meta = D["meta"]; Y = D["Y"]; sp = meta.split.to_numpy(); src = meta.source.to_numpy()
    m, ck = VP.load_model(tag); mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
    real = src == "real"
    # train priors + LR on scenario-general features (real train rows, full-history features)
    tr = real & (sp == "train"); prior, lrG = {}, {}
    for hi, h in enumerate(HEADS):
        mm = tr & ~np.isnan(Y[:, hi]); y = Y[mm, hi].astype(int); K = max(2, HEAD_DIMS[h])
        prior[h] = np.bincount(y, minlength=K) / len(y); lrG[h] = LogisticRegression(max_iter=500).fit(D["X"][mm], y)
    # warm predictions (full history) for every real row
    warm_rows = np.flatnonzero(real); wl, _ = VP.predict(tag, D, warm_rows, conv_mask=real); wpos = {r: k for k, r in enumerate(warm_rows)}
    def collect(mask, lr):
        rec = []
        for si, seg in enumerate(segments(meta, mask)):
            Xs = cold_features(meta, seg, mu, sd); Xs[:, FG.NB:] = D["X"][seg, FG.NB:]   # conversation-level context (participants) is known at join time
            lz = model_logits(m, Xs)
            ad, cnt = run_segment(meta, Y, seg, lz, temps, lr)
            for hi, h in enumerate(HEADS):
                k = HEAD_DIMS[h]; T = temps.get(h, 1.0)
                for i, r in enumerate(seg):
                    if np.isnan(Y[r, hi]): continue
                    zc = lz[h][i] / T; zw = wl[h][wpos[r]] / T
                    pc = 1 / (1 + np.exp(-zc[0])) if k == 1 else softmax(zc); pw = 1 / (1 + np.exp(-zw[0])) if k == 1 else softmax(zw)
                    c = cnt[h][i]; pb = (c + PRIOR_N * prior[h]) / (c.sum() + PRIOR_N)
                    pl = lrG[h].predict_proba(Xs[i:i + 1])[0]; pp = prior[h]
                    if k == 1: pb, pl, pp = pb[1], pl[1], pp[1]
                    rec.append(dict(h=h, seg=si, pos=i, y=Y[r, hi], warm=pw, cold=pc, adapt=ad[h][i], base_online=pb, lr_G=pl, prior=pp))
        return rec
    # choose the adaptation learning rate on real VAL (seen groups, later time) by mean log-loss on positions <= 100
    def ll(rec, key):
        v = []
        for e in rec:
            p = e[key]; y = int(e["y"])
            v.append(-np.log(np.clip(p if HEAD_DIMS[e["h"]] == 1 and y == 1 else (1 - p if HEAD_DIMS[e["h"]] == 1 else p[y]), 1e-9, 1)))
        return float(np.mean(v))
    lr_grid = [0.0, 0.02, 0.05, 0.1, 0.2]; val_ll = {h: {} for h in HEADS}
    for lr in lr_grid:
        rec = [e for e in collect(real & (sp == "val"), lr) if e["pos"] < 100]
        for h in HEADS: val_ll[h][lr] = ll([e for e in rec if e["h"] == h], "adapt")
    lr_best = {h: min(v, key=v.get) for h, v in val_ll.items()}            # per head; 0.0 = no adaptation
    rec = collect(real & (sp == "test_group"), lr_best)
    systems = ["warm", "cold", "adapt", "base_online", "lr_G", "prior"]
    rep = dict(tag=tag, join_every=JOIN, horizons_s=HORIZON, adapt_lr_val_logloss=val_ll, adapt_lr=lr_best,
               note="held-out groups only (2 groups never used in training); join points every %d events; blocks = join segments" % JOIN,
               n_segments=len(set(e["seg"] for e in rec)), heads={})
    for h in HEADS:
        b = HEAD_DIMS[h] == 1; H = [e for e in rec if e["h"] == h]; rep["heads"][h] = {}
        for N in NS + (JOIN,):
            S = [e for e in H if e["pos"] < N]
            if len(S) < 20: continue
            y = np.array([e["y"] for e in S]).astype(int); blocks = np.array([e["seg"] for e in S])
            P = {s: np.array([e[s] for e in S], float) for s in systems}
            if b and not (0 < y.sum() < len(y)): rep["heads"][h][f"first_{N}"] = dict(n=len(y), pos=int(y.sum()), note="single class"); continue
            r = dict(n=int(len(y)), pos=int(y.sum()) if b else np.bincount(y, minlength=HEAD_DIMS[h]).tolist(), systems={}, deltas={})
            for s in systems:
                p = P[s]
                if b: r["systems"][s] = dict(pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)), ece=M.ece(y, p),
                                             logloss=float(-np.mean(y * np.log(np.clip(p, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-9, 1)))))
                else: r["systems"][s] = dict(macro_f1=float(f1_score(y, p.argmax(1), average="macro", labels=list(range(HEAD_DIMS[h])), zero_division=0)),
                                             speak_pr_auc=float(average_precision_score(y == 0, p[:, 0])) if h == "y_act" and 0 < (y == 0).sum() < len(y) else None,
                                             ece_top=M.ece((p.argmax(1) == y).astype(int), p.max(1)), logloss=float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))))
            for a, c in [("adapt", "cold"), ("cold", "warm"), ("adapt", "warm"), ("adapt", "lr_G"), ("adapt", "base_online"), ("cold", "lr_G")]:
                r["deltas"][f"{a}-{c}"] = M.paired_delta(y, P[a], P[c], .5, .5, blocks, "binary" if b else "multi", n_boot=boot)
                # log-loss delta (negative = a better), same blocks
                def lfn(idx, a=a, c=c):
                    def L(p):
                        return -np.mean(y[idx] * np.log(np.clip(p[idx], 1e-9, 1)) + (1 - y[idx]) * np.log(np.clip(1 - p[idx], 1e-9, 1))) if b else -np.mean(np.log(np.clip(p[idx][np.arange(len(idx)), y[idx]], 1e-9, 1)))
                    return (L(P[a]) - L(P[c]),)
                bs = M.boot(lfn, blocks, boot, 0); r["deltas"][f"{a}-{c}"]["d_logloss_mean"] = float(np.nanmean(bs[:, 0])); r["deltas"][f"{a}-{c}"]["d_logloss_ci"] = M.ci(bs[:, 0])
            rep["heads"][h][f"first_{N}"] = r
    json.dump(rep, open("reports/coldstart_phase2.json", "w"), indent=1, default=float)
    print("adapt lr per head (val):", lr_best)
    for h, hr in rep["heads"].items():
        for k, r in hr.items():
            if "systems" not in r: print(h, k, r); continue
            key = "pr_auc" if "pr_auc" in r["systems"]["cold"] else "macro_f1"
            d = r["deltas"]
            print(f"{h:9s} {k:10s} n={r['n']:4d} pos={r['pos']} " + " ".join(f"{s}={v[key]:.3f}" for s, v in r["systems"].items()) +
                  "  adapt-cold=%+.3f[%+.3f,%+.3f] cold-warm=%+.3f[%+.3f,%+.3f] adapt-lrG=%+.3f[%+.3f,%+.3f] dLL(adapt-cold)=%+.3f[%+.3f,%+.3f]" % (
                      d["adapt-cold"]["d_primary_mean"], *d["adapt-cold"]["d_primary_ci"], d["cold-warm"]["d_primary_mean"], *d["cold-warm"]["d_primary_ci"],
                      d["adapt-lr_G"]["d_primary_mean"], *d["adapt-lr_G"]["d_primary_ci"], d["adapt-cold"]["d_logloss_mean"], *d["adapt-cold"]["d_logloss_ci"]))

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
