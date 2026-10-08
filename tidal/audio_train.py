"""Phase 3: train + evaluate the streaming audio front end on two-speaker, separately recorded conversations
(MagicData multi-stream Chinese; research-only license -> weights are never published).

  python -m tidal.audio_train prep         # wav -> log-mel cache + segment labels (data/public/proc/audio_md/)
  python -m tidal.audio_train cv           # 3-fold cross-validation by speaker pair (held-out pair = unseen speakers)
Evaluation events (all from the listener's = "self" point of view, both channel assignments):
  shift/hold : at the end of the other speaker's segment (+200 ms of mutual silence), does self take the next turn?
  backchannel: at each 100 ms tick while other talks and self is silent, does self start a backchannel within 0.5 s?
Baselines: class prior; logistic regression on hand-crafted prosodic/timing features (same events, trained on the
training folds). CIs: block bootstrap over conversations."""
import glob, json, os, re, sys, time, wave, numpy as np, torch, torch.nn.functional as Fn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score
from tidal.audio_fe import AudioEncoder, logmel, stack_frames, p_shift, BINS, STEP, NMEL, STACK, HOP, SR, FRAME_READY
from tidal.metrics import boot, ci
from tidal.public_data.manifest import DATA

SRC = DATA / "magicdata_ms"; CACHE = DATA / "proc" / "audio_md"
CACHE_VERSION = 2  # BOM handling and non-speech filtering
from tidal.turn_features import BC_CHARS, TAG, is_bc

def read_wav(p):
    with wave.open(str(p)) as w:
        assert w.getframerate() == SR and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0

def read_txt(p):
    seg = []
    for l in open(p, encoding="utf-8-sig"):
        m = re.match(r"\[([\d.]+),([\d.]+)\]\s+\S+\s+\S+\s*(.*)$", l.strip())
        if m:
            start, end, text = float(m.group(1)), float(m.group(2)), m.group(3)
            if end > start and TAG.sub("", text):
                seg.append((start, end, text))
    return seg


def shift_events(segments, self_ch, observed_until, bc_fn=is_bc, symmetric=False):
    """Exact timestamp candidates; exclude speech through decision, never after it."""
    other = segments[1-self_ch]; own = segments[self_ch]
    for segment in other:
        if bc_fn(segment):
            continue
        end = segment[1]; decision = end + .2
        if decision > observed_until:
            continue
        if any(s[0] <= decision and s[1] > end for s in own):
            continue
        if any(s[1] > end and s[0] <= decision for s in other):
            continue
        ns = min((s[0] for s in own if s[0] > decision and not bc_fn(s)), default=np.inf)
        no = min((s[0] for s in other if s[0] > decision and (not symmetric or not bc_fn(s))), default=np.inf)
        if min(ns,no) > min(end+5.,observed_until) or ns == no:
            continue
        yield decision, float(ns < no), segment[1]-segment[0]

def prep(src=SRC, cache=CACHE):
    cache.mkdir(parents=True, exist_ok=True)
    convs = sorted({os.path.basename(p).rsplit("_0_", 1)[0] for p in glob.glob(str(src / "TXT" / "*.txt"))})
    for c in convs:
        out = cache / f"{c}.npz"
        if out.exists():
            with np.load(out) as z:
                if "cache_version" in z and int(z["cache_version"]) == CACHE_VERSION: continue
        tx = sorted(glob.glob(str(src / "TXT" / f"{c}_0_*.txt")))
        if len(tx) != 2: continue
        if any(not (src / "WAV" / (os.path.basename(t)[:-4] + ".wav")).exists() for t in tx):
            print("pending audio", c, flush=True); continue
        mels, segs = [], []
        for t in tx:
            w = src / "WAV" / (os.path.basename(t)[:-4] + ".wav"); mels.append(logmel(read_wav(w))); segs.append(read_txt(t))
        n = min(len(m) for m in mels) // STACK * STACK
        np.savez(out, cache_version=CACHE_VERSION, m0=mels[0][:n].astype(np.float16), m1=mels[1][:n].astype(np.float16),
                 s0=json.dumps(segs[0], ensure_ascii=False), s1=json.dumps(segs[1], ensure_ascii=False))
        print("prep", c, n, len(segs[0]), len(segs[1]), flush=True)

def labels(segs, T):
    """Targets at frame availability time k*STEP + FRAME_READY, not frame start."""
    va = np.zeros(T, np.float32); bco = np.zeros(T, np.float32)
    for s in segs:
        a = max(0, int(np.ceil((s[0] - FRAME_READY) / STEP - 1e-10)))
        b = max(0, int(np.ceil((s[1] - FRAME_READY) / STEP - 1e-10)))
        va[a:min(b, T)] = 1
        if is_bc(s):
            a = max(0, int(np.ceil((s[0] - .5 - FRAME_READY) / STEP - 1e-10)))
            b = max(0, int(np.ceil((s[0] - FRAME_READY) / STEP - 1e-10)))
            bco[a:min(b, T)] = 1
    cs = np.r_[0, np.cumsum(va)]
    fut = np.zeros((T, len(BINS)), np.float32)
    for j, (lo, hi) in enumerate(BINS):
        a = np.arange(T) + int(round(lo / STEP)); b = np.arange(T) + int(round(hi / STEP))
        ok = b < T; fut[ok, j] = ((cs[b[ok]] - cs[a[ok]]) / (b[ok] - a[ok]) >= 0.5).astype(np.float32); fut[~ok, j] = np.nan
    bco[max(0, T - int(round(.5 / STEP))):] = np.nan
    return va, fut, bco

def load_conv(c):
    z = np.load((DATA / "proc" / "audio_oto" / f"{c[4:]}.npz") if c.startswith("oto:") else (CACHE / f"{c}.npz")); x = stack_frames(z["m0"].astype(np.float32), z["m1"].astype(np.float32)); T = len(x)
    S = [json.loads(str(z["s0"])), json.loads(str(z["s1"]))]
    L = [labels(S[0], T), labels(S[1], T)]
    return dict(x=x, S=S, L=L, T=T)

def persp(cv, self_ch):
    """features/targets from the point of view of channel self_ch (input order: other, self)."""
    o = 1 - self_ch
    x4 = cv["x"].reshape(cv["T"], STACK, 2, NMEL)          # stack_frames layout: per step [f0:(ch0, ch1), f1:(ch0, ch1)]
    X = x4[:, :, [o, self_ch], :].reshape(cv["T"], -1)     # -> [f0:(other, self), f1:(other, self)] == AudioStream input
    vs, fs, bs = cv["L"][self_ch]; vo, fo, _ = cv["L"][o]
    return X, np.stack([vs, vo], 1), np.concatenate([fs, fo], 1), bs

def events(cv, self_ch, P=None):
    """shift/hold events at other's segment ends and backchannel ticks; optionally attach model outputs P (dict of arrays per step)."""
    o = 1 - self_ch; So, Ss = cv["S"][o], cv["S"][self_ch]; vs, _, bco = cv["L"][self_ch]; vo = cv["L"][o][0]; T = cv["T"]
    ev = []
    observed_until = (T-1)*STEP + FRAME_READY
    for t, y, dur in shift_events(cv["S"], self_ch, observed_until):
        k = int(np.floor((t-FRAME_READY+1e-10)/STEP))
        if 0 <= k < T: ev.append(("shift", k, y, dur))
    bc_need = int(round(0.5 / STEP))                            # backchannel label looks 0.5 s ahead
    for k in range(0, T, 5):                                   # 100 ms ticks
        if vo[k] and not vs[k]:
            if k + bc_need >= T: continue                       # future window not observed -> not a negative
            ev.append(("bc", k, float(bco[k]), 0.0))
    return ev

_HF = {}
def hand_feats(cv, self_ch, k, segdur):
    """prosodic / timing features available at step k (causal). Per-(conv, channel) arrays are cached."""
    key = (id(cv), self_ch)
    if key not in _HF:
        X, _, _, _ = persp(cv, self_ch); x4 = X.reshape(len(X), STACK, 2, NMEL)
        vs = cv["L"][self_ch][0]; idx = np.where(vs > 0, np.arange(len(vs)), -1); last_self = np.maximum.accumulate(idx)
        _HF.clear(); _HF[key] = (x4[:, :, 0].mean((1, 2)), x4[:, :, 1].mean((1, 2)), x4[:, :, 0, NMEL // 2:].mean((1, 2)), last_self)
    eo, es, hi_o, last_self = _HF[key]
    def m(a, lo, hi): a0, b0 = max(0, k - hi), max(1, k - lo); return float(a[a0:b0].mean()) if b0 > a0 else 0.0
    since_self = (k - last_self[k]) * STEP if last_self[k] >= 0 else 60.0
    return [m(eo, 0, 10), m(eo, 10, 50), m(eo, 0, 10) - m(eo, 10, 50), m(hi_o, 0, 10) - m(hi_o, 10, 50), m(es, 0, 10),
            np.log1p(segdur), np.log1p(min(since_self, 60.0))]

def audio_loss(outputs, va, future, bc):
    """Unknown future targets contribute no loss; current activity stays supervised."""
    loss = Fn.binary_cross_entropy_with_logits(outputs["va"], va)
    mask = torch.isfinite(future)
    if mask.any():
        loss = loss + Fn.binary_cross_entropy_with_logits(outputs["vap"][mask], future[mask])
    mask = torch.isfinite(bc)
    if mask.any():
        loss = loss + .5 * Fn.binary_cross_entropy_with_logits(
            outputs["bc"][mask], bc[mask], pos_weight=outputs["bc"].new_tensor(5.))
    return loss

def train_fold(train_convs, val_convs, seed=0, epochs=40, crop=750, bs=16, log=print, init=None, lr=2e-3, patience=6):
    """init: dict(state, mu, sd) of a pretrained encoder -> fine-tune it (keeps its input normalization)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    def _p(c, ch):
        X, VA, F, B = persp(load_conv(c) if isinstance(c, str) else c, ch); return (c, ch, X.astype(np.float16), VA, F, B)
    data = [_p(c, ch) for c in train_convs for ch in (0, 1)]      # float16 features in RAM (long corpora)
    if init is None: Xall = np.concatenate([d[2][::4].astype(np.float32) for d in data]); mu = Xall.mean(0); sd = Xall.std(0) + 1e-4; del Xall
    else: mu, sd = init["mu"], init["sd"]
    vdata = [(c, ch) + persp(load_conv(c), ch) for c in val_convs for ch in (0, 1)]
    m = AudioEncoder()
    if init is not None: m.load_state_dict(init["state"])
    opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=1e-2); best = (1e9, None, -1)
    def loss_of(X, VA, F, B):
        o, _ = m(X)
        return audio_loss(o, VA, F, B)
    for ep in range(epochs):
        m.train(); t0 = time.time(); tl = []
        for it in range(60):
            Xb, Vb, Fb, Bb = [], [], [], []
            for _ in range(bs):
                d = data[rng.integers(len(data))]; T = len(d[2]); a = rng.integers(0, T - crop)
                Xb.append((d[2][a:a + crop].astype(np.float32) - mu) / sd); Vb.append(d[3][a:a + crop]); Fb.append(d[4][a:a + crop]); Bb.append(d[5][a:a + crop])
            l = loss_of(*(torch.from_numpy(np.stack(v).astype(np.float32)) for v in (Xb, Vb, Fb, Bb)))
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); tl.append(float(l))
        m.eval(); vl = []
        with torch.no_grad():
            for d in vdata: vl.append(float(loss_of(*(torch.from_numpy(((d[2] - mu) / sd if i == 0 else d[2 + i])[None].astype(np.float32)) for i in range(4)))))
        v = float(np.mean(vl)); log(json.dumps(dict(ep=ep, train=round(float(np.mean(tl)), 4), val=round(v, 4), sec=round(time.time() - t0, 1))))
        if v < best[0] - 1e-4: best = (v, {k: t.clone() for k, t in m.state_dict().items()}, ep)
        elif ep - best[2] >= patience: break
    m.load_state_dict(best[1]); return m, mu, sd, best

@torch.no_grad()
def run_model(m, mu, sd, cv, ch):
    X, _, _, _ = persp(cv, ch); o, _ = m(torch.from_numpy(((X - mu) / sd)[None].astype(np.float32)))
    return dict(pshift=p_shift(o["vap"][0]).numpy(), bc=torch.sigmoid(o["bc"][0]).numpy(), va=torch.sigmoid(o["va"][0]).numpy())

def cv(seed=0):
    convs = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(CACHE / "*.npz"))); pairs = sorted({c.split("_")[0] for c in convs})
    rows = []; logf = open("logs/p3/audio_cv.log", "a"); log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    for test_pair in pairs:
        tr = [c for c in convs if not c.startswith(test_pair)]; te = [c for c in convs if c.startswith(test_pair)]
        val = tr[::5]; trn = [c for c in tr if c not in val]
        log(f"fold test={test_pair} train={len(trn)} val={len(val)} test={len(te)}")
        fp = f"models/audio_fe_fold_{test_pair}.pt"
        if os.path.exists(fp):                                    # resume: fold already trained
            z = torch.load(fp, weights_only=False)
            if z.get("target_version") != 2:
                raise ValueError(f"Stale audio targets in {fp}; move old checkpoints aside and retrain")
            m = AudioEncoder(); m.load_state_dict(z["state"]); m.eval(); mu, sd = z["mu"], z["sd"]
        else:
            m, mu, sd, best = train_fold(trn, val, seed, log=log)
            os.makedirs("models", exist_ok=True); torch.save(dict(state=m.state_dict(), mu=mu, sd=sd, test_pair=test_pair, best=best[0], target_version=2), fp)
        # hand-feature LR baseline trained on the training convs of this fold
        def feats_events(cl, with_model):
            out = []
            for c in cl:
                cvd = load_conv(c); _HF.clear()
                for ch in (0, 1):
                    P = run_model(m, mu, sd, cvd, ch) if with_model else None
                    for kind, k, y, sdur in events(cvd, ch):
                        r = dict(conv=c, ch=ch, kind=kind, k=k, y=y, f=hand_feats(cvd, ch, k, sdur))
                        if P is not None: r.update(pshift=float(P["pshift"][k]), pbc=float(P["bc"][k]))
                        out.append(r)
            return out
        trE = feats_events(trn, False); teE = feats_events(te, True)
        for kind in ("shift", "bc"):
            a = [r for r in trE if r["kind"] == kind]; lr = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
            lr.fit(np.array([r["f"] for r in a]), np.array([r["y"] for r in a]))
            prior = float(np.mean([r["y"] for r in a]))
            for r in teE:
                if r["kind"] == kind: r["lr"] = float(lr.predict_proba(np.array([r["f"]]))[0, 1]); r["prior"] = prior
        rows += [dict(r, fold=test_pair) for r in teE]
    json.dump(rows, open(DATA / "proc" / "audio_cv_events.json", "w"))
    return summarize(rows)

def summarize(rows, n_boot=1000):
    """AUCs are fold-stratified (n-weighted mean of per-fold AUCs): each fold has its own model, so pooling scores
    across folds would mix calibrations. The prior is a constant (AUC 0.5 by definition) and is not bootstrapped."""
    from tidal.metrics import strat_auc
    res = {}
    for kind, score in (("shift", "pshift"), ("bc", "pbc")):
        R = [r for r in rows if r["kind"] == kind]; y = np.array([r["y"] for r in R]); blocks = np.array([r["conv"] for r in R])
        fold = np.array([r["fold"] for r in R]); P = {"model": np.array([r[score] for r in R]), "lr_prosody": np.array([r["lr"] for r in R])}
        out = dict(n=len(R), pos_rate=float(y.mean()), convs=int(len(set(blocks))), per_fold_pos={f: int(y[fold == f].sum()) for f in np.unique(fold)})
        for k, p in P.items(): out[k] = dict(pr_auc=strat_auc(y, p, fold, "pr"), roc_auc=strat_auc(y, p, fold, "roc"))
        out["prior"] = dict(pr_auc=float(y.mean()), roc_auc=0.5)
        d = boot(lambda idx: (strat_auc(y[idx], P["model"][idx], fold[idx], "pr") - strat_auc(y[idx], P["lr_prosody"][idx], fold[idx], "pr"),
                              strat_auc(y[idx], P["model"][idx], fold[idx]) - strat_auc(y[idx], P["lr_prosody"][idx], fold[idx])), blocks, n_boot, 0)
        out["model_minus_lr_prosody"] = dict(d_pr_auc=float(np.nanmean(d[:, 0])), d_pr_auc_ci=ci(d[:, 0]), d_roc_auc=float(np.nanmean(d[:, 1])), d_roc_auc_ci=ci(d[:, 1]))
        d = boot(lambda idx: strat_auc(y[idx], P["model"][idx], fold[idx]) - 0.5, blocks, n_boot, 0)
        out["model_minus_chance_roc_auc"] = dict(mean=float(np.nanmean(d)), ci=ci(np.asarray(d).ravel()))
        res[kind] = out
    res["note"] = ("3-fold CV by speaker pair (test speakers unseen); MagicData multi-stream zh (research-only); fold-stratified AUCs; "
                   "block bootstrap over conversations")
    json.dump(res, open("reports/audio_phase3.json", "w"), indent=1); print(json.dumps(res, indent=1)); return res

if __name__ == "__main__":
    if sys.argv[1] == "summarize": summarize(json.load(open(DATA / "proc" / "audio_cv_events.json")))
    else: {"prep": prep, "cv": cv}[sys.argv[1]]()
