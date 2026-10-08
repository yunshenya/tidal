"""Phase 4A: text emotion / affect head on frozen bge-small-zh-v1.5 sentence embeddings (zh + en). No keyword lists.

Shared label set (8): neutral, joy, sadness, anger, fear, surprise, disgust, playful (playful only where the source
annotates it: GoEmotions 'amusement'). Multi-label sigmoid head + valence/arousal regression.
Valence/arousal targets are a documented PROXY: each category has a Russell-circumplex anchor (EMO_VA) and a text's
target is the mean anchor of its positive labels (neutral = 0,0). Arousal is then checked against HUMAN intensity
ratings (BRIGHTER track B), which the head never trains on.

Public data only (licenses verified, see reports/phase4_datasets.md); nothing is redistributed.
usage: python -m tidal.emotion embed | train | eval"""
import json, os, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
EMO = ["neutral", "joy", "sadness", "anger", "fear", "surprise", "disgust", "playful"]
EMO_VA = {"neutral": (0.0, 0.0), "joy": (0.8, 0.5), "sadness": (-0.7, -0.4), "anger": (-0.6, 0.7), "fear": (-0.6, 0.6),
          "surprise": (0.1, 0.8), "disgust": (-0.6, 0.3), "playful": (0.6, 0.6)}
VA = np.array([EMO_VA[e] for e in EMO], np.float32)
# GoEmotions 27+neutral -> 8 (Ekman grouping of Demszky et al. 2020, amusement kept apart as 'playful')
GO_NAMES = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity", "desire", "disappointment",
            "disapproval", "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"]
GO_MAP = {**{k: "anger" for k in ["anger", "annoyance", "disapproval"]}, "disgust": "disgust", "fear": "fear", "nervousness": "fear",
          **{k: "joy" for k in ["joy", "admiration", "approval", "caring", "desire", "excitement", "gratitude", "love", "optimism", "pride", "relief"]},
          "amusement": "playful", **{k: "sadness" for k in ["sadness", "disappointment", "embarrassment", "grief", "remorse"]},
          **{k: "surprise" for k in ["surprise", "realization", "confusion", "curiosity"]}, "neutral": "neutral"}
ZH_MED_MAP = {"平静": "neutral", "开心": "joy", "关心": "joy", "生气": "anger", "惊讶": "surprise", "伤心": "sadness", "厌恶": "disgust", "疑问": "surprise"}
P = "data/public"; OUT = "data/public/proc"; EMB_CACHE = "data/emb/emo_public_cache.npz"

def _row(labels, known=None):
    y = np.zeros(8, np.float32)
    for l in labels: y[EMO.index(l)] = 1
    if known is not None: y[[i for i, e in enumerate(EMO) if e not in known]] = np.nan
    return y

def build():
    rows = []
    for sp in ["train", "validation", "test"]:
        d = pd.read_parquet(f"{P}/go_emotions/simplified/{sp}-00000-of-00001.parquet")
        for t, ls in zip(d.text, d.labels):
            rows.append(("go_en", "en", {"validation": "dev"}.get(sp, sp), t, _row({GO_MAP[GO_NAMES[i]] for i in ls})))
    d = pd.read_parquet(f"{P}/go_emotions_ml/train.parquet"); d = d[d.lang == "Chinese (Simplified)"]
    for t, ls, si in zip(d.text, d.labels, d.source_index):          # machine-translated GoEmotions *train* -> zh train (+10% dev)
        rows.append(("go_zh_mt", "zh", "dev" if si % 10 == 0 else "train", t, _row({GO_MAP[l] for l in ls})))
    for lang, cfg in [("zh", "chn"), ("en", "eng")]:
        for sp in ["train", "dev", "test"]:
            d = pd.read_parquet(f"{P}/brighter_cat/{cfg}/{sp}-00000-of-00001.parquet")
            for _, r in d.iterrows():
                y = np.full(8, np.nan, np.float32); em = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
                for e in em: y[EMO.index(e)] = r[e]
                y[0] = float(np.nansum(y[1:7]) == 0) if not np.isnan(y[[EMO.index(e) for e in em if e != "disgust"]]).any() else np.nan
                rows.append((f"brighter_{lang}", lang, sp, r["text"], y))
    d = pd.read_csv(f"{P}/zh_med_s/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv").drop_duplicates("text")
    rng = np.random.default_rng(0); u = rng.random(len(d))
    for (t, l), x in zip(zip(d.text, d.label), u):
        rows.append(("zh_med", "zh", "train" if x < 0.8 else ("dev" if x < 0.9 else "test"), t, _row({ZH_MED_MAP[l]}, known=set(EMO) - {"playful"})))
    df = pd.DataFrame(rows, columns=["src", "lang", "split", "text", "y"]); df = df[df.text.astype(str).str.strip().str.len() > 0].reset_index(drop=True)
    return df

def brighter_intensity():
    out = []
    for lang, cfg in [("zh", "chn"), ("en", "eng")]:
        d = pd.read_parquet(f"{P}/brighter_int/{cfg}/test-00000-of-00001.parquet")
        em = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
        I = d[em].to_numpy(float)
        out.append(pd.DataFrame(dict(lang=lang, text=d.text, arousal_h=np.nanmax(I, 1),
                                     valence_h=I[:, 3] - np.nanmax(I[:, [0, 1, 2, 4]], 1))))
    return pd.concat(out, ignore_index=True)

def embed_all(texts, threads=2):
    from tidal.embed import Encoder, key
    k2i, E = {}, np.zeros((0, 512), np.float16)
    if os.path.exists(EMB_CACHE):
        z = np.load(EMB_CACHE); k2i = dict(zip(z["keys"].tolist(), range(len(z["keys"])))); E = z["emb"]
    todo = sorted({t for t in texts if key(t) not in k2i})
    if todo:
        t0 = time.time(); e = Encoder(threads).encode(todo); print(f"embedded {len(todo)} in {time.time() - t0:.0f}s", flush=True)
        keys = list(k2i) + [key(t) for t in todo]; E = np.concatenate([E, e.astype(np.float16)]); k2i = dict(zip(keys, range(len(keys))))
        os.makedirs("data/emb", exist_ok=True); np.savez(EMB_CACHE, keys=np.array(keys), emb=E)
    return E[[k2i[key(t)] for t in texts]].astype(np.float32)

class EmoHead(nn.Module):
    def __init__(self, d_in=512, hidden=256, dropout=0.2):
        super().__init__(); self.net = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.cls = nn.Linear(hidden, 8); self.va = nn.Linear(hidden, 2); self.register_buffer("T", torch.ones(()))
    def forward(self, e):
        h = self.net(e); return self.cls(h), torch.tanh(self.va(h))
    @torch.no_grad()
    def features(self, e):
        """runtime features: calibrated per-class probabilities (8) + valence + arousal."""
        self.eval(); lo, va = self(torch.as_tensor(e, dtype=torch.float32)); return torch.cat([torch.sigmoid(lo / self.T), va], -1).numpy()

def va_target(y):
    pos = np.nan_to_num(y) > 0.5; out = np.zeros((len(y), 2), np.float32)
    for i, p in enumerate(pos):
        if p.any(): out[i] = VA[p].mean(0)
    return out

def train(df, E, seed=0, epochs=30, log=print, train_mask=None):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tr = np.flatnonzero((df.split == "train").to_numpy() if train_mask is None else train_mask)
    dv = np.flatnonzero((df.split == "dev").to_numpy())
    Y = np.stack(df.y.to_numpy()); V = va_target(Y)
    # balance: each (source) contributes equally in expectation; human-labelled BRIGHTER upweighted x2
    src = df.src.to_numpy(); w = np.ones(len(df), np.float32)
    for s in np.unique(src[tr]): w[src == s] = len(tr) / (len(np.unique(src[tr])) * (src[tr] == s).sum())
    w[np.char.startswith(src.astype(str), "brighter")] *= 2
    m = EmoHead(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Et, Yt, Vt, Wt = (torch.from_numpy(a) for a in (E, Y, V, w)); best = (1e9, None, -1)
    def loss_on(ix, train=True):
        lo, va = m(Et[ix]); y = Yt[ix]; mk = ~torch.isnan(y)
        lc = (F.binary_cross_entropy_with_logits(lo[mk], y[mk], reduction="none") * Wt[ix].unsqueeze(1).expand_as(y)[mk]).sum() / Wt[ix].sum()
        lv = ((va - Vt[ix]) ** 2).sum(1).mul(Wt[ix]).sum() / Wt[ix].sum(); return lc + lv
    for ep in range(epochs):
        m.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), 256):
            l = loss_on(perm[b:b + 256]); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = float(loss_on(dv))
        if vl < best[0] - 1e-4: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, ep)
        elif ep - best[2] >= 4: break
    m.load_state_dict(best[1]); log(f"emo head seed={seed} best dev loss {best[0]:.4f} ep {best[2]}")
    # temperature scaling on dev (pooled binary NLL over known labels)
    with torch.no_grad(): lo, _ = m(Et[dv])
    y = Yt[dv]; mk = ~torch.isnan(y); Ts = np.linspace(0.5, 3, 51)
    nll = [float(F.binary_cross_entropy_with_logits(lo[mk] / T, y[mk])) for T in Ts]; m.T.fill_(float(Ts[int(np.argmin(nll))]))
    return m

def ece(p, y, bins=15):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1); e = 0.0
    for b in range(bins):
        s = idx == b
        if s.any(): e += s.mean() * abs(p[s].mean() - y[s].mean())
    return float(e)

def evaluate(m, df, E, boot=1000):
    from scipy.stats import spearmanr
    from sklearn.metrics import f1_score
    m.eval(); Y = np.stack(df.y.to_numpy())
    with torch.no_grad(): lo, va = m(torch.from_numpy(E))
    P_raw = torch.sigmoid(lo).numpy(); P = torch.sigmoid(lo / m.T).numpy(); va = va.numpy(); res = {"temperature": float(m.T)}
    sets = {"brighter_zh": ("brighter_zh", "test"), "brighter_en": ("brighter_en", "test"), "zh_med": ("zh_med", "test"),
            "go_en": ("go_en", "test"), "go_zh_mt": ("go_zh_mt", "dev")}
    for name, (s, sp) in sets.items():
        te = np.flatnonzero(((df.src == s) & (df.split == sp)).to_numpy())
        dvs = np.flatnonzero(((df.src == s) & (df.split == "dev")).to_numpy()) if sp == "test" else te
        labs = [j for j in range(8) if not np.isnan(Y[te, j]).all() and np.nansum(Y[te, j]) > 0]
        f1s = {}; thr = {}
        for j in labs:                                             # per-label threshold tuned on the source's dev split
            yd, pd_ = Y[dvs, j], P[dvs, j]; ok = ~np.isnan(yd); cand = np.linspace(0.05, 0.95, 37)
            thr[j] = float(cand[int(np.argmax([f1_score(yd[ok], pd_[ok] >= c, zero_division=0) for c in cand]))])
            yt = Y[te, j]; ok = ~np.isnan(yt); f1s[EMO[j]] = float(f1_score(yt[ok], P[te, j][ok] >= thr[j], zero_division=0))
        single = [i for i in te if np.nansum(Y[i]) == 1]
        acc = float(np.mean([labs[int(np.argmax(P[i, labs]))] == int(np.nanargmax(Y[i])) for i in single])) if single else None
        # bootstrap CI of macro-F1 over test items (thresholds fixed)
        rng = np.random.default_rng(0); bs = []
        for _ in range(boot if len(te) < 20000 else 200):
            ix = rng.choice(len(te), len(te)); vals = []
            for j in labs:
                yt = Y[te[ix], j]; ok = ~np.isnan(yt); vals.append(f1_score(yt[ok], P[te[ix], j][ok] >= thr[j], zero_division=0))
            bs.append(np.mean(vals))
        kn = ~np.isnan(Y[te][:, labs]); yk = Y[te][:, labs][kn]
        res[name] = dict(n=len(te), labels=[EMO[j] for j in labs], macro_f1=float(np.mean(list(f1s.values()))),
                         macro_f1_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], f1=f1s,
                         top1_acc_single_label=acc, n_single=len(single),
                         ece_raw=ece(P_raw[te][:, labs][kn], yk), ece_temp=ece(P[te][:, labs][kn], yk),
                         prior_macro_f1=float(np.mean([2 * np.nanmean(Y[te, j]) / (1 + np.nanmean(Y[te, j])) for j in labs])))
    bi = brighter_intensity(); Eb = embed_all(bi.text.tolist())
    with torch.no_grad(): _, vb = m(torch.from_numpy(Eb))
    vb = vb.numpy()
    for lang in ["zh", "en"]:
        s = (bi.lang == lang).to_numpy(); ok = s & ~np.isnan(bi.valence_h.to_numpy())
        res[f"brighter_int_{lang}"] = dict(n=int(s.sum()), spearman_arousal=float(spearmanr(vb[s, 1], bi.arousal_h[s]).correlation),
                                           spearman_valence=float(spearmanr(vb[ok, 0], bi.valence_h[ok]).correlation))
    return res

def latency(m, n=500):
    from tidal.embed import Encoder
    torch.set_num_threads(1); enc = Encoder(1); texts = ["今天好开心啊", "你怎么又迟到了", "what are you doing lol", "唉，算了吧"] * (n // 4)
    enc.encode(texts[:4], bs=1); le, lh = [], []
    for t in texts:
        t0 = time.perf_counter(); e = enc.encode([t], bs=1); t1 = time.perf_counter(); m.features(e); t2 = time.perf_counter()
        le.append((t1 - t0) * 1e3); lh.append((t2 - t1) * 1e3)
    q = lambda a, p: round(float(np.percentile(a, p)), 3)
    return dict(bge_p50_ms=q(le, 50), bge_p95_ms=q(le, 95), head_p50_ms=q(lh, 50), head_p95_ms=q(lh, 95), head_params=sum(p.numel() for p in m.parameters()))

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 2))); cmd = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if cmd == "embed":
        df = build(); df.to_pickle(f"{OUT}/emo_text.pkl"); print(df.groupby(["src", "split"]).size())
        embed_all(df.text.tolist()); embed_all(brighter_intensity().text.tolist())
    elif cmd == "train":
        df = pd.read_pickle(f"{OUT}/emo_text.pkl"); E = embed_all(df.text.tolist()); res = {}
        for seed in range(3):
            m = train(df, E, seed); torch.save(dict(state=m.state_dict(), emo=EMO, va=EMO_VA), f"models/emo_text_s{seed}.pt")
            res[f"s{seed}"] = evaluate(m, df, E)
        # ablation: human-labelled BRIGHTER train only (does the extra GoEmotions / zh dialogue data help?)
        m = train(df, E, 0, train_mask=((df.split == "train") & df.src.str.startswith("brighter")).to_numpy()); res["brighter_only_s0"] = evaluate(m, df, E)
        res["latency"] = latency(torch.load and EmoHead().eval() if False else m)
        json.dump(res, open("reports/emotion_text_phase4.json", "w"), indent=1); print(json.dumps({k: (v if k == "latency" else {n: v[n]["macro_f1"] for n in v if isinstance(v[n], dict) and "macro_f1" in v[n]}) for k, v in res.items()}, indent=1))
