"""Phase 4A (speech): emotion head on top of the streaming audio front end (tidal/audio_fe.py).

Data (public only):
  * CREMA-D (ODbL; 91 actors, 6 acted emotions, human labels + intended intensity LO/MD/HI) -> supervised, speaker-held-out
  * MagicData conversational zh segments, labelled offline by the SenseVoice-Small emotion tagger (FunASR Model License
    v1.1, used as an offline teacher via sherpa-onnx; attribution in docs) -> distillation targets
Head: input = frozen AudioEncoder hidden state (96) + its normalised mel stack (160) of the "other" channel, per 20 ms
step -> Linear(64) -> GRU(64) -> 8 emotion logits (shared label set) + valence/arousal. Streaming O(1)/step.
usage: python -m tidal.emotion_speech teacher | train"""
import glob, json, os, random, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from tidal.audio_fe import logmel, stack_frames, AudioEncoder, SR, STEP
from tidal.emotion import EMO, VA, va_target
D = "data/public"; OUT = f"{D}/proc"; CREMA = f"{D}/crema_d/data/AudioWAV"; MD = f"{D}/magicdata_ms"
CREMA_MAP = {"ANG": "anger", "DIS": "disgust", "FEA": "fear", "HAP": "joy", "NEU": "neutral", "SAD": "sadness"}
SV_MAP = {"HAPPY": "joy", "SAD": "sadness", "ANGRY": "anger", "NEUTRAL": "neutral", "FEARFUL": "fear", "DISGUSTED": "disgust", "SURPRISED": "surprise"}

def read_wav(p):
    import wave
    with wave.open(p) as w:
        sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels()).mean(1)
    if sr != SR: x = np.interp(np.arange(0, len(x) * SR / sr) * sr / SR, np.arange(len(x)), x).astype(np.float32)
    return x

def crema_items():
    out = []
    for p in sorted(glob.glob(f"{CREMA}/*.wav")):
        a, sent, emo, inten = os.path.basename(p)[:-4].split("_")
        out.append(dict(id=os.path.basename(p)[:-4], src="crema", path=p, actor=int(a), emotion=CREMA_MAP[emo], intensity=inten, start=0.0, end=None))
    return out

def md_items(n=3000, seed=0):
    from tidal.audio_train import read_txt
    rng = random.Random(seed); segs = []
    for t in sorted(glob.glob(f"{MD}/TXT/*.txt")):
        conv = os.path.basename(t).rsplit("_0_", 1)[0]
        for a, b, txt in read_txt(t):
            if 1.5 <= b - a <= 8.0: segs.append(dict(id=f"{os.path.basename(t)[:-4]}@{a:.2f}", src="md", path=f"{MD}/WAV/{os.path.basename(t)[:-4]}.wav", conv=conv, start=a, end=b))
    rng.shuffle(segs); return segs[:n]

def teacher():
    import sherpa_onnx
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(model="models/sensevoice/model.int8.onnx", tokens="models/sensevoice/tokens.txt", num_threads=1, use_itn=False)
    items = crema_items() + md_items(); out_p = f"{OUT}/emo_teacher.parquet"; done = {}
    if os.path.exists(out_p): done = {r["id"]: r for r in pd.read_parquet(out_p).to_dict("records")}
    cache = {}; rows = list(done.values()); t0 = time.time(); audio_s = 0.0
    for i, it in enumerate(items):
        if it["id"] in done: continue
        if it["path"] not in cache: cache = {it["path"]: read_wav(it["path"])}
        x = cache[it["path"]]; x = x[int(it["start"] * SR): int(it["end"] * SR) if it["end"] else None]
        s = rec.create_stream(); s.accept_waveform(SR, x); rec.decode_stream(s); r = s.result
        rows.append({**{k: v for k, v in it.items() if k != "path"}, "path": it["path"], "sv_emotion": r.emotion.strip("<|>"), "sv_event": r.event.strip("<|>"), "sv_lang": r.lang.strip("<|>")})
        audio_s += len(x) / SR
        if i % 500 == 0:
            pd.DataFrame(rows).to_parquet(out_p); print(f"{i}/{len(items)} rtf={(time.time() - t0) / max(audio_s, 1e-6):.3f}", flush=True)
    pd.DataFrame(rows).to_parquet(out_p); print("teacher done", len(rows), flush=True)

SIL = None
def _feats(enc, mu, sd, x):
    """mono PCM (speech on the 'other' channel, silent 'self') -> normalised stack [T,160], frozen encoder hidden [T,96]."""
    global SIL
    mo = logmel(x); ms = np.full_like(mo, np.log(1e-6))
    z = ((stack_frames(mo, ms) - mu) / sd).astype(np.float32)
    with torch.no_grad(): h = enc(torch.from_numpy(z[None]))[0]["h"][0].numpy()
    return z, h

def prep():
    ck = torch.load("models/audio_fe_oto.pt", weights_only=False); enc = AudioEncoder(); enc.load_state_dict(ck["state"]); enc.eval()
    mu, sd = ck["mu"], ck["sd"]; T = pd.read_parquet(f"{OUT}/emo_teacher.parquet"); Z, H, keep = [], [], []
    cache = {}
    for i, r in enumerate(T.itertuples()):
        if r.path not in cache: cache = {r.path: read_wav(r.path)}
        x = cache[r.path][int(r.start * SR): int(r.end * SR) if r.end == r.end and r.end is not None else None]
        z, h = _feats(enc, mu, sd, x)
        if len(z) < 10: continue
        Z.append(z.astype(np.float16)); H.append(h.astype(np.float16)); keep.append(i)
    T = T.iloc[keep].reset_index(drop=True); T.to_parquet(f"{OUT}/emo_speech_items.parquet")
    np.savez(f"{OUT}/emo_speech_feats.npz", lens=np.array([len(z) for z in Z]), Z=np.concatenate(Z), H=np.concatenate(H))
    print("prep", len(T), flush=True)

class SpeechEmoHead(nn.Module):
    def __init__(self, d_in=96 + 160, d=64):
        super().__init__(); self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU()); self.gru = nn.GRU(d, d, batch_first=True)
        self.cls = nn.Linear(d, 8); self.va = nn.Linear(d, 2); self.register_buffer("T", torch.ones(()))
    def forward(self, x, state=None):
        h, st = self.gru(self.inp(x), state); return self.cls(h), torch.tanh(self.va(h)), st

def _split(T):
    sp = np.where(T.src == "crema", np.where(T.actor % 5 == 0, "test", np.where(T.actor % 5 == 1, "dev", "train")), "train")
    md = (T.src == "md").to_numpy(); h = T.conv.fillna("").map(lambda c: int(c[-2:], 16) % 5 if c else 0).to_numpy() if "conv" in T else 0
    sp = np.where(md & (h == 0), "test", np.where(md & (h == 1), "dev", sp)); return sp

def _targets(T):
    """hard labels for CREMA-D (human); SOFT teacher posteriors (SenseVoice emotion logits, tidal/sv_soft.py) for MagicData."""
    S = pd.read_parquet(f"{OUT}/emo_soft.parquet").set_index("id").loc[T.id]
    y = np.full(len(T), -1); allowed = np.zeros((len(T), 8), bool); soft = np.zeros((len(T), 8), np.float32)
    for k, e in enumerate(["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]): soft[:, EMO.index(SV_MAP[e])] = S[f"sv_{e}"].to_numpy()
    for i, r in enumerate(T.itertuples()):
        if r.src == "crema": y[i] = EMO.index(r.emotion); allowed[i, [EMO.index(e) for e in CREMA_MAP.values()]] = True
        else: y[i] = int(soft[i].argmax()); allowed[i, [EMO.index(e) for e in SV_MAP.values()]] = True
    return y, allowed, soft

def train_eval(seed=0, use_teacher=True, epochs=40, log=print):
    from sklearn.metrics import f1_score, cohen_kappa_score
    from scipy.stats import spearmanr
    from tidal.emotion import ece
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    T = pd.read_parquet(f"{OUT}/emo_speech_items.parquet"); z = np.load(f"{OUT}/emo_speech_feats.npz")
    off = np.r_[0, np.cumsum(z["lens"])]; X = np.concatenate([z["H"], z["Z"]], 1).astype(np.float32)
    seqs = [X[a:b] for a, b in zip(off[:-1], off[1:])]; sp = _split(T); y, allowed, soft = _targets(T)
    is_md = (T.src == "md").to_numpy(); tgt = np.eye(8, dtype=np.float32)[np.maximum(y, 0)]; tgt[is_md] = soft[is_md]
    w = np.where(T.src == "crema", 1.0, 0.5 if use_teacher else 0.0)
    tr = np.flatnonzero((sp == "train") & (y >= 0) & (w > 0)); dv = np.flatnonzero((sp == "dev") & (T.src == "crema").to_numpy())
    m = SpeechEmoHead(); opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-2); best = (1e9, None, -1)
    vat = torch.from_numpy(VA)
    def run(ix):
        L = max(len(seqs[i]) for i in ix); xb = np.zeros((len(ix), L, X.shape[1]), np.float32); mk = np.zeros((len(ix), L), bool)
        for j, i in enumerate(ix): n = len(seqs[i]); xb[j, :n] = seqs[i]; mk[j, 15:n] = True        # loss from 0.3 s on
        lo, va, _ = m(torch.from_numpy(xb)); al = torch.from_numpy(allowed[ix])[:, None].expand_as(lo)
        lo = lo.masked_fill(~al, -1e4); return lo, va, torch.from_numpy(mk)
    def loss(ix):
        lo, va, mk = run(ix); q = torch.from_numpy(tgt[ix])[:, None].expand(*mk.shape, 8); ww = torch.from_numpy(w[ix]).float()[:, None].expand(mk.shape)
        lc = (-(q[mk] * F.log_softmax(lo[mk], -1)).sum(-1) * ww[mk]).sum() / ww[mk].sum()      # soft CE (== CE for one-hot)
        vt = q @ vat                                                                             # expected VA anchor
        lv = (((va - vt) ** 2).sum(-1)[mk] * ww[mk]).sum() / ww[mk].sum(); return lc + lv
    for ep in range(epochs):
        m.train(); perm = rng.permutation(tr)
        for b in range(0, len(perm), 64):
            l = loss(perm[b:b + 64]); opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval()
        with torch.no_grad(): vl = float(loss(dv))
        if vl < best[0] - 1e-4: best = (vl, {k: v.clone() for k, v in m.state_dict().items()}, ep)
        elif ep - best[2] >= 5: break
    m.load_state_dict(best[1]); m.eval()
    def clip_probs(ix, T_=1.0):
        P = []; A = []
        with torch.no_grad():
            for b in range(0, len(ix), 128):
                lo, va, mk = run(ix[b:b + 128])
                for j in range(len(lo)):
                    n = int(mk[j].sum()) + 15; P.append(torch.softmax(lo[j, 15:n] / T_, -1).mean(0).numpy()); A.append(va[j, 15:n].mean(0).numpy())
        return np.array(P), np.array(A)
    Pd, _ = clip_probs(dv); Ts = np.linspace(0.5, 3, 26)
    nll = [-np.mean(np.log(clip_probs(dv, t_)[0][np.arange(len(dv)), y[dv]] + 1e-9)) for t_ in Ts[::5]]; Tbest = float(Ts[::5][int(np.argmin(nll))])
    res = dict(seed=seed, use_teacher=use_teacher, best_ep=best[2], temperature=Tbest, params=sum(p.numel() for p in m.parameters()))
    te = np.flatnonzero((sp == "test") & (T.src == "crema").to_numpy()); P, A = clip_probs(te, Tbest); yt = y[te]; pr = P.argmax(1)
    six = [EMO.index(e) for e in CREMA_MAP.values()]
    rng2 = np.random.default_rng(0); actors = T.actor.to_numpy()[te]; ua = np.unique(actors); bs = []
    for _ in range(1000):                                    # speaker-cluster bootstrap
        ix = np.concatenate([np.flatnonzero(actors == a) for a in rng2.choice(ua, len(ua))]); bs.append(f1_score(yt[ix], pr[ix], labels=six, average="macro"))
    inten = T.intensity.to_numpy()[te]; lv = {"LO": 1, "MD": 2, "HI": 3}; ok = np.isin(inten, list(lv)) & (yt != EMO.index("neutral"))
    res["crema_test"] = dict(n=len(te), n_actors=len(ua), acc=float((pr == yt).mean()), macro_f1=float(f1_score(yt, pr, labels=six, average="macro")),
                             macro_f1_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], ece=ece(P.max(1), (pr == yt).astype(float)),
                             spearman_arousal_vs_intensity=float(spearmanr(A[ok, 1], [lv[i] for i in inten[ok]]).correlation), n_intensity=int(ok.sum()),
                             mean_arousal_by_class={EMO[c]: float(A[yt == c, 1].mean()) for c in six}, mean_valence_by_class={EMO[c]: float(A[yt == c, 0].mean()) for c in six})
    sv = T.sv_emotion.to_numpy()[te]; svp = np.array([EMO.index(SV_MAP[s]) if s in SV_MAP else -1 for s in sv])
    res["crema_test"]["sensevoice_hard_tag_acc"] = float((svp == yt).mean()); res["crema_test"]["sensevoice_hard_unknown_rate"] = float((sv == "EMO_UNKNOWN").mean())
    ss = soft[te][:, six]; svs = np.array(six)[ss.argmax(1)]                       # teacher soft posterior restricted to the 6 CREMA classes
    res["crema_test"]["sensevoice_soft_acc"] = float((svs == yt).mean()); res["crema_test"]["sensevoice_soft_macro_f1"] = float(f1_score(yt, svs, labels=six, average="macro"))
    res["crema_test"]["student_teacher_agreement"] = float((pr == svs).mean())
    tm = np.flatnonzero((sp == "test") & (T.src == "md").to_numpy())
    if len(tm):
        Pm, Am = clip_probs(tm, Tbest); pm = Pm.argmax(1); q = soft[tm]
        res["md_heldout_vs_teacher"] = dict(n=len(tm), argmax_agreement=float((pm == y[tm]).mean()), kappa=float(cohen_kappa_score(y[tm], pm)),
                                            mean_kl_teacher_student=float(np.mean((q * (np.log(q + 1e-9) - np.log(Pm + 1e-9))).sum(1))),
                                            teacher_argmax_dist={EMO[c]: int((y[tm] == c).sum()) for c in np.unique(y[tm])},
                                            spearman_arousal_vs_teacher_expected=float(spearmanr(Am[:, 1], q @ VA[:, 1]).correlation))
    return m, res

def latency(m, n=300):
    torch.set_num_threads(1); x = torch.randn(1, 5, 256); st = None; lat = []
    with torch.no_grad():
        for i in range(n):
            t0 = time.perf_counter(); _, _, st = m(x, st); lat.append((time.perf_counter() - t0) * 1e3)
    return dict(per_100ms_tick_p50_ms=round(float(np.percentile(lat, 50)), 3), p95_ms=round(float(np.percentile(lat, 95)), 3))

if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 1)))
    if sys.argv[1] == "teacher": teacher()
    elif sys.argv[1] == "prep": prep()
    elif sys.argv[1] == "train":
        res = {}
        for ut in [False, True]:
            for seed in range(3):
                m, r = train_eval(seed, ut); res[f"{'teacher' if ut else 'crema_only'}_s{seed}"] = r; print(json.dumps(r["crema_test"])[:300], flush=True)
                if ut: torch.save(dict(state=m.state_dict(), T=r["temperature"]), f"models/emo_speech_s{seed}.pt")
        res["latency"] = latency(m); json.dump(res, open("reports/emotion_speech_phase4.json", "w"), indent=1)
