"""Audio front end on channel-separated full-duplex English speech (otoSpeech-full-duplex-processed-141h, CC BY 4.0,
gated; local subset only) + extra evaluations.
  prep      flac 44.1 kHz stereo -> 16 kHz -> per-channel log-mel; voice activity from an energy detector (no transcripts
            in this corpus): adaptive per-channel threshold + hysteresis + gap filling. Acoustic backchannel PROXY: a
            short (<=1 s) segment that starts and ends while the other channel is speaking.
  pretrain  train the streaming AudioEncoder on oto train sessions -> models/audio_fe_oto.pt (local, never published)
  eval      (1) oto held-out sessions (energy-VAD labels): shift/hold + backchannel-proxy vs LR-prosody / prior
            (2) MagicData zh (human segments), same 3 speaker-pair folds as audio_train.cv: zero-shot oto model and
                oto-pretrained -> fine-tuned per fold, vs MagicData-only fold models (paired, same events)
            (3) Krisp turn-taking test (EVAL ONLY, license forbids training): mono clips ending in silence, label
                shift/hold; zero-shot score = P(self takes the floor) at the clip end; reference score = last-silence
                duration (no fitting). Nothing is fitted or tuned on Krisp.
-> reports/audio_oto_phase3.json"""
import os, io, sys, glob, json, tarfile, time, hashlib, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tidal.audio_fe import AudioEncoder, logmel, stack_frames, p_shift, STEP, NMEL, STACK, SR
from tidal import audio_train as AT
from tidal.metrics import boot, ci
from tidal.public_data.manifest import DATA
torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 2)))
OTO = DATA / "oto141" / "data" / "train"; CACHE = DATA / "proc" / "audio_oto"; FR = 0.01

def energy_vad(m, other_m):
    """m: [F, NMEL] log-mel (10 ms frames) -> list of (start_s, end_s) speech segments."""
    e = np.log(np.exp(m).mean(1) + 1e-9); eo = np.log(np.exp(other_m).mean(1) + 1e-9)
    lo, hi = np.percentile(e, 10), np.percentile(e, 98)
    on_t, off_t = lo + 0.40 * (hi - lo), lo + 0.30 * (hi - lo)
    act = np.zeros(len(e), bool); st = False
    for i, v in enumerate(e):                                   # hysteresis
        st = v > on_t if not st else v > off_t; act[i] = st
    act &= ~((eo - e > 3.0) & (e < lo + 0.6 * (hi - lo)))        # crosstalk guard: much louder on the other mic -> bleed
    segs = []; i = 0; n = len(act)
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]: j += 1
            segs.append([i * FR, j * FR]); i = j
        else: i += 1
    out = []
    for s in segs:                                              # fill gaps < 200 ms, drop blips < 120 ms
        if out and s[0] - out[-1][1] < 0.2: out[-1][1] = s[1]
        else: out.append(s)
    return [s for s in out if s[1] - s[0] >= 0.12]

def bc_proxy(segs, other):
    oc = np.array(other) if other else np.zeros((0, 2)); res = []
    for a, b in segs:
        inside = ((oc[:, 0] <= a - 0.3) & (oc[:, 1] >= b + 0.3)).any() if len(oc) else False
        res.append((a, b, "嗯" if (b - a) <= 1.0 and inside else "speech"))
    return res

def prep(max_sessions=60):
    CACHE.mkdir(parents=True, exist_ok=True); n = len(glob.glob(str(CACHE / "*.npz"))); stats = []
    for p in sorted(glob.glob(str(OTO / "*.tar"))):
        with tarfile.open(p) as t:
            mem = {m.name: m for m in t.getmembers()}
            for name in sorted(k for k in mem if k.endswith(".flac")):
                sid = name[:-5]; out = CACHE / f"{sid}.npz"
                if out.exists(): continue
                if n >= max_sessions: return
                meta = json.load(t.extractfile(mem[sid + ".json"])) if sid + ".json" in mem else {}
                from tidal.optional import require
                sf = require("soundfile", "audio"); resample_poly = require("scipy.signal", "audio", pip_name="scipy").resample_poly
                x, sr = sf.read(io.BytesIO(t.extractfile(mem[name]).read()), dtype="float32", always_2d=True)
                if x.shape[1] != 2: continue
                for a in meta.get("redacted_segments", []): x[int(a["start_sec"] * sr):int(a["end_sec"] * sr)] = 0.0
                ch = [resample_poly(x[:, k], 160, 441).astype(np.float32) if sr == 44100 else resample_poly(x[:, k], SR, sr).astype(np.float32) for k in (0, 1)]
                del x
                ms = [logmel(c) for c in ch]; F = min(len(m) for m in ms) // STACK * STACK; ms = [m[:F] for m in ms]
                s0, s1 = energy_vad(ms[0], ms[1]), energy_vad(ms[1], ms[0])
                s0, s1 = bc_proxy(s0, s1), bc_proxy(s1, s0)
                np.savez(out, m0=ms[0].astype(np.float16), m1=ms[1].astype(np.float16), s0=json.dumps(s0), s1=json.dumps(s1))
                sp = [sum(b - a for a, b, _ in s) for s in (s0, s1)]; n += 1
                print("prep", sid[:8], f"{F * FR / 60:.1f} min", "speech", [round(v / (F * FR), 2) for v in sp],
                      "segs", len(s0), len(s1), "bc", sum(1 for s in s0 + s1 if s[2] == "嗯"), flush=True)

def split_oto():
    ss = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(CACHE / "*.npz")))
    h = {s: int(hashlib.md5(s.encode()).hexdigest(), 16) % 100 for s in ss}
    te = [f"oto:{s}" for s in ss if h[s] < 20]; va = [f"oto:{s}" for s in ss if 20 <= h[s] < 28]; tr = [f"oto:{s}" for s in ss if h[s] >= 28]
    return tr, va, te

def pretrain(seed=0):
    tr, va, te = split_oto(); logf = open("logs/p3/audio_oto.log", "a")
    log = lambda s: (print(s, flush=True), logf.write(s + "\n"), logf.flush())
    log(f"oto pretrain train={len(tr)} val={len(va)} test={len(te)}")
    m, mu, sd, best = AT.train_fold(tr, va, seed, epochs=40, crop=750, bs=16, log=log, patience=5)
    torch.save(dict(state=m.state_dict(), mu=mu, sd=sd, best=best[0], train=tr, val=va, test=te), "models/audio_fe_oto.pt")

def _load(p):
    z = torch.load(p, weights_only=False); m = AudioEncoder(); m.load_state_dict(z["state"]); m.eval(); return m, z

def _summ(R, score_keys, n_boot=1000, pairs=None):
    """AUCs stratified by r['fold'] (n-weighted mean of per-fold AUCs; a single fold -> plain AUC). pairs: list of
    (a, b) score keys for paired block-bootstrap deltas (default: first key vs each other key)."""
    from tidal.metrics import strat_auc
    y = np.array([r["y"] for r in R]); bl = np.array([r["conv"] for r in R]); fo = np.array([r.get("fold", "all") for r in R])
    out = dict(n=len(R), pos_rate=float(y.mean()), convs=int(len(set(bl))))
    S = {k: np.array([r[k] for r in R], float) for k in score_keys}
    for k, p in S.items(): out[k] = dict(pr_auc=strat_auc(y, p, fo, "pr"), roc_auc=strat_auc(y, p, fo, "roc"))
    out["chance"] = dict(pr_auc=float(y.mean()), roc_auc=0.5)
    for a, b in (pairs or [(score_keys[0], k) for k in score_keys[1:]]):
        d = boot(lambda idx: strat_auc(y[idx], S[a][idx], fo[idx]) - strat_auc(y[idx], S[b][idx], fo[idx]), bl, n_boot, 0)
        out[f"{a}_minus_{b}_roc_auc"] = dict(mean=float(np.nanmean(d)), ci=ci(np.asarray(d).ravel()))
    for k in score_keys:
        d = boot(lambda idx: strat_auc(y[idx], S[k][idx], fo[idx]) - 0.5, bl, n_boot, 0)
        out[f"{k}_minus_chance_roc_auc"] = dict(mean=float(np.nanmean(d)), ci=ci(np.asarray(d).ravel()))
    return out

def _events(convs, models, lr=None):
    """events of AT.events with model scores; models: {name: (m, mu, sd)}"""
    rows = []
    for c in convs:
        cv = AT.load_conv(c); AT._HF.clear()
        for ch in (0, 1):
            P = {n: AT.run_model(m, mu, sd, cv, ch) for n, (m, mu, sd) in models.items()}
            for kind, k, y, sd_ in AT.events(cv, ch):
                r = dict(conv=c, ch=ch, kind=kind, k=k, y=y, f=AT.hand_feats(cv, ch, k, sd_))
                for n, Pn in P.items(): r[n] = float(Pn["pshift"][k] if kind == "shift" else Pn["bc"][k])
                rows.append(r)
    return rows

def _fit_lr(rows, kind):
    a = [r for r in rows if r["kind"] == kind]; lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    return lr.fit(np.array([r["f"] for r in a]), np.array([r["y"] for r in a])), float(np.mean([r["y"] for r in a]))

def krisp_eval(models):
    from tidal.optional import require
    pq = require("pyarrow.parquet", "audio", pip_name="pyarrow"); sf = require("soundfile", "audio")
    resample_poly = require("scipy.signal", "audio", pip_name="scipy").resample_poly
    t = pq.read_table(DATA / "krisp_tt" / "data" / "test.parquet").to_pandas(); res = []
    r = np.random.default_rng(0)
    for row in t.itertuples():
        x, sr = sf.read(io.BytesIO(row.audio["bytes"]), dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        if sr != SR: x = resample_poly(x, SR, sr).astype(np.float32)
        selfch = (r.standard_normal(len(x)) * 1e-4).astype(np.float32)
        X = stack_frames(logmel(x), logmel(selfch)); d = dict(conv=str(row.speaker_id), y=float(row.label == "shift"), last_silence=float(row.last_silence_duration))
        for n, (m, mu, sd) in models.items():
            with torch.no_grad(): o, _ = m(torch.from_numpy(((X - mu) / sd)[None].astype(np.float32)))
            d[n] = float(p_shift(o["vap"][0, -1]))
        res.append(d)
    return res

def evaluate(n_boot=1000):
    tr, va, te = split_oto(); mo, zo = _load("models/audio_fe_oto.pt"); OT = (mo, zo["mu"], zo["sd"]); out = {}
    # (1) oto held-out
    trE = _events(tr[:20], {}); teE = _events(te, {"oto_model": OT})
    for kind in ("shift", "bc"):
        lr, prior = _fit_lr(trE, kind)
        R = [dict(rr, lr_prosody=float(lr.predict_proba(np.array([rr["f"]]))[0, 1])) for rr in teE if rr["kind"] == kind]
        out[f"oto_heldout_{kind}"] = _summ(R, ["oto_model", "lr_prosody"], n_boot)
    # (2) MagicData folds: zero-shot oto, oto->ft, MagicData-only (existing fold models)
    convs = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(AT.CACHE / "*.npz"))); pairs = sorted({c.split("_")[0] for c in convs}); mdR = []
    for tp in pairs:
        trc = [c for c in convs if not c.startswith(tp)]; tec = [c for c in convs if c.startswith(tp)]; vac = trc[::5]; trn = [c for c in trc if c not in vac]
        fp = f"models/audio_fe_ft_oto_{tp}.pt"
        if not os.path.exists(fp):
            m, mu, sd, best = AT.train_fold(trn, vac, 0, epochs=30, log=lambda s: open("logs/p3/audio_oto.log", "a").write(f"[ft {tp}] {s}\n"),
                                            init=dict(state=zo["state"], mu=zo["mu"], sd=zo["sd"]), lr=5e-4, patience=5)
            torch.save(dict(state=m.state_dict(), mu=mu, sd=sd, best=best[0]), fp)
        mf, zf = _load(fp); mm, zm = _load(f"models/audio_fe_fold_{tp}.pt")
        E = _events(tec, {"oto_ft": (mf, zf["mu"], zf["sd"]), "md_only": (mm, zm["mu"], zm["sd"]), "oto_zeroshot": OT}); trE_md = _events(trn, {})
        for kind in ("shift", "bc"):
            lr, _ = _fit_lr(trE_md, kind)
            for r in E:
                if r["kind"] == kind: r["lr_prosody"] = float(lr.predict_proba(np.array([r["f"]]))[0, 1])
        mdR += [dict(r, fold=tp) for r in E]
    json.dump([{k: v for k, v in r.items() if k != "f"} for r in mdR], open(DATA / "proc" / "audio_oto_md_events.json", "w"))
    pr = [("oto_ft", "md_only"), ("oto_ft", "lr_prosody"), ("md_only", "lr_prosody"), ("oto_zeroshot", "md_only"), ("oto_zeroshot", "lr_prosody")]
    for kind in ("shift", "bc"):
        out[f"magicdata_{kind}"] = _summ([r for r in mdR if r["kind"] == kind], ["oto_ft", "md_only", "oto_zeroshot", "lr_prosody"], n_boot, pr)
    # (3) Krisp (eval only)
    mm, zm = _load(f"models/audio_fe_fold_{pairs[0]}.pt")
    KR = krisp_eval({"oto_model": OT, "md_model": (mm, zm["mu"], zm["sd"])})
    json.dump(KR, open(DATA / "proc" / "audio_krisp_scores.json", "w"))
    out["krisp_shift_vs_hold"] = _summ(KR, ["oto_model", "md_model", "last_silence"], n_boot,
                                       [("oto_model", "md_model"), ("oto_model", "last_silence"), ("md_model", "last_silence")])
    out["note"] = ("oto labels come from an energy VAD (no transcripts) and the backchannel label there is an acoustic proxy; "
                   "MagicData labels are human segments; Krisp is evaluation-only (license forbids training) and nothing was fitted on it; "
                   "Krisp blocks = speaker_id. md_model on Krisp = the MagicData fold model trained without speaker pair " + pairs[0])
    json.dump(out, open("reports/audio_oto_phase3.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__":
    {"prep": prep, "pretrain": pretrain, "eval": evaluate}[sys.argv[1]]()
