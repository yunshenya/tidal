"""Soft emotion posteriors from SenseVoice-Small (offline teacher; FunASR Model License v1.1; weights never published).
Re-implements the SenseVoice front end (Kaldi-style fbank 80, hamming, preemph 0.97, LFR 7/6, CMVN from the model
metadata) and reads the CTC logits at the emotion query position (frame 1: <lang><emo><event><itn>), softmaxed over the
7 emotion tokens (EMO_UNKNOWN excluded). Validated against sherpa-onnx's hard tag (see __main__)."""
import numpy as np, onnxruntime as ort
SV_EMO = ["HAPPY", "SAD", "ANGRY", "NEUTRAL", "FEARFUL", "DISGUSTED", "SURPRISED"]; SV_IDS = [25001, 25002, 25003, 25004, 25005, 25006, 25007]

def _melbank(n=80, nfft=512, sr=16000, lo=20.0, hi=8000.0):
    mel = lambda f: 1127.0 * np.log(1 + f / 700.0); m = np.linspace(mel(lo), mel(hi), n + 2)
    fr = mel(np.arange(nfft // 2 + 1) * sr / nfft); W = np.zeros((n, nfft // 2 + 1), np.float32)
    for i in range(n):
        l, c, r = m[i], m[i + 1], m[i + 2]
        W[i] = np.clip(np.minimum((fr - l) / (c - l), (r - fr) / (r - c)), 0, None)
    return W
_W = _melbank()

def fbank(x):
    x = x * 32768.0; L, S = 400, 160
    if len(x) < L: x = np.pad(x, (0, L - len(x)))
    n = 1 + (len(x) - L) // S; idx = np.arange(L)[None] + S * np.arange(n)[:, None]; f = x[idx].astype(np.float64)
    f = f - f.mean(1, keepdims=True); f = np.concatenate([f[:, :1] - 0.97 * f[:, :1], f[:, 1:] - 0.97 * f[:, :-1]], 1)
    f = f * np.hamming(L)[None]; p = np.abs(np.fft.rfft(f, 512)) ** 2
    return np.log(np.maximum(p @ _W.T.astype(np.float64), np.finfo(np.float32).eps)).astype(np.float32)

def lfr(F, m=7, n=6):
    T = len(F); pad = (m - 1) // 2; F = np.concatenate([np.repeat(F[:1], pad, 0), F]); out = []
    for i in range(int(np.ceil(T / n))):
        s = F[i * n: i * n + m]
        if len(s) < m: s = np.concatenate([s, np.repeat(F[-1:], m - len(s), 0)])
        out.append(s.reshape(-1))
    return np.array(out, np.float32)

class SoftTeacher:
    def __init__(self, path="models/sensevoice/model.int8.onnx", threads=1):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads; self.s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        md = self.s.get_modelmeta().custom_metadata_map
        self.nm = np.array(md["neg_mean"].split(","), np.float32); self.isd = np.array(md["inv_stddev"].split(","), np.float32)
        self.lang = int(md["lang_auto"]); self.itn = int(md["without_itn"])
    def __call__(self, x):
        X = (lfr(fbank(x)) + self.nm) * self.isd
        lg = self.s.run(None, {"x": X[None], "x_length": np.array([len(X)], np.int32), "language": np.array([self.lang], np.int32), "text_norm": np.array([self.itn], np.int32)})[0][0]
        e = lg[1, SV_IDS]; p = np.exp(e - e.max()); return p / p.sum(), int(np.argmax(lg[1]))

def validate():
    import pandas as pd
    from tidal.emotion_speech import read_wav, SR
    T = pd.read_parquet("data/public/proc/emo_teacher.parquet"); t = SoftTeacher(); agree = []
    for r in T.sample(60, random_state=0).itertuples():
        x = read_wav(r.path)[int(r.start * SR): int(r.end * SR) if r.end == r.end and r.end is not None else None]
        p, am = t(x); tok = {25001 + i: e for i, e in enumerate(SV_EMO)}; tok[25009] = "EMO_UNKNOWN"; tok[25008] = "?"
        agree.append(tok.get(am, str(am)) == r.sv_emotion)
    print("argmax agreement with sherpa-onnx hard tags:", np.mean(agree))

def run_all(out="data/public/proc/emo_soft.parquet"):
    import pandas as pd, time
    from tidal.emotion_speech import read_wav, SR
    T = pd.read_parquet("data/public/proc/emo_teacher.parquet"); t = SoftTeacher(); P = []; cache = {}; t0 = time.time()
    for i, r in enumerate(T.itertuples()):
        if r.path not in cache: cache = {r.path: read_wav(r.path)}
        x = cache[r.path][int(r.start * SR): int(r.end * SR) if r.end == r.end and r.end is not None else None]
        P.append(t(x)[0])
        if i % 1000 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(dict(id=T.id, **{f"sv_{e}": np.array(P)[:, k] for k, e in enumerate(SV_EMO)})).to_parquet(out); print("soft done", flush=True)
if __name__ == "__main__":
    import sys
    (validate if len(sys.argv) > 1 and sys.argv[1] == "validate" else run_all)()
