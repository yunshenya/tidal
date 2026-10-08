"""Model + baseline inference for shadow mode (CPU, single thread, no network).
Uses the exported phase-1 ONNX models and the phase-1 baselines refit to identical predictions (see setup.py)."""
import sys, json, os, numpy as np, pandas as pd
os.environ.setdefault("OMP_NUM_THREADS", "1")
import onnxruntime as ort, joblib
from tidal import features
from tidal.config import ROOT
from tidal.dataset import HEADS
from tidal.shadow import store as S

CTX = 64
MODELS = {"T": os.environ.get("TIDAL_SHADOW_MODEL_T", "T_rs_gru_s1"), "TS": os.environ.get("TIDAL_SHADOW_MODEL_TS", "TS_rs_gru_s1")}
FROZEN = S.STATE / "frozen.json"      # written by setup.py: norm stats, thresholds, strongest baselines, versions
P2_FROZEN = S.STATE / "frozen_p2.json"  # written by setup2.py: phase-2 VAP model (separate system, logged alongside phase 1)
RC_MID = np.array([1, 3.5, 7.5, 15, 40, 180, 300.0])

def p2_cfg():
    """phase-2 config if the phase-2 model is set up (frozen_p2.json + its ONNX), else None. Phase 1 never depends on it."""
    if not P2_FROZEN.exists(): return None
    cfg = json.loads(P2_FROZEN.read_text())
    return cfg if (ROOT / "models" / f"{cfg['tag']}.onnx").exists() else None

def ready():
    """None if models/baselines/frozen config are present, else a reason string."""
    need = [FROZEN] + [ROOT / "models" / f"{m}.onnx" for m in MODELS.values()] + [S.STATE / f"baselines_{r}.joblib" for r in MODELS]
    miss = [str(p.name) for p in need if not p.exists()]
    return ("missing: " + ", ".join(miss)) if miss else None

def has_text(t): return isinstance(t, str) and len(t) > 0

class Predictor:
    def __init__(self, db=None):
        self.db = db; self.n_encoded = 0
        if db is not None: db.execute("create table if not exists emb_cache(key text primary key, vec blob)")
        self.cfg = json.loads(FROZEN.read_text())
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = {r: ort.InferenceSession(str(ROOT / "models" / f"{MODELS[r]}.onnx"), so, providers=["CPUExecutionProvider"]) for r in MODELS}
        self.base = {r: joblib.load(S.STATE / f"baselines_{r}.joblib") for r in MODELS}
        self.enc = None; self._ecache = {}
        self.p2 = p2_cfg()
        if self.p2:
            try: self.p2_sess = ort.InferenceSession(str(ROOT / "models" / f"{self.p2['tag']}.onnx"), so, providers=["CPUExecutionProvider"])
            except Exception as e: print(f"[shadow] phase-2 model not loaded: {type(e).__name__}: {e}", file=sys.stderr); self.p2 = None

    def _embed(self, texts):
        """bge int8 embeddings, encoded one text at a time (dynamic int8 quantization makes batched outputs depend on
        batch composition) and cached persistently in the shadow DB so every text is embedded exactly once."""
        from tidal.embed import key
        todo = sorted({t for t in texts if key(t) not in self._ecache})
        if todo and self.db is not None:
            ks = [key(t) for t in todo]
            for i in range(0, len(ks), 500):
                q = ks[i:i + 500]
                for k, b in self.db.execute(f"select key, vec from emb_cache where key in ({','.join('?' * len(q))})", q):
                    self._ecache[k] = np.frombuffer(b, np.float16).astype(np.float32)
            todo = [t for t in todo if key(t) not in self._ecache]
        if todo:
            if self.enc is None:
                from tidal.embed import Encoder
                self.enc = Encoder(threads=1)
            for t in todo:
                e = self.enc.encode([t], bs=1)[0]; self._ecache[key(t)] = e.astype(np.float16).astype(np.float32)
                if self.db is not None:
                    self.db.execute("insert or ignore into emb_cache values(?,?)", (key(t), e.astype(np.float16).tobytes()))
            self.n_encoded += len(todo)
        return np.stack([self._ecache[key(t)] for t in texts]) if texts else np.zeros((0, 512), np.float32)

    def predict(self, d: pd.DataFrame, targets_T, targets_TS, targets_P2=()):
        """d: context+target events (unified frame) for the affected conversations, any order.
        targets_*: msg_ids to predict. Returns list of (msg_id, regime, system, probs_dict)."""
        d = d.sort_values(["conv", "ts"], kind="stable").reset_index(drop=True)
        X = features.compute(d)
        out = []
        if self.p2 and len(targets_P2):
            try: out = self._predict_p2(d, X, targets_P2)
            except Exception as e:  # the phase-2 system must never break phase-1 logging; unpredicted targets are retried next run
                print(f"[shadow] phase-2 system skipped this run: {type(e).__name__}: {e}", file=sys.stderr); out = []
        for regime, targets in (("T", targets_T), ("TS", targets_TS)):
            if not len(targets): continue
            keep = np.ones(len(d), bool) if regime == "T" else d.text.map(has_text).to_numpy()
            pos = {m: i for i, m in enumerate(d.msg_id)}
            rows = [pos[m] for m in targets if m in pos and keep[pos[m]]]
            if not rows: continue
            nrm = self.cfg["norm"][regime]; mu, sd = np.array(nrm["mu"], np.float32), np.array(nrm["sd"], np.float32)
            Xr = ((X[:, features.cols(regime)] - mu) / sd).astype(np.float32)
            conv = d.conv.to_numpy(); kidx = np.flatnonzero(keep)
            emb_all = None
            if regime == "TS":
                emb_all = np.zeros((len(d), 512), np.float32)
                emb_all[kidx] = self._embed(d.text.to_numpy()[kidx].tolist())
            B = len(rows); F = np.zeros((B, CTX, Xr.shape[1]), np.float32); V = np.zeros((B, CTX), bool)
            E = np.zeros((B, CTX, 512), np.float32) if regime == "TS" else None
            kc = conv[kidx]
            for b, r in enumerate(rows):
                k = np.searchsorted(kidx, r)                       # position of r among kept rows
                lo = k
                while lo > 0 and k - lo + 1 < CTX and kc[lo - 1] == conv[r]: lo -= 1
                w = kidx[lo:k + 1]; o = CTX - len(w)
                F[b, o:] = Xr[w]; V[b, o:] = True
                if E is not None: E[b, o:] = emb_all[w]
            feed = {"feats": F, "valid": V}
            if E is not None: feed["text"] = E
            res = self.sess[regime].run(None, feed)
            bp = self.base[regime]
            from tidal.baselines import predict_bundle
            BP = predict_bundle(bp, X[rows], emb_all[rows] if emb_all is not None else None)
            names = sorted({k.split("|")[1] for k in BP})
            for b, r in enumerate(rows):
                mid = d.msg_id.iat[r]
                out.append((mid, regime, "model:" + MODELS[regime], {h: _j(res[i][b]) for i, h in enumerate(HEADS)}))
                for n in names:
                    out.append((mid, regime, n, {h: _j(BP[f"{h}|{n}"][b]) for h in HEADS if f"{h}|{n}" in BP}))
        return out

    def _predict_p2(self, d, X, targets):
        """phase-2 VAP model on scenario-general inputs. Participant count / modality are not supplied by the shadow source,
        so they are left unknown (zeros + known-flag 0); training dropped these optional blocks with p=0.5 so the model
        has seen this case (effect on held-out data: reports/phase2_extra.json, participants_unknown)."""
        from tidal import features_g as FG
        G = FG.compute(d.drop(columns=["n_participants", "modality"], errors="ignore"), X)
        Z = FG.normalize(G, np.array(self.p2["mu"], np.float32), np.array(self.p2["sd"], np.float32))
        pos = {m: i for i, m in enumerate(d.msg_id)}; rows = [pos[m] for m in targets if m in pos]
        if not rows: return []
        conv = d.conv.to_numpy(); B = len(rows); F = np.zeros((B, CTX, Z.shape[1]), np.float32); V = np.zeros((B, CTX), bool)
        for b, r in enumerate(rows):
            lo = r
            while lo > 0 and r - lo + 1 < CTX and conv[lo - 1] == conv[r]: lo -= 1
            F[b, CTX - (r - lo + 1):] = Z[lo:r + 1]; V[b, CTX - (r - lo + 1):] = True
        res = []
        for a in range(0, B, 256):
            res.append(self.p2_sess.run(None, {"feats": F[a:a + 256], "valid": V[a:a + 256]}))
        res = [np.concatenate([x[i] for x in res]) for i in range(len(res[0]))]
        tau = self.p2["abstain_tau"]["y_act"]; out = []
        for b, r in enumerate(rows):
            pr = {h: _j(res[i][b]) for i, h in enumerate(HEADS)}
            pact = np.asarray(res[HEADS.index("y_act")][b]); prc = np.asarray(res[HEADS.index("y_recheck")][b])
            pr["abstain"] = int(pact.max() < tau)                                   # low confidence -> wait and recheck
            pr["recheck_after_s"] = round(float(np.clip(prc @ RC_MID, 2, 300)), 1)
            pr["p_vap"] = _j(res[-1][b])
            out.append((d.msg_id.iat[r], "T", "model:" + self.p2["tag"], pr))
        return out

def _j(v):
    v = np.asarray(v, float)
    if v.size == 1: v = v.reshape(())
    return round(float(v), 6) if v.ndim == 0 else [round(float(x), 6) for x in v]
