"""Phase-4 config loaded by shadow mode, plus the phase-5 interrupt head (shadow field only).

Wired winner: the Mamba-2-style ablation (`m3_ablate_m2`) with text emotion on (11 columns:
8 class probabilities, valence, arousal, has-emotion flag; tidal/emo_features.py) and speech
emotion off (not an input to the event model; duplex only passes it through ControlOut.affect).

The 2026-10-08 pre-registered phase-5 bake-off (fresh P5bb_* runs) selected this over
`mamba3_siso`: real head-sum val means 3.9061 vs 3.9276 (gap 0.0215 > 0.005); test also
favored the ablation. `mamba3_siso` remains a backbone option in tidal/backbones.py.
Continuum memory stays shadow-only.

Interrupt head (`y_interrupt`, adopted in phase 5): a BinHead on the encoder state h. The phase-5
heads were fit on frozen mamba3_siso states, so the shadow uses heads retrained on frozen
P4emo_m3_ablate_m2_s0 states (cache `p5_h_m2`, models/P5hd_m2_s{0,1,2}.pt, same rows, protocol and
seeds). Heads trained on another cache are refused. Its probability is written as `p_interrupt`
(mean over the loaded seed heads) to the event encoder output, ControlOut and the shadow log. It is
NOT a tick-controller input and does not change any action.
Label meaning (tidal/phase5_labels.py): the event is a finished speech segment; 1 = it started while
another speaker held the floor and took the floor, 0 = it waited or was a backchannel.
"""
import json, os
import numpy as np
import torch, torch.nn as nn
from tidal import features_g as FG

TEXT_EMO_DIM = 11   # 8 probs + valence + arousal + has_emotion
WINNER = dict(kind="m3_ablate_m2", text_emotion=True, speech_emotion=False, emo_dim=TEXT_EMO_DIM, version="phase4",
              interrupt=True)
HEAD_FEATS = "p5_h_m2"                         # cache the shadow interrupt heads must have been trained on
DEFAULT_CKPT = "models/P4emo_m3_ablate_m2_s0.pt"
DEFAULT_HEADS = tuple(f"models/P5hd_m2_s{s}.pt" for s in range(3))
SYSTEM = "model:p4_m3_ablate_m2"               # shadow-log system name (regime "P4")
REGIME = "P4"

def n_feat(cfg=None):
    cfg = WINNER if cfg is None else cfg
    return len(FG.FEAT_G) + (int(cfg.get("emo_dim", TEXT_EMO_DIM)) if cfg.get("text_emotion", True) else 0)

class InterruptHead(nn.Module):
    """Mean P(y_interrupt) over seed heads. Input h [..., 128] -> probability [...]."""
    def __init__(self, nets):
        super().__init__(); self.nets = nn.ModuleList(nets)
    def forward(self, h):
        return torch.stack([torch.sigmoid(n(h)) for n in self.nets]).mean(0)

def load_interrupt(paths=None):
    """Interrupt head from phase-5 checkpoints trained on the m3_ablate_m2 cache. No paths -> one random-init head
    (architecture / tests only). A head trained on another backbone's states raises ValueError."""
    from tidal.model import BinHead
    nets = []
    for p in paths or ():
        ck = torch.load(p, map_location="cpu", weights_only=False)
        f = ck.get("feats", "p5_h")
        if f != HEAD_FEATS:
            raise ValueError(f"{p}: y_interrupt head was trained on '{f}' states; the shadow body m3_ablate_m2 needs '{HEAD_FEATS}'")
        b = BinHead(); b.load_state_dict(ck["state"]["y_interrupt"]); nets.append(b)
    return InterruptHead(nets or [BinHead()]).eval()

def _attach(m, cfg):
    # plain dict attribute (not a registered submodule): the body's state_dict / checkpoint format stays unchanged
    m.side_heads = {"p_interrupt": load_interrupt(cfg.get("interrupt_heads"))} if cfg.get("interrupt", True) else {}
    return m

def build(cfg=None):
    """Construct an eval-mode VAPModel for the phase-4 shadow config (+ interrupt side head). Random init unless
    cfg['interrupt_heads'] names head checkpoints; call load() for a body checkpoint."""
    from tidal.vap import VAPModel
    cfg = dict(WINNER if cfg is None else {**WINNER, **cfg})
    if cfg.get("speech_emotion"):
        raise ValueError("phase-4 shadow config keeps speech emotion out of the event model")
    if cfg.get("kind") != "m3_ablate_m2":
        raise ValueError(f"phase-4 shadow backbone must be m3_ablate_m2, got {cfg.get('kind')}")
    if not cfg.get("text_emotion"):
        raise ValueError("phase-4 shadow config requires text emotion on")
    m = VAPModel(n_feat(cfg), kind="m3_ablate_m2")
    m.eval()
    return _attach(m, cfg)

def load(path=None, cfg=None):
    """Build the winner. If `path` exists, load its state dict (kind must be m3_ablate_m2; speech-* keys ignored)."""
    cfg = dict(WINNER if cfg is None else {**WINNER, **{k: v for k, v in cfg.items() if k != "ckpt"}})
    m = build(cfg)
    if path and os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        kind = ck.get("kind", "gru")
        if kind != "m3_ablate_m2":
            raise ValueError(f"checkpoint kind is {kind}, shadow phase-4 winner is m3_ablate_m2")
        sd = {k: v for k, v in ck["state"].items() if "speech" not in k}
        m.load_state_dict(sd)
    return m

def step_events(model, events, mu=None, sd=None):
    """One streaming step per event (text emotion via Event.emotion; speech emotion is not read)."""
    from tidal.duplex import EventEncoder
    mu = np.zeros(FG.NB, np.float32) if mu is None else np.asarray(mu, np.float32)
    sd = np.ones(FG.NB, np.float32) if sd is None else np.asarray(sd, np.float32)
    enc = EventEncoder(model, mu, sd)
    for e in events:
        enc.push(e)
    return enc

# ------------------------------------------------------------------ shadow log (cron tick)
def shadow_spec():
    """None unless enabled (TIDAL_SHADOW_P4=1 or shadow/state/frozen_p4.json). Missing paths fall back to the defaults
    if those files exist; otherwise random init (logged in the version string)."""
    from tidal.shadow import store as S
    f = S.STATE / "frozen_p4.json"
    if os.environ.get("TIDAL_SHADOW_P4") != "1" and not f.exists():
        return None
    spec = json.loads(f.read_text()) if f.exists() else {}
    if os.environ.get("TIDAL_SHADOW_P4_CKPT"):
        spec["ckpt"] = os.environ["TIDAL_SHADOW_P4_CKPT"]
    spec.setdefault("ckpt", DEFAULT_CKPT if os.path.exists(DEFAULT_CKPT) else None)
    spec.setdefault("interrupt_heads", [p for p in DEFAULT_HEADS if os.path.exists(p)])
    return spec

def load_shadow(spec):
    """(model with side heads, mu, sd, version dict) for the cron shadow log."""
    m = load(spec.get("ckpt"), spec)
    mu, sd = np.zeros(FG.NB, np.float32), np.ones(FG.NB, np.float32)
    if spec.get("ckpt") and os.path.exists(spec["ckpt"]):
        ck = torch.load(spec["ckpt"], map_location="cpu", weights_only=False)
        mu, sd = np.asarray(ck["mu"], np.float32), np.asarray(ck["sd"], np.float32)
    ver = dict(P4=os.path.basename(spec.get("ckpt") or "random-init"),
               interrupt=[os.path.basename(p) for p in spec.get("interrupt_heads") or []] or "random-init")
    return m, mu, sd, ver

def text_emotion_from_cache(c, texts):
    """[n, 11] text-emotion columns from bge embeddings ALREADY cached in the shadow DB (emb_cache, filled by the TS
    path). No new embedding, no network. Rows without text / without a cached embedding / without the emotion heads
    stay 0 (unknown flag 0), the same convention as training."""
    import hashlib
    out = np.zeros((len(texts), TEXT_EMO_DIM), np.float32)
    has = [i for i, t in enumerate(texts) if isinstance(t, str) and t]
    if not has or c is None or not all(os.path.exists(f"models/emo_text_s{s}.pt") for s in range(3)):
        return out
    try:
        c.execute("select 1 from emb_cache limit 1")
    except Exception:
        return out
    keys = {i: hashlib.sha1(texts[i].encode()).hexdigest() for i in has}
    got = {}
    uk = sorted(set(keys.values()))
    for a in range(0, len(uk), 500):
        q = uk[a:a + 500]
        for k, b in c.execute(f"select key, vec from emb_cache where key in ({','.join('?' * len(q))})", q):
            got[k] = np.frombuffer(b, np.float16).astype(np.float32)
    rows = [i for i in has if keys[i] in got]
    if not rows:
        return out
    from tidal.emo_features import heads
    E = np.stack([got[keys[i]] for i in rows])
    f = np.mean([h.features(E) for h in heads()], 0)
    out[rows, :10] = f; out[rows, 10] = 1.0
    return out

@torch.no_grad()
def score_windows(model, F, V):
    """Left-padded windows F [B, T, nf], V [B, T] -> per-row probabilities at the last position:
    the six heads, vap, and every side head (p_interrupt). Same window layout as phase5_encode."""
    from tidal.dataset import HEADS
    o = model(torch.from_numpy(F), torch.from_numpy(V))
    h = o["h"][:, -1]
    out = {}
    for k in HEADS:
        lg = o[k][:, -1]
        out[k] = (torch.sigmoid(lg[:, 0]) if lg.shape[-1] == 1 else torch.softmax(lg, -1)).numpy()
    out["p_vap"] = torch.sigmoid(o["vap"][:, -1]).numpy()
    for k, net in getattr(model, "side_heads", {}).items():
        out[k] = net(h).reshape(-1).numpy()
    return out

def score_frame(model, d, targets, mu, sd, emo=None, bs=256):
    """Shadow-log scoring of `targets` (msg_ids) in frame `d` (shadow events). Each target gets its own causal window of
    <= 64 events of its conversation. emo: callable(texts) -> [n, 11] or None (all unknown). Returns {msg_id: probs}."""
    from tidal import features
    from tidal.seqdata import CTX
    d = d.sort_values(["conv", "ts"], kind="stable").reset_index(drop=True)
    X = features.compute(d)
    G = FG.compute(d.drop(columns=["n_participants", "modality"], errors="ignore"), X)
    Z = FG.normalize(G, np.asarray(mu, np.float32), np.asarray(sd, np.float32))
    E = emo(d.text.tolist()) if emo is not None else np.zeros((len(d), TEXT_EMO_DIM), np.float32)
    Z = np.concatenate([Z, np.asarray(E, np.float32)], 1)
    pos = {m: i for i, m in enumerate(d.msg_id)}; rows = [pos[m] for m in targets if m in pos]
    conv = d.conv.to_numpy(); res = {}
    for a in range(0, len(rows), bs):
        rr = rows[a:a + bs]; B = len(rr)
        F = np.zeros((B, CTX, Z.shape[1]), np.float32); V = np.zeros((B, CTX), bool)
        for b, r in enumerate(rr):
            lo = r
            while lo > 0 and r - lo + 1 < CTX and conv[lo - 1] == conv[r]: lo -= 1
            F[b, CTX - (r - lo + 1):] = Z[lo:r + 1]; V[b, CTX - (r - lo + 1):] = True
        o = score_windows(model, F, V)
        for b, r in enumerate(rr):
            p = {k: (round(float(v[b]), 6) if np.ndim(v[b]) == 0 else [round(float(x), 6) for x in v[b]]) for k, v in o.items()}
            res[d.msg_id.iat[r]] = p
    return res
