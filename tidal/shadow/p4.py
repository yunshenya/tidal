"""Phase-4 config loaded by shadow mode.

Wired winner: the Mamba-2-style ablation (`m3_ablate_m2`) with text emotion on (11 columns:
8 class probabilities, valence, arousal, has-emotion flag; tidal/emo_features.py) and speech
emotion off (not an input to the event model; duplex only passes it through ControlOut.affect).

The 2026-10-08 pre-registered phase-5 bake-off (fresh P5bb_* runs) selected this over
`mamba3_siso`: real head-sum val means 3.9061 vs 3.9276 (gap 0.0215 > 0.005); test also
favored the ablation. `mamba3_siso` remains a backbone option in tidal/backbones.py.
Continuum memory stays shadow-only. The adopted interrupt head is not wired into this tick.
"""
import os
import numpy as np
from tidal import features_g as FG

TEXT_EMO_DIM = 11   # 8 probs + valence + arousal + has_emotion
WINNER = dict(kind="m3_ablate_m2", text_emotion=True, speech_emotion=False, emo_dim=TEXT_EMO_DIM, version="phase4")

def n_feat(cfg=None):
    cfg = WINNER if cfg is None else cfg
    return len(FG.FEAT_G) + (int(cfg.get("emo_dim", TEXT_EMO_DIM)) if cfg.get("text_emotion", True) else 0)

def build(cfg=None):
    """Construct an eval-mode VAPModel for the phase-4 shadow config. Random init; call load() for a checkpoint."""
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
    return m

def load(path=None, cfg=None):
    """Build the winner. If `path` exists, load its state dict (kind must be m3_ablate_m2; speech-* keys ignored)."""
    import torch
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
