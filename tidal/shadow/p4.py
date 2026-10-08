"""Phase-4 config loaded by shadow mode.

Wired winner: Mamba-3 SISO (`mamba3_siso`) with text emotion on (11 columns: 8 class
probabilities, valence, arousal, has-emotion flag; tidal/emo_features.py) and speech
emotion off (not an input to the event model; duplex only passes it through ControlOut.affect).

reports/phase4.md: if the ablation `m3_ablate_m2` is also a candidate, it had the lowest
mean real-val loss. Among the four named backbones the winner is Mamba-3 SISO, which is
the config this loader builds. Phase 5 re-ran that pair from scratch (P5bb_*); the ablation
ranked first again, and this loader still refuses to switch.
"""
import os
import numpy as np
from tidal import features_g as FG

TEXT_EMO_DIM = 11   # 8 probs + valence + arousal + has_emotion
WINNER = dict(kind="mamba3_siso", text_emotion=True, speech_emotion=False, emo_dim=TEXT_EMO_DIM, version="phase4")

def n_feat(cfg=None):
    cfg = WINNER if cfg is None else cfg
    return len(FG.FEAT_G) + (int(cfg.get("emo_dim", TEXT_EMO_DIM)) if cfg.get("text_emotion", True) else 0)

def build(cfg=None):
    """Construct an eval-mode VAPModel for the phase-4 shadow config. Random init; call load() for a checkpoint."""
    from tidal.vap import VAPModel
    cfg = dict(WINNER if cfg is None else {**WINNER, **cfg})
    if cfg.get("speech_emotion"):
        raise ValueError("phase-4 shadow config keeps speech emotion out of the event model")
    if cfg.get("kind") != "mamba3_siso":
        raise ValueError(f"phase-4 shadow backbone must be mamba3_siso, got {cfg.get('kind')}")
    if not cfg.get("text_emotion"):
        raise ValueError("phase-4 shadow config requires text emotion on")
    m = VAPModel(n_feat(cfg), kind="mamba3_siso")
    m.eval()
    return m

def load(path=None, cfg=None):
    """Build the winner. If `path` exists, load its state dict (kind must be mamba3_siso; speech-* keys ignored)."""
    import torch
    cfg = dict(WINNER if cfg is None else {**WINNER, **{k: v for k, v in cfg.items() if k != "ckpt"}})
    m = build(cfg)
    if path and os.path.exists(path):
        ck = torch.load(path, weights_only=False)
        kind = ck.get("kind", "gru")
        if kind != "mamba3_siso":
            raise ValueError(f"checkpoint kind is {kind}, shadow phase-4 winner is mamba3_siso")
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
