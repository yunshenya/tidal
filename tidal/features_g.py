"""Scenario-general per-event inputs (场景通用). No scenario enum: the model must infer the setting from stream statistics.

  timing / rate / relative role (always present, causal):  10 dims taken from tidal.features
  optional context: log(1+participants) + known flag       ( 2 dims, randomly dropped in training)
  optional modality: known flag + text/voice/visual/other  ( 5 dims, randomly dropped in training)
'Unknown' is represented as zeros with the known-flag = 0, so a dropped signal and a missing signal look identical.
Speaker identity / text availability are deliberately NOT inputs (availability leakage, see reports/phase1.md §5)."""
import numpy as np, pandas as pd
from tidal import features

BASE = ["role_self", "log_gap_prev", "first_in_conv", "log_since_bot", "bot_never", "log_act_60s", "log_act_600s",
        "tod_sin", "tod_cos", "prev_is_bot"]
CTX = ["log_participants", "participants_known"]
MOD = ["mod_known", "mod_text", "mod_voice", "mod_visual", "mod_other"]
FEAT_G = BASE + CTX + MOD
NB = len(BASE)
CTX_SL = slice(NB, NB + len(CTX)); MOD_SL = slice(NB + len(CTX), len(FEAT_G))
_MODMAP = {"text": "mod_text", "voice": "mod_voice", "audio": "mod_voice", "image": "mod_visual", "video": "mod_visual",
           "visual": "mod_visual", "sticker": "mod_visual", "gift": "mod_other", "other": "mod_other", "file": "mod_other"}

def compute(d: pd.DataFrame, X_full=None) -> np.ndarray:
    """d sorted by (conv, ts). Optional columns: n_participants (float), modality (str). Returns raw [N, 17]."""
    if X_full is None: X_full = features.compute(d)
    G = np.zeros((len(d), len(FEAT_G)), np.float32)
    G[:, :NB] = X_full[:, [features.FEATURES.index(f) for f in BASE]]
    if "n_participants" in d:
        p = pd.to_numeric(d.n_participants, errors="coerce").to_numpy(float); ok = ~np.isnan(p)
        G[ok, NB] = np.log1p(p[ok]) / 4.0; G[ok, NB + 1] = 1.0
    if "modality" in d:
        mods = d.modality.to_numpy(object)
        for i, m in enumerate(mods):
            col = _MODMAP.get(m) if isinstance(m, str) else None
            if col: G[i, FEAT_G.index("mod_known")] = 1.0; G[i, FEAT_G.index(col)] = 1.0
    return G

def normalize(G, mu, sd):
    """Standardize only the timing block; optional blocks are already in [0, ~2] and must keep 0 == unknown."""
    Z = G.copy(); Z[:, :NB] = (G[:, :NB] - mu) / sd; return Z.astype(np.float32)

def drop_optional(Z, rng, p_ctx=0.5, p_mod=0.5):
    """Training-time augmentation on a [B, T, F] batch: drop each optional block per sequence."""
    B = Z.shape[0]
    Z[rng.random(B) < p_ctx, :, CTX_SL] = 0.0
    Z[rng.random(B) < p_mod, :, MOD_SL] = 0.0
    return Z
