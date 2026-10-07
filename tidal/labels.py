"""Weak labels derived from observable structure (documented in reports/phase1.md §标签).

Input: unified event frame sorted by (conv, ts) with columns
  conv, ts, role ('self'=bot | 'other'), speaker (str|None), speaker_known (bool),
  text (str|None), addr_src (1/0/NaN), bot_act_src ('speak'|'silent'|None)
Output columns (NaN = masked/unknown):
  y_eot, y_self, y_addr, y_act (0 speak,1 wait,2 silent), y_recheck (bucket 0..6), y_hreply
"""
import numpy as np, pandas as pd

N_EOT = 20.0          # s; a gap >= N ends the turn regardless of who speaks next
W_SELF = 180.0        # s; window for "same speaker comes back after interruption/pause"
W_WAIT = 60.0         # s; bot speaks at a later event within this window -> 'wait'
W_HREPLY = 60.0       # s; another human replies within this window
RECHECK_EDGES = [2, 5, 10, 20, 60, 300]   # s -> 7 buckets (time until next event in the conversation)

def compute_labels(df: pd.DataFrame, horizon: float = None) -> pd.DataFrame:
    """horizon (shadow mode): unix time up to which the stream is known to be complete. When given, an event with
    no later event yet still gets eot/recheck labels once the silence since it reaches the window length."""
    df = df.sort_values(["conv", "ts"], kind="stable").reset_index(drop=True)
    n = len(df)
    y_eot = np.full(n, np.nan); y_self = np.full(n, np.nan); y_act = np.full(n, np.nan)
    y_rc = np.full(n, np.nan); y_hr = np.full(n, np.nan)
    conv = df.conv.to_numpy(); ts = df.ts.to_numpy(float); role = df.role.to_numpy()
    spk = df.speaker.to_numpy(object); known = df.speaker_known.to_numpy(bool)
    act = df.bot_act_src.to_numpy(object)
    starts = np.r_[0, np.flatnonzero(conv[1:] != conv[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(a, b):
            j = i + 1
            if j < b:
                gap = ts[j] - ts[i]
                y_rc[i] = np.searchsorted(RECHECK_EDGES, gap, side="right")
                # --- end of turn
                if known[i]:
                    if gap >= N_EOT: y_eot[i] = 1
                    elif known[j]: y_eot[i] = float(spk[j] != spk[i])
            elif horizon is not None:
                quiet = horizon - ts[i]
                if quiet >= RECHECK_EDGES[-1]: y_rc[i] = len(RECHECK_EDGES)
                if known[i] and quiet >= N_EOT: y_eot[i] = 1
            # --- self-continuation: defined on turn-final events; same speaker returns within W_SELF
            if known[i] and y_eot[i] == 1:
                k = i + 1; found = False; unknown_seen = False
                while k < b and ts[k] - ts[i] <= W_SELF:
                    if known[k] and spk[k] == spk[i]: found = True; break
                    if not known[k]: unknown_seen = True
                    k += 1
                if found: y_self[i] = 1
                elif not unknown_seen: y_self[i] = 0
                # (unknown-speaker events inside the window could be the same person -> masked)
            # --- human-reply aux: another human (not bot, not same speaker) within W_HREPLY
            if role[i] == "other" and known[i]:
                k = i + 1; found = False; unknown_seen = False
                while k < b and ts[k] - ts[i] <= W_HREPLY:
                    if known[k] and role[k] == "other" and spk[k] != spk[i]: found = True; break
                    if not known[k]: unknown_seen = True
                    k += 1
                if found: y_hr[i] = 1
                elif not unknown_seen: y_hr[i] = 0
            # --- bot action at inbound events
            if role[i] == "other" and isinstance(act[i], str):
                if act[i] == "speak": y_act[i] = 0
                else:
                    k = i + 1; later = False
                    while k < b and ts[k] - ts[i] <= W_WAIT:
                        if role[k] == "other" and act[k] == "speak": later = True; break
                        k += 1
                    y_act[i] = 1 if later else 2
    out = df.copy()
    out["y_eot"] = y_eot; out["y_self"] = y_self; out["y_act"] = y_act; out["y_recheck"] = y_rc; out["y_hreply"] = y_hr
    # addressed-to-bot: only inbound messages in GROUP chats (private chats are trivially addressed)
    out["y_addr"] = np.where((out.role == "other") & (out.conv_type == "group"), out.addr_src, np.nan)
    return out
