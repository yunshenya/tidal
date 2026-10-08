"""Vectorized equivalent of tidal.labels.compute_labels for streams where every speaker is known (public data).
Checked against the reference implementation in tests (identical outputs)."""
import numpy as np, pandas as pd
from tidal.labels import N_EOT, W_SELF, W_WAIT, W_HREPLY, RECHECK_EDGES

def compute_labels_fast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["conv", "ts"], kind="stable").reset_index(drop=True); n = len(df)
    assert df.speaker_known.astype(bool).all(), "fast labels need all speakers known"
    conv = pd.factorize(df.conv)[0]; ts = df.ts.to_numpy(float); role = df.role.to_numpy(); spk = pd.factorize(df.speaker.astype(str) + "\x00" + df.conv.astype(str))[0]
    act = df.bot_act_src.to_numpy(object); idx = np.arange(n)
    has_next = np.r_[conv[1:] == conv[:-1], False]
    gap = np.r_[ts[1:] - ts[:-1], np.inf]
    y_rc = np.where(has_next, np.searchsorted(RECHECK_EDGES, gap, side="right"), np.nan).astype(float)
    nxt_spk = np.r_[spk[1:], -1]
    y_eot = np.where(has_next, np.where(gap >= N_EOT, 1.0, (nxt_spk != spk).astype(float)), np.nan)
    # next event of the same speaker (spk codes are per conversation)
    s = pd.Series(idx).groupby(spk).shift(-1).to_numpy()
    ok = ~np.isnan(s); ns = np.where(ok, s, 0).astype(int)
    found = ok & (ts[ns] - ts <= W_SELF)
    # conversation end time: a negative is unknown until the whole W_SELF window has been seen
    seg = np.cumsum(np.r_[True, conv[1:] != conv[:-1]]) - 1
    mx = np.full(int(seg.max()) + 1, -np.inf); np.maximum.at(mx, seg, ts); end_ts = mx[seg]
    covered_self = end_ts - ts >= W_SELF
    y_self = np.where(y_eot == 1, np.where(found, 1.0, np.where(covered_self, 0.0, np.nan)), np.nan)
    # next human (other) event by a different speaker
    y_hr = np.full(n, np.nan); O = np.flatnonzero(role == "other")
    if len(O):
        so = spk[O]; co = conv[O]
        newrun = np.r_[True, (so[1:] != so[:-1]) | (co[1:] != co[:-1])]
        run_id = np.cumsum(newrun) - 1; run_start = O[newrun]; run_conv = co[newrun]
        nr = run_id + 1; valid = nr < len(run_start)
        tgt = np.where(valid, run_start[np.minimum(nr, len(run_start) - 1)], 0)
        valid &= np.where(valid, run_conv[np.minimum(nr, len(run_start) - 1)] == co, False)
        f = valid & (ts[tgt] - ts[O] <= W_HREPLY)
        covered = end_ts[O] - ts[O] >= W_HREPLY
        y_hr[O] = np.where(f, 1.0, np.where(covered, 0.0, np.nan))
    # bot action
    y_act = np.full(n, np.nan); isstr = np.array([isinstance(a, str) for a in act])
    sp = (role == "other") & isstr & (act == "speak")
    nxt_sp = np.full(n, -1); last = -1
    for i in range(n - 1, -1, -1):                     # next speak-labelled inbound event strictly after i (same conv)
        nxt_sp[i] = last if last >= 0 and conv[last] == conv[i] else -1
        if sp[i]: last = i
    m = (role == "other") & isstr
    later = (nxt_sp >= 0) & (ts[np.maximum(nxt_sp, 0)] - ts <= W_WAIT)
    covered_wait = end_ts - ts >= W_WAIT
    y_act[m] = np.where(sp[m], 0, np.where(later[m], 1, np.where(covered_wait[m], 2, np.nan)))
    out = df.copy()
    out["y_eot"] = y_eot; out["y_self"] = y_self; out["y_act"] = y_act; out["y_recheck"] = y_rc; out["y_hreply"] = y_hr
    out["y_addr"] = np.where((out.role == "other") & (out.conv_type == "group"), out.addr_src, np.nan)
    return out
