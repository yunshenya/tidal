"""VAP-style future-event projection targets (self-supervised; timestamps + relative roles only).

For every event i and each future bin (seconds after the event) we ask: does a given party post in that bin?
Parties are RELATIVE to event i, so the same objective works in any scenario (group chat, 1:1, livestream, call):
  ch 0  self      -- the bot / host posts
  ch 1  human     -- any non-self participant posts           (always defined)
  ch 2  current   -- the author of event i posts again        (needs speaker identity; masked when unknown)
  ch 3  others    -- a non-self participant other than i's author posts (masked when unknown)
For a self (bot) event, 'current' == self and 'others' == any human. Bins beyond the end of the observed stream are
masked. Output: float32 [N, 4*len(BINS)] with NaN = unknown (channel-major)."""
import numpy as np, pandas as pd

BINS = [(0, 2), (2, 5), (5, 15), (15, 60), (60, 180)]
CHANNELS = ["self", "human", "current", "others"]
NV = len(CHANNELS) * len(BINS)
def idx(ch, b): return CHANNELS.index(ch) * len(BINS) + b

def compute(d: pd.DataFrame, end_ts=None) -> np.ndarray:
    """d sorted by (conv, ts) with columns conv, ts, role, speaker, speaker_known.
    end_ts: optional {conv: last time the stream is known to be complete}; default = last event of the conv."""
    n = len(d); V = np.full((n, NV), np.nan, np.float32)
    conv = d.conv.to_numpy(); ts = d.ts.to_numpy(float); bot = (d.role == "self").to_numpy()
    known = d.speaker_known.to_numpy(bool) & ~bot; spk = d.speaker.to_numpy(object)
    starts = np.r_[0, np.flatnonzero(conv[1:] != conv[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        t = ts[a:b]; m = b - a; isb = bot[a:b]; hk = known[a:b] & ~isb; hu = ~known[a:b] & ~isb
        cb, chk, chu = (np.r_[0, np.cumsum(x)] for x in (isb, hk, hu))
        codes = pd.Series(spk[a:b]).where(hk).astype(object)
        code_id, uniq = pd.factorize(codes)                     # -1 for unknown / bot
        pos_by_code = {c: np.flatnonzero(code_id == c) for c in range(len(uniq))}
        end = (end_ts or {}).get(conv[a], t[-1])
        ii = np.arange(m)
        for k, (lo_s, hi_s) in enumerate(BINS):
            lo = np.maximum(ii + 1, np.searchsorted(t, t + lo_s, "left")) if lo_s > 0 else ii + 1
            hi = np.searchsorted(t, t + hi_s, "left"); hi = np.maximum(hi, lo)
            nb = cb[hi] - cb[lo]; nhk = chk[hi] - chk[lo]; nhu = chu[hi] - chu[lo]; nh = nhk + nhu
            same = np.zeros(m)
            for c, P in pos_by_code.items():
                sel = P                                         # events authored by code c
                same[sel] = np.searchsorted(P, hi[sel], "left") - np.searchsorted(P, lo[sel], "left")
            ok = t + hi_s <= end                                # the whole bin has been observed
            v = np.full((m, 4), np.nan, np.float32)
            v[:, 0] = nb > 0; v[:, 1] = nh > 0
            # self events: current == self, others == any human
            v[isb, 2] = (nb > 0)[isb]; v[isb, 3] = (nh > 0)[isb]
            kh = hk                                             # known human authors
            v[kh, 2] = np.where(same[kh] > 0, 1, np.where(nhu[kh] > 0, np.nan, 0))
            v[kh, 3] = np.where(nhk[kh] - same[kh] > 0, 1, np.where(nhu[kh] > 0, np.nan, 0))
            v[~ok] = np.nan
            for ch in range(4): V[a:b, ch * len(BINS) + k] = v[:, ch]
    return V
