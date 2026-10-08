"""Script-based SYNTHETIC interaction streams for scenario-general training/evaluation (no LLM, no real data).

Every generator emits the same unified event frame as real data (see tidal/labels.py + tidal/features_g.py):
  conv, ts, role ('self' = the bot / host, 'other'), speaker, speaker_known, text (None), modality,
  n_participants (continuous context signal; optional), addr_src, bot_act_src, responds_to, initiate, msg_id,
  source ('syn_live' | 'syn_1on1'), scenario (evaluation bookkeeping ONLY -- never a model input),
  conv_type ('group' | 'private'; used ONLY by the label code to decide where 'addressed' is defined).
The numbers below are hand-set guesses of plausible dynamics, NOT fitted to any real stream. Everything produced here
is synthetic and must be labeled as such wherever it is reported."""
import math, random
import numpy as np, pandas as pd

def _zipf_pick(rng, n, s=1.1):
    w = 1.0 / np.arange(1, n + 1) ** s
    return int(rng.choices(range(n), weights=w)[0])

def livestream(seed=0, minutes=None):
    """One synthetic livestream: a host (self) talking in voice utterances, many viewers sending danmaku and gifts.
    Host picks SOME addressed danmaku to answer (responds_to), and fills dead air proactively (initiate=1)."""
    rng = random.Random(seed); nrng = np.random.default_rng(seed)
    minutes = minutes or rng.uniform(15, 40)
    viewers = int(math.exp(rng.uniform(math.log(30), math.log(3000))))
    chatters = max(5, int(viewers * rng.uniform(0.05, 0.25)))
    base_rate = min(1.2, 0.02 * chatters ** 0.8)            # danmaku / s
    dead_air = rng.uniform(12, 35)                           # host starts talking if silent this long
    T_end = minutes * 60; t = 0.0; ev = []; k = 0
    host_busy_until = 0.0; last_host = -1e9; pending = []    # addressed danmaku awaiting an answer
    next_dm = nrng.exponential(1 / base_rate)
    def add(**e):
        nonlocal k
        e.update(msg_id=f"L{seed}:{k}", conv=f"syn_live_{seed}"); ev.append(e); k += 1
        return e["msg_id"]
    while t < T_end:
        # next host decision point vs next danmaku
        t = min(next_dm, max(host_busy_until, last_host + dead_air) if not pending else host_busy_until + rng.uniform(1.5, 6))
        if t >= T_end: break
        if t == next_dm:
            spk = _zipf_pick(rng, chatters); kind = "gift" if rng.random() < 0.04 else "text"
            addressed = kind == "text" and rng.random() < 0.18
            mid = add(ts=t, role="other", speaker=f"v{spk}", modality=kind, addr_src=float(addressed), bot_act_src=None,
                      responds_to=None, initiate=0, n_participants=viewers)
            if addressed or (kind == "gift" and rng.random() < 0.5): pending.append((t, mid))
            # bursts: same viewer often follows up quickly
            if rng.random() < 0.15:
                add(ts=t + rng.uniform(0.8, 4), role="other", speaker=f"v{spk}", modality="text", addr_src=float(addressed),
                    bot_act_src=None, responds_to=None, initiate=0, n_participants=viewers)
            rate = base_rate * (2.5 if t - last_host < 8 else 1.0)       # chat reacts to the host
            next_dm = t + nrng.exponential(1 / rate)
        else:
            pending = [(pt, m) for pt, m in pending if t - pt < 40]        # stale questions are dropped
            initiate = 0; target = None
            if pending and rng.random() < 0.8:
                # host answers one pending message: prefers recent ones and gifts
                pt, target = max(pending, key=lambda x: x[0] + rng.uniform(0, 15)); pending = [p for p in pending if p[1] != target]
            elif t - last_host >= dead_air:
                initiate = 1
            else:
                host_busy_until = t + 1.0; continue
            n_utt = 1 + int(nrng.geometric(0.45)) - 1
            tt = t
            for u in range(n_utt):
                add(ts=tt, role="self", speaker="BOT", modality="voice", addr_src=np.nan, bot_act_src=None,
                    responds_to=target if u == 0 else None, initiate=initiate if u == 0 else 0, n_participants=viewers)
                tt += rng.uniform(2.0, 7.0)
            last_host = tt - rng.uniform(0.5, 1.5); host_busy_until = tt
    d = pd.DataFrame(ev).sort_values("ts", kind="stable").reset_index(drop=True)
    answered = set(d.responds_to.dropna())
    d.loc[d.role == "other", "bot_act_src"] = np.where(d.msg_id[d.role == "other"].isin(answered), "speak", "silent")
    d["conv_type"] = "group"; d["scenario"] = "livestream"; d["source"] = "syn_live"
    return d

def one_on_one(seed=0, n_sessions=None):
    """One synthetic 1:1 chat: a user sending bursts (often split messages), the bot answering after a short delay,
    sessions separated by long gaps; occasional images / voice notes."""
    rng = random.Random(10_000 + seed); ev = []; t = rng.uniform(0, 3600); k = 0
    for s in range(n_sessions or rng.randint(3, 8)):
        for turn in range(rng.randint(3, 15)):
            n = 1 + int(np.random.default_rng(seed * 997 + s * 31 + turn).geometric(0.5)) - 1
            ids = []
            for u in range(n):
                mod = rng.choices(["text", "image", "voice"], [0.85, 0.1, 0.05])[0]
                ev.append(dict(ts=t, role="other", speaker="user", modality=mod, addr_src=1.0, bot_act_src="silent",
                               responds_to=None, initiate=0)); ids.append(len(ev) - 1)
                t += rng.uniform(1.0, 9.0) if u < n - 1 else 0
            if rng.random() < 0.9:                                          # bot answers the burst
                ev[ids[-1]]["bot_act_src"] = "speak"
                t += rng.uniform(2.0, 12.0)
                for b in range(rng.choice([1, 1, 1, 2])):
                    ev.append(dict(ts=t, role="self", speaker="BOT", modality="text", addr_src=np.nan, bot_act_src=None,
                                   responds_to=ids[-1] if b == 0 else None, initiate=0)); t += rng.uniform(1.5, 5)
            t += rng.choice([rng.uniform(3, 30), rng.uniform(30, 240)])     # user's reply delay
        t += rng.uniform(1800, 6 * 3600)                                   # session gap
        if rng.random() < 0.2:                                              # bot-initiated check-in after silence
            ev.append(dict(ts=t, role="self", speaker="BOT", modality="text", addr_src=np.nan, bot_act_src=None,
                           responds_to=None, initiate=1)); t += rng.uniform(20, 600)
    d = pd.DataFrame(ev); d["msg_id"] = [f"P{seed}:{i}" for i in range(len(d))]
    d["responds_to"] = d.responds_to.map(lambda i: None if i is None or (isinstance(i, float) and np.isnan(i)) else f"P{seed}:{int(i)}")
    d["conv"] = f"syn_1on1_{seed}"; d["n_participants"] = 2; d["conv_type"] = "private"; d["scenario"] = "one_on_one"
    d["source"] = "syn_1on1"
    return d.sort_values("ts", kind="stable").reset_index(drop=True)

def build(n_live=24, n_1on1=240, seed=0):
    """All synthetic script streams with a per-stream split (train 70% / val 15% / test 15%)."""
    frames = [livestream(seed * 1000 + i) for i in range(n_live)] + [one_on_one(seed * 1000 + i) for i in range(n_1on1)]
    d = pd.concat(frames, ignore_index=True)
    d["speaker_known"] = True; d["text"] = None
    rng = np.random.default_rng(seed); convs = d.conv.unique(); u = dict(zip(convs, rng.random(len(convs))))
    d["split"] = d.conv.map(lambda c: "syn_train" if u[c] < 0.7 else ("syn_val" if u[c] < 0.85 else "syn_test"))
    return d
