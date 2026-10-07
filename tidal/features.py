"""Causal per-event features. Feature i uses ONLY events 0..i of its conversation (verified in tidal/audit.py).
Label-source columns (attention*, chosen_action, delivered, cur_addressed, cur_reply, addr_src, bot_act_src, tags)
are never read here."""
import re, math, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
TZ = timezone(timedelta(hours=8))
END_SENT = re.compile(r"[。！？!?…~～]$|\.\.\.$")
END_COMMA = re.compile(r"[，,：:、;；]$")
PARTICLE = re.compile(r"[吗呢吧啊嘛哈了呀哦啦哇耶噢嗯]$")
UNFIN = re.compile(r"(然后|还有|就是|而且|我跟你说|等下|等等|先|因为|所以|但是|不过|如果|那个|就|和|跟|的)$")
QUESTION = re.compile(r"[?？]|吗|呢|么|嘛|啥|什么|怎么|怎样|为什么|为何|是不是|对不对|好不好|行不行")
MEDIA_ONLY = re.compile(r"^(\s|\[图片\]|\[表情\]|\[动画表情\]|\[语音\]|\[视频\]|<CQ>|[\U0001F000-\U0001FAFF\u2600-\u27BF])+$")
from tidal.config import bot_name_pattern
NAME = re.compile(bot_name_pattern())
FEATURES = ["role_self", "conv_private", "has_text", "speaker_known", "log_gap_prev", "first_in_conv",
            "log_gap_same_spk", "gap_same_spk_known", "same_spk_as_prev", "same_spk_unknown", "log_burst",
            "log_since_bot", "bot_never", "log_act_60s", "log_act_600s", "tod_sin", "tod_cos", "log_len",
            "end_sent", "end_comma", "end_nopunct", "question", "end_particle", "unfinished_marker",
            "media_only", "has_at", "has_name", "has_url", "multiline", "prev_is_bot", "n_spk_recent"]
F = len(FEATURES)
def text_feats(t):
    if t is None or (isinstance(t, float) and math.isnan(t)): return [0.0] * 12
    s = t.strip()
    return [math.log1p(len(s)), float(bool(END_SENT.search(s))), float(bool(END_COMMA.search(s))),
            float(not END_SENT.search(s) and not END_COMMA.search(s) and not PARTICLE.search(s)),
            float(bool(QUESTION.search(s))), float(bool(PARTICLE.search(s))), float(bool(UNFIN.search(s))),
            float(bool(MEDIA_ONLY.match(s))), float("@" in s), float(bool(NAME.search(s))), float("<URL>" in s), float("\n" in s)]
def compute(df: pd.DataFrame) -> np.ndarray:
    """df sorted by (conv, ts). Returns [N, F] float32."""
    n = len(df); X = np.zeros((n, F), np.float32)
    conv = df.conv.to_numpy(); ts = df.ts.to_numpy(float); role = df.role.to_numpy()
    spk = df.speaker.to_numpy(object); known = df.speaker_known.to_numpy(bool); text = df.text.to_numpy(object)
    ctype = df.conv_type.to_numpy()
    starts = np.r_[0, np.flatnonzero(conv[1:] != conv[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        last_spk_t = {}; last_bot = None; burst = 0
        for i in range(a, b):
            hr = datetime.fromtimestamp(ts[i], TZ); h = hr.hour + hr.minute / 60
            gp = ts[i] - ts[i - 1] if i > a else 0.0
            if known[i] and spk[i] in last_spk_t: gs, gk = math.log1p(ts[i] - last_spk_t[spk[i]]), 1.0
            else: gs, gk = 0.0, 0.0
            if i > a and known[i] and known[i - 1]: same, unk = float(spk[i] == spk[i - 1]), 0.0
            else: same, unk = 0.0, float(i > a)
            burst = burst + 1 if (same == 1.0 and gp <= 20) else 1
            lo60 = np.searchsorted(ts[a:i], ts[i] - 60); lo600 = np.searchsorted(ts[a:i], ts[i] - 600)
            recent = [spk[k] for k in range(max(a, i - 10), i + 1) if known[k]]
            X[i] = [role[i] == "self", ctype[i] == "private", text[i] is not None and not (isinstance(text[i], float)),
                    known[i], math.log1p(gp), i == a, gs, gk, same, unk, math.log1p(burst),
                    math.log1p(ts[i] - last_bot) if last_bot is not None else 0.0, last_bot is None,
                    math.log1p(i - a - lo60), math.log1p(i - a - lo600), math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24),
                    *text_feats(text[i]), i > a and role[i - 1] == "self", len(set(recent)) / 10.0]
            if known[i]: last_spk_t[spk[i]] = ts[i]
            if role[i] == "self": last_bot = ts[i]
    return X

# ---- feature sets per evaluation regime (see reports/phase1.md §泄漏审计) ----
# Text/speaker availability in the real data is a by-product of the bot's own LLM turns (dialogue_samples exist only
# around bot engagement) => availability flags leak the bot-action label. They are therefore NEVER model inputs.
#   T  : timing + role only, all events (no text, no speaker identity)  -> clean on every event
#   TS : T + speaker + text features + bge embedding, sequences built only from text-available events
FEAT_T = ["role_self", "conv_private", "log_gap_prev", "first_in_conv", "log_since_bot", "bot_never",
          "log_act_60s", "log_act_600s", "tod_sin", "tod_cos", "prev_is_bot"]
FEAT_TS = FEAT_T + ["log_gap_same_spk", "gap_same_spk_known", "same_spk_as_prev", "log_burst", "log_len", "end_sent",
                    "end_comma", "end_nopunct", "question", "end_particle", "unfinished_marker", "media_only", "has_at",
                    "has_name", "has_url", "multiline"]
EXCLUDED = ["has_text", "speaker_known", "same_spk_unknown", "n_spk_recent"]
def cols(regime): return [FEATURES.index(f) for f in (FEAT_T if regime == "T" else FEAT_TS)]
