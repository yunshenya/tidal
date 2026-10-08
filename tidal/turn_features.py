"""Shared causal turn features; NumPy only, no training/model dependencies."""
import re
import numpy as np
from tidal.audio_spec import STACK, NMEL

BC_CHARS = set("嗯啊哦对是好行呃哈噢唔嗷诶欸的没错嘛呀噢")
TAG = re.compile(r"\[[^\]]*\]|[，。！？、,.!?…~\s]")

def is_bc(s):
    t = TAG.sub("", s[2]); return (s[1] - s[0]) <= 1.2 and 0 < len(t) <= 3 and set(t) <= BC_CHARS

CONTEXT = 150
TIMING_NAMES = (
    'log_segment_duration', 'log_since_self_end', 'self_history_missing',
    'self_occupancy_10s', 'other_occupancy_10s', 'log_self_segments_10s',
    'log_other_segments_10s', 'switch_fraction_last8', 'log_other_median_duration',
    'log_other_previous_gap', 'log_self_segments_60s', 'log_other_segments_60s',
)
PROSODY_NAMES = ('other_recent_mean', 'self_recent_mean', 'other_prev_mean',
                 'self_prev_mean', 'other_recent_std', 'self_recent_std',
                 'other_window_mean', 'self_window_mean')
EN_FEEDBACK = frozenset(('yes', 'yeah', 'yeahyeah', 'yeahum', 'yep', 'yup', 'no',
                         'okay', 'ok', 'right', 'sure', 'oh', 'ah', 'uhhuh',
                         'mmhmm', 'mhm', 'hmm', 'exactly', 'absolutely'))


def short_feedback(segment):
    if is_bc(segment):
        return True
    word = re.sub(r'[^a-z]', '', str(segment[2]).lower())
    return segment[1] - segment[0] <= 1.2 and word in EN_FEEDBACK


def _occupancy(segments, lo, hi):
    """Union, so overlapping annotations do not count speech twice."""
    total, end = 0., lo
    for start, stop, _ in sorted(segments):
        a, b = max(lo, start), min(hi, stop)
        if b > max(a, end):
            total += b - max(a, end)
            end = b
    return total / (hi - lo)


def timing_features(segments, self_ch, decision, duration):
    past = [[s for s in channel if s[1] <= decision] for channel in segments]
    own, other = past[self_ch], past[1-self_ch]
    own = sorted(own, key=lambda s: s[1]); other = sorted(other, key=lambda s: s[1])
    since_self = min(60., max(0., decision-own[-1][1])) if own else 60.
    recent = sorted([(s[1], ch) for ch, channel in enumerate(past) for s in channel])[-8:]
    switches = np.mean([a[1] != b[1] for a, b in zip(recent[:-1], recent[1:])]) if len(recent)>1 else 0.
    median = np.median([min(60., s[1]-s[0]) for s in other[-5:]]) if other else 0.
    gap = max(0., other[-1][0]-other[-2][1]) if len(other)>1 else 0.
    count = lambda channel, seconds: sum(decision-seconds < s[1] <= decision for s in channel)
    return np.array([
        np.log1p(min(60., duration)), np.log1p(since_self), float(not own),
        _occupancy(own, decision-10., decision), _occupancy(other, decision-10., decision),
        np.log1p(count(own, 10)), np.log1p(count(other, 10)), switches,
        np.log1p(median), np.log1p(min(60., gap)),
        np.log1p(count(own, 60)), np.log1p(count(other, 60)),
    ], np.float32)


def prosody_features(window):
    a = np.asarray(window, np.float32).reshape(CONTEXT, STACK, 2, NMEL).mean(1)
    return np.r_[a[-10:].mean((0, 2)), a[-50:-10].mean((0, 2)),
                 a[-10:].std((0, 2)), a.mean((0, 2))].astype(np.float32)


def window_cmvn(x):
    x = np.asarray(x, np.float32)
    return (x-x.mean(-2, keepdims=True))/np.maximum(x.std(-2, keepdims=True), .5)
