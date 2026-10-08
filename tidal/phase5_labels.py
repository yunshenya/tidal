"""Phase-5 labels from timing and gold links only. No lexicons, no embedding thresholds.

Rules are the ones locked in reports/phase5_prereg.md. Do not retune them on outcomes.
"""
import hashlib
import math
import re

import numpy as np

OVERLAP_MIN = 0.05
YIELD_HOLD = 0.20
YIELD_QUIET = 0.40
BC_MAX = 1.0
BC_CONT = 0.05
WAIT_GAP = 0.20
TEXT_GAP = 8.0
RESP_WIN = 5.0
REPLY_WIN = 60.0
NONSPEECH_MARK = re.compile(r"^\[[^\]]*\]$")


def split_hash(conv):
    """Fixed conversation split: md5(conv) % 100. train <80, val <90, else test."""
    h = int(hashlib.md5(str(conv).encode()).hexdigest(), 16) % 100
    if h < 80:
        return "pub_train"
    if h < 90:
        return "pub_val"
    return "pub_test"


def nonspeech_span(speaker, text):
    """Annotation filter for MagicData: speaker 'none' or a whole-token bracket mark. Not a word list."""
    if speaker is None or str(speaker).strip().lower() == "none":
        return True
    t = "" if text is None else str(text).strip()
    return bool(t) and bool(NONSPEECH_MARK.match(t))


def _indexed(segments):
    """Drop empty spans. Returned list is start-sorted and carries the caller's index."""
    out = []
    for i, (s, e, sp) in enumerate(segments):
        s = float(s); e = float(e)
        if not math.isfinite(s) or not math.isfinite(e) or e < s:
            continue
        out.append((s, e, sp, i))
    out.sort(key=lambda t: (t[0], t[1], t[3]))
    return out


def speech_interrupt(segments):
    """Per-segment interrupt label, aligned to the input order (not start order).

    1 floor-taking barge-in, 0 wait or backchannel, NaN otherwise. Invalid spans stay NaN.
    """
    segs = _indexed(segments)
    n = len(list(segments)) if not isinstance(segments, list) else len(segments)
    # segments may be a generator; _indexed already consumed it. Recover n from indices.
    n = (max((s[3] for s in segs), default=-1) + 1) if not isinstance(segments, list) else len(segments)
    y = np.full(n, np.nan, np.float32)
    kind = np.array(["mask"] * n, dtype=object)
    for bi, (bs, be, bsp, i) in enumerate(segs):
        incumbents = []
        for bj, (a_s, a_e, a_sp, j) in enumerate(segs):
            if a_sp == bsp:
                continue
            if a_s <= bs < a_e and bs - a_s >= OVERLAP_MIN:
                incumbents.append(bj)
        if incumbents:
            j = max(incumbents, key=lambda k: (segs[k][0], segs[k][3]))
            a_s, a_e, a_sp, _aj = segs[j]
            resumes = False
            for _bk, (s2, _e2, sp2, _k) in enumerate(segs):
                if sp2 == a_sp and a_e < s2 <= a_e + YIELD_QUIET:
                    resumes = True
                    break
            if a_e <= be and (be - a_e) >= YIELD_HOLD and not resumes:
                y[i] = 1.0
                kind[i] = "barge"
            elif (be - bs) <= BC_MAX and a_e > be + BC_CONT:
                y[i] = 0.0
                kind[i] = "bc"
            else:
                kind[i] = "mask"
            continue
        # no qualifying overlap at onset. Wait only if nothing else is ongoing and the gap is wide.
        ongoing = False
        ended = []
        spoken = False
        for a_s, a_e, a_sp, _j in segs:
            if a_sp == bsp:
                continue
            spoken = True
            if a_s <= bs < a_e:
                ongoing = True
            if a_e <= bs:
                ended.append(a_e)
        if ongoing or not spoken or not ended:
            kind[i] = "mask"
            continue
        if bs >= max(ended) + WAIT_GAP:
            y[i] = 0.0
            kind[i] = "wait"
        else:
            kind[i] = "mask"
    return y, kind


def dyad_message_value(segments):
    """1 iff the other speaker's next segment is a non-backchannel floor take, else 0 once RESP_WIN is observed.

    Output is aligned to the input order.
    """
    segs = _indexed(segments)
    n_in = len(segments) if isinstance(segments, list) else (max((s[3] for s in segs), default=-1) + 1)
    y = np.full(n_in, np.nan, np.float32)
    if not segs:
        return y
    _y_int, kind_in = speech_interrupt([(s, e, sp) for s, e, sp, _i in segs])
    # kind_in is input-order of the cleaned list only when that list was the original. Recompute on segs' own order.
    kind = []
    # speech_interrupt on the already start-sorted cleaned triples returns labels in THAT order
    # because we pass a fresh list whose indices are 0..n-1.
    triples = [(s, e, sp) for s, e, sp, _i in segs]
    _yi, kind_sorted = speech_interrupt(triples)
    conv_end = max(e for _s, e, _sp, _i in segs)
    for bi, (ss, se, sp, i) in enumerate(segs):
        nxt = None
        for bk in range(bi + 1, len(segs)):
            if segs[bk][2] != sp:
                nxt = bk
                break
        if nxt is None:
            y[i] = 0.0 if conv_end >= se + RESP_WIN else np.nan
            continue
        os_, oe, osp, _ni = segs[nxt]
        knd = kind_sorted[nxt]
        barge_against = bool(ss <= os_ < se and os_ - ss >= OVERLAP_MIN and knd == "barge")
        within = se <= os_ <= se + RESP_WIN
        if knd == "bc":
            y[i] = 0.0 if conv_end >= se + RESP_WIN else np.nan
        elif barge_against or (within and knd != "mask"):
            y[i] = 1.0
        elif knd == "mask" and within:
            y[i] = np.nan
        else:
            y[i] = 0.0 if conv_end >= se + RESP_WIN else np.nan
    return y


def text_interjection(ts, speakers):
    """tg_ru only. 1 floor-taking insertion, 0 the other side continues, else NaN. No message text."""
    ts = np.asarray(ts, float)
    sp = np.asarray(speakers, object)
    n = len(ts)
    y = np.full(n, np.nan, np.float32)
    for i in range(1, n):
        if sp[i] == sp[i - 1]:
            continue
        if ts[i] - ts[i - 1] > TEXT_GAP:
            continue
        a, b = sp[i - 1], sp[i]
        later_a = [k for k in range(i + 1, n) if sp[k] == a]
        if not later_a:
            continue
        ka = later_a[0]
        kb = next((k for k in range(i + 1, n) if sp[k] == b), None)
        if ts[ka] - ts[i] <= TEXT_GAP and (kb is None or ka < kb):
            y[i] = 0.0
        elif (ts[ka] - ts[i] > TEXT_GAP) and kb is not None and (ts[kb] - ts[i] <= TEXT_GAP):
            y[i] = 1.0
    return y


def reply_message_value(ts, msg_ids, reply_to, minute_clock=False):
    """1 iff a later gold reply link points here. 0 once the conversation is observed long enough."""
    ts = np.asarray(ts, float)
    msg_ids = list(msg_ids)
    pointed = set()
    for r in reply_to:
        if r is None:
            continue
        if isinstance(r, float) and math.isnan(r):
            continue
        pointed.add(r)
    n = len(ts)
    y = np.full(n, np.nan, np.float32)
    last = ts[-1] if n else 0.0
    id_pos = {}
    for i, m in enumerate(msg_ids):
        id_pos.setdefault(m, i)
    # a reply counts only when the pointing message is later in this conversation
    pointed_later = set()
    for i, r in enumerate(reply_to):
        if r in id_pos and id_pos[r] < i:
            pointed_later.add(r)
    for i, m in enumerate(msg_ids):
        if m in pointed_later:
            y[i] = 1.0
        elif minute_clock:
            y[i] = 0.0 if (n - i - 1) >= 2 else np.nan
        else:
            y[i] = 0.0 if last - ts[i] >= REPLY_WIN else np.nan
    return y


def active_domains(state):
    """Domains whose dialogue state holds a non-empty value. No keyword list."""
    if not isinstance(state, dict):
        return None
    found = set()
    for dom, slots in state.items():
        if not isinstance(slots, dict):
            continue
        for v in slots.values():
            if v is None:
                continue
            if isinstance(v, str) and v.strip() == "":
                continue
            if isinstance(v, (list, tuple)) and len(v) == 0:
                continue
            found.add(dom)
            break
    return found


def domain_change_labels(states):
    """y=1 iff the active-domain set differs from the previous turn that has a state.

    A turn with no dialogue state is masked and does not break the chain. The first
    turn that has a state is 0 (there is no previous set to differ from).
    """
    y = np.full(len(states), np.nan, np.float32)
    prev = None
    seen = False
    for i, st in enumerate(states):
        cur = active_domains(st)
        if cur is None:
            continue
        if not seen:
            y[i] = 0.0
            seen = True
        else:
            y[i] = 1.0 if cur != prev else 0.0
        prev = cur
    return y


# Forward barge-in (reports/phase5_barge_prereg.md). Constants are locked there.
BARGE_H_SPEECH = 1.0
BARGE_H_TEXT = 8.0
BARGE_STEP = 0.5
BARGE_MIN_HOLD = 0.5
BARGE_SELF_CAP = 60.0


def forward_barge_speech(segments, self_speaker, rec_end=None, horizon=BARGE_H_SPEECH, step=BARGE_STEP, min_hold=BARGE_MIN_HOLD):
    """Decision points while `self_speaker` does not hold the floor.

    Returns a list of (t, y, floor_start). y is 1, 0, or NaN. Every t is strictly before the
    self onset a positive predicts, and before floor_end. `rec_end` defaults to the last segment end.
    """
    segs = [(float(s), float(e), sp) for s, e, sp in segments if e >= s and math.isfinite(s) and math.isfinite(e)]
    if rec_end is None:
        rec_end = max((e for _s, e, _sp in segs), default=0.0)
    out = []
    for s, e, sp in segs:
        if sp == self_speaker or e - s < min_hold:
            continue
        t = s + min_hold
        while t < e - 1e-9:
            if any(ss <= t < se and ssp == self_speaker for ss, se, ssp in segs):
                t += step
                continue
            covering = [(ss, se) for ss, se, ssp in segs if ssp != self_speaker and ss <= t < se]
            holder = max(covering, key=lambda z: z[0])
            if holder != (s, e):
                t += step
                continue
            onsets = [ss for ss, _se, ssp in segs if ssp == self_speaker and ss > t]
            u = min(onsets) if onsets else None
            know = min(e, t + horizon)
            if u is not None and u <= t + horizon and u < e:
                y = 1.0
            elif rec_end + 1e-9 >= know:
                y = 0.0
            else:
                y = float("nan")
            out.append((t, y, s))
            t += step
    return out


def forward_barge_text(ts, roles, horizon=BARGE_H_TEXT):
    """One decision per non-self point event. Returns (index, y) for each such event."""
    ts = np.asarray(ts, float)
    roles = np.asarray(roles, object)
    n = len(ts)
    out = []
    for i in range(n):
        if roles[i] == "self":
            continue
        t = ts[i]
        if i + 1 < n and ts[i + 1] - t <= horizon:
            y = 1.0 if roles[i + 1] == "self" else 0.0
        elif i + 1 < n and ts[i + 1] - t > horizon:
            y = 0.0
        else:
            y = float("nan")
        out.append((i, y))
    return out


def barge_self_speaker(segments):
    """Speaker of the earliest segment. Ties break on (start, end, speaker), matching phase 5."""
    if not segments:
        return None
    # (start, end) only: the same tie break as phase5_data._frame_from_segs, so `self` matches those rows
    return min(segments, key=lambda z: (float(z[0]), float(z[1])))[2]
