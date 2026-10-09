"""Train-only IDF reply features and strictly observed overlap timing features."""
from functools import lru_cache
import hashlib
import math
import re

import numpy as np
from tidal.task_heads import HeadEvent, MAX_CONTEXT, PAIR_NAMES, context_at, reply_features, tokens

IDF_DIM = 4096
REPLY_NAMES = PAIR_NAMES + ('idf_cosine', 'weighted_token_overlap', 'char_bigram_overlap', 'length_ratio',
                          'exact_candidate_mention', 'exact_query_mention', 'candidate_author_count', 'query_author_count',
                          'query_previous_similarity', 'candidate_same_as_query_previous_author')
TIMING_NAMES = ('log_self_run', 'incoming_active', 'observed_incoming_fraction', 'self_history_missing',
                'incoming_history_missing', 'log_since_self_previous_end', 'log_since_incoming_previous_end',
                'self_occupancy_10s', 'incoming_occupancy_10s', 'log_self_segments_10s', 'log_incoming_segments_10s',
                'log_previous_self_duration', 'log_previous_incoming_duration', 'log_self_segments_60s',
                'log_incoming_segments_60s', 'switch_fraction_last8')


def word_bucket(word):
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), 'little') % IDF_DIM


def fit_idf(texts):
    count = 0; df = np.zeros(IDF_DIM, np.int64)
    for text in texts:
        ids = {word_bucket(word) for word in tokens(text[:4096])}
        if ids: df[list(ids)] += 1
        count += 1
    if not count: raise ValueError('No training documents')
    return (np.log((1.+count)/(1.+df))+1.).astype(np.float32)


def mentions(text, name):
    if not name: return 0.
    return float(bool(re.search(r'(?<![\w])'+re.escape(name)+r'(?![\w])', text, flags=re.IGNORECASE)))


class ReplyFeatures:
    def __init__(self, idf):
        self.idf = np.asarray(idf, np.float32).copy()
        if self.idf.shape != (IDF_DIM,) or not np.isfinite(self.idf).all() or np.any(self.idf <= 0):
            raise ValueError('Invalid train-only IDF')
        self.idf.flags.writeable = False
        self.describe = lru_cache(maxsize=4096)(self._describe)

    def _describe(self, text):
        text = text.casefold()[:4096]; words = tokens(text)
        vec = {}
        for word in words:
            k = word_bucket(word); vec[k] = float(self.idf[k])
        norm = math.sqrt(sum(v*v for v in vec.values()))
        bigrams = frozenset(text[i:i+2] for i in range(max(0, len(text)-1)))
        return words, vec, max(norm, 1.), bigrams

    def similarity(self, a, b):
        _wa, va, na, _ca = self.describe(a); _wb, vb, nb, _cb = self.describe(b)
        return sum(weight*vb.get(k, 0.) for k, weight in va.items())/(na*nb)

    def build(self, context, draft, now, speaker=''):
        h = context_at(context, now); base, mask, ids = reply_features(h, draft, now, speaker)
        x = np.zeros((MAX_CONTEXT+1, len(REPLY_NAMES)), np.float32); x[:, :len(PAIR_NAMES)] = base
        qw, _qv, _qn, qc = self.describe(draft)
        previous = next((e for e in reversed(h) if speaker and e.speaker == speaker), None)
        for i, event in enumerate(h, 1):
            cw, _cv, _cn, cc = self.describe(event.text)
            union = qw | cw; intersection = qw & cw
            weighted = sum(float(self.idf[word_bucket(w)]) for w in intersection)/max(1., sum(float(self.idf[word_bucket(w)]) for w in union))
            extra = [self.similarity(draft, event.text), weighted, len(qc & cc)/max(1, len(qc | cc)),
                     min(len(draft), len(event.text))/max(1, max(len(draft), len(event.text))),
                     mentions(draft, event.speaker), mentions(event.text, speaker),
                     sum(e.speaker == event.speaker for e in h)/MAX_CONTEXT,
                     sum(e.speaker == speaker for e in h)/MAX_CONTEXT if speaker else 0.,
                     self.similarity(previous.text, event.text) if previous else 0.,
                     float(previous is not None and previous.speaker == event.speaker)]
            x[i, len(PAIR_NAMES):] = extra
        return x, mask, ids


def timing_features(history, incoming_start, self_start, decision, incoming_active, observed_incoming_duration):
    """history contains completed observed spans only: [self, incoming]."""
    if len(history) != 2 or not isinstance(incoming_active, (bool, np.bool_)):
        raise ValueError('Expected observed self/incoming histories and VAD status')
    if any(not math.isfinite(t) or t < 0 for t in (incoming_start, self_start, decision, observed_incoming_duration)):
        raise ValueError('Invalid observed timing')
    if abs(decision-incoming_start-.2) > 1e-6 or self_start > incoming_start-.5:
        raise ValueError('Wrong fixed overlap opportunity')
    if observed_incoming_duration > .2+1e-6 or (incoming_active and abs(observed_incoming_duration-.2) > 1e-6):
        raise ValueError('Future incoming duration forbidden')
    clean = []
    for channel in history:
        out = []
        for start, end in channel:
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or end > decision:
                raise ValueError('History must contain completed observed segments')
            out.append((start, end))
        clean.append(sorted(out, key=lambda span: (span[1], span[0])))
    own, other = clean
    own_last = own[-1] if own else None; other_last = other[-1] if other else None
    def occupancy(channel):
        spans = sorted((max(s, decision-10.), e) for s, e in channel if e > decision-10.)
        total = 0.; end = decision-10.
        for start, stop in spans:
            total += max(0., stop-max(start, end)); end = max(end, stop)
        return min(1., total/10.)
    past = sorted([(e, 0) for _, e in own]+[(e, 1) for _, e in other])[-8:]
    switches = sum(a[1] != b[1] for a, b in zip(past, past[1:]))/max(1, len(past)-1)
    own_spans = own+[(self_start, decision)]
    other_spans = other+[(incoming_start, incoming_start+observed_incoming_duration)] if observed_incoming_duration else other
    return np.array([math.log1p(decision-self_start), float(incoming_active), observed_incoming_duration/.2,
                     float(not own), float(not other), math.log1p(decision-own_last[1]) if own_last else 0.,
                     math.log1p(decision-other_last[1]) if other_last else 0.,
                     occupancy(own_spans), occupancy(other_spans),
                     math.log1p(sum(e >= decision-10. for _, e in own)), math.log1p(sum(e >= decision-10. for _, e in other)),
                     math.log1p(own_last[1]-own_last[0]) if own_last else 0.,
                     math.log1p(other_last[1]-other_last[0]) if other_last else 0.,
                     math.log1p(sum(e >= decision-60. for _, e in own)), math.log1p(sum(e >= decision-60. for _, e in other)), switches], np.float32)


def reference_observation(segments, point):
    """Convert gold segment boundaries to what could be observed at the fixed decision.

    Current active flags are observable VAD proxies; no future endpoint is returned.
    """
    t = point['t']; ch = point['incoming']; start = point['start']
    current = next(s for s in segments[ch] if abs(s[0]-start) < 1e-9)
    active = current[1] > t
    duration = min(current[1], t)-start
    history = [[(s, e) for s, e, _ in segments[c] if e <= t and not (c == ch and abs(s-start) < 1e-9)] for c in (1-ch, ch)]
    return dict(history=history, incoming_start=start, self_start=point['self_start'], decision=t,
                incoming_active=bool(active), observed_incoming_duration=max(0., duration))
