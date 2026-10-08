"""Causal features and dependency-light shadow inference for explicit task contracts."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
from functools import lru_cache
import json
import math
from pathlib import Path
import re

import numpy as np

VERSION = 1
MAX_CONTEXT = 32
HASH_DIM = 256
CLASSES = {
    'action_policy': ('speak', 'wait', 'silent'),
    'eot_semantic': ('incomplete', 'complete'),
    'recheck_policy': ('0_2s', '2_5s', '5_10s', '10_20s', '20_60s', '60_300s', '300s_plus'),
    'overlap_intent': ('backchannel', 'floor_claim', 'stop_request', 'correction', 'unrelated'),
    'overlap_outcome': ('backchannel', 'floor_claim'),
}
PAIR_NAMES = ('null', 'cosine', 'token_overlap', 'mentions_candidate', 'candidate_mentions_query',
              'same_speaker', 'log_gap', 'distance', 'candidate_length', 'query_length',
              'candidate_question', 'query_question')
OVERLAP_NAMES = tuple(f'{ch}_{stat}' for ch in ('incoming', 'self')
                      for stat in ('recent_mean', 'previous_mean', 'recent_std', 'previous_std', 'rise', 'spectral_std')) + ('log_self_run', 'incoming_elapsed')
POLICY_DIM = HASH_DIM * 3 + 8

@dataclass(frozen=True)
class HeadEvent:
    id: str
    ts: float                     # arrival time, not future segment completion
    text: str = ''
    speaker: str = ''
    role: str = 'other'


def context_at(events, now):
    if not math.isfinite(now) or now < 0:
        raise ValueError('Invalid decision time')
    events = list(events)
    ids = set()
    last = -math.inf
    for e in events:
        if not isinstance(e, HeadEvent) or not isinstance(e.id, str) or not e.id or e.id in ids:
            raise ValueError('Expected unique event IDs')
        if not math.isfinite(e.ts) or e.ts < 0 or e.ts < last:
            raise ValueError('Events must have ordered finite arrival times')
        if not isinstance(e.text, str) or not isinstance(e.speaker, str) or e.role not in ('self', 'other', 'current'):
            raise ValueError('Invalid event fields')
        last = e.ts; ids.add(e.id)
    return [e for e in events if e.ts <= now][-MAX_CONTEXT:]


@lru_cache(maxsize=4096)
def tokens(text):
    return frozenset(re.findall(r'\w+', text.casefold()))


@lru_cache(maxsize=4096)
def text_vector(text):
    """Stable signed hashing; no vocabulary fit, identity table, or remote encoder."""
    if not isinstance(text, str):
        raise ValueError('Expected text string')
    text = text.casefold()[:4096]
    units = re.findall(r'\w+', text) + [text[i:i+3] for i in range(max(0, len(text)-2))]
    out = np.zeros(HASH_DIM, np.float32)
    for unit in units:
        h = hashlib.blake2b(unit.encode(), digest_size=8).digest()
        out[int.from_bytes(h[:4], 'little') % HASH_DIM] += 1. if h[4] & 1 else -1.
    out = out / max(float(np.linalg.norm(out)), 1.)
    out.flags.writeable = False
    return out


def pair_features(query, candidate, now, distance):
    if candidate is None:
        return np.array([1.] + [0.] * (len(PAIR_NAMES)-1), np.float32)
    qv, cv = text_vector(query.text), text_vector(candidate.text)
    qt, ct = tokens(query.text), tokens(candidate.text)
    overlap = len(qt & ct) / max(1, len(qt | ct))
    return np.asarray([0., float(qv @ cv), overlap,
                       float(bool(candidate.speaker) and candidate.speaker.casefold() in qt),
                       float(bool(query.speaker) and query.speaker.casefold() in ct),
                       float(bool(query.speaker) and query.speaker == candidate.speaker),
                       math.log1p(now-candidate.ts), distance/MAX_CONTEXT,
                       math.log1p(len(candidate.text))/10., math.log1p(len(query.text))/10.,
                       float('?' in candidate.text or '？' in candidate.text),
                       float('?' in query.text or '？' in query.text)], np.float32)


def reply_features(context, draft, now, speaker=''):
    """Condition on an already available draft. Candidate 0 is always null."""
    if not isinstance(draft, str) or not isinstance(speaker, str): raise ValueError('Invalid draft or speaker')
    history = context_at(context, now)
    query = HeadEvent('__draft__', now, draft, speaker, 'self')
    x = np.zeros((MAX_CONTEXT+1, len(PAIR_NAMES)), np.float32)
    mask = np.zeros(MAX_CONTEXT+1, bool); mask[0] = True
    x[0] = pair_features(query, None, now, 0)
    for i, e in enumerate(history, 1):
        x[i] = pair_features(query, e, now, len(history)-i+1); mask[i] = True
    return x, mask, [None]+[e.id for e in history]


def policy_features(context, now, prefix='', speaking=False, elapsed_s=0., remaining_s=0., incoming_elapsed_s=0.):
    h = context_at(context, now)
    times = [elapsed_s, remaining_s, incoming_elapsed_s]
    if any(not math.isfinite(t) or t < 0 for t in times):
        raise ValueError('Invalid observed playback timing')
    if not isinstance(prefix, str):
        raise ValueError('Expected available transcript prefix')
    vecs = [text_vector(e.text) for e in h]
    mean = np.mean(vecs, axis=0) if h else np.zeros(HASH_DIM, np.float32)
    latest = vecs[-1] if h else np.zeros(HASH_DIM, np.float32)
    state = [len(h)/MAX_CONTEXT, math.log1p(now-h[-1].ts) if h else 0., float(bool(h)),
             float(speaking), math.log1p(elapsed_s), math.log1p(remaining_s),
             math.log1p(incoming_elapsed_s), float(bool(prefix))]
    return np.r_[latest, mean, text_vector(prefix), state].astype(np.float32)


def overlap_features(window, self_run, incoming_elapsed=.2):
    """Last 1s of complete stacked frames [incoming, self], not future segment ends."""
    w = np.asarray(window, np.float32)
    if w.shape != (50, 160) or not np.isfinite(w).all() or self_run < 0 or not math.isfinite(self_run):
        raise ValueError('Expected 50 finite complete stereo frames and observed self run')
    if abs(incoming_elapsed-.2) > 1e-6:
        raise ValueError('Outcome model is trained at exactly 200ms after incoming onset')
    w = w.reshape(50, 2, 2, 40)
    out = []
    for ch in (0, 1):
        recent = w[-10:, :, ch, :]; previous = w[-20:-10, :, ch, :]
        out.extend([recent.mean(), previous.mean(), recent.std(), previous.std(),
                    recent.mean()-previous.mean(), recent.mean((0, 1)).std()])
    return np.r_[out, math.log1p(self_run), incoming_elapsed].astype(np.float32)


class TaskHeadShadow:
    """Only optional scores; this class never returns a control action."""
    def __init__(self, directory):
        import onnxruntime as ort
        self.directory = Path(directory); self.models = {}; self.metadata = {}
        if not self.directory.is_dir(): raise FileNotFoundError(self.directory)
        for path in sorted(self.directory.glob('*_bundle.json')):
            meta = json.loads(path.read_text()); task = meta.get('task')
            expected = PAIR_NAMES if task == 'reply_to' else OVERLAP_NAMES if task == 'overlap_outcome' else None
            if task not in (*CLASSES, 'reply_to') or meta.get('schema_version') != VERSION or meta.get('shadow_only') is not True or meta.get('automatic_action_enabled') is not False:
                raise ValueError('Unsupported task bundle')
            if meta.get('classes') != list(CLASSES.get(task, ())):
                raise ValueError('Class schema mismatch')
            if expected and meta.get('feature_names') != list(expected):
                raise ValueError('Feature schema mismatch')
            if meta.get('feature_contract') != 'task_heads_v1' or meta.get('max_context') != MAX_CONTEXT or meta.get('hash_dim') != HASH_DIM:
                raise ValueError('Feature contract mismatch')
            name = meta['model_file']
            if Path(name).name != name or not name.endswith('.onnx'):
                raise ValueError('Invalid model filename')
            model_path = self.directory/name
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta['model_sha256']:
                raise ValueError('Model checksum mismatch')
            options = ort.SessionOptions(); options.intra_op_num_threads = 2; options.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(model_path), sess_options=options, providers=['CPUExecutionProvider'])
            expected_inputs = ['features', 'mask'] if task == 'reply_to' else ['features']
            if [i.name for i in sess.get_inputs()] != expected_inputs:
                raise ValueError('Model input schema mismatch')
            self.models[task] = sess; self.metadata[task] = meta

    def _score(self, task, features, mask=None):
        if task not in self.models:
            return None
        x = np.asarray(features, np.float32)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite features')
        feed = {'features': x[None]}
        if mask is not None: feed['mask'] = np.asarray(mask, bool)[None]
        p = np.asarray(self.models[task].run(None, feed)[0][0])
        n = MAX_CONTEXT+1 if task == 'reply_to' else len(CLASSES[task])
        if p.shape != (n,) or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1., atol=1e-5):
            raise ValueError('Invalid model probability output')
        return p

    def reply(self, context, draft, now, speaker=''):
        if draft is None: return None
        if not isinstance(draft, str): raise ValueError('Expected draft text')
        if not draft.strip(): return None
        x, m, ids = reply_features(context, draft, now, speaker)
        p = self._score('reply_to', x, m)
        if p is None: return None
        if np.any(p[~m] > 1e-6): raise ValueError('Invalid masked probabilities')
        return dict(target=ids[int(p.argmax())], probabilities={str(k) if k is not None else '__null__': float(p[i]) for i, k in enumerate(ids)},
                    shadow_only=True, draft_conditioned=True)

    def policy(self, context, now, **observed):
        x = policy_features(context, now, **observed); out = {}
        for task in ('action_policy', 'eot_semantic', 'recheck_policy', 'overlap_intent'):
            if task == 'overlap_intent' and (not observed.get('speaking', False) or not observed.get('prefix')): continue
            p = self._score(task, x)
            if p is not None: out[task] = dict(zip(CLASSES[task], map(float, p)))
        return out or None

    def overlap(self, window, self_run, incoming_elapsed=.2):
        p = self._score('overlap_outcome', overlap_features(window, self_run, incoming_elapsed))
        return None if p is None else dict(probabilities=dict(zip(CLASSES['overlap_outcome'], map(float, p))),
                                          weak_behavior_proxy=True, shadow_only=True)


class OverlapStream:
    """Bounded stereo PCM cache; scores only the fixed causal 200ms opportunity."""
    def __init__(self, shadow):
        self.shadow = shadow; self.reset()

    def reset(self, start_time=0.):
        from collections import deque
        if not math.isfinite(start_time) or start_time < 0: raise ValueError('Invalid stream origin')
        self.origin = start_time; self.frames = deque(maxlen=100)
        self.received = [0, 0]; self.buffers = [np.empty(0, np.float32), np.empty(0, np.float32)]; self.next_step = 0

    def push(self, incoming_pcm, self_pcm):
        from tidal.audio_spec import SR, WIN, HOP, STACK, numpy_logmel, stack_frames
        chunks = [np.asarray(p, np.float32) for p in (incoming_pcm, self_pcm)]
        if any(p.ndim != 1 or not np.isfinite(p).all() for p in chunks): raise ValueError('Invalid PCM')
        received = [self.received[i]+len(chunks[i]) for i in range(2)]
        if abs(received[0]-received[1]) > SR*5: raise ValueError('Channel skew exceeds five seconds')
        buffers = [np.concatenate([self.buffers[i], chunks[i]]) for i in range(2)]
        n = min(map(len, buffers)); count = max(0, (n-WIN)//HOP+1)//STACK*STACK
        if count:
            end = (count-1)*HOP+WIN
            m = [numpy_logmel(p[:end]).astype(np.float16).astype(np.float32) for p in buffers]
            x = stack_frames(*m)
            for row in x:
                self.frames.append((self.next_step, row.copy())); self.next_step += 1
            buffers = [p[count*HOP:] for p in buffers]
        self.buffers = buffers; self.received = received

    def score(self, now, incoming_onset, self_start, self_speaking):
        from tidal.audio_spec import STEP, FRAME_READY, SR
        if incoming_onset is None or self_start is None or not self_speaking: return None
        if any(not math.isfinite(t) or t < 0 for t in (now, incoming_onset, self_start)):
            raise ValueError('Invalid online onset timing')
        if abs(now-incoming_onset-.2) > 1e-6 or self_start > incoming_onset-.5:
            return None
        if now > self.origin+min(self.received)/SR+1e-9: return None
        rows = [(k, x) for k, x in self.frames if self.origin+k*STEP+FRAME_READY <= now+1e-9]
        if len(rows) < 50: return None
        rows = rows[-50:]
        if rows[-1][0]-rows[0][0] != 49: return None
        return self.shadow.overlap(np.stack([x for _, x in rows]), now-self_start)
