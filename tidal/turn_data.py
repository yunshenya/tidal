"""Causal turn examples and features shared by training and shadow inference.

Future segments may define targets; features use only segments ending by decision.
Public ASR segmentation and a short-feedback lexicon are weak labels, not human
judgements of when an assistant should speak.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tidal.audio_fe import FRAME_READY, STEP, STACK, NMEL, stack_frames
from tidal.audio_train import CACHE_VERSION, TAG, shift_events
from tidal.audio_turn import available_index
from tidal.public_data.manifest import DATA

from tidal.turn_features import (CONTEXT, TIMING_NAMES, PROSODY_NAMES, EN_FEEDBACK,
                                 short_feedback, timing_features, prosody_features, window_cmvn)

def examples(segments, channel, observed_until):
    return shift_events(segments, channel, observed_until,
                        bc_fn=short_feedback, symmetric=True)


@dataclass
class TurnData:
    rows: list
    features: np.ndarray
    y: np.ndarray
    windows: np.ndarray | None = None


def audio_data(groups):
    """Load only explicitly permitted groups, so validation selection cannot read test."""
    rows, features, windows, targets = [], [], [], []
    for language, names in groups.items():
        src = DATA/('magicdata_ms' if language=='zh' else 'magicdata_en')
        cache = DATA/'proc'/('audio_md' if language=='zh' else 'audio_en')
        manifest = json.loads((src/'_done.json').read_text())
        if manifest['ok'] != manifest['files']:
            raise ValueError(f'Incomplete {src}')
        for path in sorted(cache.glob('*.npz')):
            conv = path.stem; group = conv.split('_')[0]
            if group not in names:
                continue
            with np.load(path) as z:
                if int(z.get('cache_version', -1)) != CACHE_VERSION:
                    raise ValueError(f'Stale cache {path}')
                x = stack_frames(z['m0'].astype(np.float32), z['m1'].astype(np.float32))
                segments = [json.loads(str(z['s0'])), json.loads(str(z['s1']))]
            speakers = sorted(f'{language}:{p.stem.rsplit("_", 1)[-1]}' for p in (src/'TXT').glob(f'{conv}_0_*.txt'))
            if len(speakers) != 2:
                raise ValueError(f'Cannot verify speaker IDs in {conv}')
            observed = (len(x)-1)*STEP+FRAME_READY
            for channel in (0, 1):
                order = [1-channel, channel]
                xx = x.reshape(len(x), STACK, 2, NMEL)[:, :, order, :].reshape(len(x), -1)
                for t, y, duration in examples(segments, channel, observed):
                    k = available_index(t)
                    if k < CONTEXT-1:
                        continue
                    window = xx[k-CONTEXT+1:k+1]
                    feat = np.r_[timing_features(segments, channel, t, duration), prosody_features(window)]
                    rows.append(dict(conv=f'{language}:{conv}', group=f'{language}:{group}',
                                     speakers=speakers, ch=channel, t=t))
                    windows.append(window); features.append(feat); targets.append(y)
    if not rows:
        raise ValueError('No selected audio examples')
    found = {r['group'] for r in rows}
    expected = {f'{lang}:{g}' for lang, gs in groups.items() for g in gs}
    if found != expected:
        raise ValueError(f'Missing groups: {expected-found}')
    return TurnData(rows, np.asarray(features, np.float32), np.asarray(targets, np.float32),
                    np.asarray(windows, np.float32))


def candor_split(conv):
    bucket = int(hashlib.sha256(conv.encode()).hexdigest()[:8], 16)%100
    return 'train' if bucket<70 else 'val' if bucket<85 else 'test'


def candor_data(split):
    source = DATA/'candor_tt'/'data'/'train-00000-of-00001.parquet'
    frame = pd.read_parquet(source, columns=['conversation_id', 'channel', 'offset', 'duration', 'text'])
    frame = frame[frame.conversation_id.map(candor_split)==split]
    rows, features, targets = [], [], []
    for conv, df in frame.groupby('conversation_id', sort=True):
        channels = sorted(df.channel.unique())
        if len(channels)!=2:
            continue
        segments=[]
        for ch in channels:
            arr=[]
            for r in df[df.channel==ch].itertuples():
                if pd.isna(r.text) or not TAG.sub('', str(r.text)):
                    continue
                start, end = float(r.offset), float(r.offset+r.duration)
                if np.isfinite(start) and np.isfinite(end) and start>=0 and end>start:
                    arr.append((start, end, str(r.text)))
            segments.append(sorted(arr))
        observed=max((s[1] for ss in segments for s in ss), default=0.)
        for channel in (0, 1):
            for t, y, duration in examples(segments, channel, observed):
                rows.append(dict(conv=str(conv), ch=channel, t=t))
                features.append(timing_features(segments, channel, t, duration)); targets.append(y)
    if not rows:
        raise ValueError(f'No CANDOR {split} examples')
    return TurnData(rows, np.asarray(features, np.float32), np.asarray(targets, np.float32))


def assert_disjoint(*datasets, speakers=False):
    sets=[{speaker for row in d.rows for speaker in row['speakers']} if speakers
          else {r['conv'] for r in d.rows} for d in datasets]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            if a & b:
                raise ValueError(f'Overlapping split identities: {sorted(a & b)}')
