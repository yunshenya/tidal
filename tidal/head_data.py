"""Task datasets: human judgements, gold reply links, and causal audio weak outcomes."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from tidal.task_heads import (CLASSES, MAX_CONTEXT, HeadEvent, context_at, reply_features,
                              policy_features, overlap_features)
from tidal.public_data.manifest import DATA


def annotation_split(conv):
    bucket = int(hashlib.sha256(conv.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 70 else 'val' if bucket < 85 else 'test'


def read_annotations(path, split=None, task_filter=None):
    """Unknown labels are masked; conflicting duplicate judgements are masked as a group."""
    records = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip(): continue
        r = json.loads(line); task = r.get('task')
        if task_filter is not None and task != task_filter: continue
        if split is not None:
            conv = r.get('conversation')
            if not isinstance(conv, str) or not conv: raise ValueError('Conversation ID required')
            if annotation_split(conv) != split: continue
        if task not in CLASSES or task == 'overlap_outcome':
            raise ValueError(f'Line {line_no}: unsupported human task')
        if r.get('label_source') not in ('human', 'user_feedback') or not isinstance(r.get('annotator'), str) or not r['annotator'].strip():
            raise ValueError(f'Line {line_no}: human provenance required')
        if not isinstance(r.get('conversation'), str) or not r['conversation']:
            raise ValueError('Conversation ID required')
        t = float(r['decision_time'])
        context = [HeadEvent(**e) for e in r['context']]
        visible = context_at(context, t)
        if any(e.ts > t for e in context):
            raise ValueError('Future context is forbidden')
        prefix = r.get('prefix', '')
        if prefix:
            available = float(r['prefix_available_at'])
            if not math.isfinite(available) or available < 0 or available > t:
                raise ValueError('Future transcript prefix is forbidden')
        state = r.get('self_state', {})
        allowed = {'speaking', 'elapsed_s', 'remaining_s', 'incoming_elapsed_s'}
        if set(state)-allowed: raise ValueError('Unknown state fields')
        if 'speaking' in state and not isinstance(state['speaking'], bool): raise ValueError('speaking must be boolean')
        if task == 'overlap_intent' and (not state.get('speaking', False) or not prefix):
            raise ValueError('Overlap intent requires active own speech and an available transcript prefix')
        label = r.get('label')
        if label is not None and r['annotator'] == 'pending': raise ValueError('Replace pending with actual annotator ID')
        if label is not None and label not in CLASSES[task]: raise ValueError('Invalid class label')
        x = policy_features(visible, t, prefix=prefix, **state)
        records.append(dict(task=task, conv=r['conversation'], t=t, x=x,
                            label=None if label is None else CLASSES[task].index(label),
                            key=(task, r['conversation'], t)))
    groups = {}
    for r in records: groups.setdefault(r['key'], []).append(r)
    out = []; conflicts = 0
    for group in groups.values():
        labels = {r['label'] for r in group if r['label'] is not None}
        same_features = all(np.array_equal(group[0]['x'], r['x']) for r in group)
        if len(labels) > 1 or not same_features:
            conflicts += 1; continue
        if labels:
            row = dict(group[0], label=next(iter(labels))); out.append(row)
    return out, dict(raw_records=len(records), accepted_decisions=len(out), conflicting_decisions=conflicts)


def human_data(path, task, split):
    rows, _ = read_annotations(path, split=split, task_filter=task)
    return dict(x=np.array([r['x'] for r in rows], np.float32),
                y=np.array([r['label'] for r in rows], np.int64),
                groups=np.array([r['conv'] for r in rows], object))


_IRC = re.compile(r'^\[(\d\d):(\d\d)\]\s+<([^>]+)>\s*(.*)$')

def irc_data(split):
    """Gold adjacency is symmetric; only backward links / explicit self-roots are targets."""
    filename = 'validation' if split == 'val' else split
    df = pd.read_parquet(DATA/'irc_dis'/'ubuntu'/f'{filename}-00000-of-00001.parquet')
    xs, masks, targets, groups = [], [], [], []
    counts = dict(annotated_queries=0, skipped_unannotated=0, skipped_nonmessage=0,
                  with_visible_target=0, with_outside_target=0, roots=0)
    for date, frame in df.groupby('date', sort=True):
        past = []; previous_minute = -1; day_offset = 0
        for r in frame.sort_values('id').itertuples():
            match = _IRC.match(r.raw)
            if not match:
                counts['skipped_nonmessage'] += 1; continue
            hh, mm, nick, text = match.groups(); minute = int(hh)*60+int(mm)
            if minute < previous_minute: day_offset += 86400
            previous_minute = minute; now = day_offset + minute*60.
            links = {int(k) for k in r.connections if int(k) <= int(r.id)}
            history = context_at(past, now)
            if links:
                counts['annotated_queries'] += 1
                x, mask, ids = reply_features(history, text, now, nick)
                visible = {int(k): i for i, k in enumerate(ids) if k is not None}
                gold = np.zeros(MAX_CONTEXT+1, bool)
                for k in links:
                    if k == int(r.id) or k not in visible: gold[0] = True
                    else: gold[visible[k]] = True
                counts['roots'] += int(int(r.id) in links)
                counts['with_visible_target'] += int(gold[1:].any())
                counts['with_outside_target'] += int(any(k != int(r.id) and k not in visible for k in links))
                xs.append(x); masks.append(mask); targets.append(gold); groups.append(str(date))
            else:
                counts['skipped_unannotated'] += 1
            past.append(HeadEvent(str(int(r.id)), now, text, nick, 'other'))
            past = past[-MAX_CONTEXT:]
    if not xs: raise ValueError(f'No IRC {split} examples')
    return dict(x=np.array(xs), mask=np.array(masks), y=np.array(targets), groups=np.array(groups, object), counts=counts)


AUDIO_GROUPS = {
    'train': {'zh': ('A1012', 'A1091'), 'en': ('Group0006', 'Group0030')},
    'val': {'zh': ('A1102',), 'en': ('Group0046',)},
    'test': {'en': ('Group0078',)},
}

def overlap_points(segments, observed_until):
    """Candidates depend only on onsets and speech still active at onset+200ms.

    Future endpoints classify subsequent floor outcome; ambiguous observations stay masked.
    """
    for incoming in (0, 1):
        own = 1-incoming
        for start, end, _text in segments[incoming]:
            decision = start+.2
            if decision > observed_until or end <= start:
                continue
            incumbents = [s for s in segments[own] if s[0] <= start-.5 and s[1] > decision]
            if not incumbents: continue
            if any(s[0] < start < s[1] for s in segments[incoming]): continue
            floor = max(incumbents, key=lambda s: s[0]); floor_start, floor_end, _ = floor
            label = None
            if end-start <= 1. and floor_end > end+.05 and observed_until >= end+.05:
                label = 0
            elif end >= floor_end+.2 and observed_until >= max(end, floor_end+.4):
                resumed = any(floor_end < s[0] <= floor_end+.4 for s in segments[own])
                if not resumed: label = 1
            yield dict(t=decision, start=start, self_start=floor_start, incoming=incoming, label=label)


def overlap_data(split):
    from tidal.audio_spec import STEP, FRAME_READY, STACK, NMEL, stack_frames
    from tidal.audio_train import CACHE_VERSION
    xs, ys, groups, speakers = [], [], [], set()
    found_groups = set()
    counts = dict(candidates=0, ambiguous=0, insufficient_context=0)
    for language, names in AUDIO_GROUPS[split].items():
        source = DATA/('magicdata_ms' if language == 'zh' else 'magicdata_en')
        cache = DATA/'proc'/('audio_md' if language == 'zh' else 'audio_en')
        for path in sorted(cache.glob('*.npz')):
            conv = path.stem
            if conv.split('_')[0] not in names: continue
            found_groups.add((language, conv.split('_')[0]))
            with np.load(path) as z:
                if int(z.get('cache_version', -1)) != CACHE_VERSION: raise ValueError('Stale audio cache')
                x = stack_frames(z['m0'].astype(np.float32), z['m1'].astype(np.float32))
                seg = [json.loads(str(z['s0'])), json.loads(str(z['s1']))]
            found = [f'{language}:{p.stem.rsplit("_", 1)[-1]}' for p in (source/'TXT').glob(f'{conv}_0_*.txt')]
            if len(found) != 2: raise ValueError('Cannot identify speakers')
            speakers.update(found)
            observed = (len(x)-1)*STEP+FRAME_READY
            for p in overlap_points(seg, observed):
                counts['candidates'] += 1
                if p['label'] is None: counts['ambiguous'] += 1; continue
                k = int(np.floor((p['t']-FRAME_READY)/STEP+1e-9))
                if k < 49: counts['insufficient_context'] += 1; continue
                order = [p['incoming'], 1-p['incoming']]
                window = x[k-49:k+1].reshape(50, STACK, 2, NMEL)[:, :, order, :].reshape(50, -1)
                xs.append(overlap_features(window, p['t']-p['self_start']))
                ys.append(p['label']); groups.append(f'{language}:{conv}')
    expected = {(lang, group) for lang, names in AUDIO_GROUPS[split].items() for group in names}
    if found_groups != expected: raise ValueError(f'Missing audio groups: {expected-found_groups}')
    if not xs: raise ValueError(f'No overlap {split} examples')
    return dict(x=np.asarray(xs, np.float32), y=np.asarray(ys, np.int64), groups=np.asarray(groups, object),
                speakers=sorted(speakers), counts=counts)
