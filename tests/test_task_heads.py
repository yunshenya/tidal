"""Synthetic structural tests; public task accuracy is measured by the separate pipeline."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest
import torch

from tidal.task_heads import HeadEvent, reply_features, policy_features, OverlapStream, MAX_CONTEXT
from tidal.head_data import read_annotations, overlap_points
from tidal.head_train import loss_vector, disjoint, block_delta


def test_reply_future_is_not_a_candidate_and_null_is_always_present():
    history = [HeadEvent('older', 1., 'how to fix audio?', 'a'), HeadEvent('newer', 2., 'hello', 'b')]
    x, m, ids = reply_features(history, 'a: use this setting', 3.)
    xx, mm, ii = reply_features(history+[HeadEvent('future', 4., 'answer', 'c')], 'a: use this setting', 3.)
    assert np.array_equal(x, xx) and np.array_equal(m, mm) and ids == ii
    assert ids[0] is None and m[0] and x[0, 0] == 1.
    assert x[1, 3] == 1 and x[2, 3] == 0
    assert not m[len(ids):].any()


def test_context_bounds_duplicate_ids_and_invalid_times():
    history = [HeadEvent(str(i), float(i)) for i in range(100)]
    _, mask, ids = reply_features(history, '', 100.)
    assert len(ids) == MAX_CONTEXT+1 and ids[1] == '68' and mask.all()
    with pytest.raises(ValueError): reply_features([HeadEvent('a', 0.), HeadEvent('a', 1.)], '', 2.)
    with pytest.raises(ValueError): policy_features([], float('nan'))
    with pytest.raises(ValueError): policy_features([], 1., remaining_s=-1.)


def _annotation():
    return dict(task='action_policy', conversation='conv', decision_time=2.,
                context=[asdict(HeadEvent('a', 1., 'question'))], label='wait',
                label_source='human', annotator='reviewer', self_state={'speaking': False})


def test_human_provenance_future_prefix_unknown_and_conflicts(tmp_path):
    path = tmp_path/'labels.jsonl'; r = _annotation()
    path.write_text(json.dumps(r)+'\n')
    rows, stats = read_annotations(path); assert len(rows) == 1 and rows[0]['label'] == 1
    changed = dict(r, label='speak'); path.write_text(json.dumps(r)+'\n'+json.dumps(changed)+'\n')
    rows, stats = read_annotations(path); assert rows == [] and stats['conflicting_decisions'] == 1
    for changed in [dict(r, label_source='historical_behavior'), dict(r, annotator='pending'),
                    dict(r, prefix='future full utterance', prefix_available_at=3.),
                    dict(r, context=[asdict(HeadEvent('future', 3.))])]:
        path.write_text(json.dumps(changed)+'\n')
        with pytest.raises(ValueError): read_annotations(path)
    path.write_text(json.dumps(dict(r, label=None))+'\n'); assert read_annotations(path)[0] == []


def test_behavior_does_not_manufacture_policy_labels():
    from tidal.labels import task_label_contract
    d = pd.DataFrame({'y_act': [0., 2.], 'y_recheck': [1., 3.]})
    out = task_label_contract(d.copy()); assert out.y_act_policy.isna().all()
    assert np.array_equal(out.y_act_behavior, d.y_act) and np.array_equal(out.y_next_gap, d.y_recheck)
    d['policy_act_src'] = ['wait', None]; d['policy_label_source'] = ['human', None]
    assert task_label_contract(d).y_act_policy.iloc[0] == 1
    d.loc[0, 'policy_label_source'] = 'behavior'
    with pytest.raises(ValueError): task_label_contract(d)


def test_pointer_loss_accepts_multiple_targets_and_rejects_padding():
    z = torch.tensor([[[0.], [1.], [2.], [500.]]], requires_grad=True)
    mask = torch.tensor([[True, True, True, False]])
    gold = torch.tensor([[False, True, True, False]])
    loss = loss_vector(z, gold, mask); expected = -torch.log(torch.softmax(z[0, :3, 0], 0)[1:].sum())
    assert torch.allclose(loss[0], expected)
    loss.sum().backward(); assert z.grad[0, 3, 0] == 0.
    with pytest.raises(ValueError): loss_vector(z, torch.tensor([[False, False, False, True]]), mask)


def test_overlap_candidates_exist_before_their_outcome_is_known():
    # Robot has held the floor for >500ms; incoming onset, not incoming end, fixes t.
    segments = [[(2., 2.6, 'uh huh')], [(0., 4., 'incumbent')]]
    p = list(overlap_points(segments, 5.)); assert p[0]['t'] == 2.2 and p[0]['label'] == 0
    changed = [[(2., 4.5, 'different FUTURE complete text')], [(0., 4., 'incumbent')]]
    q = list(overlap_points(changed, 5.)); assert q[0]['t'] == p[0]['t'] and q[0]['label'] == 1
    assert list(overlap_points(changed, 4.1))[0]['label'] is None
    # No candidate when robot already stopped by the actual decision time.
    assert not list(overlap_points([[(2., 3., '')], [(0., 2.1, '')]], 4.))


class RecordingModel:
    def overlap(self, window, self_run, incoming_elapsed=.2):
        return {'mean': float(window.mean()), 'run': self_run}


def test_overlap_pcm_chunking_future_and_bounded_state():
    rng = np.random.default_rng(2); a = rng.normal(size=32000).astype(np.float32); b = a*.5
    streams = []
    for chunk in (32000, 1600, 711):
        s = OverlapStream(RecordingModel())
        for i in range(0, len(a), chunk): s.push(a[i:i+chunk], b[i:i+chunk])
        streams.append(s)
    scores = [s.score(2., 1.8, .2, True) for s in streams]
    assert scores[0] == scores[1] == scores[2]
    s = streams[-1]; s.push(a[:8000], b[:8000]); assert s.score(2., 1.8, .2, True) == scores[0]
    assert s.score(2., 1.8, .2, False) is None and s.score(2., 1.9, .2, True) is None
    assert len(s.frames) <= 100 and max(map(len, s.buffers)) < 400
    before = s.received.copy()
    with pytest.raises(ValueError): s.push(np.array([np.nan]), np.array([0.]))
    assert before == s.received


def test_splits_reject_shared_speakers_and_block_interval():
    with pytest.raises(ValueError): disjoint({'groups': np.array(['a']), 'speakers': ['u']}, {'groups': np.array(['b']), 'speakers': ['u']})
    y = np.array([0, 1, 0, 1]); p = np.array([[.9, .1], [.1, .9]]*2); b = 1-p
    delta = block_delta(p, b, y, np.array(['a', 'a', 'b', 'b']))
    assert delta['accuracy_improvement_ci'][0] == 1 and delta['nll_improvement_ci'][0] > 0


def test_optional_task_shadow_cannot_change_controller_actions():
    from tidal import features_g as FG
    from tidal.vap import VAPModel
    from tidal.duplex import EventEncoder, TickModel, DuplexController, TickInput, Event
    class Stub:
        def reset(self): pass
        def observe(self, ti): return {'reply_to': {'target': 'different'}, 'policy': {'action_policy': {'speak': 1.}}}
    torch.manual_seed(3); model = VAPModel(len(FG.FEAT_G)); tick = TickModel()
    def ctl(shadow): return DuplexController(tick, EventEncoder(model, np.zeros(FG.NB), np.ones(FG.NB)), task_shadow=shadow)
    a, b = ctl(None), ctl(Stub())
    for i in range(5):
        ti = TickInput(10.+i*.1, [Event(str(i), 10.+i*.1, 'other', text='question')])
        x, z = a.step(ti), b.step(ti)
        assert (x.action, x.address_event, x.stop_tts, x.request_content_for, x.probs) == (z.action, z.address_event, z.stop_tts, z.request_content_for, z.probs)
        assert z.task_shadow is not None
    class Broken(Stub):
        def observe(self, ti): raise ValueError('invalid prefix')
    b = ctl(Broken()); b.reset(); out = b.step(TickInput(11.))
    assert out.task_shadow_error == 'ValueError' and out.task_shadow is None


def test_irc_symmetric_future_links_do_not_become_targets(monkeypatch):
    from tidal.head_data import irc_data
    # row0 has only a future incoming edge: context, not an annotated negative.
    frame = pd.DataFrame(dict(id=[0, 1, 2, 3, 4], date=['day']*5,
        raw=['[10:00] <a> first', '[10:01] <b> root', '[10:02] <c> reply to first',
             '[10:03] <d> reply to two', '[10:04] <e> unlabelled'],
        connections=[[2], [1, 3], [0, 3], [1, 2], []]))
    monkeypatch.setattr(pd, 'read_parquet', lambda *a, **k: frame)
    d = irc_data('train')
    assert len(d['y']) == 3 and d['counts']['skipped_unannotated'] == 2
    assert d['y'][0, 0] and d['y'][0].sum() == 1  # explicit root
    assert d['y'][1, 1] and d['y'][1].sum() == 1  # backward link to first
    assert d['y'][2, 2] and d['y'][2, 3] and d['y'][2].sum() == 2


def test_annotation_split_loader_does_not_parse_test_labels(tmp_path):
    from tidal.head_data import annotation_split, human_data
    train = next(f'c{i}' for i in range(1000) if annotation_split(f'c{i}') == 'train')
    test = next(f'c{i}' for i in range(1000) if annotation_split(f'c{i}') == 'test')
    a = dict(_annotation(), conversation=train); b = dict(_annotation(), conversation=test, label='invalid_test_label')
    path = tmp_path/'a.jsonl'; path.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
    assert len(human_data(path, 'action_policy', 'train')['y']) == 1
    with pytest.raises(ValueError): human_data(path, 'action_policy', 'test')


def test_real_adapter_rejects_unavailable_text_before_history_mutation(monkeypatch):
    import tidal.head_shadow as HS
    from tidal.duplex import TickInput, Event
    class Fake:
        def __init__(self, directory): pass
        def policy(self, *a, **k): return None
        def reply(self, *a, **k): return None
    monkeypatch.setattr(HS, 'TaskHeadShadow', Fake)
    shadow = HS.ControllerTaskShadow('unused')
    ti = TickInput(2., [Event('a', 1., 'other', text='arrived')], draft_text='future draft', draft_available_at=3.)
    with pytest.raises(ValueError): shadow.observe(ti)
    assert len(shadow.history) == 0
    ti.draft_available_at = 2.; out = shadow.observe(ti)
    assert len(shadow.history) == 1 and out['reply_to'] is None
    with pytest.raises(ValueError): shadow.observe(TickInput(3., incoming_onset=4.))
    assert len(shadow.history) == 1


def test_overlap_semantic_labels_require_a_real_prefix(tmp_path):
    path = tmp_path/'a.jsonl'
    r = dict(_annotation(), task='overlap_intent', label='stop_request', self_state={'speaking': True})
    path.write_text(json.dumps(r)+'\n')
    with pytest.raises(ValueError): read_annotations(path)
    r.update(prefix='stop', prefix_available_at=2.)
    path.write_text(json.dumps(r)+'\n'); assert len(read_annotations(path)[0]) == 1


def test_overlap_score_due_uses_fixed_prefix_and_delivers_once():
    from tidal.task_heads import OverlapStream
    class M:
        def overlap(self, window, self_run, incoming_elapsed=.2): return {'mean': float(window.mean())}
    s=OverlapStream(M()); rng=np.random.default_rng(8); a=rng.normal(size=24000).astype(np.float32); b=a*.2
    s.push(a,b)
    assert s.score_due(1.19, 1., .5, True) is None
    first=s.score_due(1.2, 1., .5, True); assert first is not None and first['decision_time']==1.2
    assert s.score_due(1.3, 1., .5, True) is None
    other=s.score_due(1.3, 1.1, .5, True); assert other is not None and other['decision_time']==1.3
    s.reset(); s.push(a, b); assert s.score_due(1.2, 1., .5, True) is not None
