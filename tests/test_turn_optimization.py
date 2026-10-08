import json
from pathlib import Path
import numpy as np
import pytest
from tidal.turn_data import (TurnData, timing_features, short_feedback, examples,
                             window_cmvn, assert_disjoint, candor_split)
from tidal.turn_optimize import calibration, apply_calibration, verify_selection, digest


def test_future_segments_cannot_change_features():
    s = [[(0., 1., 'long statement')], [(.1, .5, 'answer')]]
    expected = timing_features(s, 1, 1.2, 1.)
    future = [s[0]+[(2., 20., 'future')], s[1]+[(1.3, 15., 'later')]]
    assert np.array_equal(expected, timing_features(future, 1, 1.2, 1.))


def test_relative_roles_do_not_depend_on_channel_identity():
    s = [[(0., 1., 'long statement')], [(.1, .5, 'answer')]]
    assert np.array_equal(timing_features(s, 1, 1.2, 1.), timing_features(s[::-1], 0, 1.2, 1.))


def test_overlapping_history_is_not_double_counted():
    s = [[(0., 4., 'a'), (2., 6., 'b')], []]
    f = timing_features(s, 1, 6.2, 4.)
    assert f[4] == pytest.approx(.6)
    assert f[2] == 1 and np.isfinite(f).all()


def test_feedback_lexicon_is_short_and_symmetric():
    assert short_feedback((0., .7, 'Yeah, um.'))
    assert not short_feedback((0., 3., 'yes'))
    assert not short_feedback((0., .7, 'a substantive sentence'))
    # Ignore the original owner's feedback before the listener's real turn.
    s = [[(0., 1., 'long sentence'), (1.4, 1.8, 'yeah')], [(2., 3., 'my new topic')]]
    assert next(examples(s, 1, 4.))[:2] == (1.2, 1.)


def test_cmvn_removes_stationary_device_gain():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(2,150,160)).astype(np.float32)
    gain=rng.normal(size=(1,1,160)).astype(np.float32)*4
    assert np.allclose(window_cmvn(x), window_cmvn(x+gain), atol=1e-5)
    assert np.isfinite(window_cmvn(np.zeros_like(x))).all()


def test_identity_guard_checks_speakers_not_only_conversations():
    def data(conv, speakers):
        return TurnData([dict(conv=conv, speakers=speakers)], np.zeros((1,12)), np.zeros(1))
    a,b=data('one',['a','b']),data('two',['a','c'])
    assert_disjoint(a,b)
    with pytest.raises(ValueError, match='Overlapping'):
        assert_disjoint(a,b,speakers=True)


def test_hash_split_is_stable_and_all_partitions_exist():
    convs=[f'conversation-{i}' for i in range(1000)]
    first={c:candor_split(c) for c in convs}
    assert first == {c:candor_split(c) for c in reversed(convs)}
    assert set(first.values()) == {'train','val','test'}


def test_calibration_preserves_ranking_and_reduces_overconfidence():
    rng=np.random.default_rng(0)
    z=np.linspace(-8,8,1000)
    y=(rng.random(len(z)) < 1/(1+np.exp(-z*.25))).astype(float)
    params=calibration(z,y); p=apply_calibration(z,params)
    assert np.all(np.diff(p)>0)
    before=np.mean(np.logaddexp(0,z)-y*z)
    adjusted=params['scale']*z+params['bias']
    after=np.mean(np.logaddexp(0,adjusted)-y*adjusted)
    assert after < before*.8


def test_frozen_selection_rejects_changed_weights(tmp_path,monkeypatch):
    import tidal.turn_optimize as mod
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    p=tmp_path/'weights';p.write_bytes(b'initial')
    frozen=dict(signature='sig',artifact_sha256={'weights':digest(p)})
    verify_selection(frozen,'sig')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen artifact changed'):
        verify_selection(frozen,'sig')
