import numpy as np
import torch
from tidal.audio_turn import FRAME_READY, available_index, shift_events, TurnModel
from tidal.audio_fe import STEP


def test_audio_frame_availability():
    for k in (0, 1, 37, 10000):
        ready = k * STEP + FRAME_READY
        assert available_index(ready) == k
        assert available_index(ready - 1e-6) == k - 1
    assert available_index(.2) * STEP + FRAME_READY <= .2


def test_candidate_filter_does_not_peek_after_decision():
    # Resumption just after 200ms silence stays an eligible hold example.
    segs = [[(0., 1., '长句'), (1.205, 2., '继续说')], [(3., 4., '回答')]]
    assert list(shift_events(segs, 1, 5.))[0] == (1.2, 0., 1.)
    segs[0][1] = (1.195, 2., '继续说')
    assert all(t != 1.2 for t, _, _ in shift_events(segs, 1, 5.))


def test_unobserved_future_is_not_negative():
    segs = [[(0., 1., '长句')], [(3., 4., '回答')]]
    assert list(shift_events(segs, 1, 2.)) == []
    assert list(shift_events(segs, 1, 5.)) == [(1.2, 1., 1.)]


def test_turn_model_is_causal_and_backpropagates():
    torch.manual_seed(0)
    m = TurnModel().eval()
    x = torch.randn(2, 12, 160)
    duration = torch.zeros(2)
    loss = m(x, duration).sum()
    loss.backward()
    assert m.head[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        a, _ = m.encoder(x)
        altered = x.clone(); altered[:, 6:] += 10
        b, _ = m.encoder(altered)
    assert torch.allclose(a['h'][:, :6], b['h'][:, :6], atol=1e-6)


def test_transcript_bom_and_non_speech(tmp_path):
    from tidal.audio_train import read_txt
    p = tmp_path / 'channel.txt'
    p.write_text('\ufeff[0.000,1.000]\ts1\tMale\t你好\n[1.000,2.000]\ts1\tMale\t[noise]\n[3.000,2.000]\ts1\tMale\t坏时间\n', encoding='utf-8')
    assert read_txt(p) == [(0., 1., '你好')]


def test_cost_threshold_can_choose_wait():
    from tidal.audio_turn import select_threshold, decisions
    y = np.array([0., 1.]); p = np.array([.9, .1])
    threshold = select_threshold(y, p)
    assert np.isinf(threshold)
    assert decisions(y, p, threshold)['take_count'] == 0
    assert select_threshold(y, p[::-1]) == .9


def test_overlapping_other_segments_do_not_make_false_silence():
    segs = [[(0., 1., '长句'), (.9, 2., '重叠片段')], [(3., 4., '回答')]]
    assert all(t != 1.2 for t, _, _ in shift_events(segs, 1, 5.))


def test_audio_targets_use_frame_completion_time():
    from tidal.audio_train import labels
    # Frame 0 finishes at 35ms: speech started at 30ms is already current.
    va, future, bc = labels([(.030, .050, '嗯')], 40)
    assert va[0] == 1 and va[1] == 0
    # BC at 50ms lies in frame 0's next half second but not frame 1's past.
    _, _, bc = labels([(.050, .100, '嗯')], 40)
    assert bc[0] == 1 and bc[1] == 0


def test_audio_event_uses_latest_completed_frame():
    from tidal.audio_train import labels, events
    S = [[(0., 1., '长句')], [(3., 4., '回答')]]
    cv = dict(S=S, T=300, L=[labels(s, 300) for s in S])
    kind, k, y, _ = next(e for e in events(cv, 1) if e[0] == 'shift')
    assert y == 1
    assert k*STEP + FRAME_READY <= 1.2
    assert (k+1)*STEP + FRAME_READY > 1.2


def test_audio_future_windows_are_fully_observed():
    from tidal.audio_train import labels
    _, f, _ = labels([(0., .8, '长句')], 40)
    assert np.isfinite(f[29, 0])
    assert np.isnan(f[30, 0])


def test_backchannel_training_targets_are_censored():
    from tidal.audio_train import labels
    _, _, b = labels([], 40)
    assert b[14] == 0
    assert np.isnan(b[15:]).all()


def test_audio_loss_does_not_train_unknown_future_as_negative():
    from tidal.audio_train import audio_loss
    out = dict(va=torch.zeros(1,2,2,requires_grad=True), vap=torch.zeros(1,2,8,requires_grad=True), bc=torch.zeros(1,2,requires_grad=True))
    loss = audio_loss(out, torch.zeros(1,2,2), torch.full((1,2,8),float('nan')), torch.full((1,2),float('nan')))
    loss.backward()
    assert torch.isfinite(loss)
    assert out['va'].grad is not None
    assert out['vap'].grad is None and out['bc'].grad is None
