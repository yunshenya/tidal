"""Shadow mode can build and step the phase-4 winner: mamba3_siso, text emotion on, speech emotion off."""
import torch
from tidal.duplex import Event
from tidal.shadow.p4 import WINNER, build, n_feat, step_events
from tidal.shadow.infer import load_phase4_winner

def test_winner_config_is_siso_text_emotion_only():
    assert WINNER["kind"] == "mamba3_siso"
    assert WINNER["text_emotion"] is True and WINNER["speech_emotion"] is False
    assert WINNER["emo_dim"] == 11

def test_build_and_step_phase4_winner():
    m = build()
    assert m.kind == "mamba3_siso" and m.fproj.in_features == n_feat() == n_feat(WINNER)
    emo = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1]   # 8 probs + valence + arousal
    events = [
        Event(id="a", ts=1_700_000_000.0, role="other", emotion=emo),
        Event(id="b", ts=1_700_000_012.0, role="self"),
    ]
    enc = step_events(m, events)
    assert enc.last["id"] == "b" and enc.last_affect == [float(x) for x in emo]
    # a second pass with the same events is finite (streaming state, not a crash)
    enc2 = step_events(load_phase4_winner(), events)
    assert torch.isfinite(torch.tensor(enc2.last["p_eot"]))

def test_speech_emotion_is_rejected():
    try:
        build({"speech_emotion": True})
    except ValueError as e:
        assert "speech" in str(e)
    else:
        raise AssertionError("speech emotion must not be a phase-4 shadow input")
