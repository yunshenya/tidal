"""A backchannel tick whose 0.5 s future is past the recording is not stored as a negative."""
import numpy as np
from tidal.audio_train import STEP, events

def test_bc_incomplete_future_is_dropped():
    need = int(round(0.5 / STEP))
    T = need + 8
    vs = np.zeros(T, np.float32); vo = np.ones(T, np.float32); bco = np.zeros(T, np.float32)
    cv = {"S": [[], []], "L": [(vs, None, bco), (vo, None, None)], "T": T}
    bc = [e for e in events(cv, 0) if e[0] == "bc"]
    assert bc, "ticks with a full future window should still be labeled"
    assert all(k + need <= T for _, k, y, _ in bc)
    assert all(y == 0.0 for _, _, y, _ in bc)
    T2 = need - 1
    short = {"S": [[], []], "L": [(np.zeros(T2, np.float32), None, np.zeros(T2)), (np.ones(T2, np.float32), None, None)], "T": T2}
    assert [e for e in events(short, 0) if e[0] == "bc"] == []
