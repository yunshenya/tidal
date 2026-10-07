import numpy as np, pandas as pd
from tidal.labels import compute_labels

def frame(rows):
    d = pd.DataFrame(rows, columns=["ts", "role", "speaker"])
    d["conv"] = "c1"; d["conv_type"] = "group"; d["speaker_known"] = True; d["text"] = "x"
    d["addr_src"] = np.nan; d["bot_act_src"] = None
    return d

def test_eot_and_self():
    L = compute_labels(frame([(0, "other", "a"), (3, "other", "a"), (10, "other", "b"), (40, "other", "a")]))
    assert L.y_eot.tolist()[:3] == [0, 1, 1]
    assert L.y_self.iloc[1] == 1          # a comes back within 180 s after b interrupts

def test_horizon_closes_open_windows():
    d = frame([(0, "other", "a")])
    assert np.isnan(compute_labels(d).y_eot.iloc[0])
    L = compute_labels(d, horizon=1000)
    assert L.y_eot.iloc[0] == 1 and L.y_recheck.iloc[0] == 6
