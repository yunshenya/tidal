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

def test_incomplete_future_window_is_unknown_not_negative():
    """A missing reply / continuation / later bot turn is not a negative until the window has actually elapsed."""
    # human reply window is 60s; the log only covers 5s, and the next event is the bot (not another human)
    d = frame([(0, "other", "a"), (5, "self", "bot")])
    L = compute_labels(d)
    assert np.isnan(L.y_hreply.iloc[0])
    assert compute_labels(d, horizon=100).y_hreply.iloc[0] == 0
    # a real reply inside the window is positive even if the window is not otherwise complete
    d2 = frame([(0, "other", "a"), (5, "other", "b")])
    assert compute_labels(d2).y_hreply.iloc[0] == 1
    # end-of-turn is positive (gap >= 20s) but self-continuation looks 180s ahead — not yet observed
    d3 = frame([(0, "other", "a"), (25, "other", "b")])
    L3 = compute_labels(d3)
    assert L3.y_eot.iloc[0] == 1 and np.isnan(L3.y_self.iloc[0])
    assert compute_labels(d3, horizon=200).y_self.iloc[0] == 0
    # bot action: "silent" is not finalized until W_WAIT has been seen
    d4 = frame([(0, "other", "a"), (5, "other", "b")])
    d4["bot_act_src"] = ["silent", None]
    assert np.isnan(compute_labels(d4).y_act.iloc[0])
    assert compute_labels(d4, horizon=80).y_act.iloc[0] == 2

def test_vap_bin_past_observed_end_is_masked():
    from tidal import vap_targets as VT
    d = frame([(0, "other", "a"), (1, "other", "b")])
    V = VT.compute(d)
    # bin (0, 2] is not fully observed (stream ends at t=1) -> unknown, not a negative
    assert np.isnan(V[0, VT.idx("human", 0)])
    assert np.isnan(V[0, VT.idx("self", 0)])
