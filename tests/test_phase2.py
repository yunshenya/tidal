"""Phase-2 unit tests (synthetic only; no real data, no trained weights)."""
import numpy as np, pandas as pd, torch
from tidal import vap_targets as VT, features_g as FG, scenarios
from tidal.vap import VAPModel
from tidal.duplex import (EventFeaturizer, EventEncoder, TickModel, DuplexController, TickInput, Event, AudioFrame, SelfState, ACTIONS)

def _frame(rows):
    d = pd.DataFrame(rows, columns=["ts", "role", "speaker"]); d["conv"] = "c"; d["speaker_known"] = d.speaker.notna()
    return d

def test_vap_targets_channels_and_masking():
    d = _frame([(0, "other", "a"), (1, "self", "BOT"), (3, "other", "a"), (4, "other", "b"), (100, "other", None), (400, "other", "a")])
    V = VT.compute(d)
    assert V.shape == (6, VT.NV)
    assert V[0, VT.idx("self", 0)] == 1 and V[0, VT.idx("current", 1)] == 1 and V[0, VT.idx("others", 1)] == 1
    assert V[0, VT.idx("human", 0)] == 0
    assert np.isnan(V[3, VT.idx("current", 4)])          # unknown speaker inside the 60-180 s bin -> masked
    assert np.isnan(V[5]).all()                          # nothing observed after the last event

def test_causal_vap_model_streaming_equals_batch():
    torch.manual_seed(0); m = VAPModel(len(FG.FEAT_G)).eval(); x = torch.randn(1, 12, len(FG.FEAT_G))
    with torch.no_grad():
        full = m(x, torch.ones(1, 12, dtype=torch.bool)); st = None
        for t in range(12): o, st = m.step(x[:, t], st)
        assert torch.allclose(o["vap"], full["vap"][:, -1], atol=1e-5)
        cut = m(x[:, :7], torch.ones(1, 7, dtype=torch.bool))       # future inputs never change past outputs
        assert torch.allclose(cut["y_act"][:, -1], full["y_act"][:, 6], atol=1e-5)
    assert sum(p.numel() for p in m.parameters()) < 2_000_000

def test_event_featurizer_matches_batch_features():
    d = scenarios.one_on_one(5).sort_values(["conv", "ts"], kind="stable")
    d = d[d.conv == d.conv.iloc[0]].reset_index(drop=True); d["text"] = None; d["speaker_known"] = True
    G = FG.compute(d); f = EventFeaturizer(np.zeros(FG.NB), np.ones(FG.NB))
    S = np.stack([f(Event(str(i), r.ts, "self" if r.role == "self" else "other", r.speaker, r.modality), r.n_participants) for i, r in d.iterrows()])
    assert np.abs(S - G).max() < 1e-5

def test_optional_blocks_are_droppable_and_zero_means_unknown():
    Z = np.ones((4, 3, len(FG.FEAT_G)), np.float32); FG.drop_optional(Z, np.random.default_rng(0), 1.0, 1.0)
    assert (Z[:, :, FG.CTX_SL] == 0).all() and (Z[:, :, FG.MOD_SL] == 0).all() and (Z[:, :, :FG.NB] == 1).all()

def test_duplex_controller_tick():
    torch.manual_seed(0); enc = EventEncoder(VAPModel(len(FG.FEAT_G)), np.zeros(FG.NB), np.ones(FG.NB))
    ctl = DuplexController(TickModel(), enc)
    for k in range(30):
        evs = [Event(f"e{k}", 1.7e9 + k * .1, "other", "u", "text")] if k % 7 == 0 else []
        out = ctl.step(TickInput(1.7e9 + k * .1, evs, AudioFrame(vad_other=float(k % 5 == 0)), None, SelfState(speaking=k > 20), None))
        assert out.action in ACTIONS and abs(sum(out.probs.values()) - 1) < 1e-3
    assert out.address_event is not None

def test_coldstart_adapter_is_causal_in_label_time():
    from tidal import coldstart as C
    from tidal.dataset import HEADS
    from tidal.model import HEAD_DIMS
    n = 30; ts = np.arange(n) * 10.0; meta = pd.DataFrame(dict(ts=ts)); seg = np.arange(n)
    Y = np.zeros((n, len(HEADS)), np.float32); Y[:, HEADS.index("y_act")] = 2; Y[:, HEADS.index("y_recheck")] = 3
    rng = np.random.default_rng(0)
    lz = {h: rng.normal(size=(n, HEAD_DIMS[h])).astype(np.float32) for h in HEADS}
    a, _ = C.run_segment(meta, Y, seg, lz, {}, 0.2)
    Y2 = Y.copy(); Y2[0, HEADS.index("y_eot")] = 1                      # label of event 0 revealed at ts 0 + 20 s
    b, _ = C.run_segment(meta, Y2, seg, lz, {}, 0.2)
    k = HEADS.index("y_eot"); first = int(np.searchsorted(ts, ts[0] + C.HORIZON["y_eot"]))
    assert np.allclose(a["y_eot"][:first], b["y_eot"][:first])            # nothing changes before the label is known
    assert not np.allclose(a["y_eot"][first:], b["y_eot"][first:])
    z, _ = C.run_segment(meta, Y, seg, lz, {}, 0.0)                       # lr 0 == the un-adapted model
    assert np.allclose(z["y_eot"], 1 / (1 + np.exp(-lz["y_eot"][:, 0])), atol=1e-6)

def test_rule_controller_runs_on_tick_features():
    from tidal import duplex_sim as DS
    from tidal.duplex import N_TICK
    X = np.zeros((50, N_TICK), np.float32); out = DS.rule_controller(X)
    assert out.shape == (50,) and set(out.tolist()) <= set(range(len(ACTIONS)))
