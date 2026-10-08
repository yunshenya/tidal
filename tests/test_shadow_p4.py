"""Shadow mode can build and step the phase-4 winner (m3_ablate_m2, text emotion on, speech emotion off) with the
phase-5 interrupt side head, and p_interrupt reaches the shadow outputs without changing any action."""
import json, sqlite3, time
import numpy as np, pytest, torch
from tidal import features_g as FG
from tidal.duplex import Event, TickModel, DuplexController, EventEncoder, TickInput, AudioFrame, SelfState
from tidal.model import BinHead
from tidal.shadow import p4
from tidal.shadow.p4 import WINNER, build, n_feat, step_events
from tidal.shadow.infer import load_phase4_winner

EMO = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.1]   # 8 probs + valence + arousal

def _events():
    return [Event(id="a", ts=1_700_000_000.0, role="other", emotion=EMO),
            Event(id="b", ts=1_700_000_012.0, role="self"),
            Event(id="c", ts=1_700_000_015.0, role="other")]

def test_winner_config_is_m2_ablation_text_emotion_only():
    assert WINNER["kind"] == "m3_ablate_m2"
    assert WINNER["text_emotion"] is True and WINNER["speech_emotion"] is False
    assert WINNER["emo_dim"] == 11 and WINNER["interrupt"] is True

def test_build_and_step_phase4_winner_with_interrupt_head():
    torch.manual_seed(0)
    m = build()
    assert m.kind == "m3_ablate_m2" and m.fproj.in_features == n_feat() == n_feat(WINNER)
    assert set(m.side_heads) == {"p_interrupt"}
    assert not any("side" in k or "nets" in k for k in m.state_dict())     # body checkpoint format unchanged
    enc = step_events(m, _events()[:2])
    assert enc.last["id"] == "b" and enc.last_affect == [float(x) for x in EMO]
    assert 0.0 < enc.last["p_interrupt"] < 1.0 and all("p_interrupt" in b for b in enc.buf)
    enc2 = step_events(load_phase4_winner(), _events())
    assert torch.isfinite(torch.tensor(enc2.last["p_eot"])) and 0.0 < enc2.last["p_interrupt"] < 1.0

def test_controller_logs_p_interrupt_without_changing_action():
    torch.manual_seed(1)
    m = build(); heads = m.side_heads
    tm = TickModel(enc_dim=m.vap[0].in_features)
    def run(side):
        m.side_heads = side
        ctl = DuplexController(tm, EventEncoder(m, np.zeros(FG.NB), np.ones(FG.NB))); outs = []
        for k, e in enumerate(_events()):
            outs.append(ctl.step(TickInput(e.ts, [e], AudioFrame(0.2, 0, 0, 0.1), None, SelfState(), 2.0)))
        outs.append(ctl.step(TickInput(1_700_000_015.1, [], None, None, SelfState(), 2.0)))
        return outs, ctl
    on, ctl = run(heads); off, _ = run({})
    for a, b in zip(on, off):
        assert a.action == b.action and a.probs == b.probs and a.address_event == b.address_event
        assert b.p_interrupt is None and 0.0 < a.p_interrupt < 1.0
    assert on[-1].p_interrupt == ctl.enc.last["p_interrupt"]          # quiet tick repeats the last event's value

def test_speech_emotion_is_rejected():
    with pytest.raises(ValueError, match="speech"):
        build({"speech_emotion": True})

def test_interrupt_head_from_another_backbone_is_refused(tmp_path):
    torch.manual_seed(2)
    b = BinHead(); st = {"y_interrupt": b.state_dict()}
    torch.save(dict(state=st), tmp_path / "siso.pt")                         # old phase-5 heads carry no 'feats' -> p5_h
    torch.save(dict(state=st, feats="p5_h"), tmp_path / "siso2.pt")
    torch.save(dict(state=st, feats=p4.HEAD_FEATS), tmp_path / "m2.pt")
    for bad in ("siso.pt", "siso2.pt"):
        with pytest.raises(ValueError, match="m3_ablate_m2"):
            build({"interrupt_heads": [str(tmp_path / bad)]})
    m = build({"interrupt_heads": [str(tmp_path / "m2.pt")] * 2})
    h = torch.randn(3, 128)
    assert torch.allclose(m.side_heads["p_interrupt"](h), torch.sigmoid(b(h)), atol=1e-6)   # mean over identical seeds

@torch.no_grad()
def test_p_interrupt_is_pad_invariant():
    torch.manual_seed(3)
    m = build(); x = torch.randn(2, 5, n_feat())
    full = p4.score_windows(m, x.numpy(), np.ones((2, 5), bool))
    n = 7
    xp = torch.cat([torch.randn(2, n, n_feat()), x], 1).numpy()            # nonzero junk in pad slots
    v = np.concatenate([np.zeros((2, n), bool), np.ones((2, 5), bool)], 1)
    pad = p4.score_windows(m, xp, v)
    for k in ("p_interrupt", "y_eot", "y_act", "p_vap"):
        assert np.allclose(full[k], pad[k], atol=1e-5), k
    # streaming: a pad step (valid=False) does not move the state, so the next event's p_interrupt is unchanged
    head = m.side_heads["p_interrupt"]
    o1, s1 = m.step(x[:, 0], None)
    _, s0 = m.step(torch.randn(2, n_feat()), None, valid=torch.zeros(2, dtype=torch.bool))
    o2, _ = m.step(x[:, 0], s0, valid=torch.ones(2, dtype=torch.bool))
    assert torch.allclose(head(o1["h"]), head(o2["h"]), atol=1e-5)

def test_cron_tick_logs_p_interrupt(tmp_path, monkeypatch):
    from tidal.shadow import run, store as S
    monkeypatch.setattr(S, "STATE", tmp_path / "state"); monkeypatch.setattr(S, "DB", tmp_path / "state" / "shadow.db")
    monkeypatch.setenv("TIDAL_SHADOW_P4", "1")
    t0 = time.time() - 4 * 3600
    ev = [dict(msg_id=f"q{i}", conv="g9", conv_type="group", ts=t0 + i * 20, role="self" if i % 5 == 4 else "other",
               speaker=None if i % 5 == 4 else f"p{i % 3}", text=f"消息{i}") for i in range(12)]
    p = tmp_path / "ev.jsonl"; p.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ev))
    monkeypatch.setenv("TIDAL_SHADOW_JSONL", str(p))
    assert run.main(["--source", "jsonl"]) == 0
    c = sqlite3.connect(S.DB)
    rows = c.execute("select msg_id, probs from predictions where regime=? and system=?", (p4.REGIME, p4.SYSTEM)).fetchall()
    assert len(rows) == 12
    for _, pr in rows:
        pr = json.loads(pr); assert 0.0 < pr["p_interrupt"] < 1.0 and len(pr["y_act"]) == 3
    st = json.loads(c.execute("select stats from runs").fetchone()[0])
    assert st["predictions_P4"] == 12 and "p4_error" not in st
    assert run.main(["--source", "jsonl"]) == 0                              # idempotent: nothing re-predicted
    assert c.execute("select count(*) from predictions where system=?", (p4.SYSTEM,)).fetchone()[0] == 12
