import numpy as np, pandas as pd, torch
from pathlib import Path
from tidal import corpus_fmt as CF
from tidal.labels import compute_labels
from tidal.public_data.fastlabels import compute_labels_fast
from tidal.audio_fe import AudioEncoder, AudioStream, logmel, stack_frames, p_shift, SR

EX = Path(__file__).resolve().parents[1] / "examples" / "corpus_example.txt"

def test_corpus_parser_example():
    sn = CF.parse(EX.read_text(encoding="utf-8"))
    assert [s.sid for s in sn] == ["demo-weather", "demo-greeting"]
    a = sn[0]; assert a.purpose == "离线参考" and a.roles == ["小潮", "阿明", "小红"]
    assert a.sources[0][1] == [12 * 60 + 30] and len(a.turns) == 4 and a.turns[3][2] is True
    ev = CF.to_events(sn, {"小潮"}, "demo")
    assert ev.role.tolist()[:4] == ["other", "self", "other", "self"]
    assert set(ev.speaker) <= {"s0", "s1", "s2"} and "小潮" not in " ".join(ev.speaker)
    assert ev.ts.iloc[3] - ev.ts.iloc[2] > 100          # omitted-turns gap is a time break
    assert ev.responds_to.iloc[1] == ev.msg_id.iloc[0] and pd.isna(ev.responds_to.iloc[3])

def test_fast_labels_equal_reference():
    r = np.random.default_rng(3); rows = []
    for c in range(6):
        t = 1.7e9
        for k in range(150):
            t += float(r.exponential(40)); spk = f"p{r.integers(4)}"
            rows.append(dict(conv=f"c{c}", conv_type="group", ts=t, role="self" if spk == "p0" else "other", speaker=spk,
                             speaker_known=True, text="x", addr_src=np.nan, bot_act_src=None))
    d = pd.DataFrame(rows); a = compute_labels(d); b = compute_labels_fast(d)
    for col in ("y_eot", "y_self", "y_act", "y_recheck", "y_hreply"):
        if col in a: assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True), col

def test_audio_stream_matches_batch():
    torch.manual_seed(0); m = AudioEncoder().eval(); r = np.random.default_rng(0)
    o = (r.standard_normal(SR * 2) * 0.05).astype(np.float32); s = (r.standard_normal(SR * 2) * 0.01).astype(np.float32)
    x = stack_frames(logmel(o), logmel(s)); mu, sd = x.mean(0), x.std(0) + 1e-4
    with torch.no_grad():
        ob, _ = m(torch.from_numpy(((x - mu) / sd)[None].astype(np.float32)))
    st = AudioStream(m, mu, sd); outs = []
    for k in range(0, len(o), SR // 10): outs.append(st.push(o[k:k + SR // 10], s[k:k + SR // 10]))
    last = outs[-1]
    # the stream has consumed all complete 20 ms steps; compare with the batch output at the same step
    T = len(x); assert abs(last["shift"] - float(p_shift(ob["vap"][0, T - 1]))) < 1e-4
    assert abs(last["vad_other"] - float(torch.sigmoid(ob["va"][0, T - 1, 1]))) < 1e-4
