"""Shadow report on a synthetic store: --boot 0 skips bootstrap CIs instead of crashing; P4 rows stay out of the
phase-1 comparison and only feed the p_interrupt summary."""
import json, time
import numpy as np, pytest
from tidal.shadow import report, store as S

HEADS = ["y_eot", "y_self", "y_addr", "y_hreply", "y_act", "y_recheck"]

def _probs(r, k):
    p = {h: float(r.random()) for h in ("y_eot", "y_self", "y_addr", "y_hreply")}
    a = r.random(3); p["y_act"] = (a / a.sum()).tolist(); b = r.random(7); p["y_recheck"] = (b / b.sum()).tolist()
    return p

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "STATE", tmp_path / "state"); monkeypatch.setattr(S, "DB", tmp_path / "state" / "shadow.db")
    c = S.connect(); r = np.random.default_rng(0); t0 = time.time() - 6 * 3600
    systems = ["model:T_x", "base_rate"]
    (S.STATE / "frozen.json").write_text(json.dumps(dict(
        strongest_baseline={"T": {h: "base_rate" for h in HEADS}},
        thresholds={"T": {h: {s: 0.5 for s in systems} for h in HEADS}})))
    S.set_meta(c, "predict_after", t0 - 1)
    for i in range(60):
        m, ts = f"m{i}", t0 + i * 120
        c.execute("insert into events(msg_id, conv, conv_type, ts, role) values(?,?,?,?,?)", (m, f"g{i % 3}", "group", ts, "self" if i % 6 == 5 else "other"))
        c.execute("insert into labels values(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (m, f"g{i % 3}", "group", ts, i % 2, (i // 2) % 2, int(i % 5 == 0), i % 3, i % 7, (i // 3) % 2, 1, ts, "t"))
        for s in systems:
            c.execute("insert into predictions values(?,?,?,?,?,?,?,?,?)", (m, "T", s, json.dumps(_probs(r, i)), ts, ts, 0.0, "r", "v"))
        c.execute("insert into predictions values(?,?,?,?,?,?,?,?,?)",
                  (m, "P4", "model:p4_m3_ablate_m2", json.dumps(dict(_probs(r, i), p_interrupt=float(r.random()))), ts, ts, 0.0, "r", "v"))
    c.commit()
    return c

def test_report_boot_zero_skips_cis(store):
    R = report.build(store, 0, time.time() + 1, None, 0)
    assert set(R["regimes"]) == {"T"}                                   # P4 is not a phase-1 comparison regime
    hr = R["regimes"]["T"]["heads"]["y_eot"]
    assert "f1_ci" not in hr["systems"]["model:T_x"]
    d = hr["paired"]["model:T_x"]["base_rate"]["d_roc_auc"]
    assert d["ci"] is None and np.isfinite(d["mean"])
    assert hr["paired"]["model:T_x"]["base_rate"]["d_brier_gain"]["ci"] is None
    assert R["regimes"]["T"]["heads"]["y_act"]["paired"]["model:T_x"]["base_rate"]["d_macro_f1"]["ci"] is None
    assert sum(s["n"] for s in R["p4_interrupt"].values()) == 60
    md = report.markdown(R)
    assert "P4 系统的打断头" in md and "ΔROC-AUC" in md

def test_report_main_boot_zero_cli(store, tmp_path):
    out = tmp_path / "r.md"; js = tmp_path / "r.json"
    report.main(["--boot", "0", "--since", "2000-01-01", "--out", str(out), "--json", str(js)])
    assert "ΔROC-AUC" in out.read_text() and json.loads(js.read_text())["regimes"]["T"]

def test_report_with_bootstrap_still_has_cis(store):
    R = report.build(store, 0, time.time() + 1, None, 20)
    d = R["regimes"]["T"]["heads"]["y_eot"]["paired"]["model:T_x"]["base_rate"]["d_roc_auc"]
    assert d["ci"] is not None and len(d["ci"]) == 2
