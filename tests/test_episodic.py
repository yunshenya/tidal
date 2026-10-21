"""Episodic fact slots and their always-on Continuum readout. CPU only."""
import ast
import inspect

import numpy as np
import pytest
import torch

from tidal.episodic import EpisodicStore


def _ortho(d=8, n=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Q, _ = np.linalg.qr(X.T)
    return Q.T[:n].astype(np.float64)


def test_overwrite_same_name_rel():
    S = EpisodicStore(capacity=8, dk=4, dv=4)
    k0 = np.array([1.0, 0, 0, 0])
    v_old = np.array([1.0, 0, 0, 0])
    v_new = np.array([0.0, 1, 0, 0])
    assert S.write(k0, v_old, name="alice", rel="color")
    assert S.write(k0 * 0.5 + 0.1, v_new, name="alice", rel="color")
    assert len(S) == 1
    v, info = S.read(k0, mode="top1")
    assert info["top"] == ("alice", "color")
    assert np.allclose(v, v_new)
    assert not np.allclose(v, v_old + v_new)


def test_overwrite_replaces_key_no_shield():
    """Same identity with an orthogonal key replaces the slot. The old key is not restored."""
    S = EpisodicStore(capacity=4, dk=4, dv=4, overwrite_cos=0.99)
    k_old = np.array([1.0, 0, 0, 0])
    k_new = np.array([0.0, 1, 0, 0])
    v_new = np.array([0.0, 2.0, 0, 0])
    S.write(k_old, np.array([1.0, 0, 0, 0]), name="a", rel="r")
    S.write(k_new, v_new, name="a", rel="r")
    assert len(S) == 1
    assert np.allclose(S.slots[0].k, k_new)
    v, info = S.read(k_new, mode="top1")
    assert info["top"] == ("a", "r")
    assert np.allclose(v, v_new)


def test_identity_overwrite_beats_earlier_near_duplicate():
    S = EpisodicStore(capacity=8, dk=4, dv=4, overwrite_cos=0.9)
    k_dup = np.array([1.0, 0, 0, 0])
    k_id = np.array([0.0, 1, 0, 0])
    S.write(k_dup, np.array([1.0, 0, 0, 0]), name="other", rel="x")
    S.write(k_id, np.array([0.0, 1, 0, 0]), name="alice", rel="color")
    assert S.write(k_dup, np.array([9.0, 0, 0, 0]), name="alice", rel="color")
    assert len(S) == 2
    alice = next(s for s in S.slots if s.name == "alice")
    other = next(s for s in S.slots if s.name == "other")
    assert np.allclose(alice.k, k_dup)
    assert np.allclose(alice.v, [9, 0, 0, 0])
    assert np.allclose(other.v, [1, 0, 0, 0])


def test_overwrite_by_near_duplicate_key():
    S = EpisodicStore(capacity=8, dk=4, dv=4, overwrite_cos=0.9)
    k = np.array([1.0, 0, 0, 0])
    S.write(k, np.ones(4), name="a", rel="r1")
    S.write(k * 1.01, np.full(4, 2.0), name="b", rel="r2")
    assert len(S) == 1
    assert S.slots[0].name == "b"
    assert np.allclose(S.slots[0].v, 2.0)


def test_anonymous_slots_do_not_share_identity():
    S = EpisodicStore(capacity=4, dk=4, dv=4, overwrite_cos=0.9)
    assert S.write(np.eye(4)[0], np.ones(4))
    assert S.write(np.eye(4)[1], np.ones(4) * 2)
    assert len(S) == 2


def test_capacity_fifo_eviction():
    S = EpisodicStore(capacity=3, dk=2, dv=2)
    for i in range(5):
        th = 2 * np.pi * i / 5
        k = np.array([np.cos(th), np.sin(th)])
        assert S.write(k, k.copy(), name=f"n{i}", rel="r")
    assert len(S) == 3
    names = {s.name for s in S.slots}
    assert names == {"n2", "n3", "n4"}


def test_overwrite_refreshes_recency_for_eviction():
    S = EpisodicStore(capacity=2, dk=4, dv=4, overwrite_cos=1.0)
    keys = np.eye(4)
    S.write(keys[0], keys[0], name="a", rel="r")
    S.write(keys[1], keys[1], name="b", rel="r")
    S.write(keys[0] * 0.2, keys[2], name="a", rel="r")
    S.write(keys[3], keys[3], name="c", rel="r")
    assert {s.name for s in S.slots} == {"a", "c"}
    assert np.allclose(next(s for s in S.slots if s.name == "a").k, keys[0] * 0.2)


def test_softtopk_is_convex_combination_of_one_store():
    keys = _ortho(d=8, n=4, seed=1)
    S = EpisodicStore(capacity=16, soft_k=4, soft_temp=0.05)
    vals = [np.eye(8)[i] * (i + 1) for i in range(4)]
    for i, (k, v) in enumerate(zip(keys, vals)):
        S.write(k, v, name=f"n{i}", rel="fav")
    q = keys[2] + 0.01 * keys[0]
    v, info = S.read(q, mode="softtopk")
    assert info["top"] == ("n2", "fav")
    assert np.isclose(info["weights"].sum(), 1.0)
    recon = sum(w * S.slots[int(i)].v for w, i in zip(info["weights"], info["idx"]))
    assert np.allclose(v, recon)
    assert np.argmax(np.abs(v)) == 2
    assert abs(float(v[2]) - 3.0) < 0.5


def test_gate_skips_chat_noise_and_low_surprise():
    S = EpisodicStore(capacity=4, dk=3, dv=3, surprise_thr=0.5)
    k = np.ones(3)
    assert not S.write(k, k, name="x", rel="y", kind="chat")
    assert not S.write(k, k, name="x", rel="y", kind="noise")
    assert not S.write(k, k, name="x", rel="y", kind="fact", surprise=0.1)
    assert not S.write(np.zeros(3), k, name="x", rel="y", surprise=0.9)
    assert len(S) == 0
    assert S.write(k, np.array([1.0, 0, 0]), name="x", rel="y", kind="fact", surprise=0.9)
    assert not S.write(np.array([0.0, 1, 0]), np.array([0.0, 9, 0]), name="x", rel="y", kind="chat")
    assert len(S) == 1
    assert np.allclose(S.slots[0].v, [1, 0, 0])
    assert np.allclose(S.slots[0].k, k)


def test_empty_readout_is_zero():
    S = EpisodicStore(capacity=4, dk=4, dv=4)
    v, info = S.read(np.ones(4))
    assert v is None and info["n"] == 0
    assert np.array_equal(S.readout(np.ones(4)), np.zeros(4))


def test_group_scope_and_global_slots():
    S = EpisodicStore(capacity=4, dk=4, dv=4, soft_k=1)
    k = np.eye(4)
    S.write(k[0], k[0], name="n", rel="r", group="ga")
    S.write(k[1], k[1] * 2, name="n", rel="r", group="gb")
    S.write(k[2], k[2] * 3, name="g", rel="r", group=None)
    assert len(S) == 3
    va, ia = S.read(k[0], mode="top1", group="ga")
    vb, ib = S.read(k[1], mode="top1", group="gb")
    assert ia["top"] == ("n", "r") and np.allclose(va, k[0])
    assert ib["top"] == ("n", "r") and np.allclose(vb, k[1] * 2)
    # group ga does not retrieve gb's value
    only_b, _ = S.read(k[1], mode="top1", group="ga")
    assert not np.allclose(only_b, k[1] * 2)
    vg, ig = S.read(k[2], mode="top1", group="ga")
    assert ig["top"] == ("g", "r") and np.allclose(vg, k[2] * 3)


def test_episodic_module_does_not_import_an_external_lm():
    import tidal.episodic as ep
    tree = ast.parse(inspect.getsource(ep))
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.extend(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module.split(".")[0])
    assert {"transformers", "neuromem", "smollm", "huggingface_hub"}.isdisjoint(mods)


def _tiny():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum, _cfg, CFG_DEFAULT
    torch.manual_seed(0)
    m = VAPModel(8, d=32, hidden=32, layers=1, dropout=0).eval()
    return m, Continuum, _cfg, CFG_DEFAULT


def test_cfg_has_no_enable_switch_and_empty_heads_match():
    m, Continuum, _cfg, CFG_DEFAULT = _tiny()
    assert "enable_episodic" not in CFG_DEFAULT and "episodic_scale" not in CFG_DEFAULT
    c = Continuum(m, _cfg(dict(levels=("med",), episodic_cfg=dict(capacity=8, soft_k=2, soft_temp=0.05, overwrite_cos=0.9))))
    assert c.episodic is not None and len(c.episodic) == 0
    assert c.episodic.capacity == 8 and c.episodic.soft_k == 2
    c.new_session()
    h = torch.randn(2, c.d)
    lv = c.active("g0")
    hp = h + sum(l.delta(h) for l in lv)
    with torch.no_grad():
        o_ref = {k: mod(hp) for k, mod in m.heads.items()}
        o_ref["vap"] = m.vap(hp)
        o = c.heads(h, lv)
    for k in o:
        assert torch.allclose(o[k], o_ref[k], atol=1e-6), k


def test_disable_keys_do_not_turn_readout_off():
    m, Continuum, _cfg, _ = _tiny()
    cfg = _cfg(dict(levels=()))
    cfg["enable_episodic"] = False
    cfg["episodic_scale"] = 0.0
    c = Continuum(m, cfg)
    h = torch.zeros(1, c.d)
    h[0, 0] = 1.0
    with torch.no_grad():
        empty = c.heads(h, [])["vap"].clone()
    assert c.episodic.write(h[0].numpy(), np.full(c.d, 0.5), name="u", rel="r")
    with torch.no_grad():
        full = c.heads(h, [])["vap"]
    assert not torch.allclose(empty, full)


def test_remember_event_and_bge_pathway():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum, _cfg
    torch.manual_seed(1)
    m = VAPModel(4, d=16, hidden=16, layers=1, dropout=0).eval()
    m.tproj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(512, 16))
    c = Continuum(m, _cfg(dict(levels=())))
    rng = np.random.default_rng(0)
    feats = rng.normal(size=4).astype(np.float32)
    bge = rng.normal(size=512).astype(np.float32)
    h1 = c.embed_event(feats, bge)
    h2 = c.embed_event(feats, bge)
    h3 = c.embed_event(feats, bge + 0.5)
    assert h1.shape == (c.d,)
    assert np.allclose(h1, h2)
    assert not np.allclose(h1, h3)
    bare = VAPModel(4, d=16, hidden=16, layers=1, dropout=0).eval()
    c2 = Continuum(bare, _cfg(dict(levels=())))
    with pytest.raises(ValueError, match="tproj"):
        c2.embed_event(np.zeros(4, np.float32), np.zeros(512, np.float32))
    with pytest.raises(ValueError, match="fproj"):
        c2.embed_event(np.zeros(3, np.float32))
    with pytest.raises(ValueError, match="512"):
        c2.remember(np.ones(512), np.ones(512), name="a", rel="b")
    kf = np.zeros(4, np.float32); kf[0] = 1
    vf = np.zeros(4, np.float32); vf[1] = 1
    q = torch.from_numpy(c2.embed_event(kf).astype(np.float32)).reshape(1, -1)
    with torch.no_grad():
        before = c2.heads(q, [])["vap"].clone()
    assert c2.remember_event(kf, vf, name="ada", rel="color", group="g")
    c2._mem_group = "g"
    with torch.no_grad():
        after = c2.heads(q, [])["vap"]
    assert len(c2.episodic) == 1
    assert not torch.allclose(before, after)
    c2.new_session("g")
    assert len(c2.episodic) == 1


def test_heads_readout_is_group_scoped():
    m, Continuum, _cfg, _ = _tiny()
    c = Continuum(m, _cfg(dict(levels=())))
    d = c.d
    qa = np.zeros(d); qa[0] = 1
    qb = np.zeros(d); qb[1] = 1
    va = np.zeros(d); va[2] = 3
    vb = np.zeros(d); vb[3] = 4
    c.episodic.write(qa, va, name="n", rel="r", group="a")
    c.episodic.write(qb, vb, name="n", rel="r", group="b")
    assert len(c.episodic) == 2
    h = torch.from_numpy(qa.astype(np.float32)).reshape(1, -1)
    with torch.no_grad():
        c._mem_group = "a"
        oa = c.heads(h, [])["vap"]
        c._mem_group = "b"
        ob = c.heads(h, [])["vap"]
        expect_a = m.vap(h + torch.from_numpy(va.astype(np.float32)).reshape(1, -1))
        expect_b = m.vap(h + torch.from_numpy(vb.astype(np.float32)).reshape(1, -1))
    assert torch.allclose(oa, expect_a, atol=1e-5)
    assert torch.allclose(ob, expect_b, atol=1e-5)


def test_update_sees_episodic_residual_and_still_learns():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum, _cfg
    torch.manual_seed(2)
    m = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    c = Continuum(m, _cfg(dict(levels=("med",), surprise=False)))
    c.new_session("g")
    h = torch.zeros(1, c.d)
    h[0, 0] = 1
    assert c.episodic.write(h[0].numpy(), np.full(c.d, 0.5), name="u", rel="r", group="g")
    seen = []
    def hook(_mod, inp, _out):
        seen.append(inp[0].detach().clone())
    handle = m.vap.register_forward_hook(hook)
    v = torch.full((1, 20), float("nan"))
    v[0, 0] = 1
    c._mem_group = "g"
    lv = c.active("g")
    try:
        loss = c.update(h, v, lv)
    finally:
        handle.remove()
    assert np.isfinite(loss)
    assert torch.allclose(seen[0], h + 0.5, atol=1e-5)
    assert float(lv[0].B.abs().sum() + lv[0].b.abs().sum()) > 0


def test_replay_empty_store_matches_static_and_group_fact_stays_put():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum, replay, _cfg
    torch.manual_seed(0)
    m = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    rng = np.random.default_rng(0)
    d = m.vap[0].in_features
    n = 4
    H = rng.normal(size=(n, d)).astype(np.float32)
    V = np.zeros((n, 20), np.float32)
    ts = np.array([0.0, 1.0, 0.5, 1.5])
    conv = np.array(["a", "a", "b", "b"])
    c1 = Continuum(m, _cfg(dict(levels=())))
    s1, _, _ = replay(c1, H, V, ts, conv, None)
    with torch.no_grad():
        ref = m.vap(torch.from_numpy(H)).numpy()
    assert len(c1.episodic) == 0
    assert np.allclose(s1, ref, atol=1e-6)
    c2 = Continuum(m, _cfg(dict(levels=("med",))))
    s_med, _, _ = replay(c2, H, V, ts, conv, None)
    assert s_med.shape == (n, 20) and np.isfinite(s_med).all() and len(c2.episodic) == 0
    c3 = Continuum(m, _cfg(dict(levels=())))
    assert c3.remember(H[0], np.ones(d, np.float32), name="n", rel="r", group="a")
    s3, _, _ = replay(c3, H, V, ts, conv, None)
    assert len(c3.episodic) == 1
    assert not np.allclose(s1[0], s3[0])
    assert np.allclose(s1[2:], s3[2:], atol=1e-6)
