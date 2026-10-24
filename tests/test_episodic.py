"""Episodic fact slots. CPU only."""
import ast
import inspect

import numpy as np
import pytest

from tidal.episodic import MEMORY_MODALITY, EpisodicStore, decode_span

def _ortho(d=8, n=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    Q, _ = np.linalg.qr(X.T)
    return Q.T[:n].astype(np.float64)

def test_write_stores_answer_and_overwrite_replaces_it():
    store = EpisodicStore(capacity=4, dk=4, dv=4)
    key = np.array([1.0, 0, 0, 0])
    value = np.array([0.0, 1.0, 0, 0])
    assert store.write(key, value, name="ada", rel="birthday", answer="October 16, 1971")
    assert store.slots[0].answer == "October 16, 1971"
    assert store.slots[0].residue[0] == "birthday"
    assert "October 16, 1971" not in repr(store.slots[0].residue)
    assert store.write(key, value, name="ada", rel="birthday", answer="June 3, 1978")
    assert len(store) == 1
    assert store.slots[0].answer == "June 3, 1978"
    assert store.slots[0].residue == ("birthday", 5, 3, 1978)
    assert store.write(key, value, name="ada", rel="birthday")
    assert store.slots[0].answer is None
    assert store.slots[0].residue is None
    assert store.write(key, value, name="ada", rel="phone")
    assert store.slots[0].answer is None
    wide = EpisodicStore(capacity=4, dk=4, dv=16)
    assert wide.write(key, "5551212", name="ada", rel="phone")
    assert wide.slots[0].answer is None
    assert decode_span(wide.slots[0].v) == "5551212"
    assert store.slots[0].modality == wide.slots[0].modality == MEMORY_MODALITY == "text"

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

def test_close_keys_with_different_relations_coexist():
    S = EpisodicStore(capacity=8, dk=4, dv=4, overwrite_cos=0.9)
    k = np.array([1.0, 0, 0, 0])
    S.write(k, np.ones(4), name="a", rel="r1", text="qelvorn morvexil tundravo")
    S.write(k * 1.01, np.full(4, 2.0), name="b", rel="r2", text="qelvorn morvexil tundravo")
    assert len(S) == 2
    assert {slot.name for slot in S.slots} == {"a", "b"}

def test_same_relation_surface_repeat_overwrites_and_a_different_fact_stays():
    from tidal.episodic import token_jaccard

    shared = "qelvorn morvexil tundravo saxpenul lirvoten hendaxom"
    extra = shared + " brontelu"
    swapped = "qelvorn morvexil tundravo saxpenul lirvoten yivkarnox"
    assert token_jaccard(shared, extra) == pytest.approx(6 / 7)
    assert token_jaccard(shared, swapped) == pytest.approx(5 / 7)
    S = EpisodicStore(capacity=8, dk=4, dv=4, repeat_jaccard=0.85)
    k = np.array([1.0, 0, 0, 0])
    assert S.write(k, np.ones(4), name="a", rel="r1", text=shared)
    assert S.write(k * 1.01, np.full(4, 2.0), name="b", rel="r1", text=extra)
    assert len(S) == 1
    assert S.slots[0].name == "b"
    assert np.allclose(S.slots[0].v, 2.0)
    assert S.write(np.array([0.0, 1, 0, 0]), np.full(4, 3.0), name="c", rel="r1", text=swapped)
    assert len(S) == 2
    strict = EpisodicStore(capacity=4, dk=4, dv=4, repeat_jaccard=1.0)
    assert strict.write(k, np.ones(4), name="a", rel="r1", text=shared)
    assert strict.write(k, np.full(4, 2.0), name="b", rel="r1", text=extra)
    assert len(strict) == 2

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

def _bag(text, dim=32):
    out = np.zeros(dim)
    for b in text.encode():
        out[b % dim] += 1.0
    return out

def test_span_code_roundtrip_and_rejected_pool():
    """Mean-pooling bytes collides on a reversed phone. The positional code does not.

    L2-normalizing or adding two codes also fails the checksum. Magnitude stays put.
    """
    from tidal.episodic import decode_span, decode_tokens, encode_span, encode_tokens, span_capacity
    phone = "+1-415-555-0134"
    rev = phone[::-1]
    assert np.array_equal(_bag(phone), _bag(rev))
    assert decode_span(encode_span(phone, 32)) == phone
    assert decode_span(encode_span(rev, 32)) == rev
    assert not np.array_equal(encode_span(phone, 32), encode_span(rev, 32))
    bday = "1990-03-21"
    assert decode_span(encode_span(bday, 32)) == bday
    assert decode_span(encode_span("03/21/1990", 32)) == "03/21/1990"
    assert decode_span(encode_span("21.03.1990", 32)) == "21.03.1990"
    assert decode_span(encode_span("李思", 16)) == "李思"
    toks = ("cedar", "lantern", "north")
    swapped = ("north", "lantern", "cedar")
    assert decode_tokens(encode_tokens(toks, 32)) == toks
    assert decode_tokens(encode_tokens(swapped, 32)) == swapped
    assert decode_span(encode_tokens(toks, 32)) is None
    assert not np.array_equal(encode_tokens(toks, 32), encode_tokens(swapped, 32))
    # Joining either order uses the same bytes. A bag cannot tell them apart.
    assert np.array_equal(_bag("\x1f".join(toks)), _bag("\x1f".join(toks[::-1])))
    coded = encode_span(phone, 32)
    assert float(np.max(np.abs(coded))) <= 1.0
    assert decode_span(coded / np.linalg.norm(coded)) is None
    assert decode_span(coded * 0.5) is None
    other = encode_span("+1-415-555-0199", 32)
    assert decode_span(coded + other) is None
    assert decode_span(0.5 * coded + 0.5 * other) is None
    assert decode_span(coded.astype(np.float32)) == phone
    assert span_capacity(16) == 24
    with pytest.raises(ValueError, match="empty"):
        encode_span("", 16)
    with pytest.raises(ValueError, match="dim"):
        encode_span("a", 3)
    with pytest.raises(ValueError, match="two"):
        encode_tokens(["only"], 16)

def test_span_layout_is_positional_bytes():
    from tidal.episodic import decode_span, encode_span
    v = encode_span("ab", 6)
    assert v.shape == (6,)
    assert v[0] == np.float64(0xE915) / np.float64(65536)
    assert v[1] == np.float64(3) / np.float64(65536)  # tag + "ab"
    assert v[4] == (np.float64(0x01) + (np.float64(ord("a")) * 256)) / np.float64(65536)
    assert v[5] == np.float64(ord("b")) / np.float64(65536)
    assert decode_span(v) == "ab"

def test_span_overwrite_leaves_no_stale_phone():
    p_old, p_new, p_dist = "+1-415-555-0134", "+1-415-555-0199", "+1-202-555-0100"
    k_old, k_new = np.eye(4)[0], np.eye(4)[1]
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=1.0)
    assert S.write(k_old, p_old, name="ada", rel="phone")
    assert S.write(k_old, p_dist, name="bob", rel="phone")
    assert len(S) == 2
    assert S.write(k_new, p_new, name="ada", rel="phone")
    assert len(S) == 2
    assert S.read_span(k_new) == p_new
    # The old key now belongs to bob. It must not bring ada's previous number back.
    assert S.read_span(k_old) == p_dist
    assert all(decode_span(s.v) != p_old for s in S.slots)
    ada = next(s for s in S.slots if s.name == "ada")
    from tidal.episodic import encode_span
    assert np.array_equal(ada.v, encode_span(p_new, 32))
    assert np.array_equal(ada.k, k_new)

def test_close_keys_keep_both_spans_and_close_blend_is_not_a_phone():
    from tidal.episodic import decode_span, encode_span
    p_old, p_new = "+1-415-555-0134", "1990-03-21"
    k = np.array([1.0, 0, 0, 0])
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=0.9)
    assert S.write(k, p_old, name="a", rel="phone")
    assert S.write(k * 1.01, p_new, name="b", rel="birthday")
    assert len(S) == 2
    assert {decode_span(slot.v) for slot in S.slots} == {p_old, p_new}

    k1 = np.array([1.0, 0, 0, 0])
    k2 = np.array([0.95, np.sqrt(1.0 - 0.95 ** 2), 0.0, 0.0])
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=1.0, soft_k=2, soft_temp=0.05)
    assert S.write(k1, p_old, name="a", rel="phone")
    assert S.write(k2, p_new, name="b", rel="birthday")
    assert len(S) == 2
    assert S.read_span(k1) == p_old
    mix, info = S.read(k1, mode="softtopk")
    assert info["top"] == ("a", "phone")
    assert np.isclose(info["weights"].sum(), 1.0)
    recon = sum(w * S.slots[int(i)].v for w, i in zip(info["weights"], info["idx"]))
    assert np.allclose(mix, recon)
    assert decode_span(mix) is None
    # Orthogonal neighbor: soft top-k stays on the byte grid of the winner.
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
    assert S.write(np.eye(4)[0], p_old, name="a", rel="phone")
    assert S.write(np.eye(4)[1], p_new, name="b", rel="birthday")
    mix, info = S.read(np.eye(4)[0], mode="softtopk")
    assert decode_span(mix) == p_old
    assert abs(float(info["weights"][0]) - 1.0) < 1e-6
    assert np.allclose(mix, encode_span(p_old, 32), atol=1e-6)

def test_softtopk_truncates_and_matches_temperature():
    import math
    q = np.array([1.0, 0.0, 0.0])
    k1 = np.array([0.5, np.sqrt(0.75), 0.0])
    k1 = k1 / np.linalg.norm(k1)
    k2 = np.array([-0.25, 0.0, np.sqrt(1.0 - 0.0625)])
    k2 = k2 / np.linalg.norm(k2)
    v0 = np.array([1.0, 0.0, 0.0])
    v1 = np.array([0.0, 2.0, 0.0])
    v2 = np.array([0.0, 0.0, 1.0e6])
    S = EpisodicStore(capacity=4, soft_k=2, soft_temp=0.05, overwrite_cos=1.0)
    assert S.write(q, v0, name="a", rel="r")
    assert S.write(k1, v1, name="b", rel="r")
    assert S.write(k2, v2, name="c", rel="r")
    v, info = S.read(q, mode="softtopk")
    assert info["top"] == ("a", "r")
    assert np.max(np.abs(v)) < 10.0
    cos1 = float(np.dot(k1, q) / (np.linalg.norm(k1) * np.linalg.norm(q)))
    w1 = math.exp((cos1 - 1.0) / 0.05) / (1.0 + math.exp((cos1 - 1.0) / 0.05))
    assert np.allclose(info["weights"], [1.0 - w1, w1], atol=1e-8)
    assert abs(w1 - 4.5397868702434395e-05) < 1e-8
    assert np.allclose(v, (1.0 - w1) * v0 + w1 * v1, atol=1e-6)

def test_span_write_gate_and_failed_overwrite_keep_the_old_code():
    S = EpisodicStore(capacity=4, dk=4, dv=8, overwrite_cos=1.0)
    k = np.ones(4)
    assert S.write(k, "phone", name="a", rel="p")
    step = S._step
    with pytest.raises(ValueError, match="holds"):
        S.write(np.array([0.0, 1, 0, 0]), "x" * 30, name="a", rel="p")
    assert S._step == step and len(S) == 1
    assert S.read_span(k) == "phone"
    assert S.write(k, "x" * 30, name="a", rel="p", kind="chat") is False
    assert S.read_span(k) == "phone"
    bare = EpisodicStore(capacity=2)
    with pytest.raises(ValueError, match="dv"):
        bare.write(np.ones(4), "hi", name="a", rel="p")

def test_span_queries_do_not_see_future_writes():
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=1.0)
    k1, k2 = np.eye(4)[0], np.eye(4)[1]
    phone, bday, nxt = "+1-415-555-0134", "1990-03-21", "+1-415-555-0199"
    assert S.read_span(k1) is None
    assert phone not in {decode_span(s.v) for s in S.slots}
    assert S.write(k1, phone, name="ada", rel="phone")
    assert bday not in {decode_span(s.v) for s in S.slots}
    assert S.write(k2, bday, name="ada", rel="birthday")
    assert S.read_span(k1) == phone and S.read_span(k2) == bday
    assert S.write(np.eye(4)[2], nxt, name="ada", rel="phone")
    assert S.read_span(np.eye(4)[2]) == nxt
    assert all(decode_span(s.v) != phone for s in S.slots)
    assert S.read_span(k2) == bday

def test_group_scopes_spans():
    S = EpisodicStore(capacity=4, dk=4, dv=32, overwrite_cos=1.0, soft_k=1)
    k = np.eye(4)
    assert S.write(k[0], "+1-415-555-0100", name="ada", rel="phone", group="a")
    assert S.write(k[0], "+1-415-555-0199", name="ada", rel="phone", group="b")
    assert len(S) == 2
    assert S.read_span(k[0], group="a") == "+1-415-555-0100"
    assert S.read_span(k[0], group="b") == "+1-415-555-0199"
    assert "+1-415-555-0199" not in {decode_span(s.v) for _, s in S._visible("a")}

def test_episodic_eval_is_larger_than_n96_and_exact():
    from tidal.episodic_eval import run
    report = run()
    assert report["n_fail"] == 0, report["failures"]
    assert report["n_cases"] >= 400
    floors = {
        "overwrite": 80,
        "soft_topk": 40,
        "group": 30,
        "gate": 30,
        "causality": 100,
        "hard": 90,
        "evict": 16,
    }
    for key, floor in floors.items():
        assert report["counts"][key] >= floor, report["counts"]

def test_span_module_has_no_enable_switch():
    import tidal.episodic as ep
    src = inspect.getsource(ep)
    assert "enable_episodic" not in src and "episodic_scale" not in src
    assert "QueryKeyAdapter" not in src and "InjectResidual" not in src
    assert "install_retr" not in src and "install_inj" not in src

def test_replay_recomputes_decay_and_merges_a_same_relation_repeat():
    store = EpisodicStore(capacity=8, dk=4, dv=4, overwrite_cos=0.9, repeat_jaccard=0.85)
    keep = np.array([1.0, 0.0, 0.0, 0.0])
    other = np.array([0.0, 1.0, 0.0, 0.0])
    twin = np.array([0.99, 0.01, 0.0, 0.0])
    shared = "qelvorn morvexil tundravo"
    assert store.write(keep, np.array([1.0, 0, 0, 0]), name="ann", rel="one", text=shared)
    assert store.write(other, np.array([0, 1.0, 0, 0]), name="bob", rel="two", text=shared)
    # Same identity replaces bob. The key now sits on ann, but the relation does not.
    assert store.write(twin, np.array([0, 0, 1.0, 0]), name="bob", rel="two", text=shared)
    assert len(store) == 2
    assert store.near_duplicate_slots() == 0
    store._slots[0].salience = 0.2
    store._slots[1].salience = 0.9
    store.recompute_decay(0.5)
    assert [slot.strength for slot in store.slots] == [0.5, 0.5]
    ranked = store.ranked_by_salience()
    assert ranked[0] == 1
    store.refresh_strength(ranked[:1])
    assert store.slots[1].strength == 1.0
    assert store.slots[1].age == 0
    assert store.slots[0].strength == 0.5
    merged = store.merge_near_duplicates(limit=1, protect=ranked[:1])
    assert merged == 0
    assert len(store) == 2
    assert {slot.name for slot in store.slots} == {"ann", "bob"}
    # A leftover same-relation copy uses the same predicate as write.
    store.slots[1].rel = "one"
    assert store.near_duplicate_slots() == 2
    merged = store.merge_near_duplicates(limit=1, protect=[1])
    assert merged == 1
    assert len(store) == 1
    assert store.slots[0].name == "bob"
    assert store.slots[0].strength == 1.0
    assert store.near_duplicate_slots() == 0

def test_experience_valence_accumulates_revises_and_decays():
    store = EpisodicStore(capacity=4, dk=4, dv=4, overwrite_cos=0.9)
    key = np.array([1.0, 0.0, 0.0, 0.0])
    assert store.write(key, np.array([1.0, 0, 0, 0]), name="ada", rel="note", answer="broom")
    episode = store.next_episode()
    store._slots[0].episode_id = episode
    first = store.add_experience("ada", "note", 1, 0.5, source="ada keeps broom within acquires.", episode_id=episode)
    assert first["valence"] == 0.5
    assert first["kind"] == "experience"
    assert first["episode_id"] == episode
    assert "t" in first
    # Same identity keeps the scalar and replaces the stored answer.
    assert store.write(
        key, np.array([1.0, 0, 0, 1.0]), name="ada", rel="note", answer="ledger"
    )
    assert store.slots[0].answer == "ledger"
    assert store.slots[0].valence == 0.5
    second = store.add_experience("ada", "note", 1, 0.5, source="ada keeps broom within acquires.")
    assert store.slots[0].valence == 1.0
    assert second["episode_id"] != episode
    revised = store.add_experience("ada", "note", -1, 0.5, source="ada keeps broom within acquires.")
    assert store.slots[0].valence == 0.5
    assert revised["delta"] == -0.5
    store.recompute_decay(0.5)
    assert store.slots[0].valence == 0.25
    assert store.slots[0].strength == 0.5
    kinds = [event["kind"] for event in store.valence_history("ada", "note")]
    assert kinds == ["experience", "experience", "experience", "replay"]
    corrected = store.set_valence("ada", "note", 0.0, source="manual")
    assert corrected["kind"] == "correction"
    assert corrected["source"] == "manual"
    assert store.slots[0].valence == 0.0
    # A different identity does not inherit ada's history.
    assert store.write(np.array([0.0, 1.0, 0.0, 0.0]), np.array([0, 1.0, 0, 0]), name="bea", rel="note")
    assert store.slots[-1].valence == 0.0
    assert store.slots[-1].history == []
    # A close key and a different name do not inherit ada's valence, and they
    # do not replace ada when the surface is not a repeat.
    near = np.array([0.99, 0.01, 0.0, 0.0])
    assert store.write(near, np.array([0, 0, 1.0, 0]), name="cy", rel="note", answer="other")
    copied = [slot for slot in store.slots if slot.name == "cy"]
    assert len(copied) == 1
    assert copied[0].valence == 0.0
    assert copied[0].history == []
    assert copied[0].answer == "other"
    assert any(slot.name == "ada" for slot in store.slots)
    repeated = "ada keeps broom within acquires."
    assert store.write(near, np.array([0, 0, 1.0, 0]), name="dee", rel="note", text=repeated)
    # ada has no surface text, so this is not a repeat of ada.
    assert any(slot.name == "ada" for slot in store.slots)
    assert store.write(
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0, 0, 0, 1.0]),
        name="cy",
        rel="note",
        text=repeated,
    )
    assert len([slot for slot in store.slots if slot.text == repeated]) == 1
    kept = next(slot for slot in store.slots if slot.text == repeated)
    assert kept.name == "cy"
    assert kept.valence == 0.0

def test_clock_phase_zero_is_identity_and_a_nonzero_phase_scales_decay():
    import math

    import tidal.episodic as episodic

    store = EpisodicStore(capacity=4, dk=4, dv=4, overwrite_cos=0.9)
    key = np.array([1.0, 0.0, 0.0, 0.0])
    assert store.clock_phase() == 0.0
    assert store.write(key, np.array([1.0, 0, 0, 0]), name="ada", rel="note", answer="broom")
    episode = store.next_episode()
    store.add_experience("ada", "note", 1, 0.5, source="ada keeps broom.", episode_id=episode)
    store.recompute_decay(0.5)
    assert store.slots[0].strength == 0.5
    assert store.slots[0].valence == 0.25
    store.clear()
    assert store.clock_phase() == 0.0
    assert store.clock_log() == ()
    high = store.clock_set(math.pi / 2)
    assert high["kind"] == "set"
    assert set(high) == {"kind", "angle", "phase", "gain"}
    assert high["phase"] == 1.0
    span = float(episodic.CLOCK_SPAN)
    assert high["gain"] == pytest.approx(1.0 + span)
    assert "broom" not in repr(store.clock_log())
    assert store.write(key, np.array([1.0, 0, 0, 0]), name="ada", rel="note", answer="broom")
    store.add_experience("ada", "note", 1, 0.5, source="ada keeps broom.", episode_id=store.next_episode())
    store.recompute_decay(0.5)
    assert store.slots[0].strength == pytest.approx(0.5 ** (1.0 - span))
    # Valence decay stays the replay factor. The clock does not scale it.
    assert store.slots[0].valence == 0.25
    store.refresh_strength([0])
    assert store.slots[0].strength == 1.0
    store.clear()
    store.clock_set(-math.pi / 2)
    assert store.write(key, np.array([1.0, 0, 0, 0]), name="ada", rel="note", answer="ledger")
    store.recompute_decay(0.5)
    assert store.slots[0].strength == pytest.approx(0.5 ** (1.0 + span))
    blob = repr(store.clock_log())
    assert "ledger" not in blob
    assert "broom" not in blob
    tick = store.clock_tick()
    assert tick["kind"] == "tick"
    assert set(tick) == {"kind", "angle", "phase", "gain"}
    original = float(episodic.CLOCK_SPAN)
    episodic.CLOCK_SPAN = 1.0
    try:
        assert episodic.clock_decay_rate(1.0) == 0.0
        assert episodic.clock_write_gain(-1.0) == 0.0
    finally:
        episodic.CLOCK_SPAN = original
