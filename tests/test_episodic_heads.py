"""Residual retrieval / inject heads. CPU only.

The default last layer is zero, so ranking matches raw cosine bitwise and an
empty store is still an exact zero. A fit on tidal encoder states is scored
by ``run_report``; those weights are not installed.
"""
import ast
import inspect

import numpy as np
import torch

from tidal.episodic import (
    EpisodicStore,
    cosine_scores,
    decode_span,
    encode_span,
    episodic_head_params,
    head_hidden,
)


def _mods(src: str) -> set:
    tree = ast.parse(src)
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.extend(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module.split(".")[0])
    return set(mods)


def test_default_last_layer_is_zero_and_small():
    store = EpisodicStore(dk=8, dv=8)
    for net in (store.retr.wq, store.retr.wk, store.inj.net):
        assert net[-1].weight.dtype == torch.float64
        assert torch.count_nonzero(net[-1].weight) == 0
        assert torch.count_nonzero(net[-1].bias) == 0
    assert head_hidden(128) == 256
    assert episodic_head_params(128) == 197760
    assert episodic_head_params(192) == 296256
    assert episodic_head_params(192) < 2_000_000


def test_zero_init_matches_raw_cosine_and_phone_bitwise():
    rng = np.random.default_rng(1)
    d = 16
    store = EpisodicStore(capacity=8, dk=d, dv=d, soft_k=4, soft_temp=0.05, overwrite_cos=1.0)
    keys = rng.normal(size=(6, d))
    vals = rng.normal(size=(6, d))
    query = rng.normal(size=d)
    for i in range(6):
        assert store.write(keys[i], vals[i], name=f"n{i}", rel="r")
    assert np.array_equal(store._adapt(keys, "k"), keys)
    assert np.array_equal(store._adapt(query, "q"), query)
    scores = cosine_scores(keys, query)
    local = np.argpartition(-scores, 3)[:4]
    local = local[np.argsort(-scores[local])]
    tv = scores[local]
    weights = np.exp((tv - tv.max()) / 0.05)
    weights = weights / weights.sum()
    mix = np.zeros(d)
    for j, i in enumerate(local):
        mix = mix + weights[j] * vals[int(i)]
    got, _info = store.read(query)
    assert np.array_equal(got, mix)
    assert np.array_equal(store.readout(query), mix)
    phone = "+1-415-555-0134"
    slots = EpisodicStore(dk=4, dv=32, soft_k=1, overwrite_cos=1.0)
    key = np.zeros(4)
    key[0] = 1.0
    assert slots.write(key, phone, name="ada", rel="phone")
    code = encode_span(phone, 32)
    assert np.array_equal(slots.read(key, mode="top1")[0], code)
    assert np.array_equal(slots.readout(key, mode="top1"), code)
    assert decode_span(slots.readout(key)) == phone


def test_empty_store_skips_inject_and_read_keeps_the_raw_code():
    empty = EpisodicStore(dk=4, dv=4)
    with torch.no_grad():
        empty.inj.net[-1].bias.fill_(3.0)
    assert np.array_equal(empty.readout(np.ones(4)), np.zeros(4))
    store = EpisodicStore(dk=4, dv=4, soft_k=1, overwrite_cos=1.0)
    key = np.array([1.0, 0.0, 0.0, 0.0])
    value = np.array([0.25, -0.5, 0.0, 1.0])
    assert store.write(key, value, name="a", rel="r")
    with torch.no_grad():
        store.inj.net[-1].bias.copy_(torch.full((4,), 0.1, dtype=torch.float64))
    got, _info = store.read(key, mode="top1")
    assert np.array_equal(got, value)
    assert np.allclose(store.readout(key, mode="top1"), value + 0.1)


def test_loaded_query_bias_changes_the_winner_on_the_read_path():
    def fill(store):
        store.write(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]), name="d", rel="r")
        store.write(np.array([0.0, 1, 0, 0]), np.array([0.0, 9, 0, 0]), name="t", rel="r")

    query = np.array([1.0, 0.05, 0.0, 0.0])
    plain = EpisodicStore(dk=4, dv=4, soft_k=1, overwrite_cos=1.0)
    fill(plain)
    assert plain.read(query, mode="top1")[1]["top"] == ("d", "r")
    moved = EpisodicStore(dk=4, dv=4, soft_k=1, overwrite_cos=1.0)
    fill(moved)
    state = {k: v.detach().clone() for k, v in moved.retr.state_dict().items()}
    state["wq.2.bias"] = torch.tensor([-1.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    moved.retr.load_state_dict(state)
    value, info = moved.read(query, mode="top1")
    assert info["top"] == ("t", "r")
    assert np.allclose(value, [0.0, 9.0, 0.0, 0.0])


def test_overwrite_ignores_adapted_cosine_and_same_identity_drops_the_old_phone():
    store = EpisodicStore(capacity=4, dk=4, dv=4, overwrite_cos=0.9)
    k1 = np.array([1.0, 0.0, 0.0, 0.0])
    k2 = np.array([0.0, 1.0, 0.0, 0.0])
    assert store.write(k1, np.array([1.0, 0, 0, 0]), name="a", rel="r1")
    with torch.no_grad():
        # Same bias on every key. Adapted cosine exceeds 0.9; raw cosine is 0.
        store.retr.wk[-1].bias.copy_(torch.tensor([10.0, 10.0, 0.0, 0.0], dtype=torch.float64))
    assert store.write(k2, np.array([0.0, 2, 0, 0]), name="b", rel="r2")
    assert len(store) == 2
    phones = EpisodicStore(dk=4, dv=32, overwrite_cos=1.0, soft_k=1)
    old, new = "+1-415-555-0134", "+1-202-555-0199"
    assert phones.write(k1, old, name="ada", rel="phone")
    assert phones.write(k2, new, name="ada", rel="phone")
    assert len(phones) == 1
    assert phones.read_span(k2) == new
    assert phones.read_span(k1) != old
    assert all(decode_span(slot.v) != old for slot in phones.slots)


def test_heads_module_does_not_import_transformers():
    import tidal.episodic as ep
    import tidal.episodic_train as tr
    banned = {"transformers", "neuromem", "smollm", "huggingface_hub"}
    assert _mods(inspect.getsource(ep)).isdisjoint(banned)
    assert _mods(inspect.getsource(tr)).isdisjoint(banned)
    assert "enable_episodic" not in inspect.getsource(ep)
    assert "episodic_scale" not in inspect.getsource(ep)
    assert "enable_episodic" not in inspect.getsource(tr)


def test_empty_continuum_stays_exact_with_the_heads_present():
    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel

    torch.manual_seed(0)
    model = VAPModel(8, d=32, hidden=32, layers=1, dropout=0).eval()
    cont = Continuum(model, _cfg(dict(levels=())))
    assert cont.episodic.retr is not None and cont.episodic.inj is not None
    h = torch.zeros(1, cont.d)
    h[0, 0] = 1
    with torch.no_grad():
        static = model.vap(h)
        empty = cont.heads(h, [])["vap"]
    assert torch.equal(empty, static)


def test_trained_retr_beats_questions_and_is_not_installed():
    """Question ret@1 rises. Exact-key-among-facts falls, so nothing is loaded.

    episodic_eval's random-unit distractors stay saturated: cosine is already
    1, and the trained head does not improve that set.
    """
    from tidal.episodic_train import run_report

    report = run_report(return_state=True)
    n = report["n_test"]
    assert report["encoder_untrained_hits"] == report["encoder_cosine_hits"]
    assert report["encoder_cosine_hits"] / n < 0.45
    assert report["encoder_trained_hits"] / n >= report["encoder_cosine_hits"] / n + 0.40
    assert report["exact_cosine_hits"] == report["exact_trained_hits"] == report["n_exact"]
    assert report["unit_cosine_hits"] == report["unit_trained_hits"] == report["n_unit"]
    assert report["fact_key_cosine_hits"] == n
    assert report["fact_key_trained_hits"] <= n - 10
    assert report["install_retr"] is False
    assert report["install_inj"] is False
    assert report["loaded_into_default_store"] is False
    assert report["inject_clean_ok"] is False
    assert report["inject_max_abs_delta"] > 1e-4
    assert report["inject_raw_mix_text"] is None
    assert report["inject_denoised_text"] is None
    assert report["witness_i"] is not None
    fresh = EpisodicStore(dk=report["d"], dv=report["d"])
    assert torch.count_nonzero(fresh.retr.wq[-1].weight) == 0
    assert torch.count_nonzero(fresh.inj.net[-1].weight) == 0
    witness = report["_witness"]
    loaded = EpisodicStore(dk=report["d"], dv=report["d"], overwrite_cos=1.0, soft_k=1)
    loaded.retr.load_state_dict(report["_state"])
    for j in range(len(witness["keys"])):
        loaded.write(witness["keys"][j], np.full(report["d"], float(j + 1)), name=f"n{j}", rel="r")
    _value, info = loaded.read(witness["q"], mode="top1")
    assert info["top"] == ("n0", "r")
    plain = EpisodicStore(dk=report["d"], dv=report["d"], overwrite_cos=1.0, soft_k=1)
    for j in range(len(witness["keys"])):
        plain.write(witness["keys"][j], np.full(report["d"], float(j + 1)), name=f"n{j}", rel="r")
    assert plain.read(witness["q"], mode="top1")[1]["top"] != ("n0", "r")
