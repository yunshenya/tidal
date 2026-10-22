"""Question head stays off the Continuum cosine path. CPU only."""
import ast
import inspect

import numpy as np
import torch

from tidal.episodic import EpisodicStore, cosine_scores, decode_span


def _mods(src: str) -> set:
    tree = ast.parse(src)
    mods = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods.extend(a.name.split(".")[0] for a in n.names)
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.append(n.module.split(".")[0])
    return set(mods)


def test_sources_keep_the_paths_split():
    import tidal.continuum as cont
    import tidal.episodic as ep
    import tidal.question_head as qh

    banned = {"transformers", "neuromem", "smollm", "huggingface_hub"}
    assert _mods(inspect.getsource(ep)).isdisjoint(banned)
    assert _mods(inspect.getsource(qh)).isdisjoint(banned)
    assert _mods(inspect.getsource(cont)).isdisjoint(banned)
    ep_src = inspect.getsource(ep)
    q_src = inspect.getsource(qh)
    c_src = inspect.getsource(cont)
    for src in (ep_src, q_src, c_src):
        assert "enable_episodic" not in src and "episodic_scale" not in src
        assert "enable_question" not in src and "use_question_head" not in src
    assert "QueryKeyAdapter" not in ep_src and "InjectResidual" not in ep_src
    assert "question_head" not in inspect.getsource(ep.EpisodicStore.read)
    assert "read_scored" not in inspect.getsource(ep.EpisodicStore.read)
    assert "question" not in inspect.getsource(cont.Continuum.heads)
    assert "question" not in inspect.getsource(cont.Continuum._mix)
    assert "question" not in inspect.getsource(cont.replay)
    assert "question_scores" in inspect.getsource(cont.Continuum._scores_for)
    assert "surface_scores" in inspect.getsource(cont.Continuum._scores_for)
    assert "rank_question_scores" in inspect.getsource(cont.Continuum.query) or "_scores_for" in inspect.getsource(cont.Continuum.query)
    assert "self.query" in inspect.getsource(cont.Continuum.ask)
    assert "enable_surface" not in c_src and "use_surface" not in c_src


def test_zero_residual_matches_raw_cosine_bitwise():
    from tidal.question_head import QuestionHead, project, question_scores

    head = QuestionHead(8)
    head.eval()
    rng = np.random.default_rng(0)
    keys = rng.normal(size=(5, 8))
    query = rng.normal(size=8)
    assert np.array_equal(project(head, keys, "k"), keys)
    assert np.array_equal(project(head, query, "q"), query)
    assert np.array_equal(question_scores(head, query, keys), cosine_scores(keys, query))


def test_read_scored_matches_raw_cosine_mix_bitwise():
    rng = np.random.default_rng(1)
    d = 16
    store = EpisodicStore(capacity=8, dk=d, dv=d, soft_k=4, soft_temp=0.05, overwrite_cos=1.0)
    keys = rng.normal(size=(6, d))
    vals = rng.normal(size=(6, d))
    query = rng.normal(size=d)
    for i in range(6):
        assert store.write(keys[i], vals[i], name=f"n{i}", rel="r")
    raw = cosine_scores(keys, query)
    got, info = store.read(query, mode="softtopk")
    scored, info2 = store.read_scored(raw, mode="softtopk")
    assert np.array_equal(got, scored)
    assert np.array_equal(info["weights"], info2["weights"])
    assert info["top"] == info2["top"]
    phone = "+1-415-555-0134"
    slots = EpisodicStore(dk=4, dv=32, soft_k=1, overwrite_cos=1.0)
    key = np.zeros(4)
    key[0] = 1.0
    assert slots.write(key, phone, name="ada", rel="phone")
    top, _ = slots.read_scored(np.array([1.0]), mode="top1")
    assert decode_span(top) == phone
    assert np.array_equal(top, slots.read(key, mode="top1")[0])


def test_question_head_does_not_overwrite_or_change_replay():
    """A scrambled question map does not merge slots or move replay / heads."""
    from tidal.continuum import Continuum, _cfg, replay
    from tidal.episodic import encode_span
    from tidal.question_head import load_reference
    from tidal.vap import VAPModel

    torch.manual_seed(0)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    cont = Continuum(model, _cfg(dict(levels=())))
    assert cont.question_head.fitted is False
    k1 = np.zeros(cont.d)
    k2 = np.zeros(cont.d)
    k1[0] = 1.0
    k2[1] = 1.0
    with torch.no_grad():
        cont.question_head.wk[-1].bias.fill_(10.0)
    assert cont.remember(k1, "+1-415-555-0100", name="ada", rel="phone")
    assert cont.remember(k2, "+1-415-555-0199", name="bob", rel="city")
    assert len(cont.episodic) == 2
    assert cont.recall(k1) == "+1-415-555-0100"
    assert cont.recall(k2) == "+1-415-555-0199"

    ref = load_reference(overwrite_cos=1.0)
    assert ref.question_head.fitted is True
    assert torch.count_nonzero(ref.question_head.wq[-1].weight) > 0
    h = torch.zeros(1, ref.d)
    h[0, 0] = 1
    with torch.no_grad():
        assert torch.equal(ref.heads(h, [])["vap"], ref.m.vap(h))
    assert ref.ask(np.zeros(ref.d)) is None

    rng = np.random.default_rng(0)
    n = 4
    H = rng.normal(size=(n, ref.d)).astype(np.float32)
    V = np.zeros((n, 20), np.float32)
    ts = np.array([0.0, 1.0, 0.5, 1.5])
    conv = np.array(["a", "a", "b", "b"])
    ref.episodic.clear()
    s_empty, _, _ = replay(ref, H, V, ts, conv, None)
    with torch.no_grad():
        static = ref.m.vap(torch.from_numpy(H)).numpy()
    assert np.allclose(s_empty, static, atol=1e-6)
    phone = "+1-415-555-0134"
    assert ref.remember(H[0], phone, name="ada", rel="phone", group="a")
    s_fact, _, _ = replay(ref, H, V, ts, conv, None)
    assert not np.allclose(s_fact[0], s_empty[0])
    assert np.allclose(s_fact[2:], s_empty[2:], atol=1e-6)
    ref.question_head.wq[-1].bias.fill_(4.0)
    ref.question_head.wk[-1].weight.mul_(3.0)
    s_bad, _, _ = replay(ref, H, V, ts, conv, None)
    assert np.array_equal(s_fact, s_bad)
    code = encode_span(phone, ref.d).astype(np.float32)
    ref._mem_group = "a"
    with torch.no_grad():
        got = ref.heads(torch.from_numpy(H[0:1]), [])["vap"]
        expect = ref.m.vap(torch.from_numpy(H[0:1]) + torch.from_numpy(code))
    assert torch.allclose(got, expect, atol=1e-5)
    ref.episodic.clear()
    with torch.no_grad():
        assert torch.equal(ref.heads(h, [])["vap"], ref.m.vap(h))


def test_question_retrieval_beats_cosine_and_matches_the_bundle():
    from tidal.question_head import HIT_KEYS, build_facts, load_meta, load_reference, measure

    cont = load_reference()
    facts = build_facts(cont)
    report = measure(cont, facts)
    meta = load_meta()
    for key in HIT_KEYS:
        assert report[key] == meta[key], key
    n = report["n_test"]
    assert report["question_untrained_hits"] == report["question_cosine_hits"]
    assert report["question_cosine_hits"] / n < 0.45
    assert report["question_trained_hits"] / n >= report["question_cosine_hits"] / n + 0.40
    assert report["fact_key_cosine_hits"] == n
    assert report["fact_key_head_hits"] <= n - 10
    assert report["exact_key_cosine_hits"] == report["n_exact"]
    assert report["unit_cosine_hits"] == report["n_unit"]
    # Locked measurement from the shipped seed-0 fit. Recall does not use the head.
    assert (report["question_cosine_hits"], report["question_trained_hits"]) == (13, 47)
    assert (report["fact_key_cosine_hits"], report["fact_key_head_hits"]) == (48, 18)
    assert report["exact_key_cosine_hits"] == report["exact_key_head_hits"] == 48
    assert report["unit_cosine_hits"] == report["unit_head_hits"] == 256


def test_flat_row_ask_ignores_a_scrambled_head():
    """Sentence cosine ranks a flat row. A wrecked contrastive head does not."""
    from tidal.continuum import Continuum, _cfg
    from tidal.question_head import question_scores
    from tidal.vap import VAPModel

    torch.manual_seed(0)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    rng = np.random.default_rng(0)

    def flat():
        return rng.choice([0.2, 0.4, 0.6, 0.8], size=8).astype(np.float32)

    e0 = np.zeros(512, np.float32)
    e1 = np.zeros(512, np.float32)
    e0[0] = 1.0
    e1[1] = 1.0
    h0 = cont.embed_event(flat(), e0)
    h1 = cont.embed_event(flat(), e1)
    hq = cont.embed_event(flat(), e0.copy())
    assert not np.array_equal(h0, h1)
    assert cont.remember(h0, "phone-zero", name="a", rel="phone")
    assert cont.remember(h1, "phone-one", name="b", rel="phone")
    assert len(cont.episodic) == 2
    assert cont.episodic.slots[0].sentence is not None
    with torch.no_grad():
        cont.question_head.wq[-1].weight.zero_()
        cont.question_head.wq[-1].bias.copy_(torch.from_numpy(h1 * 50.0 - hq))
        cont.question_head.wk[-1].weight.zero_()
        cont.question_head.wk[-1].bias.zero_()
    keys = np.stack([h0, h1])
    assert int(np.argmax(question_scores(cont.question_head, hq, keys))) == 1
    assert cont.ask(hq, group=None) == "phone-zero"
    _value, info = cont.query(hq, group=None, mode="top1")
    assert info["rank"] == "sentence"
    assert cont.recall(h0) == "phone-zero"
    assert cont.recall(h1) == "phone-one"


def test_tied_sentences_follow_the_relation_logit():
    from tidal.continuum import Continuum, _cfg
    from tidal.question_surface import LABELS, iter_corpus, load_corpus_table, load_weights, surface_scores
    from tidal.vap import VAPModel

    torch.manual_seed(1)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    weights = load_weights()
    assert weights is not None and float(weights["alpha"]) > 0.0
    text = next(
        row[-1] for row in iter_corpus()
        if row[0] == "fit" and row[3] == "phone" and row[4] == "question"
    )
    vec = np.asarray(load_corpus_table()[text], dtype=np.float64)
    unit = vec / np.linalg.norm(vec)
    logits = np.concatenate([unit, np.ones(1)]) @ weights["W"]
    order = np.argsort(logits)
    assert float(logits[order[-1]]) > float(logits[order[0]])
    worse, better = LABELS[int(order[0])], LABELS[int(order[-1])]
    rng = np.random.default_rng(1)

    def flat():
        return rng.choice([0.2, 0.4, 0.6, 0.8], size=8).astype(np.float32)

    raw = vec.astype(np.float32)
    h_worse = cont.embed_event(flat(), raw)
    h_better = cont.embed_event(flat(), raw.copy())
    hq = cont.embed_event(flat(), raw.copy())
    assert cont.remember(h_worse, "worse", name="a", rel=worse)
    assert cont.remember(h_better, "better", name="b", rel=better)
    assert len(cont.episodic) == 2
    stacked = np.stack([slot.sentence for slot in cont.episodic.slots])
    rels = [slot.rel for slot in cont.episodic.slots]
    sent_q = cont._sentence_lookup(hq)
    scores = surface_scores(sent_q, stacked, rels, weights)
    assert int(np.argmax(scores)) == 1
    assert abs(float(scores[0] - scores[1])) > 1e-6
    assert cont.ask(hq, group=None) == "better"
    _value, info = cont.query(hq, group=None, mode="top1")
    assert info["rank"] == "sentence"


def test_sparse_rows_keep_the_contrastive_head():
    from tidal.continuum import Continuum, _cfg
    from tidal.question_head import _feats
    from tidal.vap import VAPModel

    torch.manual_seed(2)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=())))
    text = np.random.default_rng(2).normal(size=512).astype(np.float32)
    h = cont.embed_event(_feats(0, 0, False), text)
    q = cont.embed_event(_feats(0, 0, True), text)
    assert cont._sentence_lookup(h) is None
    assert cont._sentence_lookup(q) is None
    assert cont.remember(h, "+1-415-555-0100", name="ada", rel="phone")
    assert cont.episodic.slots[0].sentence is None
    _value, info = cont.query(q)
    assert info["rank"] == "question"


def test_surface_alpha_is_the_dev_rule_and_the_bundle_is_locked():
    import json

    from tidal.question_surface import (
        ALPHA_GRID,
        EXPECTED_BUNDLE_SHA256,
        _choose_alpha,
        bundle_sha256,
        corpus_errors,
        load_weights,
    )

    base = {"hits": {"all": 10, "en": 4, "zh": 6}}
    curve = {alpha: base for alpha in ALPHA_GRID}
    curve[0.1] = {"hits": {"all": 11, "en": 4, "zh": 7}}
    curve[0.2] = {"hits": {"all": 11, "en": 5, "zh": 6}}
    curve[0.5] = {"hits": {"all": 14, "en": 3, "zh": 11}}
    assert _choose_alpha(curve) == 0.1
    flat = {alpha: base for alpha in ALPHA_GRID}
    flat[0.5] = curve[0.5]
    assert _choose_alpha(flat) == 0.0
    assert not corpus_errors()
    assert bundle_sha256() == EXPECTED_BUNDLE_SHA256
    weights = load_weights()
    assert weights["labels"] == ("phone", "birthday", "places")
    assert weights["W"].shape == (513, 3)
    from tidal.question_surface import BUNDLE
    with np.load(BUNDLE, allow_pickle=False) as blob:
        stored = json.loads(str(blob["dev"].item()))
    parsed = {float(key): row for key, row in stored.items()}
    assert _choose_alpha(parsed) == float(weights["alpha"]) == 0.5
