"""Question head stays off the Continuum cosine path. CPU only."""
import ast
import hashlib
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
    assert info["abstain"] is None
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
    assert info["abstain"] is None


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
    assert info["abstain"] is None


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
    assert _choose_alpha(parsed) == float(weights["alpha"]) == 0.3
    digest = hashlib.sha256(np.ascontiguousarray(weights["W"], dtype=np.float64).tobytes()).hexdigest()
    assert digest == "7e3444ea198f27ce996bc1c5e10d1ef0baaaa820166eabd7219e0fb5294cdfee"


def _flat_model():
    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel

    torch.manual_seed(4)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    rng = np.random.default_rng(4)

    def flat():
        return rng.choice([0.2, 0.4, 0.6, 0.8], size=8).astype(np.float32)

    return cont, flat


def _axis(i: int) -> np.ndarray:
    vec = np.zeros(512, np.float32)
    vec[i] = 1.0
    return vec


def test_flat_row_abstain_returns_none_and_recall_does_not():
    from tidal.question_surface import is_deictic, lex_labels

    cont, flat = _flat_model()
    empty_q = cont.embed_event(flat(), _axis(0), surface="Which telephone belongs with Ingrid?")
    value, info = cont.query(empty_q, group=None, mode="top1")
    assert value is None and info["n"] == 0 and info["abstain"] is None
    assert cont.ask(empty_q, group=None) is None

    places = cont.embed_event(flat(), _axis(1), surface="Ingrid keeps a places roster.")
    phone = cont.embed_event(flat(), _axis(2), surface="Jasper keeps a telephone.")
    assert cont.remember(places, "places-value", name="Ingrid", rel="places")
    assert cont.remember(phone, "phone-value", name="Jasper", rel="phone")

    near_places = _axis(1).copy()
    near_places[20] = 0.05
    near_places /= np.linalg.norm(near_places)
    places_q = cont.embed_event(
        flat(), near_places, surface="Which places does Ingrid keep for the handoff roster?",
    )
    assert cont.ask(places_q, group=None) == "places-value"
    _value, info = cont.query(places_q, group=None, mode="top1")
    assert info["abstain"] is None and info["rank"] == "sentence"

    missing_q = cont.embed_event(
        flat(), _axis(4), surface="Which telephone belongs with Ingrid?",
    )
    assert lex_labels(cont._surface_lookup(missing_q)) == ("phone",)
    assert cont.ask(missing_q, group=None) is None
    assert cont.ask_tokens(missing_q, group=None) is None
    value, info = cont.query(missing_q, group=None, mode="top1")
    assert value is None and info["abstain"] == "missing_rel" and info["top"] is None
    assert info["weights"] is None and info["rank"] == "sentence"
    assert cont.rank_question_scores(missing_q, group=None) is not None
    assert cont.recall(missing_q, group=None) in ("places-value", "phone-value")

    absent_q = cont.embed_event(
        flat(), _axis(5), surface="Has the yard ever filed a telephone for Quentin?",
    )
    assert cont.ask(absent_q, group=None) is None
    _value, info = cont.query(absent_q, group=None, mode="top1")
    assert info["abstain"] == "absent_person"

    near_phone = _axis(2).copy()
    near_phone[21] = 0.05
    near_phone /= np.linalg.norm(near_phone)
    deictic_q = cont.embed_event(
        flat(), near_phone, surface="How should I reach you by voice tonight?",
    )
    assert cont.ask(deictic_q, group=None) == "phone-value"
    _value, info = cont.query(deictic_q, group=None, mode="top1")
    assert info["abstain"] is None

    exact = _axis(7)
    stored = cont.embed_event(flat(), exact, surface="Ingrid's places card.")
    assert cont.remember(stored, "exact-places", name="Ingrid", rel="places")
    copied = cont.embed_event(flat(), exact.copy(), surface="Which telephone belongs with Ingrid?")
    assert cont.ask(copied, group=None) == "exact-places"
    _value, info = cont.query(copied, group=None, mode="top1")
    assert info["abstain"] is None

    bare = cont.embed_event(flat(), _axis(8))
    assert cont._surface_lookup(bare) is None
    assert cont.ask(bare, group=None) is not None
    _value, info = cont.query(bare, group=None, mode="top1")
    assert info["abstain"] is None

    # Axis 12 is a phone-cue question whose abstain-ridge gap stays under
    # LEX_RIDGE_GAP, so the cue still wins. A confident disagreeing ridge is
    # covered by test_confident_ridge_overrides_a_misleading_cue.
    zh_missing = cont.embed_event(flat(), _axis(12), surface="港务台在问姜北对应的是哪一部电话。")
    assert cont.remember(
        cont.embed_event(flat(), _axis(10), surface="姜北的生日卡。"),
        "zh-day", name="姜北", rel="birthday",
    )
    assert cont.ask(zh_missing, group=None) is None
    _value, info = cont.query(zh_missing, group=None, mode="top1")
    assert info["abstain"] == "missing_rel"
    zh_absent = cont.embed_event(flat(), _axis(11), surface="场院有没有给方甯存过电话？")
    assert cont.ask(zh_absent, group=None) is None
    _value, info = cont.query(zh_absent, group=None, mode="top1")
    assert info["abstain"] == "absent_person"
    near_day = _axis(10).copy()
    near_day[22] = 0.05
    near_day /= np.linalg.norm(near_day)
    zh_you = cont.embed_event(flat(), near_day, surface="你的生辰要记在哪个清晨？")
    assert cont.ask(zh_you, group=None) == "zh-day"
    assert is_deictic("我们有没有写下姜北的电话？") is False
    assert lex_labels("To place a call through to Ingrid, what digits leave the booth?") == ("phone",)


def test_abstain_margin_is_the_dev_rule():
    from tidal.question_surface import (
        ABSTAIN_MARGIN,
        ABSTAIN_RIDGE,
        abstain_dev_curve,
        choose_abstain_margin,
        choose_abstain_ridge,
        load_corpus_table,
        load_weights,
    )

    weights = load_weights()
    table = load_corpus_table()
    ridge, Wa, errors = choose_abstain_ridge(table)
    assert ridge == ABSTAIN_RIDGE == float(weights["abstain_ridge"]) == 1.0
    assert errors == 2
    assert np.allclose(Wa, weights["Wa"])
    curve = abstain_dev_curve(weights, table)
    assert choose_abstain_margin(curve) == ABSTAIN_MARGIN == float(weights["abstain_margin"]) == 0.1
    by_margin = {row["margin"]: row for row in curve}
    shipped = by_margin[0.1]
    assert shipped["false_abstain"] == {"en": 0, "zh": 0}
    assert shipped["false_n"] == {"en": 192, "zh": 192}
    assert shipped["missing"] == {"en": 188, "zh": 189}
    assert shipped["missing_n"] == {"en": 192, "zh": 192}
    assert shipped["absent"] == {"en": 36, "zh": 36}
    assert shipped["absent_n"] == {"en": 36, "zh": 36}
    assert shipped["deictic_false"] == {"en": 0, "zh": 0}
    assert shipped["deictic_n"] == {"en": 6, "zh": 6}
    assert by_margin[0.0]["false_abstain"]["en"] == 1
    assert by_margin[0.05]["false_abstain"]["en"] == 1


def test_name_spans_keep_verbs_and_reject_closed_extensions():
    """A following verb still matches. A punctuated one-character extension does not."""
    from tidal.question_surface import _focus_names, is_deictic, name_in_text

    assert name_in_text("Ada", "What is Ada's phone number?")
    assert name_in_text("Ann", "Ann's")
    assert not name_in_text("Ann", "Anne")
    assert not name_in_text("Ann", "Ann-Marie")
    assert not name_in_text("Ann", "Ann Marie")
    assert name_in_text("Robert", "Ro bert")
    assert name_in_text("Robert", "Mr. Robert")
    assert not name_in_text("Robert", "Roberto")
    assert name_in_text("Mary Jane", "Mary  Jane")
    assert name_in_text("Mary Jane", "MaryJane")
    assert name_in_text("姜北", "港务台在问姜北对应的是哪一部电话。")
    assert name_in_text("唐宁", "唐宁选的碰面点是哪里？")
    assert name_in_text("祁安", "祁安报过路点，按原顺序重复，不要重排。")
    assert name_in_text("孙悦", "孙悦添岁是哪一天？")
    assert name_in_text("孙悦", "值班簿写着孙 悦时，地方该念哪一条？")
    assert name_in_text("刘洋", "刘洋先生")
    assert name_in_text("赵敏", "赵敏女士")
    assert not name_in_text("孙悦", "值班簿若写的是孙悦然，电话记过没有？")
    assert not name_in_text("王伟", "老王伟，")
    assert not name_in_text("Nellie", "Nellie-Jane")
    assert not name_in_text("Nellie", "Nellie Jane")
    # 王伟强的 is the same shape as 唐宁选的, so the stored name still matches.
    assert name_in_text("王伟", "名下是王伟强的话，电话存了没有？")

    assert is_deictic("For yourself, which telephone is kept?")
    assert is_deictic("您那边的电话")
    assert is_deictic("我的电话该回读哪一条？")
    assert is_deictic("他的电话")
    assert is_deictic("Their ordered locations are which?")
    assert is_deictic("我们有没有写下姜北的电话？") is False
    assert is_deictic("其他地方") is False
    assert is_deictic("if we had heard them") is False

    focused = _focus_names(
        "Of Ada and Mei, ask only Mei: which telephone is filed?",
        ["Ada", "Mei"],
        ("phone",),
    )
    assert focused == ["Mei"]


def test_lex_ridge_gap_stays_above_the_probe_disagreement():
    from tidal.question_heldout import load_text_table
    from tidal.question_surface import LEX_RIDGE_GAP, _abstain_pred, load_weights

    assert LEX_RIDGE_GAP == 0.15
    table, _meta = load_text_table()
    question = "Those other records aren't the one I want. What places did Ravi store?"
    pred, gap = _abstain_pred(table[question], load_weights()["Wa"])
    assert pred != "places"
    assert gap < LEX_RIDGE_GAP


def test_confident_ridge_overrides_a_misleading_cue():
    from tidal.question_surface import STORED_DIM, surface_abstain

    query = np.zeros(STORED_DIM, np.float64)
    query[0] = 1.0
    keys = np.zeros((1, STORED_DIM), np.float64)
    wide = np.zeros((STORED_DIM + 1, 3), np.float64)
    wide[0, 1] = 1.0  # birthday, gap 1.0 over phone and places
    weights = {"Wa": wide, "abstain_margin": 0.1, "W": wide, "alpha": 0.0, "labels": ("phone", "birthday", "places")}
    text = "Which telephone belongs with Ingrid?"
    assert surface_abstain(query, keys, ["birthday"], ["Ingrid"], weights, query_text=text) is None
    assert surface_abstain(query, keys, ["places"], ["Ingrid"], weights, query_text=text) == "missing_rel"
    narrow = np.zeros((STORED_DIM + 1, 3), np.float64)
    narrow[0, 1] = 0.05  # gap under LEX_RIDGE_GAP, so the cue still wins
    weights = {**weights, "Wa": narrow}
    assert surface_abstain(query, keys, ["birthday"], ["Ingrid"], weights, query_text=text) == "missing_rel"
    assert surface_abstain(query, keys, ["phone"], ["Ingrid"], weights, query_text=text) is None
    assert surface_abstain(
        query, keys, ["phone"], ["Ingrid"], weights,
        query_text="For yourself, which telephone is kept?",
    ) is None
    assert surface_abstain(
        query, keys, ["phone"], ["Ingrid"], weights,
        query_text="如果我们听过，方甯的地点会是哪些？",
    ) == "absent_person"
