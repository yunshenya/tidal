"""Contrastive question→fact head. CPU only. Not the Continuum replay ranker.

``EpisodicStore.read``, ``Continuum.heads``, ``Continuum.recall``, and
``replay`` rank slot keys by raw cosine. A natural-language question is a
different encoder state from the stored key (``fproj`` + ``tproj`` + one GRU
step on this frozen model). ``Continuum.ask`` / ``Continuum.query`` rank with
the maps in this file, then use the same soft top-k mix and the same
positional byte-code decode. Nothing here is added into ``heads``.

The fit is InfoNCE (τ=0.07) on synthetic facts from that encoder. The question
vector uses the reversed relation token, so it is not the stored key. A
residual inject MLP is not trained on these byte codes: it moves phone numbers
off the grid. Frozen-LM residual inject is ``tidal.llm_memory`` (see
``docs/llm_memory.md``). This module does not import ``transformers``.

The shipped file ``question_qk.npz`` is the seed-0, d=32, 400-step fit plus
the encoder it was trained on. ``load_reference`` rebuilds that pair.
``fit_question_head`` refits whatever frozen encoder a Continuum already
holds and installs the maps on ``ask`` / ``query`` only.

Run: ``python -m tidal.question_head --save``
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidal.episodic import cosine_scores

BUNDLE = Path(__file__).with_name("question_qk.npz")
DEFAULT_STEPS = 400
DEFAULT_SEED = 0
DEFAULT_D = 32
TAU = 0.07
N_NAMES = 36
N_TEST = 12
N_RELS = 4
N_DIST = 8
N_EXACT = 48
N_UNIT = 256

HIT_KEYS = (
    "question_cosine_hits",
    "question_untrained_hits",
    "question_trained_hits",
    "fact_key_cosine_hits",
    "fact_key_head_hits",
    "exact_key_cosine_hits",
    "exact_key_head_hits",
    "unit_cosine_hits",
    "unit_head_hits",
)


def head_hidden(din: int) -> int:
    """Width of each map. 128-d matches the 256-d neuromem retrieval MLP."""
    return int(min(256, max(32, int(din) * 2)))


def _residual_mlp(din: int, hid: int) -> nn.Sequential:
    """Linear → GELU → Linear. Last layer is exact zeros, so the residual adds 0."""
    net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, din))
    net.double()
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class QuestionHead(nn.Module):
    """Residual maps ``q' = q + Wq(q)``, ``k' = k + Wk(k)``.

    Untrained, the last layer is zero, so ``q' = q`` and ``k' = k``. Ask/query
    always run these maps. Replay does not. ``fitted`` is informational: it
    records that a fit for this encoder was copied in. It is not a branch.
    """

    def __init__(self, din: int, hid: Optional[int] = None):
        super().__init__()
        din = int(din)
        if din < 1:
            raise ValueError("question head din must be >= 1")
        self.din = din
        self.hid = head_hidden(din) if hid is None else int(hid)
        self.wq = _residual_mlp(self.din, self.hid)
        self.wk = _residual_mlp(self.din, self.hid)
        self.fitted = False

    def forward(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind not in ("q", "k"):
            raise ValueError(f"kind must be 'q' or 'k', got {kind!r}")
        flat = x.dim() == 1
        if flat:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.din:
            raise ValueError(f"question head input {x.shape[-1]} != din {self.din}")
        net = self.wq if kind == "q" else self.wk
        y = x + net(x.to(dtype=net[-1].weight.dtype))
        return y.squeeze(0) if flat else y


def project(module: QuestionHead, rows: np.ndarray, kind: str) -> np.ndarray:
    """Float64 rows through ``module``. 1-d in, 1-d out."""
    arr = np.asarray(rows, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    x = torch.from_numpy(np.array(arr, dtype=np.float64, copy=True))
    with torch.no_grad():
        y = module(x, kind)
    out = np.array(y.detach().numpy(), dtype=np.float64, copy=True)
    return out.reshape(-1) if single else out


def question_scores(head: QuestionHead, query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Cosine of adapted keys with the adapted question. Same denominator as slot cosine."""
    return cosine_scores(project(head, keys, "k"), project(head, query, "q"))


def _fresh_head(din: int) -> QuestionHead:
    state = torch.random.get_rng_state()
    try:
        head = QuestionHead(din)
    finally:
        torch.random.set_rng_state(state)
    head.eval()
    head.requires_grad_(False)
    return head


def encoder_modules(model):
    yield "fproj", model.fproj
    yield "inorm", model.inorm
    yield "body", model.body
    if hasattr(model, "tproj"):
        yield "tproj", model.tproj


def encoder_fingerprint(model) -> str:
    """Hash of the tensors ``embed_event`` reads. Independent of the question head."""
    h = hashlib.sha256()
    for name, mod in encoder_modules(model):
        h.update(name.encode())
        sd = mod.state_dict()
        for key in sorted(sd):
            tensor = sd[key].detach().cpu().numpy()
            arr = np.ascontiguousarray(tensor)
            h.update(key.encode())
            h.update(str(arr.dtype).encode())
            h.update(str(arr.shape).encode())
            h.update(arr.tobytes())
    return h.hexdigest()


def install_matching(cont) -> bool:
    """Copy the shipped head onto ``cont`` when the encoder hash matches.

    A different encoder keeps the zero residual. That is the untrained map,
    not a second cosine implementation and not a flag. ``heads`` never reads it.
    """
    cont.question_head.fitted = False
    if not BUNDLE.is_file():
        return False
    with np.load(BUNDLE, allow_pickle=False) as z:
        if str(z["fingerprint"].item()) != encoder_fingerprint(cont.m):
            return False
        state = {}
        for key in cont.question_head.state_dict():
            state[key] = torch.from_numpy(np.array(z[f"head.{key}"], copy=True))
        cont.question_head.load_state_dict(state)
    cont.question_head.fitted = True
    cont.question_head.eval()
    cont.question_head.requires_grad_(False)
    return True


def _unit(rng: np.random.Generator, shape: tuple) -> np.ndarray:
    x = rng.normal(size=shape)
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def _feats(name: int, rel: int, query: bool, n_feat: int = 8) -> np.ndarray:
    f = np.zeros(n_feat, np.float32)
    f[0] = 1.0 if query else -1.0
    f[1 + (name % 3)] = 1.0
    f[4 + rel] = 0.5
    return f


def _reference_shell(d: int = DEFAULT_D):
    from tidal.vap import VAPModel

    model = VAPModel(8, d=d, hidden=d, layers=1, dropout=0.0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(512, d))
    model.tproj.eval()
    return model


def build_encoder(seed: int = DEFAULT_SEED, d: int = DEFAULT_D, overwrite_cos: float = 0.9):
    """Frozen tidal event model, seeded init, with its own text projection."""
    from tidal.continuum import Continuum, _cfg

    torch.manual_seed(seed)
    model = _reference_shell(d)
    cont = Continuum(
        model,
        _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": float(overwrite_cos)})),
    )
    if cont.d != d:
        raise RuntimeError(f"Continuum width {cont.d} != {d}")
    return cont


def _load_encoder(model, z) -> None:
    for name, mod in encoder_modules(model):
        state = {}
        for key in mod.state_dict():
            state[key] = torch.from_numpy(np.array(z[f"enc.{name}.{key}"], copy=True))
        mod.load_state_dict(state)


def load_reference(*, overwrite_cos: float = 1.0):
    """Continuum whose encoder and question head are the shipped fit.

    ``overwrite_cos=1`` keeps a fixed distractor pool from collapsing. The
    live default on a hand-built Continuum is still 0.9, and that rule is
    raw cosine either way.
    """
    from tidal.continuum import Continuum, _cfg

    if not BUNDLE.is_file():
        raise FileNotFoundError(f"missing {BUNDLE}; run python -m tidal.question_head --save")
    model = _reference_shell(DEFAULT_D)
    with np.load(BUNDLE, allow_pickle=False) as z:
        _load_encoder(model, z)
    cont = Continuum(
        model,
        _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": float(overwrite_cos)})),
    )
    if not cont.question_head.fitted:
        raise RuntimeError("shipped question head did not install on the shipped encoder")
    return cont


def build_facts(
    cont,
    *,
    seed: int = DEFAULT_SEED,
    n_names: int = N_NAMES,
    n_test: int = N_TEST,
    n_rels: int = N_RELS,
) -> dict:
    """Encode every (name, rel) fact and its question through ``cont.embed_event``.

    The question uses the reversed relation token and does not include the
    value token. Both states go through ``fproj`` and ``tproj``.
    """
    if n_test >= n_names:
        raise ValueError("need at least one train name")
    if n_rels != 4:
        raise ValueError("feature layout uses four relation lanes at indices 4..7")
    prev_mkldnn = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        rng = np.random.default_rng(seed)
        tables = {
            "name": _unit(rng, (n_names, 512)),
            "rel_k": _unit(rng, (n_rels, 512)),
            "value": _unit(rng, (n_names, n_rels, 512)),
        }
        tables["rel_q"] = tables["rel_k"][:, ::-1].copy()
        names, rels, qs, ks, split = [], [], [], [], []
        for n in range(n_names):
            for r in range(n_rels):
                fact = tables["name"][n] + tables["rel_k"][r] + 0.3 * tables["value"][n, r]
                query = tables["name"][n] + tables["rel_q"][r]
                ks.append(cont.embed_event(_feats(n, r, False), fact.astype(np.float32)))
                qs.append(cont.embed_event(_feats(n, r, True), query.astype(np.float32)))
                names.append(n)
                rels.append(r)
                split.append(0 if n < n_names - n_test else 1)
    finally:
        torch.backends.mkldnn.enabled = prev_mkldnn
    return {
        "q": np.stack(qs).astype(np.float64),
        "k": np.stack(ks).astype(np.float64),
        "name": np.asarray(names, np.int64),
        "rel": np.asarray(rels, np.int64),
        "split": np.asarray(split, np.int64),
    }


def fixed_pool(name: np.ndarray, rel: np.ndarray, i: int, n_dist: int = N_DIST) -> list:
    """True index, then other rels of this name, then other names with this rel."""
    same_name = [j for j in range(len(name)) if j != i and name[j] == name[i]]
    same_rel = [j for j in range(len(name)) if name[j] != name[i] and rel[j] == rel[i]]
    extra = same_name + same_rel
    if len(extra) < n_dist:
        extra += [j for j in range(len(name)) if j != i and j not in extra]
    return [i] + extra[:n_dist]


def train_pool(name: np.ndarray, rel: np.ndarray, i: int, rng: np.random.Generator,
               n_hard_rel: int = 4, n_easy: int = 4) -> tuple:
    """Pool with the true key at index 0. Returns ``(indices, n_hard_distractors)``."""
    same_name = [j for j in range(len(name)) if j != i and name[j] == name[i]]
    same_rel = [j for j in range(len(name)) if j != i and name[j] != name[i] and rel[j] == rel[i]]
    if len(same_rel) > n_hard_rel:
        hard_rel = rng.choice(np.asarray(same_rel), size=n_hard_rel, replace=False).tolist()
    else:
        hard_rel = list(same_rel)
    hard = list(same_name) + list(hard_rel)
    forbid = {i, *hard}
    easy_idx = [j for j in range(len(name)) if j not in forbid]
    if len(easy_idx) > n_easy:
        easy = rng.choice(np.asarray(easy_idx), size=n_easy, replace=False).tolist()
    else:
        easy = easy_idx
    return [i] + hard + easy, len(hard)


def _rank_hits(head: Optional[QuestionHead], queries: np.ndarray, keys: np.ndarray, pools: list) -> int:
    hits = 0
    for i, pool in enumerate(pools):
        idx = np.asarray(pool, dtype=np.int64)
        if head is None:
            scores = cosine_scores(keys[idx], queries[i])
        else:
            scores = question_scores(head, queries[i], keys[idx])
        hits += int(np.argmax(scores) == 0)
    return hits


def exact_key_hits(head: Optional[QuestionHead], keys: np.ndarray, n_dist: int, seed: int) -> int:
    """Query equals the stored key. Distractors are random unit vectors."""
    rng = np.random.default_rng(seed)
    hits = 0
    d = keys.shape[1]
    for i in range(len(keys)):
        dist = _unit(rng, (n_dist, d))
        pool = np.concatenate([keys[i : i + 1], dist], 0)
        if head is None:
            scores = cosine_scores(pool, keys[i])
        else:
            scores = question_scores(head, keys[i], pool)
        hits += int(np.argmax(scores) == 0)
    return hits


def _loss(module: QuestionHead, query: np.ndarray, keys: np.ndarray, n_hard: int, use_hard: bool) -> torch.Tensor:
    q = torch.from_numpy(np.asarray(query, dtype=np.float64)).unsqueeze(0)
    k = torch.from_numpy(np.asarray(keys, dtype=np.float64))
    qe = F.normalize(module(q, "q"), dim=-1)
    ke = F.normalize(module(k, "k"), dim=-1)
    logits = (qe @ ke.T) / TAU
    target = torch.zeros(1, dtype=torch.long)
    loss = F.cross_entropy(logits, target)
    if use_hard and n_hard >= 1:
        sub = ke[: n_hard + 1]
        loss = loss + 0.5 * F.cross_entropy((qe @ sub.T) / TAU, target)
    return loss


def measure(cont, facts: dict, head: Optional[QuestionHead] = None, *, seed: int = DEFAULT_SEED,
            n_dist: int = N_DIST, n_exact: int = N_EXACT, n_unit: int = N_UNIT) -> dict:
    """Hit counts. ``head is None`` uses ``cont.question_head`` for the trained column.

    Cosine columns never call the head. ``fact_key_head_hits`` is what recall
    would do if it used the head; ``recall`` does not.
    """
    ranker = cont.question_head if head is None else head
    train = facts["split"] == 0
    te = {key: facts[key][~train] for key in ("q", "k", "name", "rel")}
    pools = [fixed_pool(te["name"], te["rel"], i, n_dist) for i in range(len(te["name"]))]
    n_enc = len(pools)
    n_exact = int(min(n_exact, len(te["k"])))
    fresh = _fresh_head(int(te["k"].shape[1]))
    unit_keys = _unit(np.random.default_rng(seed + 13), (n_unit, te["k"].shape[1]))
    return {
        "n_test": int(n_enc),
        "n_distractors": int(n_dist),
        "n_exact": int(n_exact),
        "n_unit": int(n_unit),
        "n_train": int(train.sum()),
        "question_cosine_hits": _rank_hits(None, te["q"], te["k"], pools),
        "question_untrained_hits": _rank_hits(fresh, te["q"], te["k"], pools),
        "question_trained_hits": _rank_hits(ranker, te["q"], te["k"], pools),
        "fact_key_cosine_hits": _rank_hits(None, te["k"], te["k"], pools),
        "fact_key_head_hits": _rank_hits(ranker, te["k"], te["k"], pools),
        "exact_key_cosine_hits": exact_key_hits(None, te["k"][:n_exact], n_dist, seed + 11),
        "exact_key_head_hits": exact_key_hits(ranker, te["k"][:n_exact], n_dist, seed + 11),
        "unit_cosine_hits": exact_key_hits(None, unit_keys, n_dist, seed + 17),
        "unit_head_hits": exact_key_hits(ranker, unit_keys, n_dist, seed + 17),
        "test_pools": pools,
        "te": te,
    }


def fit_question_head(
    cont,
    *,
    steps: int = DEFAULT_STEPS,
    bs: int = 16,
    lr: float = 1e-3,
    seed: int = DEFAULT_SEED,
    n_dist: int = N_DIST,
    install: bool = False,
    facts: Optional[dict] = None,
) -> tuple:
    """Train a fresh head on ``cont``'s frozen encoder. Replay weights are untouched.

    Returns ``(module, report)``. The module is not the store. ``install``
    copies it onto ``cont.question_head`` for ``ask`` / ``query`` only.
    """
    if facts is None:
        facts = build_facts(cont, seed=seed)
    train = facts["split"] == 0
    tr = {key: facts[key][train] for key in ("q", "k", "name", "rel")}
    d = int(tr["k"].shape[1])
    torch.manual_seed(seed + 1009)
    module = QuestionHead(d)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 3)
    module.train()
    last = 0.0
    for step in range(1, steps + 1):
        total = None
        for _ in range(bs):
            i = int(rng.integers(0, len(tr["name"])))
            pool, n_hard = train_pool(tr["name"], tr["rel"], i, rng)
            idx = np.asarray(pool, dtype=np.int64)
            loss = _loss(module, tr["q"][i], tr["k"][idx], n_hard, bool(rng.random() < 0.5))
            total = loss if total is None else total + loss
        mean = total / bs
        opt.zero_grad()
        mean.backward()
        opt.step()
        last = float(mean.detach())
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"question head step {step}/{steps} loss {last:.4f}", flush=True)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    if install:
        cont.question_head.load_state_dict(module.state_dict())
        cont.question_head.fitted = True
        cont.question_head.eval()
        cont.question_head.requires_grad_(False)
    report = measure(cont, facts, head=module, seed=seed, n_dist=n_dist)
    report["steps"] = int(steps)
    report["seed"] = int(seed)
    report["d"] = int(d)
    report["tau"] = TAU
    report["loss"] = float(last)
    report["installed"] = bool(install)
    return module, report


def _public_report(report: dict) -> dict:
    out = {k: report[k] for k in report if k not in ("test_pools", "te")}
    return out


def save_bundle(model, module: QuestionHead, report: dict, path: Path = BUNDLE) -> None:
    payload = {
        "fingerprint": np.asarray(encoder_fingerprint(model)),
        "meta": np.asarray(json.dumps(_public_report(report))),
    }
    for name, mod in encoder_modules(model):
        for key, value in mod.state_dict().items():
            payload[f"enc.{name}.{key}"] = value.detach().cpu().numpy()
    for key, value in module.state_dict().items():
        payload[f"head.{key}"] = value.detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def load_meta(path: Path = BUNDLE) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"].item()))


def format_report(r: dict) -> str:
    def rate(h, n):
        return f"{h}/{n} = {h / n:.4f}" if n else "n/a"

    n = r["n_test"]
    lines = [
        f"seed={r['seed']} d={r['d']} steps={r['steps']} train={r['n_train']} test={n} "
        f"distractors={r['n_distractors']} tau={r['tau']} loss={r['loss']:.4f}",
        f"question ret@1  cosine {rate(r['question_cosine_hits'], n)}  "
        f"untrained {rate(r['question_untrained_hits'], n)}  "
        f"head {rate(r['question_trained_hits'], n)}",
        f"fact-key ret@1  cosine {rate(r['fact_key_cosine_hits'], n)}  "
        f"head-if-misused {rate(r['fact_key_head_hits'], n)}",
        f"exact-key ret@1 cosine {rate(r['exact_key_cosine_hits'], r['n_exact'])}  "
        f"head-if-misused {rate(r['exact_key_head_hits'], r['n_exact'])}",
        f"unit-key ret@1  cosine {rate(r['unit_cosine_hits'], r['n_unit'])}  "
        f"head-if-misused {rate(r['unit_head_hits'], r['n_unit'])}",
    ]
    return "\n".join(lines)


def _phone(i: int) -> str:
    return f"+1-{(200 + (i % 700)):03d}-555-{(1000 + i):04d}"


def _cosine_ref(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    kn = np.linalg.norm(keys, axis=1)
    qn = float(np.linalg.norm(query))
    return (keys @ query) / (kn * qn + 1e-12)


def eval_questions(check) -> dict:
    """Exact checks for the question path and for an unchanged cosine path.

    ``check(section, ok, detail)`` is the episodic_eval hook. Hit counts are
    exact integers, not a rate estimate.
    """
    from tidal.continuum import replay
    from tidal.episodic import EpisodicStore, decode_span, encode_span

    cont = load_reference(overwrite_cos=1.0)
    check("question", cont.question_head.fitted, "shipped head did not install")
    check("question", cont.episodic.soft_k == 4 and cont.episodic.soft_temp == 0.05, "soft top-k defaults")
    d = cont.d
    h = torch.zeros(1, d)
    h[0, 0] = 1
    with torch.no_grad():
        static = cont.m.vap(h)
        empty = cont.heads(h, [])["vap"]
    check("question", torch.equal(empty, static), "empty heads are not the static model")
    check("question", cont.ask(h[0].numpy()) is None, "empty ask")
    facts_a = build_facts(cont)
    facts_b = build_facts(cont)
    check(
        "question",
        np.array_equal(facts_a["k"], facts_b["k"]) and np.array_equal(facts_a["q"], facts_b["q"]),
        "encoder states differ across two encodes",
    )
    scored = measure(cont, facts_a)
    meta = load_meta()
    for key in HIT_KEYS:
        check("question", int(scored[key]) == int(meta[key]), f"{key} recomputed {scored[key]} bundle {meta[key]}")
    n = scored["n_test"]
    cos_q = scored["question_cosine_hits"]
    head_q = scored["question_trained_hits"]
    check("question", scored["question_untrained_hits"] == cos_q, "zero residual is not raw cosine")
    check("question", head_q > cos_q, "question head did not beat cosine")
    check("question", cos_q / n < 0.45, f"cosine question ret {cos_q}/{n} is not the weak band")
    check("question", head_q / n >= cos_q / n + 0.40, f"question lift {head_q}/{n} vs {cos_q}/{n}")
    check("question", scored["fact_key_cosine_hits"] == n, "stored-key cosine among facts is not perfect")
    check("question", scored["fact_key_head_hits"] <= n - 10, "misusing the head on stored keys did not fall")
    check("question", scored["exact_key_cosine_hits"] == scored["n_exact"], "exact-key cosine")
    check("question", scored["unit_cosine_hits"] == scored["n_unit"], "unit-key cosine")
    check("question", int(meta["steps"]) == DEFAULT_STEPS and int(meta["seed"]) == DEFAULT_SEED, "bundle protocol")
    check("question", float(meta["tau"]) == TAU, "bundle tau")

    te = scored["te"]
    pools = scored["test_pools"]
    witness = None
    fact_miss = None
    for i, pool in enumerate(pools):
        idx = np.asarray(pool, dtype=np.int64)
        keys = te["k"][idx]
        q = te["q"][i]
        stored = te["k"][i]
        raw = cosine_scores(keys, q)
        adapted = question_scores(cont.question_head, q, keys)
        raw_key = cosine_scores(keys, stored)
        adapted_key = question_scores(cont.question_head, stored, keys)
        check("question", np.array_equal(raw, _cosine_ref(keys, q)), f"cosine bits question {i}")
        check("question", np.array_equal(raw_key, _cosine_ref(keys, stored)), f"cosine bits key {i}")
        cont.episodic.clear()
        phones = []
        for j, fact_i in enumerate(idx.tolist()):
            phone = _phone(int(fact_i) + 17 * j)
            phones.append(phone)
            check("question", cont.remember(keys[j], phone, name=f"p{j}", rel="fact") is True, f"write {i}.{j}")
        check("question", len(cont.episodic) == len(pool), f"pool collapsed {i} len {len(cont.episodic)}")
        cos_j = int(np.argmax(raw))
        head_j = int(np.argmax(adapted))
        key_j = int(np.argmax(adapted_key))
        check("question", int(np.argmax(raw_key)) == 0, f"cosine stored-key miss {i}")
        check("question", cont.recall(stored) == phones[0], f"recall stored key {i}")
        check("question", cont.recall(q) == phones[cos_j], f"recall question follows cosine {i}")
        check("question", cont.ask(q) == phones[head_j], f"ask follows the head {i}")
        check("question", cont.ask(stored) == phones[key_j], f"ask on a stored key follows the head {i}")
        mix, info = cont.episodic.read(q, mode="softtopk")
        mix2, info2 = cont.episodic.read_scored(raw, mode="softtopk")
        check("question", mix is not None and np.array_equal(mix, mix2), f"read vs cosine mix {i}")
        check("question", np.array_equal(info["weights"], info2["weights"]), f"cosine weights {i}")
        check("question", info["top"] == (f"p{cos_j}", "fact"), f"cosine top {i}")
        qmix, qinfo = cont.query(q)
        qref, qref_info = cont.episodic.read_scored(adapted, mode="softtopk")
        check("question", qinfo["rank"] == "question", f"query rank tag {i}")
        check("question", np.array_equal(qmix, qref), f"query mix {i}")
        check("question", np.array_equal(qinfo["weights"], qref_info["weights"]), f"query weights {i}")
        check("question", abs(float(qinfo["weights"].sum()) - 1.0) < 1e-12, f"query weights sum {i}")
        check("question", qinfo["top"] == (f"p{head_j}", "fact"), f"query top {i}")
        check("question", abs(float(qinfo["raw_maxcos"]) - float(raw.max())) < 1e-12, f"raw max {i}")
        # The string is the winning slot's code, not whatever the blend decodes to.
        check("question", decode_span(cont.episodic.slots[head_j].v) == phones[head_j], f"slot code {i}")
        check("question", cont.ask(q) == decode_span(cont.episodic.slots[head_j].v), f"ask decodes the winner {i}")
        if cos_j != 0 and head_j == 0 and witness is None:
            witness = i
            check("question", cont.ask(q) == phones[0], f"witness ask {i}")
            check("question", cont.recall(q) != phones[0], f"witness recall {i}")
        if int(np.argmax(raw_key)) == 0 and key_j != 0 and fact_miss is None:
            fact_miss = i
            check("question", cont.recall(stored) == phones[0], f"fact-miss recall {i}")
            check("question", cont.ask(stored) != phones[0], f"fact-miss ask {i}")
            check("question", cont.ask(stored) != cont.recall(stored), f"paths agree on a stored key {i}")
    check("question", witness is not None, "no question where the head hits and cosine misses")
    check("question", fact_miss is not None, "no stored key where the head misses and cosine hits")

    # Value-only overwrite keeps the key the question was trained against.
    hit_i = int(np.argmax([
        int(np.argmax(question_scores(
            cont.question_head, te["q"][i], te["k"][np.asarray(pools[i], dtype=np.int64)]
        )) == 0)
        for i in range(n)
    ]))
    # argmax of a 0/1 list returns the first hit. Require that it is a hit.
    pool = pools[hit_i]
    idx = np.asarray(pool, dtype=np.int64)
    keys = te["k"][idx]
    q = te["q"][hit_i]
    head_j = int(np.argmax(question_scores(cont.question_head, q, keys)))
    check("question", head_j == 0, "first question hit was not a hit")
    cont.episodic.clear()
    old, new = _phone(9000), _phone(9001)
    other = _phone(9002)
    check("question", cont.remember(keys[0], old, name="ada", rel="phone") is True, "overwrite seed")
    check("question", cont.remember(keys[1], other, name="bob", rel="phone") is True, "overwrite distractor")
    check("question", cont.ask(q) == old and cont.recall(keys[0]) == old, "before overwrite")
    check("question", cont.remember(keys[0] * 0.5, new, name="ada", rel="phone") is True, "overwrite write")
    check("question", len(cont.episodic) == 2, "overwrite changed the count")
    ada = next(s for s in cont.episodic.slots if s.name == "ada")
    check("question", np.array_equal(ada.k, keys[0] * 0.5), "overwrite kept the scaled key")
    check("question", cont.recall(keys[0] * 0.5) == new, "recall sees the new phone")
    check("question", all(decode_span(s.v) != old for s in cont.episodic.slots), "stale phone still stored")
    # Same statement key as the question was paired with: put it back, new phone.
    check("question", cont.remember(keys[0], new, name="ada", rel="phone") is True, "restore statement key")
    check("question", cont.ask(q) == new, "ask after value update")
    check("question", cont.recall(keys[0]) == new, "recall after value update")
    check("question", all(decode_span(s.v) != old for s in cont.episodic.slots), "old phone returned")

    # Group scope: a decoy that matches the question vector stays in the other group.
    cont.episodic.clear()
    decoy = _phone(9100)
    true = _phone(9101)
    check("question", cont.remember(q, decoy, name="decoy", rel="phone", group="b") is True, "decoy")
    check("question", cont.remember(keys[0], true, name="ada", rel="phone", group="a") is True, "grouped truth")
    for j in range(1, len(keys)):
        cont.remember(keys[j], _phone(9200 + j), name=f"d{j}", rel="phone", group="a")
    check("question", cont.ask(q, group="a") == true, "ask group a")
    check("question", cont.ask(q, group="b") == decoy, "ask group b")
    check("question", cont.recall(keys[0], group="a") == true, "recall group a")
    check("question", cont.recall(keys[0], group="b") != true, "recall leaked")
    visible = {decode_span(s.v) for _, s in cont.episodic._visible("a")}
    check("question", decoy not in visible, "decoy visible from a")

    # Hand-set scores: the blend of two phones is not a third phone; top-1 still decodes.
    blend = EpisodicStore(capacity=4, dk=4, dv=d, overwrite_cos=1.0, soft_k=2, soft_temp=0.05)
    p0, p1 = "+1-415-555-0134", "+1-202-555-0199"
    blend.write(np.eye(4)[0], p0, name="a", rel="phone")
    blend.write(np.eye(4)[1], p1, name="b", rel="phone")
    mixed, mix_info = blend.read_scored(np.array([0.50, 0.49]), mode="softtopk")
    top, top_info = blend.read_scored(np.array([0.50, 0.49]), mode="top1")
    check("question", mix_info["top"] == ("a", "phone") and abs(float(mix_info["weights"].sum()) - 1.0) < 1e-12, "blend weights")
    check("question", decode_span(mixed) is None, "blend decoded to a phone")
    check("question", decode_span(top) == p0 and top_info["top"] == ("a", "phone"), "top1 code")
    check("question", np.array_equal(top, encode_span(p0, d)), "top1 bytes")

    # One token list, one slot: ask_tokens reads the winning code.
    cont.episodic.clear()
    toks = ("cedar", "lantern", "north")
    kt = np.zeros(d)
    kt[0] = 1.0
    check("question", cont.remember(kt, list(toks), name="ada", rel="places") is True, "tokens")
    check("question", cont.ask_tokens(kt) == toks, "ask_tokens")
    check("question", cont.ask(kt) is None, "tokens are not a flat string")
    check("question", cont.recall_tokens(kt) == toks, "recall_tokens")

    # Replay and heads ignore the question head, including after the head is scrambled.
    cont.episodic.clear()
    cont._mem_group = None
    from tidal import vap_targets as VT

    rng = np.random.default_rng(0)
    n_rows = 4
    H = rng.normal(size=(n_rows, d)).astype(np.float32)
    V = np.zeros((n_rows, VT.NV), np.float32)
    ts = np.array([0.0, 1.0, 0.5, 1.5])
    conv = np.array(["a", "a", "b", "b"])
    s_empty, _, _ = replay(cont, H, V, ts, conv, None)
    with torch.no_grad():
        ref = cont.m.vap(torch.from_numpy(H)).numpy()
    check("question", len(cont.episodic) == 0, "replay wrote a slot")
    check("question", np.allclose(s_empty, ref, atol=1e-6), "empty replay drifted")
    phone = "+1-415-555-0134"
    check("question", cont.remember(H[0], phone, name="ada", rel="phone", group="a") is True, "replay fact")
    cont._mem_group = None
    s_fact, _, _ = replay(cont, H, V, ts, conv, None)
    check("question", not np.allclose(s_fact[0], s_empty[0]), "group fact did not move row 0")
    check("question", not np.allclose(s_fact[1], s_empty[1]), "group fact did not move row 1")
    check("question", np.allclose(s_fact[2:], s_empty[2:], atol=1e-6), "other group moved")
    code = encode_span(phone, d).astype(np.float32)
    with torch.no_grad():
        row0 = cont.m.vap(torch.from_numpy(H[0:1]) + torch.from_numpy(code)).numpy()
    check("question", np.allclose(s_fact[0], row0, atol=1e-5), "replay did not add the raw byte code")
    saved_bias = cont.question_head.wq[-1].bias.detach().clone()
    saved_weight = cont.question_head.wk[-1].weight.detach().clone()
    cont.question_head.wq[-1].bias.fill_(9.0)
    cont.question_head.wk[-1].weight.add_(0.25)
    s_scrambled, _, _ = replay(cont, H, V, ts, conv, None)
    check("question", np.array_equal(s_fact, s_scrambled), "replay changed when the question head changed")
    with torch.no_grad():
        cont._mem_group = "a"
        h_row = torch.from_numpy(H[0:1])
        before = cont.m.vap(h_row + torch.from_numpy(code))
        # heads uses _mem_group. The scrambled head must not change it.
        got = cont.heads(h_row, [])["vap"]
    check("question", torch.allclose(got, before, atol=1e-5), "heads moved with the question head")
    cont.episodic.clear()
    cont._mem_group = "a"
    with torch.no_grad():
        cleared = cont.heads(h_row, [])["vap"]
        base = cont.m.vap(h_row)
    check("question", torch.equal(cleared, base), "cleared heads are not exact zeros-on-top")
    cont.question_head.wq[-1].bias.copy_(saved_bias)
    cont.question_head.wk[-1].weight.copy_(saved_weight)
    cont.question_head.fitted = True

    out = _public_report(scored)
    out["witness_i"] = None if witness is None else int(witness)
    out["fact_miss_i"] = None if fact_miss is None else int(fact_miss)
    return out


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Fit the question head on tidal fproj/tproj states.")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--d", type=int, default=DEFAULT_D)
    p.add_argument("--save", action="store_true", help=f"write {BUNDLE.name}")
    args = p.parse_args(argv)
    cont = build_encoder(args.seed, args.d, overwrite_cos=1.0)
    facts = build_facts(cont, seed=args.seed)
    module, report = fit_question_head(cont, steps=args.steps, seed=args.seed, facts=facts, install=False)
    print(format_report(report), flush=True)
    if not args.save:
        return
    if args.d != DEFAULT_D or args.seed != DEFAULT_SEED or args.steps != DEFAULT_STEPS:
        raise SystemExit("refusing to overwrite the shipped bundle with a non-default protocol")
    save_bundle(cont.m, module, report)
    loaded = load_reference()
    if encoder_fingerprint(loaded.m) != encoder_fingerprint(cont.m):
        raise SystemExit("roundtrip fingerprint mismatch")
    again_facts = build_facts(loaded, seed=args.seed)
    if not (np.array_equal(again_facts["k"], facts["k"]) and np.array_equal(again_facts["q"], facts["q"])):
        raise SystemExit("roundtrip encoder states mismatch")
    again = measure(loaded, again_facts, seed=args.seed)
    for key in HIT_KEYS:
        if int(again[key]) != int(report[key]):
            raise SystemExit(f"roundtrip {key} {again[key]} != {report[key]}")
    print(f"saved {BUNDLE}", flush=True)


if __name__ == "__main__":
    main()
