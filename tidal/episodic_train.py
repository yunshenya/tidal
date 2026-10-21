"""Train the episodic retrieval adapter and inject MLP on tidal encoder states.

CPU only. Facts are synthetic: a frozen ``VAPModel`` (``fproj`` + ``tproj`` + one
GRU step) encodes a name, a relation, and a value token. The query uses the
reversed relation token, so it is not the stored key. Negatives are other
relations of the same name and the same relation on other names, plus a few
easy draws. The loss is InfoNCE (τ=0.07) with an extra hard-subset term, the
same shape as the neuromem v22 study. Nothing here imports SmolLM or
``transformers``.

The fit is a measurement. It does not write a checkpoint and it does not load
weights into ``EpisodicStore``. Those heads stay at a zero last layer unless a
caller copies a state dict itself. ``install_retr`` is true only when question
ret@1 beats raw cosine and three exact-key counts do not fall: random-unit
distractors (the episodic_eval recipe), random-unit keys, and the stored
encoder key among other encoded facts. ``install_inj`` is true only when a
denoised phone mixture decodes and the clean code is unchanged. On this
encoder the question task improves, the random-unit sets stay saturated, and
the stored-key-among-facts count falls, so the retrieval head is not loaded.
The inject fit moves the phone off the byte grid, so that head is not loaded
either.

Run: ``python -m tidal.episodic_train``
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from tidal.episodic import (
    EpisodicStore,
    InjectResidual,
    QueryKeyAdapter,
    cosine_scores,
    decode_span,
    encode_span,
)

DEFAULT_STEPS = 400
DEFAULT_INJECT_STEPS = 300
DEFAULT_SEED = 0
DEFAULT_D = 32
TAU = 0.07
MARGIN = 0.05


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


def build_encoder(seed: int = DEFAULT_SEED, d: int = DEFAULT_D):
    """Frozen tidal event model with its own text projection. No external LM."""
    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel

    torch.manual_seed(seed)
    model = VAPModel(8, d=d, hidden=d, layers=1, dropout=0.0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(512, d))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=())))
    if cont.d != d:
        raise RuntimeError(f"Continuum width {cont.d} != {d}")
    return cont


def build_facts(
    cont,
    *,
    seed: int = DEFAULT_SEED,
    n_names: int = 36,
    n_test: int = 12,
    n_rels: int = 4,
) -> dict:
    """Encode every (name, rel) fact and its question through ``cont.embed_event``."""
    if n_test >= n_names:
        raise ValueError("need at least one train name")
    if n_rels != 4:
        raise ValueError("feature layout uses four relation lanes at indices 4..7")
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
    return {
        "q": np.stack(qs).astype(np.float64),
        "k": np.stack(ks).astype(np.float64),
        "name": np.asarray(names, np.int64),
        "rel": np.asarray(rels, np.int64),
        "split": np.asarray(split, np.int64),
    }


def fixed_pool(name: np.ndarray, rel: np.ndarray, i: int, n_dist: int = 8) -> list:
    """True index, then other rels of this name, then other names with this rel."""
    same_name = [j for j in range(len(name)) if j != i and name[j] == name[i]]
    same_rel = [j for j in range(len(name)) if name[j] != name[i] and rel[j] == rel[i]]
    extra = same_name + same_rel
    if len(extra) < n_dist:
        extra += [j for j in range(len(name)) if j != i and j not in extra]
    return [i] + extra[:n_dist]


def train_pool(name: np.ndarray, rel: np.ndarray, i: int, rng: np.random.Generator,
               n_hard_rel: int = 4, n_easy: int = 4) -> tuple:
    """Pool with the true key at index 0. Returns (indices, n_hard_distractors)."""
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


def _adapt_scores(module: Optional[QueryKeyAdapter], query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    if module is None:
        return cosine_scores(keys, query)
    qt = torch.from_numpy(np.asarray(query, dtype=np.float64))
    kt = torch.from_numpy(np.asarray(keys, dtype=np.float64))
    with torch.no_grad():
        qa = module(qt, "q").numpy()
        ka = module(kt, "k").numpy()
    return cosine_scores(ka, qa)


def pool_hits(module: Optional[QueryKeyAdapter], q: np.ndarray, k: np.ndarray, pools: list) -> int:
    hits = 0
    for i, pool in enumerate(pools):
        scores = _adapt_scores(module, q[i], k[np.asarray(pool, dtype=np.int64)])
        hits += int(np.argmax(scores) == 0)
    return hits


def exact_fact_hits(module: Optional[QueryKeyAdapter], k: np.ndarray, pools: list) -> int:
    """Query is the stored key, distractors are the other fact keys in ``pools``."""
    hits = 0
    for i, pool in enumerate(pools):
        keys = k[np.asarray(pool, dtype=np.int64)]
        scores = _adapt_scores(module, k[i], keys)
        hits += int(np.argmax(scores) == 0)
    return hits


def exact_key_hits(module: Optional[QueryKeyAdapter], keys: np.ndarray, n_dist: int, seed: int) -> int:
    """Query equals the stored key. Distractors are random unit vectors, as in episodic_eval."""
    rng = np.random.default_rng(seed)
    hits = 0
    d = keys.shape[1]
    for i in range(len(keys)):
        dist = _unit(rng, (n_dist, d))
        pool = np.concatenate([keys[i : i + 1], dist], 0)
        scores = _adapt_scores(module, keys[i], pool)
        hits += int(np.argmax(scores) == 0)
    return hits


def _loss(module: QueryKeyAdapter, query: np.ndarray, keys: np.ndarray, n_hard: int, use_hard: bool) -> torch.Tensor:
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


def fit_retr(
    facts: dict,
    *,
    steps: int = DEFAULT_STEPS,
    bs: int = 16,
    lr: float = 1e-3,
    seed: int = DEFAULT_SEED,
    n_dist: int = 8,
) -> tuple:
    """Train a fresh adapter. The store's own module is not touched."""
    train = facts["split"] == 0
    test = ~train
    tr = {key: facts[key][train] for key in ("q", "k", "name", "rel")}
    te = {key: facts[key][test] for key in ("q", "k", "name", "rel")}
    d = int(tr["k"].shape[1])
    torch.manual_seed(seed + 1009)
    module = QueryKeyAdapter(d)
    opt = torch.optim.AdamW(module.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 3)
    test_pools = [fixed_pool(te["name"], te["rel"], i, n_dist) for i in range(len(te["name"]))]
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
            print(f"retr step {step}/{steps} loss {last:.4f}", flush=True)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module, tr, te, test_pools, last


def fit_inject(d: int = DEFAULT_D, steps: int = DEFAULT_INJECT_STEPS, seed: int = DEFAULT_SEED) -> dict:
    """Denoise a phone mixture toward the clean byte code, while keeping the clean code.

    Tidal has no token LM, so this is not the v21 next-token loss. It is the
    constraint the byte code actually has: the readout of a clean phone must
    still decode, and a mixture that does not decode would have to start decoding.
    """
    phones = ("+1-415-555-0134", "+1-202-555-0199", "+44-20-7946-0958", "1990-03-21")
    codes = [encode_span(p, d) for p in phones]
    torch.manual_seed(seed + 5)
    inj = InjectResidual(d)
    opt = torch.optim.AdamW(inj.parameters(), lr=1e-3, weight_decay=1e-4)
    inj.train()
    last = 0.0
    for step in range(steps):
        i = step % len(codes)
        j = (step + 1) % len(codes)
        clean = torch.from_numpy(codes[i])
        other = torch.from_numpy(codes[j])
        w = 0.05 if step % 2 == 0 else 0.15
        mix = (1.0 - w) * clean + w * other
        loss = torch.mean((inj(mix) - clean) ** 2) + 0.5 * torch.mean((inj(clean) - clean) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = float(loss.detach())
    inj.eval()
    target = phones[0]
    clean_np = codes[0]
    mix_np = 0.9 * clean_np + 0.1 * codes[1]
    with torch.no_grad():
        applied = inj(torch.from_numpy(clean_np)).numpy()
        denoised = inj(torch.from_numpy(mix_np)).numpy()
    return {
        "target": target,
        "clean_text": decode_span(applied),
        "clean_ok": decode_span(applied) == target,
        "max_abs_delta": float(np.max(np.abs(applied - clean_np))),
        "raw_mix_text": decode_span(mix_np),
        "denoised_text": decode_span(denoised),
        "loss": last,
        "steps": int(steps),
    }


def qualifies_retr(
    enc_cos: int,
    enc_tr: int,
    n_enc: int,
    ex_cos: int,
    ex_tr: int,
    unit_cos: int,
    unit_tr: int,
    fact_cos: int,
    fact_tr: int,
    margin: float = MARGIN,
) -> bool:
    """True only when question retrieval rises and no exact-key count falls.

    ``ex_*`` is the episodic_eval recipe (stored key, random-unit distractors).
    ``unit_*`` is the same recipe with random-unit keys. ``fact_*`` queries with
    the stored encoder key among the other encoded facts. A head that wins the
    question task by missing the key it just stored does not qualify.
    """
    if n_enc < 1:
        return False
    return bool(
        (enc_tr / n_enc) >= (enc_cos / n_enc) + margin
        and ex_tr >= ex_cos
        and unit_tr >= unit_cos
        and fact_tr >= fact_cos
    )


def qualifies_inj(clean_ok: bool, max_abs_delta: float, raw_mix_text, denoised_text, target: str) -> bool:
    """True only when the clean phone is intact and the mixture starts decoding to it."""
    return bool(
        clean_ok
        and max_abs_delta < 1e-5
        and raw_mix_text is None
        and denoised_text == target
    )


def _store_top(module: Optional[QueryKeyAdapter], query: np.ndarray, keys: np.ndarray) -> str:
    """Winner name from the live store path, after an optional state-dict load."""
    d = int(query.shape[0])
    store = EpisodicStore(capacity=max(len(keys), 1), dk=d, dv=d, overwrite_cos=1.0, soft_k=1)
    if module is not None:
        store.retr.load_state_dict(module.state_dict())
    for j in range(len(keys)):
        store.write(keys[j], np.full(d, float(j + 1)), name=f"n{j}", rel="r")
    _v, info = store.read(query, mode="top1")
    return info["top"][0]


def run_report(
    *,
    steps: int = DEFAULT_STEPS,
    inject_steps: int = DEFAULT_INJECT_STEPS,
    seed: int = DEFAULT_SEED,
    d: int = DEFAULT_D,
    n_names: int = 36,
    n_test: int = 12,
    n_dist: int = 8,
    n_exact: int = 48,
    n_unit: int = 256,
    return_state: bool = False,
) -> dict:
    """Fit both heads and score them. Does not install weights into the default store."""
    cont = build_encoder(seed, d)
    facts = build_facts(cont, seed=seed, n_names=n_names, n_test=n_test)
    module, _tr, te, test_pools, loss = fit_retr(facts, steps=steps, seed=seed, n_dist=n_dist)
    enc_cos = pool_hits(None, te["q"], te["k"], test_pools)
    # A new adapter is zero on the last layer, so it ranks like raw cosine.
    torch.manual_seed(seed + 1009)
    untrained = QueryKeyAdapter(d).eval()
    enc_un = pool_hits(untrained, te["q"], te["k"], test_pools)
    enc_tr = pool_hits(module, te["q"], te["k"], test_pools)
    n_enc = len(test_pools)
    # Exact-key sets use the test encoder keys (query = key) and, separately, random units.
    ex_cos = exact_key_hits(None, te["k"][:n_exact], n_dist, seed + 11)
    ex_tr = exact_key_hits(module, te["k"][:n_exact], n_dist, seed + 11)
    fact_cos = exact_fact_hits(None, te["k"], test_pools)
    fact_tr = exact_fact_hits(module, te["k"], test_pools)
    unit_keys = _unit(np.random.default_rng(seed + 13), (n_unit, d))
    unit_cos = exact_key_hits(None, unit_keys, n_dist, seed + 17)
    unit_tr = exact_key_hits(module, unit_keys, n_dist, seed + 17)
    witness = None
    for i, pool in enumerate(test_pools):
        keys = te["k"][np.asarray(pool, dtype=np.int64)]
        raw = _adapt_scores(None, te["q"][i], keys)
        got = _adapt_scores(module, te["q"][i], keys)
        if int(np.argmax(raw)) != 0 and int(np.argmax(got)) == 0:
            top_raw = _store_top(None, te["q"][i], keys)
            top_tr = _store_top(module, te["q"][i], keys)
            if top_raw == "n0" or top_tr != "n0":
                raise RuntimeError(
                    f"store.read disagrees with cosine_scores at test {i}: raw {top_raw}, trained {top_tr}"
                )
            witness = {"i": int(i), "q": te["q"][i], "keys": keys}
            break
    inj = fit_inject(d=d, steps=inject_steps, seed=seed)
    for i, pool in enumerate(test_pools):
        keys = te["k"][np.asarray(pool, dtype=np.int64)]
        raw_i = int(np.argmax(_adapt_scores(None, te["k"][i], keys)))
        got_i = int(np.argmax(_adapt_scores(module, te["k"][i], keys)))
        if raw_i == 0 and got_i != 0:
            top_raw = _store_top(None, te["k"][i], keys)
            top_tr = _store_top(module, te["k"][i], keys)
            if top_raw != "n0" or top_tr == "n0":
                raise RuntimeError(
                    f"store.read disagrees on an exact fact key at test {i}: raw {top_raw}, trained {top_tr}"
                )
            break
    install_retr = qualifies_retr(
        enc_cos, enc_tr, n_enc, ex_cos, ex_tr, unit_cos, unit_tr, fact_cos, fact_tr
    )
    install_inj = qualifies_inj(inj["clean_ok"], inj["max_abs_delta"], inj["raw_mix_text"], inj["denoised_text"], inj["target"])
    report = {
        "seed": int(seed),
        "d": int(d),
        "steps": int(steps),
        "inject_steps": int(inject_steps),
        "n_train": int((facts["split"] == 0).sum()),
        "n_test": int(n_enc),
        "n_distractors": int(n_dist),
        "tau": TAU,
        "loss": loss,
        "encoder_cosine_hits": int(enc_cos),
        "encoder_untrained_hits": int(enc_un),
        "encoder_trained_hits": int(enc_tr),
        "exact_cosine_hits": int(ex_cos),
        "exact_trained_hits": int(ex_tr),
        "n_exact": int(min(n_exact, len(te["k"]))),
        "fact_key_cosine_hits": int(fact_cos),
        "fact_key_trained_hits": int(fact_tr),
        "unit_cosine_hits": int(unit_cos),
        "unit_trained_hits": int(unit_tr),
        "n_unit": int(n_unit),
        "inject_clean_ok": bool(inj["clean_ok"]),
        "inject_clean_text": inj["clean_text"],
        "inject_max_abs_delta": inj["max_abs_delta"],
        "inject_raw_mix_text": inj["raw_mix_text"],
        "inject_denoised_text": inj["denoised_text"],
        "inject_loss": inj["loss"],
        "install_retr": bool(install_retr),
        "install_inj": bool(install_inj),
        "loaded_into_default_store": False,
        "witness_i": None if witness is None else witness["i"],
    }
    if return_state:
        report["_state"] = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}
        report["_witness"] = witness
    return report


def format_report(r: dict) -> str:
    def rate(h, n):
        return f"{h}/{n} = {h / n:.4f}" if n else "n/a"

    lines = [
        f"seed={r['seed']} d={r['d']} retr_steps={r['steps']} inject_steps={r['inject_steps']} "
        f"train={r['n_train']} test={r['n_test']} distractors={r['n_distractors']} tau={r['tau']}",
        f"encoder q≠k ret@1  cosine {rate(r['encoder_cosine_hits'], r['n_test'])}  "
        f"untrained {rate(r['encoder_untrained_hits'], r['n_test'])}  "
        f"trained {rate(r['encoder_trained_hits'], r['n_test'])}",
        f"exact-key ret@1    cosine {rate(r['exact_cosine_hits'], r['n_exact'])}  "
        f"trained {rate(r['exact_trained_hits'], r['n_exact'])}",
        f"fact-key ret@1     cosine {rate(r['fact_key_cosine_hits'], r['n_test'])}  "
        f"trained {rate(r['fact_key_trained_hits'], r['n_test'])}",
        f"unit-key ret@1     cosine {rate(r['unit_cosine_hits'], r['n_unit'])}  "
        f"trained {rate(r['unit_trained_hits'], r['n_unit'])}",
        f"inject clean_ok={r['inject_clean_ok']} clean_text={r['inject_clean_text']!r} "
        f"max_abs_delta={r['inject_max_abs_delta']:.6g} loss={r['inject_loss']:.6g} "
        f"raw_mix={r['inject_raw_mix_text']!r} denoised={r['inject_denoised_text']!r}",
        f"install_retr={r['install_retr']} install_inj={r['install_inj']} "
        f"loaded_into_default_store={r['loaded_into_default_store']} witness_i={r['witness_i']}",
    ]
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    p = argparse.ArgumentParser(description="Score episodic retrieval and inject heads on synthetic tidal facts.")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--inject-steps", type=int, default=DEFAULT_INJECT_STEPS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--d", type=int, default=DEFAULT_D)
    args = p.parse_args(argv)
    report = run_report(steps=args.steps, inject_steps=args.inject_steps, seed=args.seed, d=args.d)
    print(format_report(report), flush=True)


if __name__ == "__main__":
    main()
