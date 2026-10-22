"""Exact-match episodic checks. CPU only, no language model.

The neuromem held-out sample was n≈96 free-gen strings, so a 0.02 swing was
noise. This harness does not estimate a rate. Every case is an exact pass/fail
on overwrite, soft top-k, group isolation, the write gate, causal visibility,
and phone / birthday / multi-token values. Ranking on that path is raw cosine.

The ``question`` section is the separate ask/query head: question ret@1 against
raw cosine, stored-key recall staying on cosine, and replay ignoring the head.
Those hit counts are exact integers.

The ``heldout`` section is a separate measurement (``tidal.question_heldout``):
surface questions, not the training template. Its ret@1 is printed and is not
a pass/fail gate. Invariants in that section still fail the process.

Run: ``python -m tidal.episodic_eval``
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from tidal.episodic import (
    EpisodicStore,
    decode_span,
    decode_tokens,
    encode_span,
)

_WORDS = (
    "cedar", "lantern", "north", "river", "lisbon", "spring",
    "harbor", "quiet", "maple", "silver", "orbit", "garden",
)


def _phone(i: int, version: int) -> str:
    return f"+1-{200 + (i % 700):03d}-{1000 + version:04d}-{1000 + i:04d}"


def _birthday(i: int, version: int) -> str:
    kind = (i + version) % 3
    y = 1920 + i
    m = version + 1
    d = (i % 27) + 1
    if kind == 0:
        return f"{y:04d}-{m:02d}-{d:02d}"
    if kind == 1:
        return f"{m:02d}/{d:02d}/{y:04d}"
    return f"{d:02d}.{m:02d}.{y:04d}"


def _tokens(i: int, version: int) -> Tuple[str, ...]:
    return (
        _WORDS[i % len(_WORDS)],
        _WORDS[(i + version + 3) % len(_WORDS)],
        f"n{i}",
        f"v{version}",
    )


def _units(n: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    m = rng.normal(size=(n, dim))
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def _ref_soft(slots, q: np.ndarray, soft_k: int, temp: float):
    keys = np.stack([s.k for s in slots], 0)
    c = (keys @ q) / (np.linalg.norm(keys, axis=1) * float(np.linalg.norm(q)))
    kk = min(int(soft_k), int(c.size))
    idx = np.argsort(-c)[:kk]
    tv = c[idx]
    w = np.exp((tv - tv.max()) / float(temp))
    w = w / w.sum()
    out = np.zeros_like(slots[0].v)
    for j, i in enumerate(idx):
        out = out + w[j] * slots[int(i)].v
    return out, w, c


def run(seed: int = 0) -> Dict[str, Any]:
    """Return ``n_cases``, ``n_fail``, ``counts``, and a short ``failures`` list."""
    rng = np.random.default_rng(seed)
    counts: Dict[str, int] = {}
    failures: List[str] = []
    n_fail = 0

    def check(section: str, ok: bool, detail: str) -> None:
        nonlocal n_fail
        counts[section] = counts.get(section, 0) + 1
        if not ok:
            n_fail += 1
            if len(failures) < 12:
                failures.append(f"{section}: {detail}")

    dim = 64
    _overwrite(check, dim)
    _soft(check, dim, rng)
    _groups(check, dim, rng)
    _gate(check, dim)
    _causal(check, dim, rng)
    _hard(check, dim, rng)
    _evict(check, dim, rng)
    from tidal.question_head import eval_questions
    from tidal.question_heldout import eval_heldout

    question_ret = eval_questions(check)
    heldout = eval_heldout(check)
    return {
        "n_cases": int(sum(counts.values())),
        "n_fail": int(n_fail),
        "counts": counts,
        "failures": failures,
        "question_ret": question_ret,
        "heldout": heldout,
    }


def _overwrite(check: Callable[..., None], dim: int) -> None:
    s = EpisodicStore(capacity=128, dk=dim, dv=dim, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
    rng = np.random.default_rng(1)
    keys = _units(160, dim, rng)
    for i in range(80):
        old, new = _phone(i, 0), _phone(i, 1)
        name = f"p{i}"
        check("overwrite", s.write(keys[2 * i], old, name=name, rel="phone") is True, f"write old {i}")
        check("overwrite", s.write(keys[2 * i + 1], new, name=name, rel="phone") is True, f"write new {i}")
        slot = next(sl for sl in s.slots if sl.name == name)
        check("overwrite", decode_span(slot.v) == new and np.array_equal(slot.v, encode_span(new, dim)), f"code {i}")
        check("overwrite", s.read_span(keys[2 * i + 1]) == new, f"recall {i}")
        check("overwrite", all(decode_span(sl.v) != old for sl in s.slots), f"stale stored {i}")
        check("overwrite", s.read_span(keys[2 * i]) != old, f"old key resurrected {i}")
    check("overwrite", len(s) == 80, f"len {len(s)}")


def _soft(check: Callable[..., None], dim: int, rng: np.random.Generator) -> None:
    s = EpisodicStore(capacity=32, dk=dim, dv=dim, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
    keys = _units(8, dim, rng)
    for i in range(8):
        if i < 3:
            v = encode_span(_phone(100 + i, 0), dim)
        else:
            v = rng.normal(size=dim)
        check("soft_topk", s.write(keys[i], v, name=f"s{i}", rel="v") is True, f"seed {i}")
    queries = _units(24, dim, rng)
    # One query is an exact key so the top identity is known.
    queries = np.concatenate([keys[:4], queries], 0)
    for qi, q in enumerate(queries):
        got, info = s.read(q, mode="softtopk")
        ref, w, _c = _ref_soft(s.slots, q, s.soft_k, s.soft_temp)
        check("soft_topk", got is not None and np.allclose(got, ref, atol=1e-8), f"recon {qi}")
        check("soft_topk", info["weights"] is not None and np.isclose(float(info["weights"].sum()), 1.0), f"sum {qi}")
        check("soft_topk", np.allclose(info["weights"], w, atol=1e-8), f"weights {qi}")
        top = int(np.argmax(_ref_soft(s.slots, q, len(s.slots), s.soft_temp)[2]))
        check("soft_topk", info["top"] == (s.slots[top].name, s.slots[top].rel), f"top {qi}")


def _groups(check: Callable[..., None], dim: int, rng: np.random.Generator) -> None:
    s = EpisodicStore(capacity=80, dk=dim, dv=dim, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
    keys = _units(40, dim, rng)
    only_b = []
    for i in range(16):
        pa, pb = _phone(200 + i, 0), _phone(200 + i, 1)
        check("group", s.write(keys[i], pa, name=f"n{i}", rel="phone", group="a") is True, f"a {i}")
        check("group", s.write(keys[i], pb, name=f"n{i}", rel="phone", group="b") is True, f"b {i}")
        check("group", s.read_span(keys[i], group="a") == pa, f"read a {i}")
        check("group", s.read_span(keys[i], group="b") == pb, f"read b {i}")
        kb = keys[20 + (i % 16)]
        secret = _phone(400 + i, 3)
        only_b.append(secret)
        check("group", s.write(kb, secret, name=f"m{i}", rel="phone", group="b") is True, f"only b {i}")
    for secret in only_b:
        visible = {decode_span(sl.v) for sl in s.slots if sl.group is None or sl.group == "a"}
        check("group", secret not in visible, "b leaked into a")
    gphone = _phone(900, 0)
    check("group", s.write(keys[18], gphone, name="glob", rel="phone", group=None) is True, "global write")
    check("group", s.read_span(keys[18], group="a") == gphone, "global from a")
    check("group", s.read_span(keys[18], group="b") == gphone, "global from b")
    check("group", len(s) == 16 + 16 + 16 + 1, f"group len {len(s)}")


def _gate(check: Callable[..., None], dim: int) -> None:
    s = EpisodicStore(capacity=8, dk=dim, dv=dim, surprise_thr=0.5, overwrite_cos=1.0)
    k = np.zeros(dim)
    k[0] = 1.0
    for i in range(12):
        text = _phone(i, 7)
        check("gate", s.write(k, text, name="z", rel="phone", kind="chat") is False, f"chat {i}")
        check("gate", s.write(k, text, name="z", rel="phone", kind="noise") is False, f"noise {i}")
        check("gate", s.write(k, text, name="z", rel="phone", kind="fact", surprise=0.1) is False, f"low {i}")
        check("gate", s.write(np.zeros(dim), text, name="z", rel="phone", surprise=0.9) is False, f"zero {i}")
    check("gate", len(s) == 0 and s.read_span(k) is None, "gate emptied the store")
    kept = _phone(0, 1)
    check("gate", s.write(k, kept, name="z", rel="phone", kind="fact", surprise=0.9) is True, "fact")
    check("gate", s.write(k, _phone(0, 2), name="z", rel="phone", kind="chat") is False, "chat overwrite")
    check("gate", s.read_span(k) == kept and len(s) == 1, "chat left the fact")
    rejected = s.write(k, ["nope", "tokens"], name="z", rel="note", kind="noise") is False
    check("gate", rejected and len(s) == 1 and s.read_span(k) == kept, "noise tokens")


def _causal(check: Callable[..., None], dim: int, rng: np.random.Generator) -> None:
    n_id, n_ver = 36, 3
    s = EpisodicStore(capacity=64, dk=dim, dv=dim, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
    keys = _units(n_id * n_ver, dim, rng).reshape(n_id, n_ver, dim)
    texts = [[_phone(500 + i, v) for v in range(n_ver)] for i in range(n_id)]
    pending = {(i, v) for i in range(n_id) for v in range(n_ver)}
    current: Dict[int, int] = {}
    for i in range(8):
        check("causality", s.read_span(keys[i, 0]) is None, f"empty query {i}")
    for ver in range(n_ver):
        for i in range(n_id):
            wrote = s.write(keys[i, ver], texts[i][ver], name=f"c{i}", rel="phone")
            check("causality", wrote is True, f"write {i}.{ver}")
            pending.discard((i, ver))
            current[i] = ver
            slot = next(sl for sl in s.slots if sl.name == f"c{i}")
            check("causality", decode_span(slot.v) == texts[i][ver], f"visible {i}.{ver}")
            for old in range(ver):
                check("causality", decode_span(slot.v) != texts[i][old], f"stale {i}.{old}")
            future = {texts[j][v] for (j, v) in pending}
            stored = {decode_span(sl.v) for sl in s.slots}
            check("causality", future.isdisjoint(stored), f"future after {i}.{ver}")
            got = s.read_span(keys[i, ver])
            check("causality", got == texts[i][ver], f"query {i}.{ver}")


def _hard(check: Callable[..., None], dim: int, rng: np.random.Generator) -> None:
    cases: List[Tuple[str, Any, str]] = []
    for i in range(36):
        cases.append(("phone", _phone(i, 0), "phone"))
    for i in range(24):
        cases.append(("birthday", _birthday(i, 0), "birthday"))
    for i in range(24):
        cases.append(("tokens", _tokens(i, 0), "note"))
    distractors = _units(12, dim, rng)
    for n, (kind, value, rel) in enumerate(cases):
        s = EpisodicStore(capacity=24, dk=dim, dv=dim, overwrite_cos=1.0, soft_k=4, soft_temp=0.05)
        for j in range(8):
            s.write(distractors[j], _phone(800 + j, n % 5), name=f"d{j}", rel="phone", group="g")
        key = _units(1, dim, np.random.default_rng(10_000 + n))[0]
        check("hard", s.write(key, value, name="ada", rel=rel, group="g") is True, f"write {kind} {n}")
        if kind == "tokens":
            got = s.read_tokens(key, group="g")
            check("hard", got == value, f"recall {kind} {n}")
            check("hard", s.read_span(key, group="g") is None, f"tokens are not a flat string {n}")
        else:
            got = s.read_span(key, group="g")
            check("hard", got == value, f"recall {kind} {n}")
        check("hard", s.read_span(key, group="other") != value and s.read_tokens(key, group="other") != value, f"other group {n}")
        updated: Any
        if kind == "phone":
            updated = _phone(n, 1)
        elif kind == "birthday":
            updated = _birthday(n, 1)
        else:
            updated = _tokens(n, 1)
        key2 = _units(1, dim, np.random.default_rng(20_000 + n))[0]
        check("hard", s.write(key2, updated, name="ada", rel=rel, group="g") is True, f"update {kind} {n}")
        if kind == "tokens":
            check("hard", s.read_tokens(key2, group="g") == updated, f"new tokens {n}")
            check("hard", all(decode_tokens(sl.v) != value for sl in s.slots), f"stale tokens {n}")
        else:
            check("hard", s.read_span(key2, group="g") == updated, f"new text {n}")
            check("hard", all(decode_span(sl.v) != value for sl in s.slots), f"stale text {n}")


def _evict(check: Callable[..., None], dim: int, rng: np.random.Generator) -> None:
    s = EpisodicStore(capacity=8, dk=dim, dv=dim, overwrite_cos=1.0)
    keys = _units(24, dim, rng)
    phones = [_phone(i, 4) for i in range(21)]
    for i in range(20):
        check("evict", s.write(keys[i], phones[i], name=f"e{i}", rel="phone") is True, f"fill {i}")
    kept = {decode_span(sl.v) for sl in s.slots}
    for i in range(12):
        check("evict", phones[i] not in kept, f"evicted stayed {i}")
    for i in range(12, 20):
        check("evict", phones[i] in kept, f"survivor missing {i}")
    refreshed = _phone(12, 9)
    check("evict", s.write(keys[21], refreshed, name="e12", rel="phone") is True, "refresh")
    check("evict", s.write(keys[22], phones[20], name="e20", rel="phone") is True, "newcomer")
    kept = {decode_span(sl.v) for sl in s.slots}
    names = {sl.name for sl in s.slots}
    check("evict", refreshed in kept and phones[12] not in kept, "refresh dropped the old phone")
    check("evict", phones[13] not in kept and "e13" not in names, "oldest after refresh was evicted")
    check("evict", phones[20] in kept and "e12" in names and "e20" in names, "refreshed slot survived")
    check("evict", len(s) == 8, f"cap {len(s)}")


def main() -> None:
    report = run()
    counts = " ".join(f"{k}={v}" for k, v in sorted(report["counts"].items()))
    print(f"n_cases={report['n_cases']} n_fail={report['n_fail']} {counts}")
    qr = report.get("question_ret") or {}
    if qr:
        n = qr["n_test"]
        print(
            f"question ret@1 cosine {qr['question_cosine_hits']}/{n} "
            f"head {qr['question_trained_hits']}/{n} "
            f"fact-key cosine {qr['fact_key_cosine_hits']}/{n} "
            f"head-if-used-on-stored-keys {qr['fact_key_head_hits']}/{n}"
        )
    heldout = report.get("heldout")
    if heldout:
        from tidal.question_heldout import format_heldout
        print(format_heldout(heldout))
    for line in report["failures"]:
        print(line)
    raise SystemExit(0 if report["n_fail"] == 0 else 1)


if __name__ == "__main__":
    main()
