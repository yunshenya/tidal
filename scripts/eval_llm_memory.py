"""Before/after held-out eval of llm_memory keys on frozen SmolLM2-135M.

The "before" column is the old key: mean-pool every content token, no sink
drop, no center. The "after" column writes ``LLMMemory.encode_key`` into the
store (sink token dropped, fixed center subtracted). Queries for retrieval
are the natural questions in ``tidal.llm_memory_data``. Free generation uses
the untrained inject and a completion prefix (``"Nell's phone number is "``),
because a natural question makes this model emit a leading newline and the
v22 stop rule then scores every system as empty. Fact-in-context and bge RAG
append that same prefix after the fact line.

CPU bfloat16 matmul in this torch build is about 1.7s per forward.
``dtype=float32`` is the same checkpoint and the dtype the original harness
forced with ``fp32patch``. The shipped centers still apply.

    python scripts/eval_llm_memory.py --n 32 --Ns 0,4,16,32 --k-reps 4
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from tidal.episodic import EpisodicStore
from tidal.llm_memory import KEY_SINK_RATIO, LLMMemory, pool_key_tokens, pool_masked
from tidal.llm_memory_data import chat, fact, fact_stream, match, norm, session

MAX_NEW_TOKENS = 16


def prefix_of(turn: dict) -> str:
    """Fact text with the answer span removed, including the trailing space."""
    text, answer = turn["text"], turn["a"]
    j = text.rfind(answer)
    if j < 0:
        raise ValueError("answer is not in the fact text")
    return text[:j]


def stop_text(tok, ids) -> str:
    """v22 free-gen stop: cut before a newline, and on sentence-ending punctuation."""
    out = []
    for nx in ids:
        s = tok.decode(out + [int(nx)])
        if "\n" in s:
            break
        out.append(int(nx))
        if s.rstrip().endswith((".", "?", "!")) and len(s.strip()) > 1:
            break
    return tok.decode(out)


def _spans(mem, text: str, answer: str | None):
    """One forward. Returns legacy pool, sink-dropped pool, and the value pool."""
    hidden, enc = mem._run_hidden(text)
    n = hidden.shape[0]
    content = mem._content_mask(enc, n)
    if answer is None:
        key_mask = content
        value = None
    else:
        answer_m = mem._answer_mask(text, enc, answer, n)
        key_mask = [c and not a for c, a in zip(content, answer_m)]
        if not any(key_mask):
            key_mask = content
        value = pool_masked(hidden, answer_m).detach().cpu().numpy().astype(np.float32, copy=False)
    legacy = pool_masked(hidden, key_mask).detach().cpu().numpy().astype(np.float32, copy=False)
    idx = [i for i, flag in enumerate(key_mask) if flag]
    raw = pool_key_tokens(hidden[idx].detach().cpu().numpy(), ratio=KEY_SINK_RATIO)
    return legacy, raw, value


def fresh_store(mem) -> EpisodicStore:
    return EpisodicStore(
        capacity=mem.store.capacity,
        dk=mem.hidden,
        dv=mem.hidden,
        overwrite_cos=mem.store.overwrite_cos,
        soft_k=mem.store.soft_k,
        soft_temp=mem.store.soft_temp,
    )


def empty_store(mem) -> EpisodicStore:
    return EpisodicStore(
        capacity=1,
        dk=mem.hidden,
        dv=mem.hidden,
        overwrite_cos=mem.store.overwrite_cos,
        soft_k=mem.store.soft_k,
        soft_temp=mem.store.soft_temp,
    )


def fill(mem, turns):
    """Write fact turns into the live store and a parallel legacy-key store."""
    mem.store.clear()
    before = fresh_store(mem)
    for turn in turns:
        if turn["kind"] != "fact":
            continue
        legacy, raw, value = _spans(mem, turn["text"], turn["a"])
        key = (raw - mem.key_mean).astype(np.float32, copy=False)
        before.write(legacy, value, name=turn["n"], rel=turn["rel"], kind="fact")
        mem.store.write(key, value, name=turn["n"], rel=turn["rel"], kind="fact")
    return before


def query_pair(mem, text: str):
    """Legacy mean-pool and centered query, from one forward."""
    legacy, raw, _value = _spans(mem, text, None)
    centered = (raw - mem.query_mean).astype(np.float64, copy=False)
    return legacy.astype(np.float64, copy=False), centered


def ret1(store, query, turn) -> bool:
    _value, info = store.read(query, mode="softtopk")
    return info.get("top") == (turn["n"], turn["rel"])


def in_store(store, turn) -> bool:
    return any((slot.name, slot.rel) == (turn["n"], turn["rel"]) for slot in store.slots)


def generate(mem, prompt: str, store, *, legacy_query: bool = False) -> str:
    """Greedy continuation against ``store``.

    ``legacy_query`` scores the old uncentered mean-pool, so a before-column
    store is not queried with the new centered vector.
    """
    saved_store = mem.store
    saved_query = mem._query_vector
    mem.store = store
    if legacy_query:
        mem._query_vector = lambda text, _mem=mem: query_pair(_mem, text)[0]
    try:
        ids = mem.continuation_ids(prompt, max_new_tokens=MAX_NEW_TOKENS)
    finally:
        mem.store = saved_store
        mem._query_vector = saved_query
    return stop_text(mem.tokenizer, ids)


def rate(hits) -> dict:
    n = len(hits)
    return {"acc": (sum(hits) / n) if n else None, "hits": int(sum(hits)), "n": n}


def summarize_store(pairs, targets) -> dict:
    slots = sum(nslot for nslot, _nfact in pairs)
    facts = sum(nfact for _nslot, nfact in pairs)
    return {
        "slots": int(slots),
        "facts_written": int(facts),
        "slots_per_fact": (slots / facts) if facts else None,
        "target_in_store": (sum(targets) / len(targets)) if targets else None,
        "n_sessions": len(pairs),
    }


class BGE:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    @torch.no_grad()
    def hit(self, turns, query: str):
        texts = [t["text"] for t in turns]
        batch = self.tok(texts + [query], padding=True, truncation=True, return_tensors="pt")
        hidden = self.model(**batch).last_hidden_state[:, 0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        scores = hidden[:-1] @ hidden[-1]
        j = int(scores.argmax())
        return turns[j]


def _offdiag(rows) -> dict:
    X = np.stack([np.asarray(r, dtype=np.float64) for r in rows])
    n = np.linalg.norm(X, axis=1, keepdims=True)
    U = X / np.maximum(n, 1e-12)
    C = U @ U.T
    v = C[~np.eye(len(X), dtype=bool)]
    return {
        "off_mean": float(v.mean()),
        "off_min": float(v.min()),
        "off_max": float(v.max()),
        "n": int(len(X)),
    }


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Measure llm_memory keys before and after the sink/center fix.")
    p.add_argument("--n", type=int, default=32, help="Sessions per N.")
    p.add_argument("--Ns", default="0,4,16,32")
    p.add_argument("--k", default="20,50,100")
    p.add_argument("--k-reps", type=int, default=4)
    p.add_argument("--n-upd", type=int, default=32)
    p.add_argument("--skip-gen", action="store_true")
    p.add_argument("--skip-bge", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    t0 = time.time()
    # float32: same weights as the default load, but CPU bfloat16 GEMM is ~50x slower
    # here. Centers shipped from the bfloat16 pass still match this geometry.
    mem = LLMMemory(dtype=torch.float32)
    print(
        f"model {mem.model_name} layer {mem.layer} hidden {mem.hidden} "
        f"overwrite {mem.store.overwrite_cos} dtype {next(mem.model.parameters()).dtype} "
        f"max_new_tokens {MAX_NEW_TOKENS}",
        flush=True,
    )
    # One fact: the helper's key matches encode_key, and the value matches encode_value.
    probe = fact(random.Random(0), "test")
    legacy, raw, value = _spans(mem, probe["text"], probe["a"])
    if not np.allclose(raw - mem.key_mean, mem.encode_key(probe["text"], probe["a"])):
        raise RuntimeError("eval key helper diverged from encode_key")
    if not np.allclose(value, mem.encode_value(probe["text"], probe["a"])):
        raise RuntimeError("eval value helper diverged from encode_value")
    q_legacy, q_new = query_pair(mem, probe["q"])
    if not np.allclose(q_new, mem._query_vector(probe["q"])):
        raise RuntimeError("eval query helper diverged from _query_vector")
    print("helpers match encode_key / encode_value / _query_vector", flush=True)

    bge = None
    if not args.skip_bge and not args.skip_gen:
        try:
            bge = BGE()
            print("bge loaded", flush=True)
        except Exception as exc:
            print("bge skipped", type(exc).__name__, exc, flush=True)

    report = {
        "args": vars(args),
        "geom": "sink8-qmean-v1",
        "dtype": "float32",
        "max_new_tokens": MAX_NEW_TOKENS,
        "gen_prompt": "completion prefix (fact text with the answer removed)",
        "ret_query": "natural question",
    }
    examples = []
    for N in [int(x) for x in args.Ns.split(",") if x != ""]:
        rng = random.Random(5000 + N)
        rows = []
        for _ in range(args.n):
            turns = session(rng, "test", N)
            facts = [t for t in turns if t["kind"] == "fact"]
            target = rng.choice(facts)
            rows.append((turns, target))
        acc = {k: [] for k in ("before_gen", "after_gen", "fact_in_context", "no_memory", "rag_bge")}
        r_before, r_after = [], []
        surv_b, surv_a, tin_b, tin_a = [], [], [], []
        rag_hit = []
        for i, (turns, target) in enumerate(rows):
            nfact = sum(t["kind"] == "fact" for t in turns)
            before = fill(mem, turns)
            q_before, q_after = query_pair(mem, target["q"])
            r_before.append(ret1(before, q_before, target))
            r_after.append(ret1(mem.store, q_after, target))
            surv_b.append((len(before), nfact))
            surv_a.append((len(mem.store), nfact))
            tin_b.append(in_store(before, target))
            tin_a.append(in_store(mem.store, target))
            if not args.skip_gen:
                pre = prefix_of(target)
                acc["no_memory"].append(match(generate(mem, pre, empty_store(mem)), target["a"]))
                acc["fact_in_context"].append(
                    match(generate(mem, target["text"] + "\n" + pre, empty_store(mem)), target["a"])
                )
                acc["before_gen"].append(match(generate(mem, pre, before, legacy_query=True), target["a"]))
                got = generate(mem, pre, mem.store)
                acc["after_gen"].append(match(got, target["a"]))
                if bge is not None:
                    picked = bge.hit(turns, target["q"])
                    rag_hit.append(picked is target)
                    acc["rag_bge"].append(
                        match(generate(mem, picked["text"] + "\n" + pre, empty_store(mem)), target["a"])
                    )
                if i < 2:
                    examples.append(
                        {
                            "N": N,
                            "q": target["q"],
                            "prefix": pre,
                            "gold": target["a"],
                            "after_gen": got,
                        }
                    )
        cell = {
            "ret1_before": rate(r_before),
            "ret1_after": rate(r_after),
            "store_before": summarize_store(surv_b, tin_b),
            "store_after": summarize_store(surv_a, tin_a),
        }
        for key, hits in acc.items():
            if hits:
                cell[key] = rate(hits)
        if rag_hit:
            cell["rag_hit"] = rate(rag_hit)
        report[f"N{N}"] = cell
        print(f"N{N}", json.dumps(cell), f"{time.time() - t0:.0f}s", flush=True)

    upd = {
        "before": {"new": [], "old": []},
        "after": {"new": [], "old": []},
        "full_session": {"new": [], "old": []},
    }
    same_cos = []
    rng = random.Random(42)
    for _ in range(args.n_upd):
        old = fact(rng, "test")
        new = fact(rng, "test", n=old["n"], rel=old["rel"])
        while norm(new["a"]) == norm(old["a"]):
            new = fact(rng, "test", n=old["n"], rel=old["rel"])
        mid = []
        for _k in range(rng.randint(3, 7)):
            mid.append(chat(rng, "test") if rng.random() < 0.6 else fact(rng, "test"))
        turns = [old] + mid + [new]
        before = fill(mem, turns)
        same_cos.append(_cos(mem.encode_key(old["text"], old["a"]), mem.encode_key(new["text"], new["a"])))
        if not args.skip_gen:
            pre = prefix_of(new)
            g_before = generate(mem, pre, before, legacy_query=True)
            g_after = generate(mem, pre, mem.store)
            g_ctx = generate(mem, "\n".join(t["text"] for t in turns) + "\n" + pre, empty_store(mem))
            upd["before"]["new"].append(match(g_before, new["a"]))
            upd["before"]["old"].append(match(g_before, old["a"]))
            upd["after"]["new"].append(match(g_after, new["a"]))
            upd["after"]["old"].append(match(g_after, old["a"]))
            upd["full_session"]["new"].append(match(g_ctx, new["a"]))
            upd["full_session"]["old"].append(match(g_ctx, old["a"]))
        slot = [s for s in mem.store.slots if (s.name, s.rel) == (new["n"], new["rel"])]
        old_slots = [s for s in before.slots if (s.name, s.rel) == (old["n"], old["rel"])]
        upd.setdefault("after_slot", []).append(len(slot) == 1)
        upd.setdefault("before_slot", []).append(len(old_slots) == 1)
        if len(slot) == 1:
            new_v = mem.encode_value(new["text"], new["a"])
            old_v = mem.encode_value(old["text"], old["a"])
            upd.setdefault("after_value_is_new", []).append(bool(np.allclose(slot[0].v, new_v)))
            upd.setdefault("after_value_is_old", []).append(bool(np.allclose(slot[0].v, old_v)))
    if not args.skip_gen:
        report["update"] = {
            key: {
                "new": rate(val["new"])["acc"],
                "old": rate(val["old"])["acc"],
                "n": len(val["new"]),
            }
            for key, val in upd.items()
            if isinstance(val, dict) and "new" in val
        }
    report["update_identity_slot"] = {
        "after": rate(upd.get("after_slot", [])),
        "before": rate(upd.get("before_slot", [])),
        "after_value_is_new": rate(upd.get("after_value_is_new", [])),
        "after_value_is_old": rate(upd.get("after_value_is_old", [])),
        "same_name_key_cos_mean": float(np.mean(same_cos)) if same_cos else None,
        "same_name_key_cos_min": float(np.min(same_cos)) if same_cos else None,
        "n": len(same_cos),
    }
    print("update", json.dumps({k: report[k] for k in report if str(k).startswith("update")}), flush=True)

    for K in [int(x) for x in args.k.split(",") if x != ""]:
        reps = args.k_reps if K < 100 else max(2, args.k_reps // 2)
        hits_b, hits_a, slots_b, slots_a = [], [], [], []
        for b in range(reps):
            stream = fact_stream(random.Random(77 + K + b), "test", K)
            before = fill(mem, stream)
            slots_b.append(len(before))
            slots_a.append(len(mem.store))
            # Query each fact once, score both stores.
            for turn in stream:
                q_before, q_after = query_pair(mem, turn["q"])
                hits_b.append(ret1(before, q_before, turn))
                hits_a.append(ret1(mem.store, q_after, turn))
        report[f"K{K}"] = {
            "ret1_before": rate(hits_b),
            "ret1_after": rate(hits_a),
            "slots_before_mean": float(np.mean(slots_b)),
            "slots_after_mean": float(np.mean(slots_a)),
            "facts_per_stream": K,
            "reps": reps,
        }
        print(f"K{K}", json.dumps(report[f"K{K}"]), f"{time.time() - t0:.0f}s", flush=True)

    stream = fact_stream(random.Random(99), "test", 20)
    raw, fixed, paras = [], [], []
    for turn in stream:
        legacy, pooled, _v = _spans(mem, turn["text"], turn["a"])
        raw.append(legacy)
        fixed.append((pooled - mem.key_mean).astype(np.float64))
        _leg_p, para_raw, _vp = _spans(mem, turn["para"], turn["a"])
        paras.append((para_raw - mem.key_mean).astype(np.float64))
    para_to_self = [_cos(a, b) for a, b in zip(fixed, paras)]
    para_to_other = []
    for i, para in enumerate(paras):
        para_to_other.append(max(_cos(para, fixed[j]) for j in range(len(fixed)) if j != i))
    report["cosine20"] = {
        "before": _offdiag(raw),
        "after": _offdiag(fixed),
        "paraphrase_to_own_mean": float(np.mean(para_to_self)),
        "paraphrase_to_own_min": float(np.min(para_to_self)),
        "paraphrase_to_other_max": float(np.max(para_to_other)),
        "paraphrase_to_other_mean": float(np.mean(para_to_other)),
    }
    print("cosine20", json.dumps(report["cosine20"]), flush=True)
    report["examples"] = examples
    report["time_s"] = time.time() - t0
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print("done", report["time_s"], flush=True)
    mem.close()


if __name__ == "__main__":
    main()
