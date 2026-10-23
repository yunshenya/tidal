"""Natural-question free-gen and soft top-k ablation on frozen SmolLM2-135M.

Same held-out sessions as ``scripts/eval_llm_memory.py`` (seed ``5000+N``,
``tidal.llm_memory_data.session``). Retrieval is reported for the natural
question and for the answer-stripped prefix, because ``generate`` queries
with the same string it feeds the LM.

Free-gen is greedy, at most ``--max-new`` new tokens, float32. Each readout
is scored with four decoders on those tokens:

- ``v22`` cuts before a newline, then on sentence punctuation (the old stop;
  a leading newline scores empty)
- ``span`` skips leading whitespace, then applies that same stop
  (``tidal.llm_memory.decode_answer``, what ``LLMMemory.generate`` returns).
  ``LLMMemory`` itself reads hard top-1; this script still arms every spec.
- ``line`` skips leading whitespace and cuts at the next newline only
- ``body`` skips leading whitespace and keeps the rest of the budget

Readout specs (injected value only; ret@1 is argmax and does not change):

- soft top-4, T=0.05 (the mix ``generate`` used before this ablation)
- hard top-1
- soft top-4, T=0.20
- soft top-2, T=0.05
- margin 0.05: top-1 when the cosine gap exceeds 0.05, else soft top-4 at 0.05

Fact-in-context and bge RAG append the chosen line, a newline, and the same
prompt, with an empty store. bge retrieves with the natural question.

    python scripts/eval_llm_question.py --n 48 --Ns 0,4,16,32
    python scripts/eval_llm_question.py --probe 2
"""
from __future__ import annotations

import argparse
import json
import random
import time

import numpy as np
import torch

from tidal.episodic import cosine_scores
from tidal.llm_memory import LLMMemory, decode_answer
from tidal.llm_memory_data import fact, match, session

MAX_NEW_DEFAULT = 16

SPECS = (
    {"name": "soft4_t05", "mode": "softtopk", "k": 4, "temp": 0.05, "margin": None},
    {"name": "top1", "mode": "top1", "k": 1, "temp": 0.05, "margin": None},
    {"name": "soft4_t20", "mode": "softtopk", "k": 4, "temp": 0.20, "margin": None},
    {"name": "soft2_t05", "mode": "softtopk", "k": 2, "temp": 0.05, "margin": None},
    {"name": "margin05", "mode": "softtopk", "k": 4, "temp": 0.05, "margin": 0.05},
)

DECODERS = ("v22", "span", "line", "body")


def prefix_of(turn: dict) -> str:
    text, answer = turn["text"], turn["a"]
    j = text.rfind(answer)
    if j < 0:
        raise ValueError("answer is not in the fact text")
    return text[:j]


def legacy_v22(tok, ids) -> str:
    out = []
    for nx in ids:
        s = tok.decode(out + [int(nx)], skip_special_tokens=True)
        if "\n" in s:
            break
        out.append(int(nx))
        if s.rstrip().endswith((".", "?", "!")) and len(s.strip()) > 1:
            break
    return tok.decode(out, skip_special_tokens=True)


def content_line(text: str) -> str:
    return text.lstrip().split("\n", 1)[0]


def decode_all(tok, ids) -> dict:
    raw = tok.decode(list(ids), skip_special_tokens=True)
    return {
        "v22": legacy_v22(tok, ids),
        "span": decode_answer(tok, ids),
        "line": content_line(raw),
        "body": raw.lstrip(),
        "raw": raw,
    }


def _scores(store, query) -> np.ndarray:
    slots = list(store.slots)
    if not slots:
        return np.zeros(0, dtype=np.float64)
    keys = np.stack([slot.k for slot in slots])
    return cosine_scores(keys, np.asarray(query, dtype=np.float64))


def cosine_gap(store, query):
    scores = _scores(store, query)
    if scores.size < 2:
        return None
    order = np.argsort(-scores)
    return float(scores[order[0]] - scores[order[1]])


def mix_value(store, query, spec) -> np.ndarray:
    """Injected value for one ablation spec. Restores the store's soft settings."""
    saved = (store.soft_k, store.soft_temp)
    store.soft_k = int(spec["k"])
    store.soft_temp = float(spec["temp"])
    try:
        if spec["margin"] is None:
            return store.readout(query, mode=spec["mode"])
        scores = _scores(store, query)
        if scores.size == 0:
            dim = store.dv if store.dv is not None else 0
            return np.zeros(int(dim), dtype=np.float64)
        order = np.argsort(-scores)
        gap = float("inf") if scores.size < 2 else float(scores[order[0]] - scores[order[1]])
        if gap > float(spec["margin"]):
            value, _info = store.read(query, mode="top1")
            if value is None:
                dim = store.dv if store.dv is not None else 0
                return np.zeros(int(dim), dtype=np.float64)
            return value
        return store.readout(query, mode="softtopk")
    finally:
        store.soft_k, store.soft_temp = saved


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def purity(store, query, spec, turn) -> float:
    value = mix_value(store, query, spec)
    for slot in store.slots:
        if (slot.name, slot.rel) == (turn["n"], turn["rel"]):
            return _cos(value, slot.v)
    return float("nan")


def cached_greedy(mem, prompt: str, max_new_tokens: int) -> list:
    """Greedy ids with KV cache. The caller has already armed ``mem._delta``."""
    tok = mem.tokenizer
    device = mem._device
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    ids = enc["input_ids"]
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ids = ids.to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        if not torch.is_tensor(attn):
            attn = torch.tensor(attn, dtype=torch.long)
        if attn.dim() == 1:
            attn = attn.unsqueeze(0)
        attn = attn.to(device)
    past = None
    new_ids: list[int] = []
    eos = getattr(tok, "eos_token_id", None)
    for _ in range(int(max_new_tokens)):
        kwargs = {"use_cache": True}
        if past is None:
            kwargs["input_ids"] = ids
        else:
            kwargs["input_ids"] = ids[:, -1:]
            kwargs["past_key_values"] = past
        if attn is not None:
            kwargs["attention_mask"] = attn
        out = mem.model(**kwargs)
        past = out.past_key_values
        nxt = int(torch.argmax(out.logits[0, -1]).item())
        if eos is not None and nxt == int(eos):
            break
        step = torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, step], dim=1)
        if attn is not None:
            attn = torch.cat(
                [attn, torch.ones((1, 1), device=attn.device, dtype=attn.dtype)],
                dim=1,
            )
        new_ids.append(nxt)
    return new_ids


def generate_ids(mem, prompt: str, spec, max_new: int, use_cache: bool) -> list:
    """Greedy ids with ``spec`` armed. Empty store adds nothing.

    Hard top-1 uses ``continuation_ids`` (one query, relation-gated spelling,
    gap scale). A soft mix still spells from hard top-1 first, then arms the
    mix once. That mix is the ablation, not the main path.
    """
    real = mem._arm

    def arm(text, _spec=spec):
        if len(mem.store) == 0:
            mem._delta = None
            return
        query = mem._query_vector(text)
        mem._delta = mem._compose(mix_value(mem.store, query, _spec), grad=False)

    main = spec.get("mode") == "top1" and spec.get("margin") is None
    mem._arm = arm
    try:
        if main:
            if use_cache:
                rank = mem._rank(prompt)
                spelled = mem._spell_ids(prompt, rank, max_new)
                if spelled is not None:
                    return spelled
                mem._set_delta(rank)
                try:
                    with torch.no_grad():
                        return cached_greedy(mem, prompt, max_new)
                finally:
                    mem._delta = None
            return mem.continuation_ids(prompt, max_new_tokens=max_new)
        scripted = mem.scripted_continuation(prompt, max_new)
        if scripted is not None:
            return scripted
        mem._arm(prompt)
        try:
            if not use_cache:
                return mem.continuation_ids(prompt, max_new_tokens=max_new, arm=False)
            with torch.no_grad():
                return cached_greedy(mem, prompt, max_new)
        finally:
            mem._delta = None
    finally:
        mem._arm = real


def remember_turns(mem, turns) -> None:
    mem.store.clear()
    for turn in turns:
        if turn["kind"] != "fact":
            continue
        mem.remember(turn["text"], turn["a"], name=turn["n"], rel=turn["rel"])


def ret1(store, query, turn) -> bool:
    _value, info = store.read(query, mode="top1")
    return info.get("top") == (turn["n"], turn["rel"])


def rate(hits) -> dict:
    n = len(hits)
    return {"acc": (sum(hits) / n) if n else None, "hits": int(sum(hits)), "n": n}


def _nums(xs):
    vals = []
    for x in xs:
        if x is None:
            continue
        try:
            value = float(x)
        except (TypeError, ValueError):
            continue
        if value == value:
            vals.append(value)
    return vals


def _mean(xs) -> float | None:
    vals = _nums(xs)
    if not vals:
        return None
    return float(np.mean(vals))


def _median(xs) -> float | None:
    vals = _nums(xs)
    if not vals:
        return None
    return float(np.median(vals))


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


def _blank_hits():
    return {dec: [] for dec in DECODERS}


def _empty_hits():
    return []


def cache_matches(mem, use_cache_candidate: bool) -> bool:
    """One armed prompt: cached ids must equal the uncached main-path loop."""
    if not use_cache_candidate:
        return False
    probe = fact(random.Random(0), "test")
    mem.store.clear()
    mem.remember(probe["text"], probe["a"], name=probe["n"], rel=probe["rel"])
    spec = SPECS[0]
    prompts = (probe["q"], probe["text"] + "\n" + prefix_of(probe))
    try:
        ok = True
        for prompt in prompts:
            plain = generate_ids(mem, prompt, spec, 12, use_cache=False)
            cached = generate_ids(mem, prompt, spec, 12, use_cache=True)
            if plain != cached:
                ok = False
                print("cache mismatch", repr(prompt[:80]), plain, cached, flush=True)
    except Exception as exc:
        print("cache disabled", type(exc).__name__, exc, flush=True)
        return False
    finally:
        mem.store.clear()
    print("cache_matches", ok, flush=True)
    if not ok:
        print("plain", plain, flush=True)
        print("cached", cached, flush=True)
    return ok


def sessions_for(n: int, N: int):
    rng = random.Random(5000 + N)
    rows = []
    for _ in range(n):
        turns = session(rng, "test", N)
        facts = [t for t in turns if t["kind"] == "fact"]
        target = rng.choice(facts)
        rows.append((turns, target))
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Question free-gen and soft top-k ablation.")
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--Ns", default="0,4,16,32")
    p.add_argument("--max-new", type=int, default=MAX_NEW_DEFAULT)
    p.add_argument("--probe", type=int, default=0, help="Print this many raw sessions at N0 and N16, then exit.")
    p.add_argument("--skip-bge", action="store_true")
    p.add_argument("--skip-prefix", action="store_true", help="Score natural questions only.")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)
    if args.probe:
        args.n = int(args.probe)
        args.Ns = "0,16"
        args.max_new = max(int(args.max_new), 32)

    torch.set_num_threads(4)
    torch.manual_seed(0)
    t0 = time.time()
    mem = LLMMemory(dtype=torch.float32)
    print(
        f"model {mem.model_name} layer {mem.layer} hidden {mem.hidden} "
        f"dtype {next(mem.model.parameters()).dtype} soft_k {mem.store.soft_k} "
        f"soft_temp {mem.store.soft_temp} max_new {args.max_new}",
        flush=True,
    )
    use_cache = cache_matches(mem, not args.no_cache)

    bge = None
    if not args.skip_bge:
        try:
            bge = BGE()
            print("bge loaded", flush=True)
        except Exception as exc:
            print("bge skipped", type(exc).__name__, exc, flush=True)

    prompts = ("question",) if args.skip_prefix else ("question", "prefix")
    specs = SPECS if not args.probe else SPECS[:2]
    report = {
        "args": vars(args),
        "dtype": str(next(mem.model.parameters()).dtype).replace("torch.", ""),
        "max_new_tokens": args.max_new,
        "use_cache": use_cache,
        "specs": [spec["name"] for spec in specs],
        "decoders": list(DECODERS),
        "note": (
            "span is LLMMemory.generate. v22 is the old newline-first stop. "
            "ret@1 is argmax cosine and does not depend on the mix."
        ),
    }
    examples = []

    for N in [int(x) for x in args.Ns.split(",") if x != ""]:
        rows = sessions_for(args.n, N)
        ret_q, ret_p = [], []
        gap_q, gap_p = [], []
        pure = {spec["name"]: {prompt: [] for prompt in prompts} for spec in specs}
        gen = {spec["name"]: {prompt: _blank_hits() for prompt in prompts} for spec in specs}
        empty = {spec["name"]: {prompt: _empty_hits() for prompt in prompts} for spec in specs}
        base = {
            name: {prompt: _blank_hits() for prompt in prompts}
            for name in ("no_memory", "fact_in_context", "rag_bge")
        }
        base_empty = {
            name: {prompt: _empty_hits() for prompt in prompts}
            for name in ("no_memory", "fact_in_context", "rag_bge")
        }
        rag_hit = []
        for i, (turns, target) in enumerate(rows):
            example_at = len(examples)
            remember_turns(mem, turns)
            q_text = target["q"]
            pre = prefix_of(target)
            text_for = {"question": q_text, "prefix": pre}
            q_vec = mem._query_vector(q_text)
            p_vec = mem._query_vector(pre)
            ret_q.append(ret1(mem.store, q_vec, target))
            ret_p.append(ret1(mem.store, p_vec, target))
            gap_q.append(cosine_gap(mem.store, q_vec))
            gap_p.append(cosine_gap(mem.store, p_vec))
            vec_for = {"question": q_vec, "prefix": p_vec}
            for spec in specs:
                for prompt in prompts:
                    pure[spec["name"]][prompt].append(
                        purity(mem.store, vec_for[prompt], spec, target)
                    )

            def _score(bucket, empty_bucket, prompt_name, ids):
                decoded = decode_all(mem.tokenizer, ids)
                for dec in DECODERS:
                    bucket[prompt_name][dec].append(match(decoded[dec], target["a"]))
                empty_bucket[prompt_name].append(decoded["span"].strip() == "")
                return decoded

            saved = mem.store
            try:
                mem.store = type(saved)(
                    capacity=1,
                    dk=mem.hidden,
                    dv=mem.hidden,
                    overwrite_cos=saved.overwrite_cos,
                    soft_k=saved.soft_k,
                    soft_temp=saved.soft_temp,
                )
                for prompt in prompts:
                    ids = generate_ids(mem, text_for[prompt], specs[0], args.max_new, use_cache)
                    decoded = _score(base["no_memory"], base_empty["no_memory"], prompt, ids)
                    if args.probe and prompt == "question":
                        examples.append({"N": N, "spec": "no_memory", "prompt": prompt, "q": q_text, "gold": target["a"], "raw": decoded["raw"], "span": decoded["span"], "v22": decoded["v22"]})
                    ctx = target["text"] + "\n" + text_for[prompt]
                    ids = generate_ids(mem, ctx, specs[0], args.max_new, use_cache)
                    decoded = _score(base["fact_in_context"], base_empty["fact_in_context"], prompt, ids)
                    if args.probe and prompt == "question":
                        examples.append({"N": N, "spec": "fact_in_context", "prompt": prompt, "q": q_text, "gold": target["a"], "raw": decoded["raw"], "span": decoded["span"], "v22": decoded["v22"]})
                    if bge is not None:
                        picked = bge.hit(turns, q_text)
                        if prompt == "question":
                            rag_hit.append(picked is target)
                        rag_prompt = picked["text"] + "\n" + text_for[prompt]
                        ids = generate_ids(mem, rag_prompt, specs[0], args.max_new, use_cache)
                        _score(base["rag_bge"], base_empty["rag_bge"], prompt, ids)
            finally:
                mem.store = saved

            for spec in specs:
                for prompt in prompts:
                    ids = generate_ids(mem, text_for[prompt], spec, args.max_new, use_cache)
                    decoded = _score(gen[spec["name"]], empty[spec["name"]], prompt, ids)
                    if args.probe or (i < 2 and spec["name"] in ("soft4_t05", "top1") and prompt == "question"):
                        examples.append(
                            {
                                "N": N,
                                "spec": spec["name"],
                                "prompt": prompt,
                                "q": q_text,
                                "gold": target["a"],
                                "raw": decoded["raw"],
                                "v22": decoded["v22"],
                                "span": decoded["span"],
                                "line": decoded["line"],
                                "body": decoded["body"],
                                "ret_q": bool(ret_q[-1]),
                                "ret_prefix": bool(ret_p[-1]),
                            }
                        )
            if args.probe:
                print("PROBE", json.dumps(examples[example_at:], ensure_ascii=False), flush=True)
            elif (i + 1) % 4 == 0 or i + 1 == len(rows):
                print(f"N{N} {i + 1}/{len(rows)} {time.time() - t0:.0f}s", flush=True)

        def _pack(hits):
            return {dec: rate(hits[dec]) for dec in DECODERS}

        cell = {
            "ret1_question": rate(ret_q),
            "ret1_prefix": rate(ret_p),
            "gap_question": {
                "mean": _mean(gap_q),
                "median": _median(gap_q),
                "frac_gt_0.05": _mean([g > 0.05 for g in gap_q if g is not None]),
                "n_defined": sum(g is not None for g in gap_q),
            },
            "gap_prefix": {
                "mean": _mean(gap_p),
                "median": _median(gap_p),
                "frac_gt_0.05": _mean([g > 0.05 for g in gap_p if g is not None]),
                "n_defined": sum(g is not None for g in gap_p),
            },
            "purity": {
                name: {prompt: _mean(pure[name][prompt]) for prompt in prompts} for name in pure
            },
            "purity_lt_0.90": {
                name: {
                    prompt: rate([c < 0.90 for c in pure[name][prompt]]) for prompt in prompts
                }
                for name in pure
            },
            "empty_span": {
                name: {prompt: rate(empty[name][prompt]) for prompt in prompts} for name in empty
            },
        }
        for name in gen:
            cell[name] = {prompt: _pack(gen[name][prompt]) for prompt in prompts}
        for name in base:
            if any(base[name][prompt]["span"] for prompt in prompts):
                cell[name] = {prompt: _pack(base[name][prompt]) for prompt in prompts}
                cell[name + "_empty_span"] = {
                    prompt: rate(base_empty[name][prompt]) for prompt in prompts
                }
        if rag_hit:
            cell["rag_hit"] = rate(rag_hit)
        report[f"N{N}"] = cell
        print(f"N{N}", json.dumps(cell), f"{time.time() - t0:.0f}s", flush=True)
        if args.out:
            report["examples"] = examples
            report["time_s"] = time.time() - t0
            with open(args.out, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(report, indent=1))

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
