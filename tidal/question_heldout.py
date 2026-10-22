"""Realistic held-out questions for ``Continuum.ask``. CPU only.

The shipped head was fit by ``question_head.build_facts`` (seed 0, 36 names,
4 relation ids). That fit has no sentences. A training question is

    embed_event(_feats(name, rel, True), name_unit + reversed(rel_unit))

and the stored key is the same name unit plus the forward relation unit plus
``0.3 * value_unit``, with Gaussian unit vectors from ``default_rng(0)``.
``heads`` / ``recall`` / replay stay on raw cosine. On these flat rows
``ask`` / ``query`` rank by sentence cosine plus the relation logit from
``question_surface`` (the sentence vector is the one passed into
``embed_event``). The same path returns None when the nearest stored person
does not have the asked relation. Sparse name/relation rows still use the
contrastive head and do not abstain.
The decode is the same top-1 byte code either way.

This probe does not refit that head. It asks Chinese and English surface
questions (``question_heldout_data``), embeds each string with frozen
bge-small-zh-v1.5 (CLS, L2, 64 tokens, no retrieval instruction — the same
contract as ``tidal.embed``), and runs the vector through ``embed_event``.
Flat feature rows (this probe's 8-d hash) take that function's constant-offset
correction; the training ``_feats`` rows do not. The cache
``question_heldout_emb.npz`` keeps the run offline. Feature rows are an 8-d
hash, not ``_feats``.

Disjointness is checked against that training construction: feature rows,
pre-encoder text vectors, encoder states, and string literals in
``question_head.py``. Hit counts are measurements. They are not pass/fail
gates. Empty-store zeros, same-identity overwrite (no stale code), chat
rejection, and replay ignoring the question head still fail the process.

Run: ``python -m tidal.question_heldout``
Also printed from ``python -m tidal.episodic_eval``.
Rebuild the embedding cache (needs transformers, not used by tests):
``python -m tidal.question_heldout --cache``
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tidal.question_heldout_data import (
    ABSENT,
    CATEGORIES,
    PEOPLE,
    build_cases,
    corpus_summary,
    probe_strings,
    validate_corpus,
    _tokens,
)
from tidal.question_surface import (
    EXPECTED_BUNDLE_SHA256,
    abstain_dev_curve,
    bundle_sha256,
    choose_abstain_margins,
    corpus_errors,
    corpus_strings,
    load_corpus_table,
    load_weights,
    names_used,
)

BGE_MODEL = "BAAI/bge-small-zh-v1.5"
BGE_MAXLEN = 64
CACHE = Path(__file__).with_name("question_heldout_emb.npz")
# Filled after the cache is written. Empty skips the lock; tests require it set.
EXPECTED_CACHE_SHA256 = "8413bb1a4a964d26042ca61338745ccef1a2494643b54eafa074a5d5f7c20bb3"

_THRESHOLDS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
_TALLY_FIELDS = (
    "n", "cosine_hits", "head_hits", "identity_cosine", "identity_head",
    "both", "head_only", "cosine_only", "neither",
    "phone_n", "phone_cosine", "phone_head",
    "date_n", "date_cosine", "date_head",
    "token_n", "token_cosine", "token_head",
    "stale_n", "stale_cosine", "stale_head",
    "false_n", "false_cosine", "false_head",
    "abstain_cosine", "abstain_head", "false_abstain",
    "gold_present", "lost_09",
    "jaccard_n",
)


def _new_tally() -> Dict[str, float]:
    row = {key: 0 for key in _TALLY_FIELDS}
    row["jaccard_sum"] = 0.0
    return row


def _add_tally(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key in _TALLY_FIELDS:
        dst[key] += src[key]
    dst["jaccard_sum"] += src["jaccard_sum"]


def _kind_fields(rel: str) -> Tuple[str, str, str]:
    if rel == "phone":
        return "phone_n", "phone_cosine", "phone_head"
    if rel == "birthday":
        return "date_n", "date_cosine", "date_head"
    return "token_n", "token_cosine", "token_head"


def surface_feats(text: str, role: str) -> np.ndarray:
    """8-d row in ``{0.2, 0.4, 0.6, 0.8}``. Not the training ``_feats`` pattern.

    Training rows use ``f[0] in {+1, -1}``, a single 1 in ``f[1:4]``, a single
    0.5 in ``f[4:8]``, and zeros elsewhere. These rows have no zeros.
    """
    digest = hashlib.sha256(f"{role}\n{text}".encode("utf-8")).digest()
    feats = np.empty(8, np.float32)
    for i in range(8):
        feats[i] = np.float32((1 + (digest[i] % 4)) * 0.2)
    return feats


def _unit_row(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-12:
        return arr
    return arr / nrm


def _name_in_text(name: str, text: str) -> bool:
    """Whole-name hit. Latin names are not prefixes; CJK names are exact spans."""
    import re
    if any("\u4e00" <= ch <= "\u9fff" for ch in name):
        return name in text
    return re.search(rf"(?i)(?<![A-Za-z]){re.escape(name)}(?![A-Za-z])", text) is not None


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _literals_in_question_head() -> List[str]:
    import tidal.question_head as qh
    tree = ast.parse(Path(qh.__file__).read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append(node.value)
    return found


def _training_text_and_feats():
    """Rebuild the pre-encoder inputs ``build_facts`` consumes, in its order."""
    from tidal.question_head import DEFAULT_SEED, N_NAMES, N_RELS, _feats, _unit

    rng = np.random.default_rng(DEFAULT_SEED)
    name = _unit(rng, (N_NAMES, 512))
    rel_k = _unit(rng, (N_RELS, 512))
    value = _unit(rng, (N_NAMES, N_RELS, 512))
    rel_q = rel_k[:, ::-1].copy()
    feat_rows = []
    text_rows = []
    feat_k, text_k, feat_q, text_q = [], [], [], []
    for n in range(N_NAMES):
        for r in range(N_RELS):
            fk = _feats(n, r, False)
            fq = _feats(n, r, True)
            tk = (name[n] + rel_k[r] + 0.3 * value[n, r]).astype(np.float32)
            tq = (name[n] + rel_q[r]).astype(np.float32)
            feat_k.append(fk)
            text_k.append(tk)
            feat_q.append(fq)
            text_q.append(tq)
            feat_rows.extend((fk, fq))
            text_rows.extend((tk, tq))
    return {
        "feat_k": np.stack(feat_k),
        "text_k": np.stack(text_k),
        "feat_q": np.stack(feat_q),
        "text_q": np.stack(text_q),
        "feat_rows": np.stack(feat_rows),
        "text_rows": np.stack(text_rows),
    }


def load_text_table(path: Path = CACHE) -> Tuple[Dict[str, np.ndarray], dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path.name}; build it with python -m tidal.question_heldout --cache"
        )
    with np.load(path, allow_pickle=False) as blob:
        texts = json.loads(str(blob["texts"].item()))
        emb = np.array(blob["emb"], dtype=np.float32, copy=True)
        meta = {
            "model": str(blob["model"].item()),
            "revision": str(blob["revision"].item()),
            "maxlen": int(blob["maxlen"]),
        }
    if emb.ndim != 2 or emb.shape[1] != 512 or emb.shape[0] != len(texts):
        raise RuntimeError(f"embedding cache shape {emb.shape} does not match {len(texts)} texts")
    if meta["model"] != BGE_MODEL or meta["maxlen"] != BGE_MAXLEN:
        raise RuntimeError(f"embedding cache contract {meta} != {BGE_MODEL} maxlen {BGE_MAXLEN}")
    table = {text: emb[i] for i, text in enumerate(texts)}
    return table, meta


def cache_sha256(path: Path = CACHE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_embedding_cache(path: Path = CACHE) -> dict:
    """Encode ``probe_strings()`` with bge-small-zh-v1.5. Not called by tests."""
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer

    texts = list(probe_strings())
    tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL)
    model = AutoModel.from_pretrained(BGE_MODEL)
    model.eval()
    rows = []
    for start in range(0, len(texts), 32):
        chunk = texts[start:start + 32]
        batch = tokenizer(
            chunk, padding=True, truncation=True, max_length=BGE_MAXLEN, return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
        rows.append(hidden.cpu().numpy().astype(np.float32))
    emb = np.concatenate(rows, 0)
    revision = model_info(BGE_MODEL).sha
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb=emb,
        texts=np.asarray(json.dumps(texts)),
        model=np.asarray(BGE_MODEL),
        revision=np.asarray(revision),
        maxlen=np.int64(BGE_MAXLEN),
    )
    return {"n": len(texts), "revision": revision, "sha256": cache_sha256(path), "path": str(path)}


def _stored_values(slots) -> List[Any]:
    from tidal.episodic import decode_span, decode_tokens

    out = []
    for slot in slots:
        text = decode_span(slot.v)
        out.append(text if text is not None else decode_tokens(slot.v))
    return out


def _pct(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    out = {str(p): float(np.percentile(arr, p)) for p in (10, 25, 50, 75, 90)}
    out["min"] = float(arr.min())
    out["max"] = float(arr.max())
    out["mean"] = float(arr.mean())
    return out


def _sweep(margins: Sequence[float], hits: Sequence[bool], unanswerable: Sequence[float]) -> List[dict]:
    ans_m = np.asarray(list(margins), dtype=np.float64)
    ans_h = np.asarray(list(hits), dtype=bool)
    un_m = np.asarray(list(unanswerable), dtype=np.float64)
    rows = []
    for threshold in _THRESHOLDS:
        keep = ans_m >= threshold if ans_m.size else ans_m
        still = un_m >= threshold if un_m.size else un_m
        rows.append({
            "t": float(threshold),
            "answerable_kept": int(np.sum(keep)) if ans_m.size else 0,
            "answerable_n": int(ans_m.size),
            "answerable_hits_kept": int(np.sum(ans_h & keep)) if ans_m.size else 0,
            "unanswerable_still": int(np.sum(still)) if un_m.size else 0,
            "unanswerable_n": int(un_m.size),
        })
    return rows


def _resid_ratio(head, rows: np.ndarray, kind: str) -> np.ndarray:
    from tidal.question_head import project

    projected = project(head, rows, kind)
    num = np.linalg.norm(projected - rows, axis=1)
    den = np.linalg.norm(rows, axis=1) + 1e-12
    return num / den


def _margin(scores: np.ndarray) -> Tuple[float, float]:
    order = np.argsort(-scores)
    top = float(scores[order[0]])
    second = float(scores[order[1]]) if scores.size > 1 else top
    return top, top - second


class _States:
    def __init__(self, cont, table: Dict[str, np.ndarray]):
        self.cont = cont
        self.table = table
        self.cache: Dict[Tuple[str, str], np.ndarray] = {}
        self.feats: Dict[Tuple[str, str], np.ndarray] = {}

    def get(self, text: str, role: str) -> np.ndarray:
        key = (role, text)
        if key not in self.cache:
            feats = surface_feats(text, role)
            self.feats[key] = feats
            self.cache[key] = self.cont.embed_event(feats, self.table[text])
        return self.cache[key]


def _api_answer(cont, question: np.ndarray, which: str):
    if which == "head":
        text = cont.ask(question, group=None)
        if text is not None:
            return text
        return cont.ask_tokens(question, group=None)
    text = cont.recall(question, group=None)
    if text is not None:
        return text
    return cont.recall_tokens(question, group=None)


def _top_value(cont, scores: np.ndarray):
    from tidal.episodic import decode_span, decode_tokens

    value, info = cont.episodic.read_scored(scores, mode="top1", group=None)
    if value is None:
        return None, info
    text = decode_span(value)
    if text is not None:
        return text, info
    return decode_tokens(value), info


def _decode_query_value(value):
    from tidal.episodic import decode_span, decode_tokens

    if value is None:
        return None
    text = decode_span(value)
    if text is not None:
        return text
    return decode_tokens(value)


def eval_heldout(check) -> Dict[str, Any]:
    """Measure ``ask`` on surface questions. ``check`` records invariants only.

    A low ret@1 does not call ``check(..., False, ...)``.
    """
    import torch

    from tidal.episodic import EpisodicStore, cosine_scores
    from tidal.question_head import build_facts, load_meta, load_reference

    cases = build_cases()
    summary = corpus_summary(cases)
    probe_errors = validate_corpus(cases)
    check("heldout", not probe_errors, "; ".join(probe_errors[:4]) or "corpus")
    check("heldout", summary["n"] >= 300, f"n={summary['n']}")
    for cat in CATEGORIES:
        check("heldout", summary["frames"].get(cat, 0) >= 12, f"{cat} frames {summary['frames'].get(cat)}")

    table, emb_meta = load_text_table()
    missing = [text for text in probe_strings(cases) if text not in table]
    check("heldout", not missing, f"cache missing {missing[:3]}")
    if missing:
        raise RuntimeError(f"embedding cache is missing {len(missing)} probe strings")
    norms = np.linalg.norm(np.stack([table[t] for t in probe_strings(cases)]), axis=1)
    check("heldout", float(np.max(np.abs(norms - 1.0))) < 1e-3, "bge rows are not unit length")
    sha = cache_sha256()
    if EXPECTED_CACHE_SHA256:
        check("heldout", sha == EXPECTED_CACHE_SHA256, f"cache sha {sha}")

    prev_mkldnn = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        cont = load_reference(overwrite_cos=1.0)
        check("heldout", cont.question_head.fitted is True, "shipped head did not install")
        check("heldout", cont.episodic.overwrite_cos == 1.0, "ranking probe overwrite_cos")
        facts = build_facts(cont)
        rebuilt = _training_text_and_feats()
        recon_ok = True
        for i in range(len(facts["k"])):
            got_k = cont.embed_event(rebuilt["feat_k"][i], rebuilt["text_k"][i])
            got_q = cont.embed_event(rebuilt["feat_q"][i], rebuilt["text_q"][i])
            if not (np.array_equal(got_k, facts["k"][i]) and np.array_equal(got_q, facts["q"][i])):
                recon_ok = False
                break
        check("heldout", recon_ok, "training-input reconstruction does not match build_facts")

        states = _States(cont, table)
        # Touch every probe string once so disjointness sees the full set.
        for case in cases:
            states.get(case.question, "question")
            if case.statement:
                states.get(case.statement, "statement")
            for write in case.writes:
                role = "chat" if write.kind == "chat" else "statement"
                states.get(write.text, role)

        feat_hit = 0
        train_feats = rebuilt["feat_rows"]
        for feats in states.feats.values():
            if np.any(np.all(train_feats == feats, axis=1)):
                feat_hit += 1
        state_hit = 0
        train_states = np.concatenate([facts["q"], facts["k"]], 0)
        for row in states.cache.values():
            if np.any(np.all(train_states == row, axis=1)):
                state_hit += 1
        text_mat = np.stack([
            np.asarray(table[text], dtype=np.float64) for _role, text in states.cache
        ])
        train_text = rebuilt["text_rows"].astype(np.float64)
        train_text = train_text / np.linalg.norm(train_text, axis=1, keepdims=True)
        text_unit = text_mat / np.linalg.norm(text_mat, axis=1, keepdims=True)
        max_text_cos = float(np.max(text_unit @ train_text.T))
        literals = _literals_in_question_head()
        meta_blob = json.dumps(load_meta())
        literal_hits = 0
        questions = [case.question for case in cases]
        for question in questions:
            if question in literals or question in meta_blob:
                literal_hits += 1
                continue
            for literal in literals:
                if len(literal) >= 24 and (literal in question or question in literal):
                    literal_hits += 1
                    break
        check("heldout", feat_hit == 0, f"feature rows match training _feats ({feat_hit})")
        check("heldout", state_hit == 0, f"encoder states match build_facts ({state_hit})")
        check("heldout", max_text_cos < 0.5, f"bge cosine vs training text {max_text_cos:.4f}")
        check("heldout", literal_hits == 0, f"question strings collide with question_head literals ({literal_hits})")
        surf_err = corpus_errors()
        check("heldout", not surf_err, "; ".join(surf_err[:4]) or "surface corpus")
        surf_texts = corpus_strings()
        probe_set = set(probe_strings(cases))
        surf_literal = sum(1 for text in surf_texts if text in probe_set)
        probe_names = [name for lang in ("en", "zh") for name in (*PEOPLE[lang], *ABSENT[lang])]
        surf_name_hits = 0
        for name in probe_names:
            if any(_name_in_text(name, text) for text in surf_texts):
                surf_name_hits += 1
                continue
            if any(_name_in_text(name, other) or _name_in_text(other, name) for other in names_used()):
                surf_name_hits += 1
        surf_table = load_corpus_table()
        missing_surf = [text for text in surf_texts if text not in surf_table]
        check("heldout", not missing_surf, f"surface bundle missing {missing_surf[:2]}")
        probe_mat = np.stack([np.asarray(table[text], dtype=np.float64) for text in probe_strings(cases)])
        surf_mat = np.stack([_unit_row(surf_table[text]) for text in surf_texts])
        probe_mat = probe_mat / np.linalg.norm(probe_mat, axis=1, keepdims=True)
        surf_cos = surf_mat @ probe_mat.T
        max_surf_cos = float(surf_cos.max())
        flat_i = int(np.argmax(surf_cos))
        si, pi = divmod(flat_i, surf_cos.shape[1])
        max_jaccard = 0.0
        for left in surf_texts:
            for right in probe_set:
                max_jaccard = max(max_jaccard, _jaccard(left, right))
        weights = load_weights()
        check("heldout", weights is not None, "surface relation bundle missing")
        check(
            "heldout",
            bundle_sha256() == EXPECTED_BUNDLE_SHA256,
            f"surface bundle sha {bundle_sha256()}",
        )
        check("heldout", surf_literal == 0, f"surface strings copy the probe ({surf_literal})")
        check("heldout", surf_name_hits == 0, f"surface names collide with the probe ({surf_name_hits})")
        check("heldout", max_surf_cos < 0.98, f"surface bge cosine vs probe {max_surf_cos:.4f}")
        check("heldout", max_jaccard < 0.65, f"surface/probe token jaccard {max_jaccard:.3f}")
        check(
            "heldout",
            cont._sentence_conflicts == 0,
            f"one encoder state mapped to two sentence vectors ({cont._sentence_conflicts})",
        )

        held_q = np.stack([states.get(case.question, "question") for case in cases])
        train_q_ratio = _resid_ratio(cont.question_head, facts["q"], "q")
        held_q_ratio = _resid_ratio(cont.question_head, held_q, "q")
        train_k_ratio = _resid_ratio(cont.question_head, facts["k"], "k")
        held_statements = [case.statement for case in cases if case.statement]
        held_k = np.stack([states.get(text, "statement") for text in held_statements])
        held_k_ratio = _resid_ratio(cont.question_head, held_k, "k")

        cont.episodic.clear()
        cont._mem_group = None
        h = torch.zeros(1, cont.d)
        h[0, 0] = 1
        with torch.no_grad():
            static = cont.m.vap(h)
            empty = cont.heads(h, [])["vap"]
        check("heldout", torch.equal(empty, static), "empty heads are not the static model")
        q0 = states.get(cases[0].question, "question")
        check("heldout", cont.ask(q0, group=None) is None, "empty ask returned a string")
        check("heldout", cont.recall(q0, group=None) is None, "empty recall returned a string")

        tallies = {cat: _new_tally() for cat in CATEGORIES}
        subtypes: Dict[str, Dict[str, float]] = {}
        ans_head_m: List[float] = []
        ans_cos_m: List[float] = []
        ans_head_top: List[float] = []
        ans_cos_top: List[float] = []
        ans_head_hit: List[bool] = []
        ans_cos_hit: List[bool] = []
        un_head_m: List[float] = []
        un_cos_m: List[float] = []
        un_head_top: List[float] = []
        un_cos_top: List[float] = []
        examples: List[dict] = []
        seen_example = set()
        wire_fail = 0
        sent_missing = 0
        q_missing = 0
        fact_fail = 0
        chat_fail = 0
        id_fail = 0
        present_fail = 0
        stale_fail = 0
        update_09_fail = 0
        pair_hi = 0
        max_pair = 0.0
        n_pairs = 0

        for case in cases:
            cont.episodic.clear()
            cont._mem_group = None
            keys = []
            for write in case.writes:
                role = "chat" if write.kind == "chat" else "statement"
                key = states.get(write.text, role)
                keys.append(key)
                if write.kind == "chat":
                    ok = cont.remember(
                        key, write.value, name=write.name, rel=write.rel, kind="chat", group=None,
                    )
                    if ok is not False:
                        chat_fail += 1
                else:
                    ok = cont.remember(
                        key, write.value, name=write.name, rel=write.rel, kind="fact", group=None,
                    )
                    if ok is not True:
                        fact_fail += 1
            fact_writes = [w for w in case.writes if w.kind == "fact"]
            expect_ids = {(w.name, w.rel) for w in fact_writes}
            got_ids = {(s.name, s.rel) for s in cont.episodic.slots}
            if got_ids != expect_ids or len(cont.episodic) != len(expect_ids):
                id_fail += 1
            values_now = _stored_values(cont.episodic.slots)
            if case.gold is not None and case.gold not in values_now:
                present_fail += 1
            if case.old is not None and case.old in values_now:
                stale_fail += 1
            if len(cont.episodic) > 1:
                mat = np.stack([s.k for s in cont.episodic.slots])
                sim = cosine_scores_matrix(mat)
                iu = np.triu_indices(sim.shape[0], 1)
                vals = sim[iu]
                n_pairs += int(vals.size)
                pair_hi += int(np.sum(vals > 0.9))
                max_pair = max(max_pair, float(vals.max()))

            shadow = EpisodicStore(
                capacity=256, dk=cont.d, dv=cont.d, overwrite_cos=0.9, soft_k=4, soft_temp=0.05,
            )
            for write, key in zip(case.writes, keys):
                shadow.write(key, write.value, name=write.name, rel=write.rel, kind=write.kind)
            shadow_values = _stored_values(shadow.slots)
            lost = False
            if case.gold is not None and case.gold not in shadow_values:
                lost = True
            if case.old is not None and (case.gold not in shadow_values or case.old in shadow_values):
                update_09_fail += 1

            q = states.get(case.question, "question")
            if cont._sentence_lookup(q) is None:
                q_missing += 1
            head_ans = _api_answer(cont, q, "head")
            cos_ans = _api_answer(cont, q, "cosine")
            _q_value, q_info = cont.query(q, group=None, mode="top1")
            visible = cont.episodic._visible(None)
            visible_keys = np.stack([s.k for _, s in visible], 0)
            if any(slot.sentence is None for _, slot in visible):
                sent_missing += 1
            raw = cosine_scores(visible_keys, q)
            adapted = cont.rank_question_scores(q, group=None)
            head_top, head_info = _top_value(cont, adapted)
            cos_top, cos_info = _top_value(cont, raw)
            # ask follows query, including None when the sentence ranker abstains.
            # When it answers, that string is still the score argmax. recall stays cosine.
            if _decode_query_value(_q_value) != head_ans or cos_top != cos_ans:
                wire_fail += 1
            if q_info.get("abstain"):
                if head_ans is not None or q_info.get("top") is not None:
                    wire_fail += 1
            elif head_top != head_ans:
                wire_fail += 1
            head_m_top, head_m = _margin(adapted)
            cos_m_top, cos_m = _margin(raw)
            head_hit = case.gold is not None and head_ans == case.gold
            cos_hit = case.gold is not None and cos_ans == case.gold
            head_id = bool(head_info.get("top") == (case.name, case.rel)) if case.gold is not None else False
            cos_id = bool(cos_info.get("top") == (case.name, case.rel)) if case.gold is not None else False

            row = tallies[case.category]
            before = {key: row[key] for key in row}
            row["n"] += 1
            if case.statement:
                row["jaccard_n"] += 1
                row["jaccard_sum"] += _jaccard(case.question, case.statement)
            if case.gold is not None:
                if case.gold in values_now:
                    row["gold_present"] += 1
                if lost:
                    row["lost_09"] += 1
                if head_hit:
                    row["head_hits"] += 1
                if cos_hit:
                    row["cosine_hits"] += 1
                if head_id:
                    row["identity_head"] += 1
                if cos_id:
                    row["identity_cosine"] += 1
                if head_hit and cos_hit:
                    row["both"] += 1
                elif head_hit:
                    row["head_only"] += 1
                elif cos_hit:
                    row["cosine_only"] += 1
                else:
                    row["neither"] += 1
                n_key, c_key, h_key = _kind_fields(case.rel)
                row[n_key] += 1
                if cos_hit:
                    row[c_key] += 1
                if head_hit:
                    row[h_key] += 1
                ans_head_m.append(head_m)
                ans_cos_m.append(cos_m)
                ans_head_top.append(head_m_top)
                ans_cos_top.append(cos_m_top)
                ans_head_hit.append(head_hit)
                ans_cos_hit.append(cos_hit)
                if head_ans is None:
                    row["false_abstain"] += 1
                if not head_hit and case.category not in seen_example:
                    seen_example.add(case.category)
                    examples.append({
                        "category": case.category,
                        "rel": case.rel,
                        "subtype": case.subtype,
                        "question": case.question,
                        "gold": _show(case.gold),
                        "cosine": _show(cos_ans),
                        "head": _show(head_ans),
                    })
            else:
                row["false_n"] += 1
                if head_ans is not None:
                    row["false_head"] += 1
                else:
                    row["abstain_head"] += 1
                if cos_ans is not None:
                    row["false_cosine"] += 1
                else:
                    row["abstain_cosine"] += 1
                un_head_m.append(head_m)
                un_cos_m.append(cos_m)
                un_head_top.append(head_m_top)
                un_cos_top.append(cos_m_top)
            if case.old is not None:
                row["stale_n"] += 1
                if head_ans == case.old:
                    row["stale_head"] += 1
                if cos_ans == case.old:
                    row["stale_cosine"] += 1
            if case.subtype:
                for key in (f"{case.category}/{case.subtype}", case.subtype):
                    bucket = subtypes.setdefault(key, _new_tally())
                    for field, value in row.items():
                        bucket[field] += value - before[field]

        check("heldout", fact_fail == 0, f"fact writes rejected: {fact_fail}")
        check("heldout", chat_fail == 0, f"chat writes stored: {chat_fail}")
        check("heldout", id_fail == 0, f"identity set changed at overwrite_cos=1: {id_fail}")
        check("heldout", present_fail == 0, f"gold value missing after write: {present_fail}")
        check("heldout", stale_fail == 0, f"stale value still stored after overwrite: {stale_fail}")
        check("heldout", update_09_fail == 0, f"live 0.9 overwrite kept a stale update: {update_09_fail}")
        check("heldout", wire_fail == 0, f"ask/recall diverged from top-1 decode: {wire_fail}")
        check("heldout", sent_missing == 0, f"probe fact missing its sentence vector ({sent_missing})")
        check("heldout", q_missing == 0, f"probe question missing its sentence vector ({q_missing})")
        _replay_unchanged(cont, check)

        answerable = _new_tally()
        unanswerable = _new_tally()
        everything = _new_tally()
        by_lang = {
            "answerable_en": _new_tally(),
            "answerable_zh": _new_tally(),
            "unanswerable_en": _new_tally(),
            "unanswerable_zh": _new_tally(),
        }
        for cat, row in tallies.items():
            _add_tally(everything, row)
            is_un = cat.endswith("unanswerable")
            if is_un:
                _add_tally(unanswerable, row)
            else:
                _add_tally(answerable, row)
            lang = "en" if cat.startswith("en_") else "zh"
            _add_tally(by_lang[("unanswerable_" if is_un else "answerable_") + lang], row)
        dev_curve = abstain_dev_curve(weights)
        dev_rm, dev_pm = choose_abstain_margins(dev_curve)
        report = {
            "n": int(summary["n"]),
            "categories": {cat: _public_tally(tallies[cat]) for cat in CATEGORIES},
            "rollups": {
                "all": _public_tally(everything),
                "answerable": _public_tally(answerable),
                "unanswerable": _public_tally(unanswerable),
                **{key: _public_tally(row) for key, row in by_lang.items()},
            },
            "abstain_dev": {
                "rel_margin": dev_rm,
                "person_margin": dev_pm,
                "curve": dev_curve,
            },
            "subtypes": {key: _public_tally(row) for key, row in sorted(subtypes.items())},
            "frames": {key: int(value) for key, value in summary["frames"].items()},
            "disjoint": {
                "feature_collisions": int(feat_hit),
                "state_collisions": int(state_hit),
                "max_text_cosine_vs_training": float(max_text_cos),
                "literal_hits": int(literal_hits),
                "n_training_pairs": int(len(facts["k"])),
                "surface_literal_hits": int(surf_literal),
                "surface_name_hits": int(surf_name_hits),
                "surface_max_bge_cosine": float(max_surf_cos),
                "surface_max_jaccard": float(max_jaccard),
                "surface_closest_train": surf_texts[si],
                "surface_closest_probe": list(probe_strings(cases))[pi],
                "surface_alpha": None if weights is None else float(weights["alpha"]),
                "surface_n_strings": int(len(surf_texts)),
                "training": (
                    "build_facts seed 0: _feats(name, rel, query) plus Gaussian "
                    "name + rel_k + 0.3*value for keys, name + reversed(rel_k) for questions. "
                    "No surface strings."
                ),
            },
            "residual": {
                "train_question_mean": float(train_q_ratio.mean()),
                "train_question_median": float(np.median(train_q_ratio)),
                "heldout_question_mean": float(held_q_ratio.mean()),
                "heldout_question_median": float(np.median(held_q_ratio)),
                "train_key_mean": float(train_k_ratio.mean()),
                "heldout_statement_mean": float(held_k_ratio.mean()),
                "heldout_statement_median": float(np.median(held_k_ratio)),
            },
            "confidence": {
                "answerable_head_margin": _pct(ans_head_m),
                "answerable_cosine_margin": _pct(ans_cos_m),
                "answerable_head_top": _pct(ans_head_top),
                "answerable_cosine_top": _pct(ans_cos_top),
                "unanswerable_head_margin": _pct(un_head_m),
                "unanswerable_cosine_margin": _pct(un_cos_m),
                "unanswerable_head_top": _pct(un_head_top),
                "unanswerable_cosine_top": _pct(un_cos_top),
                "head_sweep": _sweep(ans_head_m, ans_head_hit, un_head_m),
                "cosine_sweep": _sweep(ans_cos_m, ans_cos_hit, un_cos_m),
            },
            "store": {
                "ranking_overwrite_cos": 1.0,
                "live_overwrite_cos": 0.9,
                "co_resident_pairs": int(n_pairs),
                "pairs_cosine_above_0.9": int(pair_hi),
                "max_pair_cosine": float(max_pair),
            },
            "examples": examples,
            "embedder": {
                "model": emb_meta["model"],
                "revision": emb_meta["revision"],
                "maxlen": emb_meta["maxlen"],
                "pooling": "cls",
                "normalized": True,
                "query_instruction": False,
                "cache_sha256": sha,
            },
        }
        return report
    finally:
        torch.backends.mkldnn.enabled = prev_mkldnn


def cosine_scores_matrix(keys: np.ndarray) -> np.ndarray:
    keys = np.asarray(keys, dtype=np.float64)
    norms = np.linalg.norm(keys, axis=1)
    return (keys @ keys.T) / (norms[:, None] * norms[None, :] + 1e-12)


def _show(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, tuple):
        return " ".join(value)
    return str(value)


def _public_tally(row: Dict[str, float]) -> Dict[str, Any]:
    out: Dict[str, Any] = {key: int(row[key]) for key in _TALLY_FIELDS}
    jn = int(row["jaccard_n"])
    out["jaccard_mean"] = None if jn == 0 else float(row["jaccard_sum"] / jn)
    return out


def _replay_unchanged(cont, check) -> None:
    """Empty store matches the static model, and scrambling ask/query does not move replay."""
    import torch

    from tidal import vap_targets as VT
    from tidal.continuum import replay
    from tidal.episodic import encode_span

    cont.episodic.clear()
    cont._mem_group = None
    rng = np.random.default_rng(0)
    n_rows = 4
    hidden = rng.normal(size=(n_rows, cont.d)).astype(np.float32)
    targets = np.zeros((n_rows, VT.NV), np.float32)
    ts = np.array([0.0, 1.0, 0.5, 1.5])
    conv = np.array(["a", "a", "b", "b"])
    empty, _, _ = replay(cont, hidden, targets, ts, conv, None)
    with torch.no_grad():
        static = cont.m.vap(torch.from_numpy(hidden)).numpy()
    check("heldout", len(cont.episodic) == 0, "replay wrote a slot")
    check("heldout", np.allclose(empty, static, atol=1e-6), "empty replay drifted")
    phone = "+1-415-555-0134"
    check("heldout", cont.remember(hidden[0], phone, name="ada", rel="phone", group="a") is True, "replay fact")
    cont._mem_group = None
    fact, _, _ = replay(cont, hidden, targets, ts, conv, None)
    code = encode_span(phone, cont.d).astype(np.float32)
    with torch.no_grad():
        row0 = cont.m.vap(torch.from_numpy(hidden[0:1]) + torch.from_numpy(code)).numpy()
    check("heldout", np.allclose(fact[0], row0, atol=1e-5), "replay did not add the byte code")
    check("heldout", np.allclose(fact[2:], empty[2:], atol=1e-6), "other group moved")
    saved_bias = cont.question_head.wq[-1].bias.detach().clone()
    saved_weight = cont.question_head.wk[-1].weight.detach().clone()
    try:
        cont.question_head.wq[-1].bias.fill_(9.0)
        cont.question_head.wk[-1].weight.add_(0.25)
        scrambled, _, _ = replay(cont, hidden, targets, ts, conv, None)
        check("heldout", np.array_equal(fact, scrambled), "replay changed when the question head changed")
        with torch.no_grad():
            cont._mem_group = "a"
            h_row = torch.from_numpy(hidden[0:1])
            before = cont.m.vap(h_row + torch.from_numpy(code))
            got = cont.heads(h_row, [])["vap"]
        check("heldout", torch.allclose(got, before, atol=1e-5), "heads moved with the question head")
    finally:
        cont.question_head.wq[-1].bias.copy_(saved_bias)
        cont.question_head.wk[-1].weight.copy_(saved_weight)
        cont.question_head.fitted = True
        cont.episodic.clear()
        cont._mem_group = None


def _frac(hits: int, n: int, show: bool) -> str:
    if not show:
        return "—"
    return f"{int(hits)}/{int(n)}"


def _pair(cosine: int, head: int, n: int, show: bool) -> str:
    if not show:
        return "—"
    return f"{int(cosine)}/{int(n)} · {int(head)}/{int(n)}"


def _row_line(name: str, row: Dict[str, Any]) -> str:
    answerable = int(row["phone_n"] + row["date_n"] + row["token_n"])
    jacc = "—" if row["jaccard_mean"] is None else f"{row['jaccard_mean']:.2f}"
    lost = "—" if answerable == 0 else f"{int(row['lost_09'])}/{answerable}"
    cells = [
        name,
        str(int(row["n"])),
        jacc,
        _frac(row["cosine_hits"], answerable, answerable > 0),
        _frac(row["head_hits"], answerable, answerable > 0),
        _frac(row["head_only"], answerable, answerable > 0),
        _frac(row["cosine_only"], answerable, answerable > 0),
        _pair(row["phone_cosine"], row["phone_head"], row["phone_n"], row["phone_n"] > 0),
        _pair(row["date_cosine"], row["date_head"], row["date_n"], row["date_n"] > 0),
        _pair(row["token_cosine"], row["token_head"], row["token_n"], row["token_n"] > 0),
        _pair(row["stale_cosine"], row["stale_head"], row["stale_n"], row["stale_n"] > 0),
        _pair(row["false_cosine"], row["false_head"], row["false_n"], row["false_n"] > 0),
        lost,
    ]
    return "| " + " | ".join(cells) + " |"


def format_heldout(report: Dict[str, Any]) -> str:
    """Markdown table plus the abstain distributions. Measurements, not a gate."""
    header = (
        "| category | n | jaccard | cosine ret@1 | ask ret@1 | ask only | cosine only "
        "| phone cos·ask | date cos·ask | tokens cos·ask | stale cos·ask | false cos·ask "
        "| gold lost @0.9 |"
    )
    rule = "|" + "---|" * 13
    lines = [
        header,
        rule,
    ]
    for cat in CATEGORIES:
        lines.append(_row_line(cat, report["categories"][cat]))
    for name in ("answerable", "unanswerable", "all"):
        lines.append(_row_line(name, report["rollups"][name]))
    lines.append("")
    lines.append(
        "Paired cells are encoder-state cosine · ask. ret@1 is an exact decode of the current value."
    )
    lines.append(
        "ask on these flat rows is sentence cosine plus the relation logit. "
        "The cosine column is still the encoder state, which recall uses."
    )
    lines.append("Stale is an answer equal to the value from before the same-identity overwrite.")
    lines.append(
        "False is an unanswerable question that still returned some stored fact. "
        "Ask abstains (returns None) when the nearest stored person lacks the asked relation."
    )
    lines.append("Ranking uses overwrite_cos=1 so a near-duplicate key cannot delete another identity.")
    lines.append("gold lost @0.9 replays the same writes at the live threshold 0.9.")
    d = report["disjoint"]
    e = report["embedder"]
    lines.append(
        f"Disjoint from training: feature collisions {d['feature_collisions']}, "
        f"encoder-state collisions {d['state_collisions']}, "
        f"max cosine of bge rows vs training text vectors {d['max_text_cosine_vs_training']:.4f}, "
        f"literal hits {d['literal_hits']}. "
        f"Surface-training strings {d['surface_n_strings']}: literal hits {d['surface_literal_hits']}, "
        f"name hits {d['surface_name_hits']}, "
        f"max bge cosine vs probe {d['surface_max_bge_cosine']:.4f}, "
        f"max token jaccard {d['surface_max_jaccard']:.3f}, "
        f"relation alpha {d['surface_alpha']}."
    )
    lines.append(
        f"Embedder: {e['model']} rev {e['revision']} CLS L2 maxlen {e['maxlen']}, "
        f"no query instruction. Cache sha256 {e['cache_sha256']}."
    )
    r = report["residual"]
    lines.append(
        "Question-map residual ‖Δq‖/‖q‖ mean "
        f"train {r['train_question_mean']:.4f} (median {r['train_question_median']:.4f}), "
        f"held-out {r['heldout_question_mean']:.4f} (median {r['heldout_question_median']:.4f}). "
        f"Key-map residual mean train {r['train_key_mean']:.4f}, "
        f"held-out statements {r['heldout_statement_mean']:.4f}."
    )
    store = report["store"]
    lines.append(
        f"Co-resident key pairs {store['co_resident_pairs']}, "
        f"cosine > 0.9: {store['pairs_cosine_above_0.9']}, "
        f"max {store['max_pair_cosine']:.4f}."
    )
    lines.append("")
    lines.append("Subtype rows (same definitions):")
    lines.append(header)
    lines.append(rule)
    for name, row in report["subtypes"].items():
        if "/" in name:
            continue
        lines.append(_row_line(name, row))
    lines.append("")
    conf = report["confidence"]
    lines.append("Score distributions (top-1 cosine of the ranker, and top-1 minus top-2):")
    for label in (
        "answerable_head_margin", "answerable_cosine_margin",
        "unanswerable_head_margin", "unanswerable_cosine_margin",
        "answerable_head_top", "answerable_cosine_top",
        "unanswerable_head_top", "unanswerable_cosine_top",
    ):
        stats = conf[label]
        if not stats:
            continue
        ordered = []
        for key in ("min", "10", "25", "50", "75", "90", "max", "mean"):
            if key in stats:
                ordered.append(f"{key}={stats[key]:.4f}")
        lines.append(f"- {label}: " + " ".join(ordered))
    lines.append("")
    lines.append("Shipped abstention on this probe (ask returned None):")
    lines.append("| slice | n | abstained | still a fact | ret@1 | false abstain |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key in (
        "en_unanswerable/missing_rel",
        "en_unanswerable/absent_person",
        "zh_unanswerable/missing_rel",
        "zh_unanswerable/absent_person",
        "missing_rel",
        "absent_person",
    ):
        row = report["subtypes"].get(key)
        if not row:
            continue
        n_un = int(row["false_n"])
        lines.append(
            f"| {key} | {n_un} | {int(row['abstain_head'])}/{n_un} "
            f"| {int(row['false_head'])}/{n_un} | — | — |"
        )
    for key in ("answerable_en", "answerable_zh", "answerable"):
        row = report["rollups"][key]
        n_ans = int(row["phone_n"] + row["date_n"] + row["token_n"])
        lines.append(
            f"| {key} | {n_ans} | — | — | {int(row['head_hits'])}/{n_ans} "
            f"| {int(row['false_abstain'])}/{n_ans} |"
        )
    dev = report.get("abstain_dev")
    if dev:
        lines.append("")
        lines.append(
            "Abstain margins were chosen on generated dev pools, not on this probe. "
            f"Shipped relation margin {dev['rel_margin']}, person margin {dev['person_margin']}. "
            "A grid point is kept only when neither language loses an answerable hit "
            "and neither language abstains on an answerable question. The most unanswerable "
            "abstentions win; a tie keeps the larger relation margin, then the larger person margin."
        )
        lines.append("| person margin | rel margin | unanswerable abstained | false abstain | hits lost | allowed |")
        lines.append("|---:|---:|---:|---:|---:|---|")
        for row in dev["curve"]:
            caught = int(row["caught_total"])
            false_abs = int(row["false_abstain"]["en"] + row["false_abstain"]["zh"])
            lost = int(row["lost"]["en"] + row["lost"]["zh"])
            allowed = "yes" if lost == 0 and false_abs == 0 else "no"
            lines.append(
                f"| {row['person_margin']:.2f} | {row['rel_margin']:.2f} | {caught} "
                f"| {false_abs} | {lost} | {allowed} |"
            )
        chosen = next(
            row for row in dev["curve"]
            if row["rel_margin"] == dev["rel_margin"] and row["person_margin"] == dev["person_margin"]
        )
        for lang in ("en", "zh"):
            hits = chosen["hits"][lang]
            n_ans = chosen["answerable_n"][lang]
            lines.append(
                f"Dev {lang}: answerable hits {hits}/{n_ans} (lost {chosen['lost'][lang]}), "
                f"false abstain {chosen['false_abstain'][lang]}/{n_ans}, "
                f"missing_rel {chosen['caught'][lang]['missing_rel']}/{chosen['caught_n'][lang]['missing_rel']}, "
                f"absent_person {chosen['caught'][lang]['absent_person']}/{chosen['caught_n'][lang]['absent_person']}."
            )
    lines.append("")
    lines.append(
        "If abstain were 'margin < t' instead of the shipped relation check "
        "(not what ask does):"
    )
    lines.append("| ranker | t | answerable kept | exact among kept | unanswerable still answered |")
    lines.append("|---|---:|---:|---:|---:|")
    for ranker, key in (("head", "head_sweep"), ("cosine", "cosine_sweep")):
        for sweep in conf[key]:
            lines.append(
                f"| {ranker} | {sweep['t']:.2f} | {sweep['answerable_kept']}/{sweep['answerable_n']} "
                f"| {sweep['answerable_hits_kept']}/{sweep['answerable_kept']} "
                f"| {sweep['unanswerable_still']}/{sweep['unanswerable_n']} |"
            )
    if report["examples"]:
        lines.append("")
        lines.append("First ask miss in each answerable category:")
        for ex in report["examples"]:
            lines.append(
                f"- {ex['category']} ({ex['rel']}): {ex['question']} "
                f"gold={ex['gold']} cosine={ex['cosine']} head={ex['head']}"
            )
    lines.append("")
    lines.append("These hit counts are not a pass/fail gate.")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Held-out surface questions for Continuum.ask")
    parser.add_argument("--cache", action="store_true", help="rebuild question_heldout_emb.npz")
    args = parser.parse_args(argv)
    if args.cache:
        info = build_embedding_cache()
        print(f"wrote {info['path']} n={info['n']} rev={info['revision']} sha256={info['sha256']}")
        return
    failures: List[str] = []

    def check(section: str, ok: bool, detail: str) -> None:
        if not ok:
            failures.append(f"{section}: {detail}")

    report = eval_heldout(check)
    print(format_heldout(report), flush=True)
    for line in failures:
        print(line, flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
