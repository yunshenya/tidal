"""Sentence-space ranking for surface questions. CPU only.

A flat feature row has no name or relation code in it. ``embed_event`` still
returns a ``d``-dimensional encoder state (that state is the slot key, so
overwrite and ``recall`` stay on raw cosine), and it keeps the 512-d sentence
vector that produced the state. ``ask`` / ``query`` rank those slots by
cosine of the sentence vectors plus a relation logit. The same logit decides
abstention: ``ask`` / ``query`` return no fact when the nearest stored person
does not have the asked relation. Sparse name/relation rows, the ones
``question_head.build_facts`` uses, do not carry a sentence vector, so those
asks stay on the contrastive head and do not abstain.

The logit is a ridge map fit on generated paraphrases. Names and strings in
that corpus are not the held-out probe. ``python -m tidal.question_surface
--fit`` rebuilds ``question_surface.npz`` (needs transformers). Runtime only
reads the file. There is no enable flag: a question state with a stored
sentence vector is ranked here, and a bare encoder state is ranked by the
contrastive head. Abstention uses the dev pools below, not the probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from tidal.episodic import cosine_scores

BUNDLE = Path(__file__).with_name("question_surface.npz")
BGE_MODEL = "BAAI/bge-small-zh-v1.5"
BGE_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
BGE_MAXLEN = 64
LABELS: Tuple[str, ...] = ("phone", "birthday", "places")
# An alpha that drops either language below plain cosine is discarded.
# A tie on the total keeps the smaller alpha, so a flat curve stays at 0.
# The probe is not an input to this choice.
ALPHA_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
# Abstention grid. A point is kept only when neither language loses an
# answerable hit and neither language abstains on an answerable question.
# The kept point with the most unanswerable abstentions wins. A tie keeps
# the larger relation margin, then the larger person margin. The probe is
# not an input. Person margin 0 means a unique nearest person is enough;
# an exact cosine tie unions those people's relations.
ABSTAIN_REL_GRID: Tuple[float, ...] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
ABSTAIN_PERSON_GRID: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
# Shipped point from ``choose_abstain_margins`` on the dev pools.
ABSTAIN_REL_MARGIN = 0.2
ABSTAIN_PERSON_MARGIN = 0.0
# A query this close to a stored sentence is that sentence. Not a fit.
ABSTAIN_EXACT_COS = 0.999
ABSTAIN_TIE_EPS = 1e-8
# sha256 of question_surface.npz. Refit, then update this, if the corpus changes.
EXPECTED_BUNDLE_SHA256 = "10e098a204e555ebf5a1a7cf884e629b7629ed88aa01e07e71426430e4ec08c5"
RIDGE = 1e-2

FIT_NAMES = {
    "en": ("Arthur", "Beatrice", "Cecilia", "Dominic", "Elaine", "Farid", "Greta", "Hugo"),
    "zh": ("祁安", "裴昭", "谢迟", "陆深", "丁兰", "范舟", "彭栗", "崔远"),
}
DEV_NAMES = {
    "en": ("Ingrid", "Jasper", "Keiko", "Lionel", "Maren", "Nadia", "Oscar", "Pauline"),
    "zh": ("姜北", "罗屿", "蔡禾", "潘夏", "杜衡", "魏迟", "任潇", "卢卡"),
}

# Fit questions only. Wording is situational so it is not a one-word edit of
# the probe frames ("What is {name}'s phone number?", "On what date was {name} born?").
FIT_Q = {
    "en": {
        "phone": (
            "The switchboard is holding for {name}. Which extension do I give them?",
            "I misplaced the callback string we kept for {name}.",
            "{name} stepped out. What do I punch in to raise them by voice?",
            "Front desk wants the digits that terminate at {name}.",
        ),
        "birthday": (
            "Which calendar square belongs to {name}'s natal day?",
            "The badge form wants the day {name} first drew breath.",
            "When do the candles go on the cake for {name}?",
            "The yearbook line for {name} wants the day they arrived.",
        ),
        "places": (
            "Read me the rendezvous sequence {name} left on the card.",
            "Where should the courier stand for {name}, first stop to last?",
            "{name} dictated waypoints. Repeat them without sorting.",
            "The route card for {name} lists which stops?",
        ),
    },
    "zh": {
        "phone": (
            "总机在等{name}，我该报哪一个分机？",
            "我把留给{name}的回电串弄丢了。",
            "{name}出去了，用语音怎么叫到人？",
            "前台要打到{name}本人的那一串数字。",
        ),
        "birthday": (
            "{name}的出生日该填在日历的哪一格？",
            "胸牌表要{name}落地后第一次呼吸的那天。",
            "给{name}插蜡烛是哪一天？",
            "年鉴里{name}那一行要写到达的日子。",
        ),
        "places": (
            "把{name}留在卡片上的汇合顺序念一遍。",
            "信使找{name}，从第一站站到最后一站该站哪？",
            "{name}报过路点，按原顺序重复，不要重排。",
            "{name}的路线卡上写了哪些站？",
        ),
    },
}
DEV_Q = {
    "en": {
        "phone": (
            "Which cradle should the night clerk try for {name}?",
            "The courier needs {name} live. What string do we hand over?",
        ),
        "birthday": (
            "The orchard book wants {name}'s natal morning.",
            "Which dawn does the bakery ice a cake for {name}?",
        ),
        "places": (
            "The chit under the blotter is {name}'s stand order. Read it.",
            "Which stands did {name} want kept from the top of the slip downward?",
        ),
    },
    "zh": {
        "phone": (
            "夜班该用哪部座机去叫{name}？",
            "信使要和{name}通上话，我们交出哪一串？",
        ),
        "birthday": (
            "果园名册要{name}出生的那个早晨。",
            "面包房哪一天给{name}抹蛋糕？",
        ),
        "places": (
            "吸墨纸下的条子是{name}的站序，念出来。",
            "{name}要留的站点是从纸条上端往下的哪些？",
        ),
    },
}
DEV_S = {
    "en": {
        "phone": (
            "{name} pinned the callback on the slate beside the cistern.",
            "Voice meant for {name} ends at the cradle in the chart loft.",
        ),
        "birthday": (
            "The natal morning {name} claims is penciled on the spare compass card.",
            "{name} arrived on the date etched under the barn clock.",
        ),
        "places": (
            "{name} listed the stands in spoken order and forbade reshuffling.",
            "Keep {name}'s waypoints as they were read off the chit.",
        ),
    },
    "zh": {
        "phone": (
            "{name}把回电钉在水池边的石板上。",
            "打给{name}的语音停在海图阁楼的座机上。",
        ),
        "birthday": (
            "{name}认的那个出生早晨写在备用罗盘卡上。",
            "{name}落地的日子刻在谷仓钟下面。",
        ),
        "places": (
            "{name}按口头顺序报了站点，还不许重排。",
            "按从条子上读出来的样子留住{name}的路点。",
        ),
    },
}

_WEIGHTS: Optional[dict] = None


def names_used() -> Tuple[str, ...]:
    out: List[str] = []
    for bank in (FIT_NAMES, DEV_NAMES):
        for lang in ("en", "zh"):
            out.extend(bank[lang])
    return tuple(out)


def iter_corpus() -> Iterable[Tuple[str, str, str, str, str, str]]:
    """``(split, lang, name, rel, role, text)`` in a fixed order."""
    for split, names, qbank, sbank in (
        ("fit", FIT_NAMES, FIT_Q, None),
        ("dev", DEV_NAMES, DEV_Q, DEV_S),
    ):
        for lang in ("en", "zh"):
            for rel in LABELS:
                for name in names[lang]:
                    for tmpl in qbank[lang][rel]:
                        yield split, lang, name, rel, "question", tmpl.format(name=name)
                    if sbank is not None:
                        for tmpl in sbank[lang][rel]:
                            yield split, lang, name, rel, "statement", tmpl.format(name=name)


def corpus_strings() -> Tuple[str, ...]:
    seen: List[str] = []
    bag = set()
    for row in iter_corpus():
        text = row[-1]
        if text not in bag:
            bag.add(text)
            seen.append(text)
    return tuple(seen)


def corpus_errors() -> Tuple[str, ...]:
    """Internal problems in the generated corpus. Empty means it is usable."""
    errors = []
    texts = [row[-1] for row in iter_corpus()]
    if len(texts) != len(set(texts)):
        errors.append("duplicate surface-training string")
    if len(set(names_used())) != len(names_used()):
        errors.append("duplicate surface-training name")
    for split, lang, name, rel, role, text in iter_corpus():
        if name not in text:
            errors.append(f"name missing from {split} {lang} {rel} {role}: {text}")
        if text.strip() != text or text == "":
            errors.append(f"blank surface string: {text!r}")
    return tuple(errors)


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm < 1e-12:
        return arr
    return arr / nrm


def load_weights(path: Path = BUNDLE) -> Optional[dict]:
    """Shipped ridge map, or None when the file is absent.

    Absence is not a switch the caller sets. A checkout without the file
    ranks surface slots by sentence cosine alone.
    """
    global _WEIGHTS
    if _WEIGHTS is not None and path == BUNDLE:
        return _WEIGHTS
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as blob:
        weights = {
            "W": np.array(blob["W"], dtype=np.float64, copy=True),
            "alpha": float(blob["alpha"]),
            "labels": tuple(json.loads(str(blob["labels"].item()))),
        }
    if weights["W"].shape != (513, len(LABELS)) or weights["labels"] != LABELS:
        raise RuntimeError(f"surface bundle has shape {weights['W'].shape} labels {weights['labels']}")
    if path == BUNDLE:
        _WEIGHTS = weights
    return weights


def surface_scores(
    query_emb: np.ndarray,
    key_embs: np.ndarray,
    rels: Sequence[Optional[str]],
    weights: Optional[dict] = None,
) -> np.ndarray:
    """Cosine of sentence vectors, plus ``alpha`` times the relation logit.

    ``rels`` aligns with the rows of ``key_embs``. An unknown relation adds
    nothing. ``alpha`` is 0 when the bundle is missing or the dev set did not
    beat plain cosine.
    """
    if len(rels) != len(key_embs):
        raise ValueError(f"{len(rels)} relations for {len(key_embs)} sentence vectors")
    scores = cosine_scores(key_embs, _unit(query_emb))
    if weights is None:
        weights = load_weights()
    if weights is None or float(weights["alpha"]) == 0.0:
        return scores
    labels = weights["labels"]
    x = np.concatenate([_unit(query_emb), np.ones(1, dtype=np.float64)])
    logits = x @ weights["W"]
    bonus = np.zeros(len(rels), dtype=np.float64)
    for i, rel in enumerate(rels):
        if rel in labels:
            bonus[i] = float(logits[labels.index(rel)])
    return scores + float(weights["alpha"]) * bonus


def relation_logits(query_emb: np.ndarray, weights: Optional[dict] = None) -> np.ndarray:
    """Logit for each shipped relation, from the unit query sentence plus a bias."""
    if weights is None:
        weights = load_weights()
    if weights is None:
        raise RuntimeError("surface bundle is missing")
    x = np.concatenate([_unit(query_emb), np.ones(1, dtype=np.float64)])
    return x @ weights["W"]


def surface_abstain(
    query_emb: np.ndarray,
    key_embs: np.ndarray,
    rels: Sequence[Optional[str]],
    names: Sequence[Optional[str]],
    weights: Optional[dict] = None,
    *,
    rel_margin: Optional[float] = None,
    person_margin: Optional[float] = None,
) -> Optional[str]:
    """``missing_rel`` when the nearest stored person lacks the asked relation.

    The asked relation is the argmax of ``relation_logits``. Slots are grouped
    by ``name``. The nearest person is the group with the highest sentence
    cosine; groups within ``ABSTAIN_TIE_EPS`` are one tie and their relations
    are unioned. Abstain when that union does not contain the asked relation
    and the logit margin is at least ``rel_margin`` (the shipped value when
    omitted). ``person_margin`` is the gap to the next person outside the tie;
    the shipped value is 0, so a unique nearest person is enough. A cosine of
    at least ``ABSTAIN_EXACT_COS`` is the stored sentence itself and is not
    an abstention. Returns None when the bundle is missing, which leaves the
    ranker on sentence cosine alone. No enable flag.
    """
    if len(rels) != len(key_embs) or len(names) != len(rels):
        raise ValueError(
            f"{len(names)} names and {len(rels)} relations for {len(key_embs)} sentence vectors"
        )
    if len(rels) == 0:
        return None
    if weights is None:
        weights = load_weights()
    if weights is None:
        return None
    rm = ABSTAIN_REL_MARGIN if rel_margin is None else float(rel_margin)
    pm = ABSTAIN_PERSON_MARGIN if person_margin is None else float(person_margin)
    labels = weights["labels"]
    logits = relation_logits(query_emb, weights)
    order = np.argsort(-logits)
    pred = labels[int(order[0])]
    gap = float(logits[order[0]] - logits[order[1]]) if len(labels) > 1 else float(logits[order[0]])
    if gap < rm:
        return None
    cos = cosine_scores(key_embs, _unit(query_emb))
    groups: Dict[Optional[str], List[int]] = {}
    for i, name in enumerate(names):
        groups.setdefault(name, []).append(i)
    person_cos = {name: float(np.max(cos[idx])) for name, idx in groups.items()}
    best = max(person_cos.values())
    if best >= ABSTAIN_EXACT_COS:
        return None
    tied = [name for name, score in person_cos.items() if best - score <= ABSTAIN_TIE_EPS]
    tied_rels = {rels[i] for name in tied for i in groups[name]}
    if pred in tied_rels:
        return None
    outsiders = [score for name, score in person_cos.items() if name not in tied]
    second = max(outsiders) if outsiders else best
    if best - second < pm:
        return None
    return "missing_rel"


def _dev_statement_table(table: Dict[str, np.ndarray]) -> Dict[Tuple[str, str, str], Tuple[str, np.ndarray]]:
    out = {}
    for lang in ("en", "zh"):
        for rel in LABELS:
            bank = DEV_S[lang][rel]
            for i, name in enumerate(DEV_NAMES[lang]):
                text = bank[i % len(bank)].format(name=name)
                out[(lang, name, rel)] = (text, np.asarray(table[text], dtype=np.float64))
    return out


def _name_ring(names: Sequence[str]) -> List[Tuple[str, str]]:
    return [(name, rel) for name in names for rel in LABELS]


def _dev_distractors(names: Sequence[str], name: str, rel: str, k: int, *, skip_name: bool) -> List[Tuple[str, str]]:
    items = _name_ring(names)
    start = items.index((name, rel))
    out: List[Tuple[str, str]] = []
    step = 1
    while len(out) < k:
        item = items[(start + step) % len(items)]
        step += 1
        if item == (name, rel):
            continue
        if skip_name and item[0] == name:
            continue
        out.append(item)
        if step > len(items) + 5:
            break
    return out


def abstain_dev_pools(table: Optional[Dict[str, np.ndarray]] = None) -> List[dict]:
    """Answerable, missing-relation, and absent-person pools from the dev split.

    Eight names, three relations, two question templates per language. The
    answerable store is the gold fact plus four distractors. ``missing_rel``
    keeps the person's other relations and fills to six facts without the
    asked pair. ``absent_person`` keeps six facts and omits the asked name.
    Strings are the dev corpus already in the bundle. The probe is not read.
    """
    if table is None:
        table = load_corpus_table()
    statements = _dev_statement_table(table)
    pools = []
    for lang in ("en", "zh"):
        names = DEV_NAMES[lang]
        for name in names:
            for rel in LABELS:
                for tmpl in DEV_Q[lang][rel]:
                    question = tmpl.format(name=name)
                    q = np.asarray(table[question], dtype=np.float64)
                    builds = {
                        "answerable": [(name, rel)] + _dev_distractors(names, name, rel, 4, skip_name=False),
                        "missing_rel": [(name, other) for other in LABELS if other != rel],
                        "absent_person": _dev_distractors(names, name, rel, 6, skip_name=True),
                    }
                    missing = builds["missing_rel"]
                    for item in _dev_distractors(names, name, rel, 6, skip_name=True):
                        if len(missing) >= 6:
                            break
                        missing.append(item)
                    for kind, slots in builds.items():
                        texts = [statements[(lang, n, r)][0] for n, r in slots]
                        embs = np.stack([statements[(lang, n, r)][1] for n, r in slots])
                        gold = None
                        if kind == "answerable":
                            gold = 0
                        pools.append({
                            "lang": lang,
                            "kind": kind,
                            "rel": rel,
                            "name": name,
                            "question": question,
                            "q": q,
                            "embs": embs,
                            "rels": [r for _n, r in slots],
                            "names": [n for n, _r in slots],
                            "texts": texts,
                            "gold": gold,
                        })
    return pools


def _pool_hit(pool: dict, weights: dict) -> bool:
    if pool["gold"] is None:
        return False
    scores = surface_scores(pool["q"], pool["embs"], pool["rels"], weights)
    return int(np.argmax(scores)) == int(pool["gold"])


def score_abstain_margins(pools: Sequence[dict], weights: dict, rel_margin: float, person_margin: float) -> dict:
    """Hit loss, false abstentions, and unanswerable catches at one grid point."""
    lost = {"en": 0, "zh": 0}
    false_abstain = {"en": 0, "zh": 0}
    hits = {"en": 0, "zh": 0}
    answerable_n = {"en": 0, "zh": 0}
    caught = {"en": {"missing_rel": 0, "absent_person": 0}, "zh": {"missing_rel": 0, "absent_person": 0}}
    caught_n = {"en": {"missing_rel": 0, "absent_person": 0}, "zh": {"missing_rel": 0, "absent_person": 0}}
    for pool in pools:
        lang = pool["lang"]
        reason = surface_abstain(
            pool["q"], pool["embs"], pool["rels"], pool["names"], weights,
            rel_margin=rel_margin, person_margin=person_margin,
        )
        abstained = reason is not None
        if pool["kind"] == "answerable":
            answerable_n[lang] += 1
            hit = _pool_hit(pool, weights)
            if hit:
                hits[lang] += 1
            if abstained:
                false_abstain[lang] += 1
                if hit:
                    lost[lang] += 1
        else:
            caught_n[lang][pool["kind"]] += 1
            if abstained:
                caught[lang][pool["kind"]] += 1
    total = sum(caught[lang][kind] for lang in ("en", "zh") for kind in ("missing_rel", "absent_person"))
    return {
        "rel_margin": float(rel_margin),
        "person_margin": float(person_margin),
        "hits": hits,
        "lost": lost,
        "false_abstain": false_abstain,
        "answerable_n": answerable_n,
        "caught": caught,
        "caught_n": caught_n,
        "caught_total": int(total),
    }


def abstain_dev_curve(weights: Optional[dict] = None, pools: Optional[Sequence[dict]] = None) -> List[dict]:
    """Every grid point on the dev pools. Does not read the probe."""
    if weights is None:
        weights = load_weights()
    if weights is None:
        raise RuntimeError("surface bundle is missing")
    if pools is None:
        pools = abstain_dev_pools()
    return [
        score_abstain_margins(pools, weights, rel_margin, person_margin)
        for person_margin in ABSTAIN_PERSON_GRID
        for rel_margin in ABSTAIN_REL_GRID
    ]


def choose_abstain_margins(curve: Optional[Sequence[dict]] = None) -> Tuple[float, float]:
    """Operating point. See ``ABSTAIN_REL_GRID`` for the rule."""
    if curve is None:
        curve = abstain_dev_curve()
    allowed = [
        row for row in curve
        if row["lost"]["en"] == 0 and row["lost"]["zh"] == 0
        and row["false_abstain"]["en"] == 0 and row["false_abstain"]["zh"] == 0
    ]
    if not allowed:
        raise RuntimeError("no abstain margin keeps both languages' answerable hits")
    best = max(allowed, key=lambda row: (row["caught_total"], row["rel_margin"], row["person_margin"]))
    return float(best["rel_margin"]), float(best["person_margin"])


def _fit_matrix(emb: np.ndarray, rels: Sequence[str]) -> np.ndarray:
    y = np.stack([(np.asarray(rels) == lab).astype(np.float64) for lab in LABELS], 1)
    x = np.concatenate([emb, np.ones((len(emb), 1), dtype=np.float64)], 1)
    gram = x.T @ x + RIDGE * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def _dev_pools(table: Dict[str, np.ndarray]) -> List[dict]:
    """Fixed pools. Gold index varies, but the draw is seeded."""
    pools = []
    rng = np.random.default_rng(0)
    for lang in ("en", "zh"):
        names = DEV_NAMES[lang]
        statements = {}
        for rel in LABELS:
            bank = DEV_S[lang][rel]
            for i, name in enumerate(names):
                text = bank[i % len(bank)].format(name=name)
                statements[(name, rel)] = np.asarray(table[text], dtype=np.float64)
        for rel in LABELS:
            for name in names:
                for tmpl in DEV_Q[lang][rel]:
                    others = [n for n in names if n != name]
                    picks = [(name, other) for other in LABELS if other != rel]
                    picks += [(n, rel) for n in others[:4]]
                    rest = [(n, r) for n in others for r in LABELS if (n, r) not in picks]
                    rng.shuffle(rest)
                    while len(picks) < 6:
                        picks.append(rest.pop())
                    picks = picks[:6]
                    gold_at = int(rng.integers(0, 7))
                    ordered = picks[:gold_at] + [(name, rel)] + picks[gold_at:]
                    pools.append({
                        "lang": lang,
                        "q": np.asarray(table[tmpl.format(name=name)], dtype=np.float64),
                        "K": np.stack([statements[pair] for pair in ordered]),
                        "rels": [pair[1] for pair in ordered],
                        "gi": gold_at,
                    })
    return pools


def _hits(pools: Sequence[dict], weights: dict, alpha: float) -> dict:
    use = dict(weights)
    use["alpha"] = float(alpha)
    total = {"all": 0, "en": 0, "zh": 0}
    count = {"all": 0, "en": 0, "zh": 0}
    for pool in pools:
        j = int(np.argmax(surface_scores(pool["q"], pool["K"], pool["rels"], use)))
        ok = int(j == pool["gi"])
        total["all"] += ok
        count["all"] += 1
        total[pool["lang"]] += ok
        count[pool["lang"]] += 1
    return {"hits": total, "n": count}


def _choose_alpha(curve: dict) -> float:
    """Largest total that does not drop either language below plain cosine."""
    base = curve[0.0]
    allowed = []
    for alpha in ALPHA_GRID:
        row = curve[alpha]
        if row["hits"]["en"] < base["hits"]["en"] or row["hits"]["zh"] < base["hits"]["zh"]:
            continue
        allowed.append(alpha)
    best = max(allowed, key=lambda alpha: (curve[alpha]["hits"]["all"], -alpha))
    return float(best)


def _embed(texts: Sequence[str]) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
    model = AutoModel.from_pretrained(BGE_MODEL, revision=BGE_REVISION)
    model.eval()
    rows = []
    for start in range(0, len(texts), 32):
        chunk = list(texts[start:start + 32])
        batch = tok(chunk, padding=True, truncation=True, max_length=BGE_MAXLEN, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0]
            hidden = F.normalize(hidden, p=2, dim=-1)
        rows.append(hidden.cpu().numpy().astype(np.float32))
    return np.concatenate(rows, 0)


def fit_bundle(path: Path = BUNDLE) -> dict:
    """Fit the relation map and choose alpha on the dev pools. Not used by tests."""
    errors = corpus_errors()
    if errors:
        raise RuntimeError("; ".join(errors))
    texts = corpus_strings()
    emb = _embed(texts).astype(np.float64)
    table = {text: emb[i] for i, text in enumerate(texts)}
    fit_rows = [row for row in iter_corpus() if row[0] == "fit" and row[4] == "question"]
    fit_emb = np.stack([table[row[-1]] for row in fit_rows])
    fit_rels = [row[3] for row in fit_rows]
    W = _fit_matrix(fit_emb, fit_rels)
    pools = _dev_pools(table)
    curve = {}
    for alpha in ALPHA_GRID:
        curve[float(alpha)] = _hits(pools, {"W": W, "alpha": alpha, "labels": LABELS}, alpha)
    chosen = _choose_alpha(curve)
    payload = {
        "W": W.astype(np.float64),
        "alpha": np.float64(chosen),
        "labels": np.asarray(json.dumps(list(LABELS))),
        "texts": np.asarray(json.dumps(list(texts))),
        "emb": emb.astype(np.float32),
        "model": np.asarray(BGE_MODEL),
        "revision": np.asarray(BGE_REVISION),
        "dev": np.asarray(json.dumps({str(k): v for k, v in curve.items()})),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)
    global _WEIGHTS
    _WEIGHTS = None
    return {
        "alpha": chosen,
        "n_fit": len(fit_rows),
        "n_dev": len(pools),
        "curve": {str(k): v for k, v in curve.items()},
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }


def bundle_sha256(path: Path = BUNDLE) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corpus_table(path: Path = BUNDLE) -> Dict[str, np.ndarray]:
    """Sentence vectors stored with the fit, keyed by training string."""
    with np.load(path, allow_pickle=False) as blob:
        texts = json.loads(str(blob["texts"].item()))
        emb = np.array(blob["emb"], dtype=np.float64, copy=True)
    if emb.shape != (len(texts), 512):
        raise RuntimeError(f"surface corpus shape {emb.shape} for {len(texts)} strings")
    return {text: emb[i] for i, text in enumerate(texts)}


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Fit the surface relation logit")
    parser.add_argument("--fit", action="store_true", help="rebuild question_surface.npz")
    args = parser.parse_args(argv)
    if not args.fit:
        weights = load_weights()
        print("missing bundle" if weights is None else f"alpha {weights['alpha']}")
        return
    info = fit_bundle()
    print(f"alpha {info['alpha']} fit {info['n_fit']} dev {info['n_dev']} sha256 {info['sha256']}")
    for key, row in info["curve"].items():
        hits, n = row["hits"], row["n"]
        print(f"  a={key} {hits['all']}/{n['all']} en {hits['en']}/{n['en']} zh {hits['zh']}/{n['zh']}")


if __name__ == "__main__":
    main()
