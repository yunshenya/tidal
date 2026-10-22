"""Sentence-space ranking for surface questions. CPU only.

A flat feature row has no name or relation code in it. ``embed_event`` still
returns a ``d``-dimensional encoder state (that state is the slot key, so
overwrite and ``recall`` stay on raw cosine), and it keeps the 512-d sentence
vector that produced the state. ``ask`` / ``query`` rank those slots by
cosine of the sentence vectors plus a relation logit. Sparse name/relation
rows, the ones ``question_head.build_facts`` uses, do not carry a sentence
vector, so those asks stay on the contrastive head.

The logit is a ridge map fit on generated paraphrases. Names and strings in
that corpus are not the held-out probe. ``python -m tidal.question_surface
--fit`` rebuilds ``question_surface.npz`` (needs transformers). Runtime only
reads the file. There is no enable flag: a question state with a stored
sentence vector is ranked here, and a bare encoder state is ranked by the
contrastive head.
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
