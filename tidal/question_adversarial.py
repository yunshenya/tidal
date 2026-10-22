"""Score flat-row ``ask`` abstain on the adversarial split. CPU only.

The test split is not an input to the fit. ``--split dev`` scores the dev bank
the same way, for a change that has already been chosen. Disjointness is
literal equality, token Jaccard (the probe tokenizer), and sentence cosine of
the frozen ``multilingual-e5-small`` vectors.

Run: ``python -m tidal.question_adversarial``
Rebuild vectors: ``python -m tidal.question_adversarial --cache``
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_heldout import surface_feats
from tidal.question_heldout_data import probe_strings
from tidal.question_stress import (
    StressCase,
    build_cases,
    case_strings,
    jaccard_report,
    summary,
    validate_cases,
)
from tidal.question_surface import (
    STORED_DIM,
    corpus_strings,
    embed_sentences,
    lex_labels,
    load_corpus_table,
    load_weights,
    _abstain_pred,
    _matched_names,
)

CACHE = Path(__file__).with_name("question_adversarial_emb.npz")
REPORT = Path(__file__).resolve().parents[1] / "reports" / "ask_abstain_adversarial.md"


def _banks(which: str) -> List[Tuple[str, dict]]:
    if which == "test":
        return [("test", TEST_BANK)]
    if which == "dev":
        return [("dev", DEV_BANK)]
    return [("test", TEST_BANK), ("dev", DEV_BANK)]


def load_cases(which: str) -> Dict[str, Tuple[StressCase, ...]]:
    built = {name: build_cases(bank) for name, bank in _banks(which)}
    if "test" in built and "dev" in built:
        problems = validate_cases(built["test"], other=built["dev"])
        problems += validate_cases(built["dev"], other=built["test"])
    else:
        only = next(iter(built.values()))
        problems = validate_cases(only)
    if problems:
        raise RuntimeError("; ".join(problems[:6]))
    return built


def all_strings(cases: Dict[str, Sequence[StressCase]]) -> Tuple[str, ...]:
    seen = []
    bag = set()
    for group in cases.values():
        for text in case_strings(group):
            if text not in bag:
                bag.add(text)
                seen.append(text)
    return tuple(seen)


def load_table(path: Path = CACHE) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path.name}; build it with --cache")
    with np.load(path, allow_pickle=False) as blob:
        texts = json.loads(str(blob["texts"].item()))
        emb = np.array(blob["emb"], dtype=np.float32, copy=True)
    if emb.shape != (len(texts), STORED_DIM):
        raise RuntimeError(f"adversarial cache shape {emb.shape}")
    return {text: emb[i] for i, text in enumerate(texts)}


def build_cache(cases: Dict[str, Sequence[StressCase]], path: Path = CACHE) -> dict:
    texts = list(all_strings(cases))
    emb = embed_sentences(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb=emb.astype(np.float32),
        texts=np.asarray(json.dumps(texts)),
    )
    return {"n": len(texts), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "path": str(path)}


def _stack(table: Dict[str, np.ndarray], texts: Sequence[str]) -> np.ndarray:
    rows = []
    for text in texts:
        vec = np.asarray(table[text], dtype=np.float64)
        nrm = float(np.linalg.norm(vec))
        rows.append(vec / nrm if nrm > 1e-12 else vec)
    return np.stack(rows)


def _max_cosine(left: np.ndarray, right: np.ndarray) -> Tuple[float, int, int]:
    scores = left @ right.T
    flat = int(np.argmax(scores))
    row, col = divmod(flat, scores.shape[1])
    return float(scores[row, col]), int(row), int(col)


def cosine_report(cases: Dict[str, Sequence[StressCase]], table: Dict[str, np.ndarray]) -> dict:
    """Sentence cosine of each split against the probe, the surface corpus, and the other split."""
    probe = probe_strings()
    surface = corpus_strings()
    probe_table, _meta = _probe_table()
    surface_table = load_corpus_table()
    missing_probe = [text for text in probe if text not in probe_table]
    missing_surface = [text for text in surface if text not in surface_table]
    if missing_probe or missing_surface:
        raise RuntimeError(f"reference vectors missing probe={missing_probe[:2]} surface={missing_surface[:2]}")
    mats = {
        "probe": _stack(probe_table, probe),
        "surface": _stack(surface_table, surface),
    }
    split_texts = {name: case_strings(group) for name, group in cases.items()}
    for name, texts in split_texts.items():
        missing = [text for text in texts if text not in table]
        if missing:
            raise RuntimeError(f"{name} cache missing {missing[:2]}")
        mats[name] = _stack(table, texts)
    out = {}
    for name, texts in split_texts.items():
        block = {}
        for other, labels in (
            ("probe", probe),
            ("surface", surface),
        ):
            score, i, j = _max_cosine(mats[name], mats[other])
            block[other] = {"max": score, "left": texts[i], "right": labels[j]}
        for other_name, other_texts in split_texts.items():
            if other_name == name:
                continue
            score, i, j = _max_cosine(mats[name], mats[other_name])
            block[other_name] = {"max": score, "left": texts[i], "right": other_texts[j]}
        out[name] = block
    return out


def _probe_table():
    from tidal.question_heldout import load_text_table

    return load_text_table()


def _blank() -> dict:
    return {
        "n": 0,
        "abstained": 0,
        "still_answered": 0,
        "ask_hits": 0,
        "false_abstain": 0,
        "as_missing_rel": 0,
        "as_absent_person": 0,
    }


def _bump(bucket: dict, row: dict) -> None:
    bucket["n"] += 1
    if row["reason"]:
        bucket["abstained"] += 1
        if row["reason"] == "missing_rel":
            bucket["as_missing_rel"] += 1
        elif row["reason"] == "absent_person":
            bucket["as_absent_person"] += 1
    else:
        bucket["still_answered"] += 1
    if row["hit"]:
        bucket["ask_hits"] += 1
    if row["false_abstain"]:
        bucket["false_abstain"] += 1


def score_split(cases: Sequence[StressCase], table: Dict[str, np.ndarray]) -> List[dict]:
    """Run shipped ``ask`` on each case. Does not fit anything."""
    import torch

    from tidal.episodic import decode_span, decode_tokens
    from tidal.question_head import load_reference

    weights = load_weights()
    if weights is None or weights.get("Wa") is None:
        raise RuntimeError("abstain bundle missing")
    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    try:
        cont = load_reference(overwrite_cos=1.0)
        rows = []
        for case in cases:
            cont.episodic.clear()
            cont._mem_group = None
            for write in case.writes:
                feats = surface_feats(write.text, "statement")
                key = cont.embed_event(feats, table[write.text], surface=write.text)
                ok = cont.remember(key, write.value, name=write.name, rel=write.rel, kind="fact", group=None)
                if ok is not True:
                    raise RuntimeError(f"fact rejected: {write.text}")
            feats = surface_feats(case.question, "question")
            query = cont.embed_event(feats, table[case.question], surface=case.question)
            value, info = cont.query(query, group=None, mode="top1")
            reason = info.get("abstain")
            if value is None:
                answer = None
            else:
                answer = decode_span(value)
                if answer is None:
                    answer = decode_tokens(value)
            hit = case.label == "answerable" and answer in case.golds
            pred, gap = _abstain_pred(table[case.question], weights["Wa"])
            names = [write.name for write in case.writes]
            matched = _matched_names(case.question, names)
            rows.append({
                "slice": case.slice,
                "lang": case.lang,
                "rel": case.rel,
                "label": case.label,
                "subtype": case.subtype,
                "question": case.question,
                "reason": reason or "",
                "answer": answer if isinstance(answer, str) else ("" if answer is None else " ".join(answer)),
                "golds": list(case.golds),
                "hit": bool(hit),
                "false_abstain": bool(case.label == "answerable" and reason),
                "lex": list(lex_labels(case.question)),
                "matched": matched,
                "ridge_pred": pred,
                "ridge_gap": gap,
                "top": None if reason else info.get("top"),
                "canonical": case.canonical,
                "focus": case.focus,
                "pool": case.pool,
            })
        return rows
    finally:
        torch.backends.mkldnn.enabled = prev


def aggregate(rows: Sequence[dict]) -> dict:
    cells: Dict[str, dict] = {}
    subtypes: Dict[str, dict] = {}
    for row in rows:
        keys = (
            f"{row['slice']}|{row['lang']}|{row['label']}",
            f"{row['slice']}|{row['lang']}|*",
            f"{row['slice']}|*|{row['label']}",
            f"*|{row['lang']}|{row['label']}",
        )
        for key in keys:
            _bump(cells.setdefault(key, _blank()), row)
        sub = f"{row['slice']}|{row['lang']}|{row['subtype']}|{row['label']}"
        _bump(subtypes.setdefault(sub, _blank()), row)
    return {"cells": cells, "subtypes": subtypes}


def format_report(payload: dict) -> str:
    lines = [
        "# Adversarial ask abstain",
        "",
        "Shipped flat-row `ask` on a held-out set written apart from the cue list,",
        "the probe, and the surface fit/dev phrasing. Not a pass/fail gate.",
        "",
    ]
    dis = payload["disjoint"]
    jac = payload["jaccard"]
    lines.append("## Disjointness")
    lines.append("")
    lines.append("| split | against | literal | max token Jaccard | closest Jaccard pair | max sentence cosine | closest cosine pair |")
    lines.append("|---|---|---:|---:|---|---:|---|")
    for split, block in dis.items():
        jblock = jac[split]
        for other in block:
            jrow = jblock.get(other, {})
            cos = block[other]
            literal = 0
            if other in ("probe", "surface"):
                literal = payload["literal"].get(split, {}).get(other, 0)
            elif other in payload["literal"].get(split, {}):
                literal = payload["literal"][split][other]
            pair_j = jrow.get("pair", ("", ""))
            lines.append(
                f"| {split} | {other} | {literal} | {jrow.get('max', float('nan')):.3f} | "
                f"{pair_j[0]} ‖ {pair_j[1]} | {cos['max']:.3f} | "
                f"{cos['left']} ‖ {cos['right']} |"
            )
    lines.append("")
    lines.append("## Sizes")
    lines.append("")
    for split, info in payload["summaries"].items():
        lines.append(f"{split}: n={info['n']} sha256=`{info['sha256']}`")
        lines.append("")
        lines.append("| slice | n |")
        lines.append("|---|---:|")
        for name, n in sorted(info["slices"].items()):
            lines.append(f"| {name} | {n} |")
        lines.append("")
        lines.append(f"by language {info['by_lang']}, by label {info['by_label']}")
        lines.append("")
    lines.append("## Ask")
    lines.append("")
    lines.append("Abstained: `ask` returned no fact (`missing_rel` or `absent_person`).")
    lines.append("Still answered: a stored fact was returned.")
    lines.append("Ask hits: answerable and the decode is the gold value (either gold, when a pronoun could be either person).")
    lines.append("False abstain: answerable and abstained.")
    lines.append("On an unanswerable row, still answered is the missed abstain. False abstain stays zero there by the label.")
    lines.append("")
    for split, agg in payload["scores"].items():
        lines.append(f"### {split}")
        lines.append("")
        lines.append("| slice | lang | label | n | abstained | still answered | ask hits | false abstain |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        cells = agg["cells"]
        order = sorted(key for key in cells if key.count("|") == 2 and "*" not in key)
        for key in order:
            slice_name, lang, label = key.split("|")
            bucket = cells[key]
            lines.append(
                f"| {slice_name} | {lang} | {label} | {bucket['n']} | {bucket['abstained']} | "
                f"{bucket['still_answered']} | {bucket['ask_hits']} | {bucket['false_abstain']} |"
            )
        lines.append("")
        lines.append("| slice | lang | subtype | label | n | abstained | still answered | ask hits | false abstain |")
        lines.append("|---|---|---|---|---:|---:|---:|---:|---:|")
        for key in sorted(agg["subtypes"]):
            slice_name, lang, subtype, label = key.split("|")
            bucket = agg["subtypes"][key]
            lines.append(
                f"| {slice_name} | {lang} | {subtype} | {label} | {bucket['n']} | {bucket['abstained']} | "
                f"{bucket['still_answered']} | {bucket['ask_hits']} | {bucket['false_abstain']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _literal_hits(cases: Dict[str, Sequence[StressCase]]) -> dict:
    probe = set(probe_strings())
    surface = set(corpus_strings())
    texts = {name: set(case_strings(group)) for name, group in cases.items()}
    out = {}
    for name, bag in texts.items():
        block = {
            "probe": len(bag & probe),
            "surface": len(bag & surface),
        }
        for other, other_bag in texts.items():
            if other != name:
                block[other] = len(bag & other_bag)
        out[name] = block
    return out


def evaluate(which: str = "both", *, table: Optional[Dict[str, np.ndarray]] = None) -> dict:
    cases = load_cases(which)
    if table is None:
        table = load_table()
    jacc = {}
    built_list = list(cases.items())
    for name, group in built_list:
        other = ()
        for other_name, other_group in built_list:
            if other_name != name:
                other = other_group
        report = jaccard_report(group, other=other)
        # jaccard_report nests probe/surface/other_split. Rename other_split to the sibling.
        sibling = next((n for n in cases if n != name), None)
        if sibling and "other_split" in report:
            report[sibling] = report.pop("other_split")
        jacc[name] = report
    payload = {
        "summaries": {name: summary(group) for name, group in cases.items()},
        "jaccard": {
            name: {key: value for key, value in block.items() if key not in ("n_strings", "literal_hits")}
            for name, block in jacc.items()
        },
        "literal": _literal_hits(cases),
        "cosine": cosine_report(cases, table),
        "scores": {},
        "rows": {},
    }
    for name, group in cases.items():
        rows = score_split(group, table)
        payload["rows"][name] = rows
        payload["scores"][name] = aggregate(rows)
    payload["disjoint"] = payload["cosine"]
    return payload


def _json_default(obj):
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score adversarial ask abstain")
    parser.add_argument("--cache", action="store_true", help="rebuild the sentence cache")
    parser.add_argument("--split", choices=("test", "dev", "both"), default="both")
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cache:
        # Cache every string either split will ask, so a later --split does not re-embed.
        info = build_cache(load_cases("both"))
        print(f"wrote {info['path']} n={info['n']} sha256={info['sha256']}")
    payload = evaluate(args.split)
    text = format_report(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # The markdown table is the readable report. Rows stay in JSON.
    public = {key: value for key, value in payload.items() if key != "rows"}
    args.out.write_text(text, encoding="utf-8")
    json_path = args.json
    if json_path is None:
        json_path = args.out.with_suffix(".json")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(text)
    print(f"wrote {args.out} and {json_path}")


if __name__ == "__main__":
    main()
