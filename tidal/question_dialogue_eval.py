"""Score the dialogue-state dev bank. CPU only.

``ask`` sees a per-session mention list. This run asks each episode twice:
once after the episode's turns and earlier questions, and once with that
list empty (fact writes only). The empty pass is the single-turn path.

The frozen adversarial questions are not an input. Rebuild vectors with
``python -m tidal.question_dialogue_eval --cache``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from tidal.question_dialogue_dev_data import (
    Episode,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_heldout import surface_feats
from tidal.question_heldout_data import _tokens
from tidal.question_surface import (
    SENTENCE_MAXLEN,
    SENTENCE_MODEL,
    SENTENCE_POOLING,
    SENTENCE_REVISION,
    STORED_DIM,
)

CACHE = Path(__file__).with_name("question_dialogue_emb.npz")
PRIOR = Path(__file__).resolve().parents[1] / "tests" / "question_dialogue_prior.json"


def _sha(texts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def prior_corpora() -> Dict[str, dict]:
    """Earlier banks the dev strings must not overlap. Questions and statements."""
    from tidal.question_abstain_dev_data import BANK as DEV_BANK
    from tidal.question_adversarial_data import BANK as TEST_BANK
    from tidal.question_bare_dev_data import questions as bare_questions
    from tidal.question_bare_dev_data import questions_sha256 as bare_sha
    from tidal.question_cjk_dev_data import questions as cjk_questions
    from tidal.question_cjk_dev_data import questions_sha256 as cjk_sha
    from tidal.question_cue_dev_data import questions as cue_questions
    from tidal.question_cue_dev_data import questions_sha256 as cue_sha
    from tidal.question_focus_dev_data import bank_strings as focus_strings
    from tidal.question_focus_dev_data import questions_sha256 as focus_sha
    from tidal.question_heldout_data import probe_strings
    from tidal.question_indirect_dev_data import bank_sha256 as indirect_bank_sha
    from tidal.question_indirect_dev_data import bank_strings as indirect_bank
    from tidal.question_indirect_dev_data import questions as indirect_questions
    from tidal.question_indirect_dev_data import questions_sha256 as indirect_sha
    from tidal.question_nickname_dev_data import questions as nick_questions
    from tidal.question_nickname_dev_data import questions_sha256 as nick_sha
    from tidal.question_stress import build_cases, case_strings, questions_sha256
    from tidal.question_surface import corpus_strings
    from tidal.question_two_name_dev_data import bank_strings as two_strings
    from tidal.question_two_name_dev_data import questions_sha256 as two_sha

    def _literal(path: str, name: str) -> Tuple[str, ...]:
        import ast

        src = Path(path).read_text(encoding="utf-8")
        for node in ast.parse(src).body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if getattr(target, "id", "") == name:
                    return tuple(ast.literal_eval(node.value))
        raise RuntimeError(f"{name} missing in {path}")

    typo = _literal("tests/test_question_indirect_dev.py", "_TYPO_DEV_QUESTIONS")
    residual = _literal("tests/test_question_two_name.py", "_RESIDUAL_QUESTIONS")
    frozen = build_cases(TEST_BANK)
    abstain = build_cases(DEV_BANK)
    prior = json.loads(PRIOR.read_text(encoding="utf-8"))
    bags = {
        "probe": {"sha": "", "strings": tuple(probe_strings())},
        "surface": {"sha": "", "strings": tuple(corpus_strings())},
        "abstain_dev": {
            "sha": questions_sha256(abstain),
            "strings": tuple(case_strings(abstain)),
        },
        "frozen": {
            "sha": questions_sha256(frozen),
            "strings": tuple(case_strings(frozen)),
        },
        "cjk": {"sha": cjk_sha(), "strings": tuple(cjk_questions())},
        "nickname": {"sha": nick_sha(), "strings": tuple(nick_questions())},
        "typo": {"sha": _sha(typo), "strings": typo},
        "indirect_questions": {
            "sha": indirect_sha(),
            "strings": tuple(indirect_questions()),
        },
        "indirect_bank": {
            "sha": indirect_bank_sha(),
            "strings": tuple(indirect_bank()),
        },
        "cue": {"sha": cue_sha(), "strings": tuple(cue_questions())},
        "bare": {"sha": bare_sha(), "strings": tuple(bare_questions())},
        "residual": {"sha": _sha(residual), "strings": residual},
        "two_name": {"sha": two_sha(), "strings": tuple(two_strings())},
        "focus": {"sha": focus_sha(), "strings": tuple(focus_strings())},
        "typo_edit": {
            "sha": prior["typo_edit_3447aaaa"]["sha256"],
            "strings": tuple(prior["typo_edit_3447aaaa"]["strings"]),
        },
        "lastname_questions": {
            "sha": prior["lastname_064d7052_questions"]["sha256"],
            "strings": tuple(prior["lastname_064d7052_questions"]["strings"]),
        },
        "lastname_bank": {
            "sha": _sha(prior["lastname_064d7052_bank"]["strings"]),
            "strings": tuple(prior["lastname_064d7052_bank"]["strings"]),
        },
    }
    return bags


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def jaccard_report(own: Sequence[str], bags: Dict[str, dict]) -> Dict[str, dict]:
    """Literal overlap and max token Jaccard of ``own`` against each prior bag."""
    own_tokens = [_tokens(text) for text in own]
    own_set = set(own)
    report = {}
    for name, bag in bags.items():
        texts = bag["strings"]
        literal = sum(1 for text in own if text in set(texts))
        best = 0.0
        pair = ("", "")
        right = [_tokens(text) for text in texts]
        for src, src_tok in zip(own, own_tokens):
            for dst, dst_tok in zip(texts, right):
                score = _jaccard(src_tok, dst_tok)
                if score > best:
                    best = score
                    pair = (src, dst)
        report[name] = {
            "sha": bag["sha"],
            "n": len(texts),
            "literal": literal,
            "max_jaccard": best,
            "pair": pair,
            "own_n": len(own_set),
        }
    return report


def load_table(path: Path = CACHE) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {path.name}; build it with --cache")
    with np.load(path, allow_pickle=False) as blob:
        texts = json.loads(str(blob["texts"].item()))
        emb = np.array(blob["emb"], dtype=np.float32, copy=True)
        meta = {
            "model": str(blob["model"].item()),
            "revision": str(blob["revision"].item()),
            "pooling": str(blob["pooling"].item()),
            "maxlen": int(blob["maxlen"]),
        }
    if emb.shape != (len(texts), STORED_DIM):
        raise RuntimeError(f"dialogue cache shape {emb.shape}")
    if (
        meta["model"] != SENTENCE_MODEL
        or meta["revision"] != SENTENCE_REVISION
        or meta["pooling"] != SENTENCE_POOLING
        or meta["maxlen"] != SENTENCE_MAXLEN
    ):
        raise RuntimeError(f"dialogue cache contract {meta}")
    return {text: emb[i] for i, text in enumerate(texts)}


def build_cache(rows: Sequence[Episode], path: Path = CACHE) -> dict:
    from tidal.question_surface import embed_sentences

    texts = list(bank_strings(rows))
    emb = embed_sentences(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        emb=emb.astype(np.float32),
        texts=np.asarray(json.dumps(texts)),
        model=np.asarray(SENTENCE_MODEL),
        revision=np.asarray(SENTENCE_REVISION),
        pooling=np.asarray(SENTENCE_POOLING),
        maxlen=np.int64(SENTENCE_MAXLEN),
    )
    return {
        "n": len(texts),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "path": str(path),
    }


def _answer(value):
    from tidal.episodic import decode_span, decode_tokens

    if value is None:
        return None
    text = decode_span(value)
    if text is not None:
        return text
    tokens = decode_tokens(value)
    if tokens is None:
        return None
    return " ".join(tokens)


def _ok(expect: str, golds: Sequence[str], answer, reason: str) -> bool:
    if expect == "answerable":
        return reason == "" and answer in golds
    if expect == "absent_person":
        return reason == "absent_person" and answer is None
    if expect == "missing_rel":
        return reason == "missing_rel" and answer is None
    raise RuntimeError(expect)


def score_rows(rows: Sequence[Episode], table: Dict[str, np.ndarray]) -> dict:
    """Each episode with its dialogue, then again with an empty mention list."""
    import torch

    from tidal.question_head import load_reference

    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    kinds: Dict[str, dict] = {}
    failures: List[dict] = []
    try:
        cont = load_reference(overwrite_cos=1.0)
        for row in rows:
            slot = kinds.setdefault(row.kind, {"n": 0, "state": 0, "empty": 0})
            slot["n"] += 1
            for mode in ("state", "empty"):
                got = _score_one(cont, table, row, with_state=mode == "state")
                expect = row.label if mode == "state" else row.empty_label
                golds = row.golds if mode == "state" else row.empty_golds
                if _ok(expect, golds, got["answer"], got["reason"]):
                    slot[mode] += 1
                else:
                    failures.append({
                        "kind": row.kind,
                        "lang": row.lang,
                        "mode": mode,
                        "expect": expect,
                        "reason": got["reason"],
                        "answer": got["answer"],
                        "golds": list(golds),
                        "question": row.question,
                    })
    finally:
        torch.backends.mkldnn.enabled = prev
    return {"kinds": kinds, "failures": failures}


def _score_one(cont, table, row: Episode, *, with_state: bool) -> dict:
    cont.episodic.clear()
    cont._mem_group = None
    for slot in row.slots:
        key = cont.embed_event(
            surface_feats(slot.text, "statement"),
            table[slot.text],
            surface=slot.text,
        )
        ok = cont.remember(key, slot.value, name=slot.name, rel=slot.rel, kind="fact", group=None)
        if ok is not True:
            raise RuntimeError(f"fact rejected: {slot.text}")
    if with_state:
        for turn in row.turns:
            cont.note_turn(turn, group=None)
        for prior in row.prior_asks:
            query = cont.embed_event(
                surface_feats(prior, "question"),
                table[prior],
                surface=prior,
            )
            cont.query(query, group=None, mode="top1")
    query = cont.embed_event(
        surface_feats(row.question, "question"),
        table[row.question],
        surface=row.question,
    )
    value, info = cont.query(query, group=None, mode="top1")
    reason = info.get("abstain") or ""
    return {"reason": reason, "answer": _answer(value)}


def format_report(rows: Sequence[Episode], disjoint: dict, scored: Optional[dict]) -> str:
    lines = [
        "# Dialogue dev",
        "",
        f"questions `{questions_sha256(rows)}` n={len(rows)}",
        f"bank `{bank_sha256(rows)}` n={len(bank_strings(rows))}",
        "",
        "## Disjointness",
        "",
        "| against | sha | n | literal | max Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    for name, row in disjoint.items():
        sha = row["sha"][:12] + "…" if row["sha"] else "—"
        lines.append(
            f"| {name} | {sha} | {row['n']} | {row['literal']} | {row['max_jaccard']:.3f} |"
        )
    if scored is not None:
        lines.extend([
            "",
            "## Ask",
            "",
            "A count is episodes whose label for that pass was met. A typo hit",
            "is correct with the log when it answers, and correct with an empty",
            "log when it abstains `absent_person`.",
            "",
            "| kind | n | state correct | empty correct |",
            "|---|---:|---:|---:|",
        ])
        state_n = empty_n = total = 0
        for kind, slot in scored["kinds"].items():
            lines.append(
                f"| {kind} | {slot['n']} | {slot['state']}/{slot['n']} | {slot['empty']}/{slot['n']} |"
            )
            state_n += slot["state"]
            empty_n += slot["empty"]
            total += slot["n"]
        lines.append(f"| total | {total} | {state_n}/{total} | {empty_n}/{total} |")
        lines.append("")
        lines.append(f"failures {len(scored['failures'])}")
        for fail in scored["failures"][:12]:
            lines.append(
                f"- {fail['kind']} {fail['lang']} {fail['mode']}: "
                f"expect {fail['expect']} got {fail['reason'] or fail['answer']}"
            )
    lines.append("")
    lines.append(f"kinds {kind_counts(rows)}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the dialogue-state dev bank")
    parser.add_argument("--cache", action="store_true", help="rebuild question_dialogue_emb.npz")
    parser.add_argument("--disjoint-only", action="store_true")
    args = parser.parse_args(argv)
    rows = build_rows()
    bags = prior_corpora()
    disjoint = jaccard_report(bank_strings(rows), bags)
    if args.cache:
        info = build_cache(rows)
        print(f"wrote {info['path']} n={info['n']} sha256={info['sha256']}")
    scored = None
    if not args.disjoint_only:
        table = load_table()
        missing = [text for text in bank_strings(rows) if text not in table]
        if missing:
            raise RuntimeError(f"cache missing {len(missing)} strings; rerun --cache")
        scored = score_rows(rows, table)
    text = format_report(rows, disjoint, scored)
    print(text)


if __name__ == "__main__":
    main()
