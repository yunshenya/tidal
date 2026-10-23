"""Score the held-out dialogue bank and sweep the mention window.

The shipped window is whatever ``tidal.dialogue_state.TURN_WINDOW`` is.
This module does not change it. A sweep sets the global for one pass
and puts it back.

Filler counts 2, 3, 5, 6, and 7 (``role == "lag"``) are reported as a
boundary ladder. They are not part of the pass/fail that picks a window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from tidal.question_dialogue_eval import (
    _ok,
    _score_one,
    jaccard_report,
    load_table,
    prior_corpora,
)
from tidal.question_dialogue_heldout_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_surface import (
    SENTENCE_MAXLEN,
    SENTENCE_MODEL,
    SENTENCE_POOLING,
    SENTENCE_REVISION,
    STORED_DIM,
)

CACHE = Path(__file__).with_name("question_dialogue_heldout_emb.npz")
WINDOWS = (2, 4, 6, 8)

# Dialogue-dev kinds, for the sweep of that bank. It has no ``role`` field.
_DEV_ROLE = {
    "typo_hit_utterance": "hit",
    "typo_hit_prior_ask": "hit",
    "typo_abstain_stale": "aged",
    "typo_abstain_tie": "tie",
    "typo_abstain_silent": "silent",
    "typo_abstain_other": "other",
    "spelling_hit": "pos",
    "nickname_hit": "pos",
    "focus_hit": "pos",
    "dismiss_hit": "pos",
    "open_either": "pos",
    "indirect_hit": "pos",
    "latin_extension": "neg",
    "cjk_superstring": "neg",
    "cjk_prefix": "neg",
    "indirect_noself": "neg",
    "indirect_missing": "neg",
}


def heldout_corpora() -> Dict[str, dict]:
    """Prior banks plus the dialogue dev split this held-out set must miss."""
    from tidal.question_dialogue_dev_data import bank_sha256 as dev_bank_sha
    from tidal.question_dialogue_dev_data import bank_strings as dev_bank_strings
    from tidal.question_dialogue_dev_data import build_rows as dev_rows
    from tidal.question_dialogue_dev_data import questions_sha256 as dev_qsha

    bags = prior_corpora()
    rows = dev_rows()
    bags["dialogue_dev_questions"] = {
        "sha": dev_qsha(rows),
        "strings": tuple(row.question for row in rows),
    }
    bags["dialogue_dev_bank"] = {
        "sha": dev_bank_sha(rows),
        "strings": tuple(dev_bank_strings(rows)),
    }
    return bags


def build_cache(rows: Sequence, path: Path = CACHE) -> dict:
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


def role_of(row) -> str:
    role = getattr(row, "role", None)
    if role:
        return role
    if row.kind not in _DEV_ROLE:
        raise RuntimeError(f"no sweep role for {row.kind}")
    return _DEV_ROLE[row.kind]


def _answered(got: dict) -> bool:
    return got["reason"] == "" and got["answer"] is not None


def score_pass(cont, table: Dict[str, np.ndarray], rows: Sequence, *, with_state: bool) -> List[dict]:
    out = []
    for row in rows:
        got = _score_one(cont, table, row, with_state=with_state)
        expect = row.label if with_state else row.empty_label
        golds = row.golds if with_state else row.empty_golds
        ok = None if expect == "observe" else _ok(expect, golds, got["answer"], got["reason"])
        out.append({
            "kind": row.kind,
            "lang": row.lang,
            "role": role_of(row),
            "lag": getattr(row, "lag", None),
            "expect": expect,
            "ok": ok,
            "reason": got["reason"],
            "answer": got["answer"],
            "answered": _answered(got),
            "gold_answer": got["answer"] in golds if got["answer"] is not None else False,
            "question": row.question,
            "who": getattr(row, "who", ""),
        })
    return out


def _cell(rows: Sequence[dict], pred) -> str:
    chosen = [row for row in rows if pred(row)]
    if not chosen:
        return "—"
    good = sum(1 for row in chosen if row["ok"])
    return f"{good}/{len(chosen)}"


def summarize(scored: Sequence[dict]) -> dict:
    """Column counts. ``lag`` rows are outside the gate."""
    gated = [row for row in scored if row["role"] != "lag"]
    false = [
        row for row in gated
        if row["expect"] in ("absent_person", "missing_rel") and row["answered"]
    ]
    return {
        "n": len(scored),
        "gated": len(gated),
        "typo_hit": _cell(gated, lambda row: row["role"] == "hit"),
        "aged": _cell(gated, lambda row: row["role"] == "aged"),
        "tie": _cell(gated, lambda row: row["role"] == "tie"),
        "silent": _cell(gated, lambda row: row["role"] == "silent"),
        "other": _cell(gated, lambda row: row["role"] == "other"),
        "pos": _cell(gated, lambda row: row["role"] == "pos"),
        "neg": _cell(gated, lambda row: row["role"] == "neg"),
        "false_accept": len(false),
        "gated_correct": sum(1 for row in gated if row["ok"]),
        "false_rows": [
            {"kind": row["kind"], "lang": row["lang"], "who": row["who"], "answer": row["answer"]}
            for row in false[:8]
        ],
    }


def kind_table(scored: Sequence[dict]) -> List[dict]:
    order = []
    buckets: Dict[str, List[dict]] = {}
    for row in scored:
        if row["kind"] not in buckets:
            order.append(row["kind"])
            buckets[row["kind"]] = []
        buckets[row["kind"]].append(row)
    out = []
    for kind in order:
        group = buckets[kind]
        gated = [row for row in group if row["expect"] != "observe"]
        answered = sum(1 for row in group if row["answered"])
        gold = sum(1 for row in group if row["gold_answer"])
        out.append({
            "kind": kind,
            "n": len(group),
            "correct": None if not gated else sum(1 for row in gated if row["ok"]),
            "gated": len(gated),
            "answered": answered,
            "gold_answer": gold,
        })
    return out


def ladder(scored: Sequence[dict]) -> List[dict]:
    """Utterance lag only: target mention, then that many other-person turns."""
    rows = [
        row for row in scored
        if row["kind"] in ("typo_hit", "typo_aged", "lag_observe") and row["lag"] is not None
    ]
    lags = sorted({row["lag"] for row in rows})
    out = []
    for lag in lags:
        group = [row for row in rows if row["lag"] == lag]
        out.append({
            "lag": lag,
            "n": len(group),
            "answered": sum(1 for row in group if row["answered"]),
            "gold_answer": sum(1 for row in group if row["gold_answer"]),
            "abstained": sum(1 for row in group if row["reason"] == "absent_person"),
        })
    return out


def sweep(cont, table: Dict[str, np.ndarray], rows: Sequence) -> dict:
    import tidal.dialogue_state as dialogue_state

    saved = dialogue_state.TURN_WINDOW
    try:
        empty = score_pass(cont, table, rows, with_state=False)
        passes = {"empty": empty}
        for window in WINDOWS:
            dialogue_state.TURN_WINDOW = window
            passes[str(window)] = score_pass(cont, table, rows, with_state=True)
        return passes
    finally:
        dialogue_state.TURN_WINDOW = saved


def format_disjoint(rows: Sequence, disjoint: dict) -> str:
    lines = [
        f"questions `{questions_sha256(rows)}` n={len(rows)}",
        f"bank `{bank_sha256(rows)}` n={len(bank_strings(rows))}",
        "",
        "| against | sha | n | literal | max Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    worst = 0.0
    for name, row in disjoint.items():
        sha = row["sha"][:12] + "…" if row["sha"] else "—"
        lines.append(
            f"| {name} | {sha} | {row['n']} | {row['literal']} | {row['max_jaccard']:.3f} |"
        )
        worst = max(worst, row["max_jaccard"])
    lines.append("")
    lines.append(f"max Jaccard {worst:.3f}")
    lines.append(f"kinds {kind_counts(rows)}")
    return "\n".join(lines)


def format_kind(scored: Sequence[dict], title: str) -> str:
    lines = [
        f"## {title}",
        "",
        "| kind | n | correct | answered | gold answer |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in kind_table(scored):
        correct = "—" if row["correct"] is None else f"{row['correct']}/{row['gated']}"
        lines.append(
            f"| {row['kind']} | {row['n']} | {correct} | {row['answered']}/{row['n']} | "
            f"{row['gold_answer']}/{row['n']} |"
        )
    misses = [row for row in scored if row["ok"] is False]
    lines.append("")
    lines.append(f"misses {len(misses)}")
    for row in misses[:16]:
        lines.append(
            f"- {row['kind']} {row['lang']} expect {row['expect']} "
            f"got {row['reason'] or row['answer']} who {row['who']}"
        )
    return "\n".join(lines)


def format_sweep(name: str, passes: dict) -> str:
    lines = [
        f"## Window sweep: {name}",
        "",
        "Typo hit counts episodes that returned the stored value. Aged, tie,",
        "silent, and other count abstentions. Positive controls are exact",
        "spelling, nickname, focus, dismiss, open two-name, and a resolved",
        "hearer. Negative controls are extensions, a CJK prefix, a hearer",
        "with no first person, and a hearer whose relation is missing.",
        "False accepts are abstentions that returned a value. Lag rows are",
        "not in these columns.",
        "",
        "| setting | typo hits | aged abstain | tie abstain | silent abstain | other abstain | positive controls | negative controls | false accepts | gated correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["empty", *(str(window) for window in WINDOWS)]
    summaries = {}
    for key in order:
        if key not in passes:
            continue
        item = summarize(passes[key])
        summaries[key] = item
        label = "empty log" if key == "empty" else f"W={key}"
        lines.append(
            f"| {label} | {item['typo_hit']} | {item['aged']} | {item['tie']} | "
            f"{item['silent']} | {item['other']} | {item['pos']} | {item['neg']} | "
            f"{item['false_accept']} | {item['gated_correct']}/{item['gated']} |"
        )
    # Ladder from the state passes. Empty has no window.
    if any(row.get("lag") not in (None, -1) for row in passes.get("4", [])):
        lines.extend([
            "",
            "Utterance lag is the number of later turns after the intended",
            "person was named. Answered / n. This ladder is not the gate.",
            "",
            "| fillers after mention | " + " | ".join(
                ["empty log", *(f"W={window}" for window in WINDOWS)]
            ) + " |",
            "|---:|" + "---:|" * (1 + len(WINDOWS)),
        ])
        lag_ids = sorted({
            row["lag"] for row in passes["4"]
            if row["kind"] in ("typo_hit", "typo_aged", "lag_observe") and isinstance(row["lag"], int) and row["lag"] >= 0
        })
        for lag in lag_ids:
            cells = []
            for key in order:
                group = [
                    row for row in passes[key]
                    if row["kind"] in ("typo_hit", "typo_aged", "lag_observe") and row["lag"] == lag
                ]
                if not group:
                    cells.append("—")
                else:
                    cells.append(f"{sum(1 for row in group if row['answered'])}/{len(group)}")
            lines.append(f"| {lag} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _load_reference():
    import torch

    from tidal.question_head import load_reference

    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    cont = load_reference(overwrite_cos=1.0)
    return cont, prev


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the held-out dialogue bank")
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--disjoint-only", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    rows = build_rows()
    disjoint = jaccard_report(bank_strings(rows), heldout_corpora())
    print(format_disjoint(rows, disjoint))
    if args.disjoint_only and not (args.cache or args.score or args.sweep):
        return
    if args.cache:
        info = build_cache(rows)
        print(f"wrote {info['path']} n={info['n']} sha256={info['sha256']}")
    if not args.score and not args.sweep:
        return
    table = load_table(CACHE)
    missing = [text for text in bank_strings(rows) if text not in table]
    if missing:
        raise RuntimeError(f"cache missing {len(missing)} strings; rerun --cache")
    import torch

    cont, prev = _load_reference()
    payload = {}
    try:
        if args.score or args.sweep:
            shipped = score_pass(cont, table, rows, with_state=True)
            empty = score_pass(cont, table, rows, with_state=False)
            print()
            print(format_kind(shipped, "Held-out at the shipped window, with the log"))
            print()
            print(format_kind(empty, "Held-out with an empty log"))
            payload["shipped"] = kind_table(shipped)
            payload["empty_kinds"] = kind_table(empty)
            payload["shipped_summary"] = summarize(shipped)
            payload["empty_summary"] = summarize(empty)
        if args.sweep:
            from tidal.question_dialogue_dev_data import build_rows as dev_rows
            from tidal.question_dialogue_eval import CACHE as DEV_CACHE
            from tidal.question_dialogue_eval import bank_strings as dev_bank_strings

            dev = dev_rows()
            dev_table = load_table(DEV_CACHE)
            missing_dev = [text for text in dev_bank_strings(dev) if text not in dev_table]
            if missing_dev:
                raise RuntimeError(f"dev cache missing {len(missing_dev)} strings")
            held_passes = sweep(cont, table, rows)
            dev_passes = sweep(cont, dev_table, dev)
            print()
            print(format_sweep("held-out", held_passes))
            print()
            print(format_sweep("dialogue dev", dev_passes))
            payload["heldout_sweep"] = {key: summarize(val) for key, val in held_passes.items()}
            payload["dev_sweep"] = {key: summarize(val) for key, val in dev_passes.items()}
            payload["heldout_ladder"] = {
                key: ladder(val) for key, val in held_passes.items() if key != "empty"
            }
    finally:
        torch.backends.mkldnn.enabled = prev
    if args.json is not None:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
