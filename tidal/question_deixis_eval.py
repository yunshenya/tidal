"""Score the deixis dev bank. CPU only.

``ask`` sees the per-session hearer and the conversational slice. Each
episode is asked after its turns, then again with those turns omitted.
The second pass is the empty log. The frozen adversarial questions are
not an input.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from tidal.question_deixis_dev_data import (
    Episode,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_dialogue_eval import jaccard_report, prior_corpora


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including the dialogue dev split and its held-out split."""
    from tidal.question_dialogue_dev_data import bank_sha256 as dev_bank_sha
    from tidal.question_dialogue_dev_data import bank_strings as dev_bank
    from tidal.question_dialogue_dev_data import build_rows as dev_rows
    from tidal.question_dialogue_dev_data import questions_sha256 as dev_sha
    from tidal.question_dialogue_heldout_data import bank_sha256 as held_bank_sha
    from tidal.question_dialogue_heldout_data import bank_strings as held_bank
    from tidal.question_dialogue_heldout_data import build_rows as held_rows
    from tidal.question_dialogue_heldout_data import questions_sha256 as held_sha

    bags = prior_corpora()
    bags["dialogue_dev_questions"] = {
        "sha": dev_sha(),
        "strings": tuple(row.question for row in dev_rows()),
    }
    bags["dialogue_dev_bank"] = {"sha": dev_bank_sha(), "strings": tuple(dev_bank())}
    bags["dialogue_heldout_questions"] = {
        "sha": held_sha(),
        "strings": tuple(row.question for row in held_rows()),
    }
    bags["dialogue_heldout_bank"] = {"sha": held_bank_sha(), "strings": tuple(held_bank())}
    return bags


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


def score_rows(rows: Sequence[Episode]) -> dict:
    """Tiny encoder. Sentence vectors are relation directions, not e5.

    Same-relation slots tie, so an unscoped deictic ask returns the first
    slot of that relation. The bank puts the decoy there.
    """
    import numpy as np
    import torch

    from tidal.continuum import Continuum, _cfg
    from tidal.question_heldout import surface_feats
    from tidal.question_surface import LABELS as RELS
    from tidal.question_surface import load_weights
    from tidal.vap import VAPModel

    weights = load_weights()
    Wa = np.asarray(weights["Wa"], dtype=np.float64)
    directions = {}
    for i, lab in enumerate(RELS):
        direction = Wa[:-1, i].astype(np.float64)
        directions[lab] = direction / np.linalg.norm(direction)

    def slot_vec(rel: str, i: int) -> np.ndarray:
        # One vector per relation. Tied slots keep insertion order, so an
        # unscoped deictic ask returns the first slot of that relation.
        del i
        base = directions[rel]
        noise = np.zeros_like(base)
        noise[0] = 1.0
        noise -= base * float(np.dot(noise, base))
        noise /= np.linalg.norm(noise)
        vec = 0.95 * base + 0.05 * noise
        vec /= np.linalg.norm(vec)
        return vec.astype(np.float32)

    torch.manual_seed(4)
    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    kinds: Dict[str, dict] = {}
    failures: List[dict] = []
    try:
        for row in rows:
            slot = kinds.setdefault(row.kind, {"n": 0, "state": 0, "empty": 0})
            slot["n"] += 1
            for mode in ("state", "empty"):
                got = _score_one(cont, directions, slot_vec, surface_feats, row, with_state=mode == "state")
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
                        "who": row.who,
                    })
    finally:
        torch.backends.mkldnn.enabled = prev
    return {"kinds": kinds, "failures": failures}


def _score_one(cont, directions, slot_vec, surface_feats, row: Episode, *, with_state: bool) -> dict:
    import numpy as np

    cont.episodic.clear()
    cont._mem_group = None
    for i, slot in enumerate(row.slots):
        key = cont.embed_event(
            surface_feats(slot.text, "statement"),
            slot_vec(slot.rel, i + 1),
            surface=slot.text,
        )
        ok = cont.remember(key, slot.value, name=slot.name, rel=slot.rel, kind="fact", group=None)
        if ok is not True:
            raise RuntimeError(f"fact rejected: {slot.text}")
    if with_state:
        for turn in row.turns:
            cont.note_turn(
                turn.text,
                speaker=turn.speaker or None,
                hearer=turn.hearer or None,
                group=None,
            )
    query = cont.embed_event(
        surface_feats(row.question, "question"),
        directions[row.rel].astype(np.float32),
        surface=row.question,
    )
    value, info = cont.query(query, group=None, mode="top1")
    reason = info.get("abstain") or ""
    return {"reason": reason, "answer": _answer(value)}


def format_report(rows: Sequence[Episode], disjoint: dict, scored: Optional[dict]) -> str:
    lines = [
        "# Deixis",
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
            "State is the episode's turns. Empty is the same facts with no turn.",
            "A count is episodes whose label for that pass was met.",
            "",
            "| kind | n | state | empty |",
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
        for fail in scored["failures"][:16]:
            lines.append(
                f"- {fail['kind']} {fail['lang']} {fail['mode']}: "
                f"expect {fail['expect']} got {fail['reason'] or fail['answer']}"
            )
    lines.append("")
    lines.append(f"kinds {kind_counts(rows)}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the deixis dev bank")
    parser.add_argument("--disjoint-only", action="store_true")
    args = parser.parse_args(argv)
    rows = build_rows()
    disjoint = jaccard_report(bank_strings(rows), prior_bags())
    scored = None if args.disjoint_only else score_rows(rows)
    print(format_report(rows, disjoint, scored))


if __name__ == "__main__":
    main()
