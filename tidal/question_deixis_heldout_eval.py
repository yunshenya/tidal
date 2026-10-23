"""Score the deixis held-out bank. CPU only.

The rule is the one already on this branch. This module does not change it.
Each episode is asked after its turns and any earlier name-ask, then again
with that log omitted. The second pass is the empty log.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from tidal.question_deixis_eval import prior_bags as dev_prior_bags
from tidal.question_deixis_heldout_data import (
    Episode,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_dialogue_eval import jaccard_report


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including the 48-episode deixis dev split."""
    from tidal.question_deixis_dev_data import bank_sha256 as dev_bank_sha
    from tidal.question_deixis_dev_data import bank_strings as dev_bank
    from tidal.question_deixis_dev_data import questions_sha256 as dev_sha
    from tidal.question_deixis_dev_data import questions as dev_questions

    bags = dev_prior_bags()
    bags["deixis_dev_questions"] = {"sha": dev_sha(), "strings": tuple(dev_questions())}
    bags["deixis_dev_bank"] = {"sha": dev_bank_sha(), "strings": tuple(dev_bank())}
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


def _false_accept(expect: str, answer) -> bool:
    return expect in ("absent_person", "missing_rel") and answer is not None


def score_rows(rows: Sequence[Episode]) -> dict:
    """Tiny encoder. Same-relation slots tie, so an unscoped deictic ask returns the first."""
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
            slot = kinds.setdefault(row.kind, {
                "n": 0, "state": 0, "empty": 0, "state_fa": 0, "empty_fa": 0,
            })
            slot["n"] += 1
            for mode in ("state", "empty"):
                got = _score_one(
                    cont, directions, slot_vec, surface_feats, row, with_state=mode == "state",
                )
                expect = row.label if mode == "state" else row.empty_label
                golds = row.golds if mode == "state" else row.empty_golds
                if _false_accept(expect, got["answer"]):
                    slot["state_fa" if mode == "state" else "empty_fa"] += 1
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
        for text, rel in row.prior_asks:
            query = cont.embed_event(
                surface_feats(text, "question"),
                directions[rel].astype(np.float32),
                surface=text,
            )
            cont.query(query, group=None, mode="top1")
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
        f"questions `{questions_sha256(rows)}` n={len(rows)}",
        f"bank `{bank_sha256(rows)}` n={len(bank_strings(rows))}",
        "",
        "| against | sha | n | literal | max Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    closest = ("", "", 0.0)
    for name, row in disjoint.items():
        sha = row["sha"][:12] + "…" if row["sha"] else "—"
        lines.append(
            f"| {name} | {sha} | {row['n']} | {row['literal']} | {row['max_jaccard']:.3f} |"
        )
        if row["max_jaccard"] > closest[2]:
            closest = (name, row["pair"], row["max_jaccard"])
    lines.append("")
    lines.append(f"max Jaccard {closest[2]:.3f} against {closest[0]}")
    if closest[1]:
        lines.append(f"closest `{closest[1][0]}` ‖ `{closest[1][1]}`")
    if scored is not None:
        lines.extend([
            "",
            "| kind | n | state | empty | state false accept | empty false accept |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        totals = {"n": 0, "state": 0, "empty": 0, "state_fa": 0, "empty_fa": 0}
        for kind, slot in scored["kinds"].items():
            lines.append(
                f"| {kind} | {slot['n']} | {slot['state']}/{slot['n']} | "
                f"{slot['empty']}/{slot['n']} | {slot['state_fa']} | {slot['empty_fa']} |"
            )
            for key in totals:
                totals[key] += slot[key]
        lines.append(
            f"| total | {totals['n']} | {totals['state']}/{totals['n']} | "
            f"{totals['empty']}/{totals['n']} | {totals['state_fa']} | {totals['empty_fa']} |"
        )
        lines.append("")
        lines.append(f"failures {len(scored['failures'])}")
        for fail in scored["failures"]:
            lines.append(
                f"- {fail['kind']} {fail['lang']} {fail['mode']} {fail['who']}: "
                f"expect {fail['expect']} got {fail['reason'] or fail['answer']}"
            )
    lines.append("")
    lines.append(f"kinds {kind_counts(rows)}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the deixis held-out bank")
    parser.add_argument("--disjoint-only", action="store_true")
    args = parser.parse_args(argv)
    rows = build_rows()
    disjoint = jaccard_report(bank_strings(rows), prior_bags())
    scored = None if args.disjoint_only else score_rows(rows)
    print(format_report(rows, disjoint, scored))


if __name__ == "__main__":
    main()
