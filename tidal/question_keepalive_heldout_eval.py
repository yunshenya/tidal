"""Score the keep-alive held-out bank. CPU only.

State mode walks ``note_turn`` and ``query``. Empty mode asks each question
on a fresh store after that walk. The instrument is the tied relation
encoder, not e5. Development rows on the #35 bank are scored by
``question_dialogue_real_eval`` and are not mixed into this table.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from tidal.question_dialogue_eval import jaccard_report
from tidal.question_dialogue_real_eval import prior_bags as real_prior_bags
from tidal.question_keepalive_heldout_data import (
    Ask,
    Episode,
    Turn,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including the #35 real-dialogue questions and statements."""
    from tidal.question_dialogue_real_data import bank_sha256 as real_bank_sha
    from tidal.question_dialogue_real_data import bank_strings as real_bank
    from tidal.question_dialogue_real_data import questions as real_questions
    from tidal.question_dialogue_real_data import questions_sha256 as real_sha

    bags = real_prior_bags()
    bags["dialogue_real_questions"] = {
        "sha": real_sha(),
        "strings": tuple(real_questions()),
    }
    bags["dialogue_real_bank"] = {
        "sha": real_bank_sha(),
        "strings": tuple(real_bank()),
    }
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


def _blank() -> dict:
    return {"n": 0, "state": 0, "empty": 0, "human": 0, "state_fa": 0, "empty_fa": 0}


def score_rows(rows: Sequence[Episode]) -> dict:
    """Tiny encoder. A hit is the string decision, not a retrieval score."""
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
    for lab in RELS:
        direction = Wa[:-1, RELS.index(lab)].astype(np.float64)
        directions[lab] = direction / np.linalg.norm(direction)

    def slot_vec(rel: str, i: int) -> np.ndarray:
        base = directions[rel]
        noise = np.zeros_like(base)
        noise[0] = 1.0
        noise -= base * float(np.dot(noise, base))
        noise /= np.linalg.norm(noise)
        tilt = 0.05 + (i * 1e-6)
        vec = 0.95 * base + tilt * noise
        vec /= np.linalg.norm(vec)
        return vec.astype(np.float32)

    torch.manual_seed(4)
    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    by_kind: Dict[str, dict] = {}
    by_lang: Dict[str, dict] = {}
    misses: List[dict] = []
    try:
        for row in rows:
            _store(cont, slot_vec, surface_feats, row)
            pending = []
            for event in row.events:
                if isinstance(event, Turn):
                    cont.note_turn(
                        event.text,
                        speaker=event.speaker or None,
                        hearer=event.hearer or None,
                        group=None,
                    )
                    continue
                if not isinstance(event, Ask):
                    raise RuntimeError(type(event))
                pending.append((event, _query(cont, directions, surface_feats, event)))
            for event, got in pending:
                empty = _empty_query(cont, directions, slot_vec, surface_feats, row, event)
                _record(by_kind, by_lang, misses, row, event, got, empty)
    finally:
        torch.backends.mkldnn.enabled = prev
    return {"kind": by_kind, "lang": by_lang, "misses": misses}


def _store(cont, slot_vec, surface_feats, row: Episode) -> None:
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


def _query(cont, directions, surface_feats, ask: Ask) -> dict:
    import numpy as np

    query = cont.embed_event(
        surface_feats(ask.question, "question"),
        directions[ask.rel].astype(np.float32),
        surface=ask.question,
    )
    value, info = cont.query(query, group=None, mode="top1")
    return {"reason": info.get("abstain") or "", "answer": _answer(value)}


def _empty_query(cont, directions, slot_vec, surface_feats, row: Episode, ask: Ask) -> dict:
    _store(cont, slot_vec, surface_feats, row)
    return _query(cont, directions, surface_feats, ask)


def _record(by_kind, by_lang, misses, row, ask, got, empty) -> None:
    slots = (by_kind.setdefault(ask.kind, _blank()), by_lang.setdefault(ask.lang, _blank()))
    state_ok = _ok(ask.rule, ask.rule_golds, got["answer"], got["reason"])
    empty_ok = _ok(ask.empty_rule, ask.empty_golds, empty["answer"], empty["reason"])
    human_ok = _ok(ask.human, ask.human_golds, got["answer"], got["reason"])
    state_fa = _false_accept(ask.rule, got["answer"])
    empty_fa = _false_accept(ask.empty_rule, empty["answer"])
    for slot in slots:
        slot["n"] += 1
        slot["state"] += int(state_ok)
        slot["empty"] += int(empty_ok)
        slot["human"] += int(human_ok)
        slot["state_fa"] += int(state_fa)
        slot["empty_fa"] += int(empty_fa)
    if not state_ok or not empty_ok:
        misses.append({
            "eid": row.eid,
            "kind": ask.kind,
            "question": ask.question,
            "state_ok": state_ok,
            "empty_ok": empty_ok,
            "state_reason": got["reason"],
            "state_answer": got["answer"],
            "empty_reason": empty["reason"],
            "empty_answer": empty["answer"],
            "who_rule": ask.who_rule,
        })


def format_report(rows: Sequence[Episode], disjoint: dict, scored: Optional[dict]) -> str:
    lines = [
        "# Pronoun keep-alive and leading vocatives",
        "",
        "Development rows are the #35 real-dialogue bank. This file is the fresh "
        "held-out. No threshold, window, or name-head weight was fit. `TURN_WINDOW` "
        "stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is "
        "no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is "
        "not an input.",
        "",
        f"Held-out episodes {len(rows)}. Scored asks {sum(kind_counts(rows).values())}.",
        f"Questions `{questions_sha256(rows)}`.",
        f"Bank `{bank_sha256(rows)}` ({len(bank_strings(rows))} strings).",
        "",
        "Rule labels are the replay of `apply_utterance`, `refresh_resolved`, and the "
        "deixis selectors. They were checked against the pre-registered human labels "
        "when the bank was built. State mode walks `note_turn` and `query`. Empty mode "
        "asks each question on a fresh store. A false accept is a rule abstain that "
        "returned a value.",
        "",
        "## Disjointness",
        "",
        "Literal overlap 0 and max token Jaccard under 0.50 against every earlier bank, "
        "including the #35 real-dialogue bank. Tokenizer is `tidal.question_heldout_data._tokens`.",
        "",
        "| against | sha | n | literal | max Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    closest = ("", ("", ""), 0.0)
    for name, row in disjoint.items():
        sha = row["sha"][:12] + "…" if row["sha"] else "—"
        lines.append(
            f"| {name} | {sha} | {row['n']} | {row['literal']} | {row['max_jaccard']:.3f} |"
        )
        if row["max_jaccard"] > closest[2]:
            closest = (name, row["pair"], row["max_jaccard"])
    lines.append("")
    lines.append(f"Closest pair is {closest[2]:.3f} against {closest[0]}.")
    if closest[1][0]:
        lines.append(f"`{closest[1][0]}`")
        lines.append(f"`{closest[1][1]}`")
    if scored is None:
        lines.append("")
        lines.append(f"kinds {kind_counts(rows)}")
        return "\n".join(lines) + "\n"
    lines.extend(["", "## Held-out", "", "| kind | n | state rule | empty rule | state human | state FA | empty FA |", "|---|---:|---:|---:|---:|---:|---:|"])
    total = _blank()
    for name, slot in scored["kind"].items():
        lines.append(
            f"| {name} | {slot['n']} | {slot['state']}/{slot['n']} | "
            f"{slot['empty']}/{slot['n']} | {slot['human']}/{slot['n']} | "
            f"{slot['state_fa']} | {slot['empty_fa']} |"
        )
        for key in total:
            total[key] += slot[key]
    lines.append(
        f"| total | {total['n']} | {total['state']}/{total['n']} | "
        f"{total['empty']}/{total['n']} | {total['human']}/{total['n']} | "
        f"{total['state_fa']} | {total['empty_fa']} |"
    )
    lines.extend(["", "## Spec misses", ""])
    if scored["misses"]:
        for item in scored["misses"]:
            lines.append(
                f"- `{item['eid']}` {item['kind']}: state {item['state_reason'] or item['state_answer']} "
                f"(rule {item['who_rule']}, ok {item['state_ok']}); "
                f"empty {item['empty_reason'] or item['empty_answer']} (ok {item['empty_ok']}). "
                f"{item['question']}"
            )
    else:
        lines.append("None. Every held-out ask matched its rule label on both passes.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the keep-alive held-out bank")
    parser.add_argument("--disjoint-only", action="store_true")
    parser.add_argument("--write", default="")
    args = parser.parse_args(argv)
    rows = build_rows()
    disjoint = jaccard_report(bank_strings(rows), prior_bags())
    scored = None if args.disjoint_only else score_rows(rows)
    text = format_report(rows, disjoint, scored)
    if args.write:
        from pathlib import Path

        Path(args.write).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
