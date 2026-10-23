"""Score the positive keep-alive slice once. The rule is not edited.

Labels stay the ones committed in ``question_keepalive_pos_data``. A miss is
reported. It is not relabeled and it is not patched.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from tidal.question_dialogue_eval import jaccard_report
from tidal.question_dialogue_real_data import _human_outcome, _predict
from tidal.question_keepalive_expand_data import bank_sha256 as expand_bank_sha
from tidal.question_keepalive_expand_data import bank_strings as expand_bank
from tidal.question_keepalive_expand_data import questions as expand_questions
from tidal.question_keepalive_expand_data import questions_sha256 as expand_questions_sha
from tidal.question_keepalive_expand_eval import prior_bags as expand_prior_bags
from tidal.question_keepalive_heldout_data import Ask, Episode, Slot, Turn
from tidal.question_keepalive_heldout_eval import (
    _blank,
    _empty_query,
    _false_accept,
    _ok,
    _query,
    _store,
)
from tidal.question_keepalive_pos_data import Ask as PosAsk
from tidal.question_keepalive_pos_data import Turn as PosTurn
from tidal.question_keepalive_pos_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_surface import apply_utterance, ask_people, refresh_resolved


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including the 42-ask bank, the 94-ask expansion, and #35."""
    bags = expand_prior_bags()
    bags["keepalive_expand_questions"] = {
        "sha": expand_questions_sha(),
        "strings": tuple(expand_questions()),
    }
    bags["keepalive_expand_bank"] = {
        "sha": expand_bank_sha(),
        "strings": tuple(expand_bank()),
    }
    return bags


def replay_rows(rows: Optional[Sequence] = None) -> tuple:
    """Attach the frozen rule to the committed listener labels. Labels stay."""
    from tidal.dialogue_state import DialogueState

    out = []
    for row in build_rows() if rows is None else rows:
        names = [slot.name for slot in row.slots]
        surfaces = [slot.text for slot in row.slots]
        slots = tuple(Slot(slot.name, slot.rel, slot.value, slot.text) for slot in row.slots)
        state = DialogueState()
        events = []
        for event in row.events:
            if isinstance(event, PosTurn):
                apply_utterance(
                    state, event.text, names, surfaces,
                    speaker=event.speaker or None,
                    hearer=event.hearer or None,
                )
                events.append(Turn(event.text, event.speaker, event.hearer))
                continue
            if not isinstance(event, PosAsk):
                raise RuntimeError(type(event))
            rule, rule_golds, who = _predict(state, slots, event.question, event.rel)
            empty_rule, empty_golds, _empty_who = _predict(
                DialogueState(), slots, event.question, event.rel,
            )
            human, human_golds, who_human = _human_outcome(event.listener, slots, event.rel)
            events.append(Ask(
                event.kind, event.lang, event.rel, event.question,
                rule, human, rule_golds, human_golds,
                empty_rule, empty_golds, who, who_human,
            ))
            mentioned = ask_people(
                event.question, names, surfaces,
                state.conversational(ignore_surface=event.question),
            )
            state.note_ask(mentioned, event.question)
            refresh_resolved(state, event.question, names, surfaces)
        out.append(Episode(row.eid, row.lang, slots, tuple(events)))
    return tuple(out)


def _named(slots: Sequence[Slot], rel: str, value) -> str:
    if value is None:
        return "—"
    for slot in slots:
        if slot.rel == rel and slot.value == value:
            return f"{slot.name} {value}"
    return str(value)


def score_rows(rows: Sequence[Episode]) -> dict:
    """Same tied encoder as the 42-ask bank. One pass. Misses are kept."""
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
    listener_misses: List[dict] = []
    encoder_misses: List[dict] = []
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
                _record(by_kind, by_lang, listener_misses, encoder_misses, row, event, got, empty)
    finally:
        torch.backends.mkldnn.enabled = prev
    return {
        "kind": by_kind,
        "lang": by_lang,
        "listener_misses": listener_misses,
        "encoder_misses": encoder_misses,
    }


def _record(by_kind, by_lang, listener_misses, encoder_misses, row, ask, got, empty) -> None:
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
    item = {
        "eid": row.eid,
        "kind": ask.kind,
        "lang": ask.lang,
        "rel": ask.rel,
        "question": ask.question,
        "turns": tuple(event.text for event in row.events if isinstance(event, Turn)),
        "listener": ask.who_human,
        "listener_label": ask.human,
        "rule": ask.who_rule,
        "rule_label": ask.rule,
        "state_answer": _named(row.slots, ask.rel, got["answer"]),
        "state_reason": got["reason"],
        "empty_answer": _named(row.slots, ask.rel, empty["answer"]),
        "empty_reason": empty["reason"],
        "state_fa": state_fa,
        "empty_fa": empty_fa,
    }
    if not human_ok:
        listener_misses.append(item)
    if not state_ok or not empty_ok:
        encoder_misses.append(item)


def _table(scored: dict, key: str, order: Sequence[str]) -> List[str]:
    lines = [
        "| | n | state rule | empty rule | listener | state FA | empty FA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    total = _blank()
    for name in order:
        slot = scored[key][name]
        lines.append(
            f"| {name} | {slot['n']} | {slot['state']}/{slot['n']} | "
            f"{slot['empty']}/{slot['n']} | {slot['human']}/{slot['n']} | "
            f"{slot['state_fa']} | {slot['empty_fa']} |"
        )
        for field in total:
            total[field] += slot[field]
    lines.append(
        f"| total | {total['n']} | {total['state']}/{total['n']} | "
        f"{total['empty']}/{total['n']} | {total['human']}/{total['n']} | "
        f"{total['state_fa']} | {total['empty_fa']} |"
    )
    return lines


def _miss_block(items: Sequence[dict]) -> List[str]:
    if not items:
        return ["None.", ""]
    lines = []
    for item in items:
        lines.append(
            f"- `{item['eid']}` {item['kind']} {item['lang']} {item['rel']}: "
            f"listener {item['listener']} ({item['listener_label']}), "
            f"rule {item['rule']} ({item['rule_label']}). "
            f"State {item['state_reason'] or item['state_answer']}. "
            f"Empty {item['empty_reason'] or item['empty_answer']}."
        )
        for turn in item["turns"]:
            lines.append(f"  - {turn}")
        lines.append(f"  - {item['question']}")
    lines.append("")
    return lines


def format_report(raw_rows, scored_rows, disjoint: dict, scored: Optional[dict]) -> str:
    lines = [
        "# Positive keep-alive slice",
        "",
        f"Episodes {len(raw_rows)}. Scored asks {sum(kind_counts(raw_rows).values())}.",
        f"Questions `{questions_sha256(raw_rows)}`.",
        f"Bank `{bank_sha256(raw_rows)}` ({len(bank_strings(raw_rows))} strings).",
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
        return "\n".join(lines) + "\n"
    lines.extend(["", "## By kind", ""])
    lines.extend(_table(scored, "kind", ("pos_keep", "pos_voc", "neg_near")))
    lines.extend(["", "## By language", ""])
    lines.extend(_table(scored, "lang", ("en", "zh")))
    lines.extend(["", "## Listener misses", ""])
    lines.extend(_miss_block(scored["listener_misses"]))
    lines.extend(["## Encoder misses", ""])
    if scored["encoder_misses"]:
        lines.extend(_miss_block(scored["encoder_misses"]))
    else:
        lines.append("None. State and empty matched the replayed rule on every ask.")
        lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the positive keep-alive slice")
    parser.add_argument("--disjoint-only", action="store_true")
    parser.add_argument("--write", default="")
    args = parser.parse_args(argv)
    raw = build_rows()
    disjoint = jaccard_report(bank_strings(raw), prior_bags())
    scored = None
    scored_rows = ()
    if not args.disjoint_only:
        scored_rows = replay_rows(raw)
        scored = score_rows(scored_rows)
    text = format_report(raw, scored_rows, disjoint, scored)
    if args.write:
        from pathlib import Path

        Path(args.write).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
