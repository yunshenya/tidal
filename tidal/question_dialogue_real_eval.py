"""Score the natural multi-turn bank. CPU only.

Each episode is one conversation. State mode walks turns and asks in order,
the way a caller would: ``note_turn`` then ``query``. Empty mode asks each
question on a fresh store of the same facts, so an earlier ask cannot license
a later edit. The instrument is the tied relation encoder from the deixis
dev, not e5. Same-relation slots tie, and the decoy is stored first.

The frozen adversarial questions are not an input. This module does not
change the ask path.
"""
from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence

from tidal.question_deixis_heldout_eval import prior_bags as heldout_prior_bags
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_dialogue_real_data import (
    Ask,
    Episode,
    Turn,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
    scope_counts,
)


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including both deixis splits."""
    from tidal.question_deixis_heldout_data import bank_sha256 as held_bank_sha
    from tidal.question_deixis_heldout_data import bank_strings as held_bank
    from tidal.question_deixis_heldout_data import questions as held_questions
    from tidal.question_deixis_heldout_data import questions_sha256 as held_sha

    bags = heldout_prior_bags()
    bags["deixis_heldout_questions"] = {
        "sha": held_sha(),
        "strings": tuple(held_questions()),
    }
    bags["deixis_heldout_bank"] = {
        "sha": held_bank_sha(),
        "strings": tuple(held_bank()),
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
    return {
        "n": 0,
        "state": 0,
        "empty": 0,
        "human": 0,
        "state_fa": 0,
        "empty_fa": 0,
    }


def _bump(table: Dict[str, dict], key: str) -> dict:
    return table.setdefault(key, _blank())


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
    for _i, lab in enumerate(RELS):
        direction = Wa[:-1, RELS.index(lab)].astype(np.float64)
        directions[lab] = direction / np.linalg.norm(direction)

    def slot_vec(rel: str, i: int) -> np.ndarray:
        # Same relation, one direction. A later slot carries a slightly larger
        # orthogonal part so its cosine is lower by about 1e-6 per position.
        # A batched cosine is not bit-stable: identical rows can differ by one
        # ulp, and np.argmax then returns the later person. The slope keeps
        # the earliest slot, which is the decoy for an unscoped deictic ask.
        # It is far below the gap between relations, and the cosine stays
        # under the exact-match abstain cut.
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
    by_scope: Dict[str, dict] = {}
    by_family: Dict[str, dict] = {}
    by_kind: Dict[str, dict] = {}
    by_band: Dict[str, dict] = {}
    by_lang: Dict[str, dict] = {}
    misses: List[dict] = []
    asks: List[dict] = []
    try:
        for row in rows:
            _store(cont, slot_vec, surface_feats, row)
            pending: List[tuple] = []
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
                asks.append(_record(
                    by_scope, by_family, by_kind, by_band, by_lang, misses,
                    row, event, got, empty,
                ))
    finally:
        torch.backends.mkldnn.enabled = prev
    return {
        "scope": by_scope,
        "family": by_family,
        "kind": by_kind,
        "band": by_band,
        "lang": by_lang,
        "misses": misses,
        "asks": asks,
    }


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
    reason = info.get("abstain") or ""
    return {"reason": reason, "answer": _answer(value)}


def _empty_query(cont, directions, slot_vec, surface_feats, row: Episode, ask: Ask) -> dict:
    """One ask on a fresh store. Call only after that episode's state walk."""
    _store(cont, slot_vec, surface_feats, row)
    return _query(cont, directions, surface_feats, ask)


def _record(by_scope, by_family, by_kind, by_band, by_lang, misses, row, ask, got, empty) -> dict:
    slots = (
        _bump(by_scope, ask.scope),
        _bump(by_family, ask.family),
        _bump(by_kind, ask.phenomenon),
        _bump(by_band, ask.band),
        _bump(by_lang, ask.lang),
    )
    for slot in slots:
        slot["n"] += 1
    state_ok = _ok(ask.rule, ask.rule_golds, got["answer"], got["reason"])
    empty_ok = _ok(ask.empty_rule, ask.empty_golds, empty["answer"], empty["reason"])
    human_ok = _ok(ask.human, ask.human_golds, got["answer"], got["reason"])
    state_fa = _false_accept(ask.rule, got["answer"])
    empty_fa = _false_accept(ask.empty_rule, empty["answer"])
    for slot in slots:
        slot["state"] += int(state_ok)
        slot["empty"] += int(empty_ok)
        slot["human"] += int(human_ok)
        slot["state_fa"] += int(state_fa)
        slot["empty_fa"] += int(empty_fa)
    def _who(answer: Optional[str]) -> str:
        if not answer:
            return ""
        for slot in row.slots:
            if slot.value == answer:
                return slot.name
        return answer

    item = {
        "eid": row.eid,
        "title": row.title,
        "phenomenon": ask.phenomenon,
        "scope": ask.scope,
        "band": ask.band,
        "family": ask.family,
        "lang": ask.lang,
        "question": ask.question,
        "rule": ask.rule,
        "human": ask.human,
        "who_rule": ask.who_rule,
        "who_human": ask.who_human,
        "note": ask.note,
        "state_ok": state_ok,
        "empty_ok": empty_ok,
        "human_ok": human_ok,
        "state_fa": state_fa,
        "empty_fa": empty_fa,
        "state_reason": got["reason"],
        "state_answer": got["answer"],
        "state_who": _who(got["answer"]) or got["reason"] or "none",
        "empty_reason": empty["reason"],
        "empty_answer": empty["answer"],
        "empty_who": _who(empty["answer"]) or empty["reason"] or "none",
    }
    if not state_ok or not empty_ok:
        misses.append(item)
    return item


def _totals(table: Dict[str, dict]) -> dict:
    out = _blank()
    for slot in table.values():
        for key in out:
            out[key] += slot[key]
    return out


def _line(name: str, slot: dict) -> str:
    return (
        f"| {name} | {slot['n']} | {slot['state']}/{slot['n']} | "
        f"{slot['empty']}/{slot['n']} | {slot['human']}/{slot['n']} | "
        f"{slot['state_fa']} | {slot['empty_fa']} |"
    )


def _table(title: str, table: Dict[str, dict]) -> List[str]:
    lines = [
        f"## {title}",
        "",
        "| | n | state rule | empty rule | state human | state FA | empty FA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, slot in table.items():
        lines.append(_line(name, slot))
    lines.append(_line("total", _totals(table)))
    return lines


def _show(item: dict) -> str:
    return (
        f"- `{item['eid']}` {item['phenomenon']} ({item['lang']}): "
        f"rule {item['who_rule']} / human {item['who_human']}; "
        f"ask returned {item['state_who']}. `{item['question']}` {item['note']}"
    )


def recommendation(scored: dict) -> str:
    """Pre-registered decision. A gated rule miss is the only in-scope code change."""
    gated = scored["scope"].get("gated", _blank())
    total = _totals(scored["scope"])
    gated_clean = (
        gated["n"] > 0
        and gated["state"] == gated["n"]
        and gated["empty"] == gated["n"]
        and gated["state_fa"] == 0
        and gated["empty_fa"] == 0
    )
    any_miss = bool(scored["misses"])
    if gated_clean and total["state_fa"] == 0 and total["empty_fa"] == 0:
        return (
            "Ship the caller contract, and do not widen the mechanism in this change. "
            "Gated rows match the shipped rule on both passes, and a rule abstain never "
            "returned a value. The experience gaps are the ones already outside the "
            "contract: four quiet turns age a name out, a pronoun does not refresh the "
            "log, a vocative is not a hearer unless the caller passes it, two people on "
            "the newest turn abstain with no gender, an insertion or a swap or a 老/小 "
            "prefix is not edit-1, and the speaker is not he. Open addressee forms stay "
            "out of scope. The next product step is to call note_turn with the hearer "
            "every time, and to treat the window and the closed class as the behavior, "
            "not as a full coreference model. A richer dialogue state would be a new "
            "mechanism, not a patch on this bank."
        )
    if not gated_clean:
        return (
            "Gated rule fidelity failed. That is a spec miss on the shipped path. "
            "Do not retune the window, the edit radius, or a threshold on this bank. "
            "A code change belongs in a later revision only if it is a small always-on "
            "bug, and only after a fresh held-out slice and the probe floors."
        )
    if any_miss:
        return (
            "Gated rows match and false accepts on the rule are zero, but some "
            "non-gated row disagrees with its rule label. Treat that as a spec miss "
            "to read, not as a reason to fit the window or the name head on this bank."
        )
    return "See the tables."


def format_report(rows: Sequence[Episode], disjoint: dict, scored: Optional[dict]) -> str:
    n_asks = sum(1 for row in rows for event in row.events if isinstance(event, Ask))
    lines = [
        "# Real multi-turn dialogue",
        "",
        "Measurement of the shipped ask path on natural multi-turn episodes. "
        "No threshold, window, or name-head weight was fit. `TURN_WINDOW` stays 4. "
        "`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. "
        "`LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input.",
        "",
        f"Episodes {len(rows)}. Scored asks {n_asks}.",
        f"Questions `{questions_sha256(rows)}`.",
        f"Bank `{bank_sha256(rows)}` ({len(bank_strings(rows))} strings).",
        "",
    ]
    if scored is not None:
        total = _totals(scored["scope"])
        gated = scored["scope"].get("gated", _blank())
        gap = scored["scope"].get("gap", _blank())
        oos = scored["scope"].get("oos", _blank())
        control = scored["scope"].get("control", _blank())
        observe = scored["scope"].get("observe", _blank())
        ordinary = scored["band"].get("ordinary", _blank())
        lines.extend([
            "## Result",
            "",
            f"State rule {total['state']}/{total['n']}. Empty rule {total['empty']}/{total['n']}. "
            f"False accepts {total['state_fa'] + total['empty_fa']}.",
            f"Gated rows, where the rule and the listener already agree: "
            f"{gated['state']}/{gated['n']} with the turns, {gated['empty']}/{gated['n']} on the empty log, "
            f"human {gated['human']}/{gated['n']}. Ordinary talk is {ordinary['state']}/{ordinary['n']}.",
            f"The listener's reading on the whole bank is {total['human']}/{total['n']}. "
            f"Experience gaps are {gap['human']}/{gap['n']} against the listener and "
            f"{gap['state']}/{gap['n']} against the rule. "
            f"Open addressee forms are {oos['human']}/{oos['n']} against the listener. "
            f"Empty-log controls are {control['human']}/{control['n']} against the listener. "
            f"Name-head observations are {observe['human']}/{observe['n']} against the listener. "
            "A miss in those four columns is not a regression.",
            "",
        ])
    lines.extend([
        "Rule labels are the replay of `DialogueState`, `ask_people`, and the deixis "
        "row selectors. Human labels are the conversational reading written with the "
        "episode. State mode walks `note_turn` and `query` in order. Empty mode asks "
        "each question on a fresh store of the same facts. The encoder is the tied "
        "relation instrument from the deixis dev. Same-relation slots share one "
        "direction, the decoy is stored first, and a 1e-6 insertion-order slope "
        "keeps that earliest slot when a batched cosine would otherwise move the "
        "tie by one ulp. A hit is who was selected, not an e5 score. A false "
        "accept is a rule abstain (`absent_person` or `missing_rel`) that returned a value.",
        "",
        "## Disjointness",
        "",
        "Literal overlap 0 and max token Jaccard under 0.50. Tokenizer is "
        "`tidal.question_heldout_data._tokens`. Strings are questions, turns, and fact sentences.",
        "",
        "| against | sha | n | literal | max Jaccard |",
        "|---|---|---:|---:|---:|",
    ])
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
        lines.append(f"scopes {scope_counts(rows)}")
        lines.append(f"kinds {kind_counts(rows)}")
        return "\n".join(lines) + "\n"
    lines.append("")
    lines.extend(_table("Scope", scored["scope"]))
    lines.append("")
    lines.extend(_table("Family", scored["family"]))
    lines.append("")
    lines.extend(_table("Language", scored["lang"]))
    lines.append("")
    lines.extend(_table("Band", scored["band"]))
    lines.append("")
    lines.extend(_table("Phenomenon", scored["kind"]))
    lines.append("")
    lines.append("## Where the listener and the rule disagree")
    lines.append("")
    lines.append(
        "These rows were marked gap before the score. A human miss with a rule hit "
        "is the experience gap. It is not a regression."
    )
    lines.append("")
    gaps = [item for item in scored["asks"] if item["scope"] == "gap"]
    seen = []
    for item in gaps:
        if item["phenomenon"] in seen:
            continue
        seen.append(item["phenomenon"])
        lines.append(f"### {item['phenomenon']}")
        lines.append("")
        lines.append(item["note"])
        lines.append("")
        for row in gaps:
            if row["phenomenon"] != item["phenomenon"]:
                continue
            lines.append(_show(row))
        lines.append("")
    lines.append("## Out of scope")
    lines.append("")
    lines.append(
        "Open addressee forms left out of #34. A failure here is not a regression."
    )
    lines.append("")
    for item in scored["asks"]:
        if item["scope"] == "oos":
            lines.append(_show(item))
    lines.append("")
    lines.append("## Empty log")
    lines.append("")
    lines.append(
        "No turns. The human column abstains. The rule leaves a deictic pronoun on "
        "the ranker, which returns the decoy. That is current main behavior."
    )
    lines.append("")
    for item in scored["asks"]:
        if item["scope"] == "control":
            lines.append(_show(item))
    lines.append("")
    lines.append("## Observed, not gated")
    lines.append("")
    lines.append(
        "Short name, given-name nickname, and pinyin. The head may accept or abstain. "
        "Neither result is a gate."
    )
    lines.append("")
    for item in scored["asks"]:
        if item["scope"] == "observe":
            lines.append(_show(item))
    lines.append("")
    lines.append("## Spec misses")
    lines.append("")
    if scored["misses"]:
        lines.append(
            f"{len(scored['misses'])} ask(s) disagreed with the rule label on a pass."
        )
        for item in scored["misses"]:
            lines.append(
                f"- `{item['eid']}` {item['phenomenon']} {item['scope']}: "
                f"state {item['state_reason'] or item['state_answer']} "
                f"(rule {item['who_rule']}, ok {item['state_ok']}); "
                f"empty {item['empty_reason'] or item['empty_answer']} "
                f"(ok {item['empty_ok']}). {item['question']}"
            )
    else:
        lines.append(
            "None. Every ask matched its rule label on the state pass and on the empty pass."
        )
    lines.append("")
    lines.append("## Floors")
    lines.append("")
    lines.append(
        "The ask path is unchanged, so the floors already recorded in `reports/deixis.md` "
        "still stand. They were not re-run for this measurement. Probe answerable 365/370, "
        "false abstain 0/370, unanswerable 72/72, gold lost at overwrite 0.9 is 0, "
        "co-resident cosine above 0.9 is 0 (max 0.8927). Frozen hits 188/216, false abstain "
        "19/216, near_dup_absent 30/30. Dialogue dev 102/102 and 102/102. Dialogue held-out "
        "at window 4, gated 108/108 and 108/108. Deixis dev 48/48 and 48/48. Deixis held-out "
        "128/136 with the turns (the miss is `addressee_extra` 0/8) and 136/136 on the empty log."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(recommendation(scored))
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the natural multi-turn bank")
    parser.add_argument("--disjoint-only", action="store_true")
    parser.add_argument("--write", default="", help="Write the report to this path")
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
