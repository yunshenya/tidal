"""Score the expanded keep-alive held-out. CPU only.

The 42-ask bank stays a separate prior. Development rows are not mixed in.
The instrument is the tied relation encoder, not e5. Listener gaps are the
pre-registered label against the frozen replay, and are not relabeled here.
"""
from __future__ import annotations

import argparse
from typing import Dict, Optional, Sequence

from tidal.question_dialogue_eval import jaccard_report
from tidal.question_keepalive_expand_data import (
    Episode,
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    listener_gaps,
    questions_sha256,
)
from tidal.question_keepalive_heldout_eval import prior_bags as earlier_bags
from tidal.question_keepalive_heldout_eval import score_rows


def prior_bags() -> Dict[str, dict]:
    """Earlier banks, including the frozen 42-ask keep-alive bank and #35."""
    from tidal.question_keepalive_heldout_data import bank_sha256 as old_bank_sha
    from tidal.question_keepalive_heldout_data import bank_strings as old_bank
    from tidal.question_keepalive_heldout_data import questions as old_questions
    from tidal.question_keepalive_heldout_data import questions_sha256 as old_sha

    bags = earlier_bags()
    bags["keepalive_42_questions"] = {
        "sha": old_sha(),
        "strings": tuple(old_questions()),
    }
    bags["keepalive_42_bank"] = {
        "sha": old_bank_sha(),
        "strings": tuple(old_bank()),
    }
    return bags


def format_report(rows: Sequence[Episode], disjoint: dict, scored: Optional[dict]) -> str:
    lines = [
        "# Keep-alive expansion",
        "",
        f"Episodes {len(rows)}. Scored asks {sum(kind_counts(rows).values())}.",
        f"Questions `{questions_sha256(rows)}`.",
        f"Bank `{bank_sha256(rows)}` ({len(bank_strings(rows))} strings).",
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
    gaps = listener_gaps(rows)
    lines.extend(["", f"Listener gaps against the frozen replay: {len(gaps)}.", ""])
    for gap in gaps:
        lines.append(
            f"- `{gap['eid']}` {gap['kind']}: listener {gap['listener']} "
            f"({gap['listener_label']}), rule {gap['rule']} ({gap['rule_label']}). "
            f"{gap['question']}"
        )
    if scored is None:
        lines.append("")
        return "\n".join(lines) + "\n"
    lines.extend([
        "",
        "| kind | n | state rule | empty rule | state human | state FA | empty FA |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    total = {"n": 0, "state": 0, "empty": 0, "human": 0, "state_fa": 0, "empty_fa": 0}
    for name in kind_counts(rows):
        slot = scored["kind"][name]
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
    lines.extend(["", "## Encoder misses", ""])
    if scored["misses"]:
        for item in scored["misses"]:
            lines.append(
                f"- `{item['eid']}` {item['kind']}: state {item['state_reason'] or item['state_answer']} "
                f"(ok {item['state_ok']}); empty {item['empty_reason'] or item['empty_answer']} "
                f"(ok {item['empty_ok']}). {item['question']}"
            )
    else:
        lines.append("None. Every expansion ask matched its replayed rule label on both passes.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Score the expanded keep-alive held-out")
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
