"""Print comparable E2 stage probes and, separately, pseudo-stage probes.

Reads the committed summaries, including the true-stage re-evaluation.
Does not train and does not rescore.

    python scripts/growbrain_e2_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidal.growbrain_e2_probes import (
    arm_stage_probe_means,
    handwritten_accs,
    is_pseudo_stage_probe_row,
    stage_probe_accs,
    unscored_stage_probes,
)
MULTISEED = ROOT / "reports" / "growbrain_e2_multiseed_12.json"
SCRATCH = ROOT / "reports" / "growbrain_e2_scratch_final_012.json"
REEVAL = ROOT / "reports" / "growbrain_e2_true_stage_reeval.json"


def _cell(value: float | None) -> str:
    if value is None:
        return "unscored"
    return repr(value)


def _print_stage_table(title: str, rows: list[dict]) -> None:
    print(title)
    print("arm\tseed\tprobe0_acc\tprobe1_acc\tprobe2_acc\tprobe3_acc\thandwritten\tunscored")
    comparable = 0
    for row in rows:
        probes = stage_probe_accs(row)
        if probes is None:
            continue
        comparable += 1
        seed = row.get("seed")
        hw = ",".join(f"{key}={_cell(value)}" for key, value in handwritten_accs(row).items())
        missing = ",".join(str(stage) for stage in unscored_stage_probes(row)) or "-"
        print(
            f"{row.get('arm')}\t{seed}\t"
            f"{_cell(probes['probe0_acc'])}\t{_cell(probes['probe1_acc'])}\t"
            f"{_cell(probes['probe2_acc'])}\t{_cell(probes['probe3_acc'])}\t"
            f"{hw}\t{missing}"
        )
    print(f"comparable_rows\t{comparable}")
    print()


def _print_pseudo_table(title: str, rows: list[dict]) -> None:
    print(title)
    print("arm\tseed\tprobe0_acc\tprobe1_acc\tprobe2_acc\tprobe3_acc\tnote")
    count = 0
    for row in rows:
        if not is_pseudo_stage_probe_row(row):
            continue
        count += 1
        print(
            f"{row.get('arm')}\t{row.get('seed')}\t"
            f"{_cell(row.get('probe0_acc'))}\t{_cell(row.get('probe1_acc'))}\t"
            f"{_cell(row.get('probe2_acc'))}\t{_cell(row.get('probe3_acc'))}\t"
            "shuffled pseudo-stages; not a stage ladder"
        )
    print(f"pseudo_rows\t{count}")
    print()


def _print_reeval(rows: list[dict]) -> None:
    print("true-stage re-evaluation")
    print("arm\tseed\tprobe0_acc\tprobe1_acc\tprobe2_acc\tprobe3_acc\tdata_seed\tdevice\tunscored")
    comparable = 0
    for row in rows:
        probes = stage_probe_accs(row)
        if probes is None:
            continue
        comparable += 1
        missing = ",".join(str(stage) for stage in unscored_stage_probes(row)) or "-"
        print(
            f"{row.get('arm')}\t{row.get('seed')}\t"
            f"{_cell(probes['probe0_acc'])}\t{_cell(probes['probe1_acc'])}\t"
            f"{_cell(probes['probe2_acc'])}\t{_cell(probes['probe3_acc'])}\t"
            f"{row.get('data_seed')}\t{row.get('device')}\t{missing}"
        )
    print(f"comparable_rows\t{comparable}")
    print()
    print("true-stage re-evaluation arm means")
    print("arm\tn_seeds\tprobe0\tprobe1\tprobe2\tprobe3\tmean")
    means = arm_stage_probe_means(rows)
    ordered = sorted(means.items(), key=lambda item: float(item[1]["mean"]), reverse=True)
    for arm, stats in ordered:
        print(
            f"{arm}\t{stats['n_seeds']}\t"
            f"{_cell(float(stats['probe0']))}\t{_cell(float(stats['probe1']))}\t"
            f"{_cell(float(stats['probe2']))}\t{_cell(float(stats['probe3']))}\t"
            f"{_cell(float(stats['mean']))}"
        )
    print()


def _scratch_matches_training(reeval_rows: list[dict], scratch_rows: list[dict]) -> bool:
    by_seed = {row.get("seed"): row for row in reeval_rows if row.get("arm") == "scratch_final"}
    if set(by_seed) != {row.get("seed") for row in scratch_rows}:
        return False
    for row in scratch_rows:
        other = by_seed[row.get("seed")]
        left = stage_probe_accs(row)
        right = stage_probe_accs(other)
        if left is None or right is None or left != right:
            return False
    return True


def main() -> int:
    multiseed = json.loads(MULTISEED.read_text(encoding="utf-8"))
    scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))["results"]
    reeval = json.loads(REEVAL.read_text(encoding="utf-8"))
    _print_stage_table("multi-seed true stage probes", multiseed)
    _print_pseudo_table("multi-seed pseudo-stage probes", multiseed)
    _print_stage_table("scratch_final true stage probes", scratch)
    _print_pseudo_table("scratch_final pseudo-stage probes", scratch)
    _print_reeval(reeval["rows"])
    matched = _scratch_matches_training(reeval["rows"], scratch)
    print(f"scratch_final_matches_training_probes\t{str(matched).lower()}")
    print(
        "random_order training probe0-3 are shuffled pseudo-stages "
        "and are not the true-stage re-evaluation columns"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
