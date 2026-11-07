"""Print comparable E2 stage probes and, separately, pseudo-stage probes.

Reads the committed summaries. Does not train and does not rescore.

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
    handwritten_accs,
    is_pseudo_stage_probe_row,
    stage_probe_accs,
    unscored_stage_probes,
)
MULTISEED = ROOT / "reports" / "growbrain_e2_multiseed_12.json"
SCRATCH = ROOT / "reports" / "growbrain_e2_scratch_final_012.json"


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


def main() -> int:
    multiseed = json.loads(MULTISEED.read_text(encoding="utf-8"))
    scratch = json.loads(SCRATCH.read_text(encoding="utf-8"))["results"]
    _print_stage_table("multi-seed true stage probes", multiseed)
    _print_pseudo_table("multi-seed pseudo-stage probes", multiseed)
    _print_stage_table("scratch_final true stage probes", scratch)
    _print_pseudo_table("scratch_final pseudo-stage probes", scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
