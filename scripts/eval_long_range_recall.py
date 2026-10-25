"""Run the locked long-range recall held-out protocol.

Writes ``reports/long_range_recall.json``,
``reports/long_range_recall_seeds.json`` (item-level per-seed raw),
then ``reports/long_range_recall.md`` from that JSON, so the criteria
stay the ones in ``docs/long_range_recall_prereg.md``.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidal.long_range_recall_data import (  # noqa: E402
    CONDITIONS,
    N_VALUES,
    PROTOCOL,
    SEEDS,
    bank_sha256,
)
from tidal.long_range_recall_eval import render_markdown, run, seed_report  # noqa: E402

REPORT_DIR = ROOT / "reports"
JSON_PATH = REPORT_DIR / "long_range_recall.json"
SEEDS_PATH = REPORT_DIR / "long_range_recall_seeds.json"
MD_PATH = REPORT_DIR / "long_range_recall.md"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def main() -> int:
    started = time.perf_counter()
    summary = run(progress=lambda msg: print(msg, flush=True))
    summary["elapsed_seconds"] = float(time.perf_counter() - started)
    summary["git"] = _git_sha()
    summary["bank_sha256"] = bank_sha256()
    summary["protocol"] = PROTOCOL
    summary["seeds"] = list(SEEDS)
    summary["n_values"] = list(N_VALUES)
    summary["condition_names"] = list(CONDITIONS)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    SEEDS_PATH.write_text(
        json.dumps(seed_report(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    MD_PATH.write_text(render_markdown(summary), encoding="utf-8")
    print(JSON_PATH)
    print(SEEDS_PATH)
    print(MD_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
